"""Authorised read/write operations for a person's understanding profile.

This service is the only place that decides whether a profile change is
allowed, and the only place that assigns revision numbers. Callers never write
the profile tables directly.

Rules enforced here (V0.7):

- A profile exists only while the person is ACTIVE and has granted
  ``allow_persistence``. Withdrawing persistence purges the long-term profile;
  revoking the person does the same. Neither touches the person row itself.
- Creating a profile never grants a consent flag. Learning additionally needs
  ``allow_memory_learning``.
- Explicit self-report and user correction update the effective field
  immediately; an inference only ever becomes a candidate until there is
  corroborating evidence or an explicit confirmation.
- Every accepted change bumps the profile revision and appends a revision row.
  A user correction outranks an earlier inference and rejects its pending
  candidates, so the old wrong value is no longer used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from companion.domain.enums import RegistrationStatus
from companion.domain.models import utcnow
from companion.people.profile_compiler import compile_profile_context
from companion.people.profile_models import (
    CandidateStatus,
    LIST_FIELDS,
    MAX_LIST_ITEMS,
    PersonInteractionProfile,
    ProfileCandidate,
    ProfileEntry,
    ProfileEntryStatus,
    ProfileField,
    ProfileFieldStatus,
    ProfileFieldStorage,
    ProfileMetadata,
    ProfileSource,
    profile_field_spec,
    validate_field_value,
)
from companion.people.profile_refinement import (
    FullProfileSnapshot,
    ProfileApplyResult,
    ProfileChangeProposal,
    ProfileReviewExecution,
    ProfileReviewResult,
    SnapshotEntry,
    ValidatedChange,
    is_strict_field,
    validate_change_targets,
)
from companion.people.profile_repository import PersonProfileRepository
from companion.storage.repositories import AuditRepository, PersonRepository

MAX_REFINEMENT_CHANGES = 5
_IDENTITY_LABELS = {
    "identity.name": "name",
    "identity.preferred_name": "preferred_name",
    "identity.gender_identity": "gender_identity",
    "identity.pronouns": "pronouns",
}
_SENSITIVE_CONTENT = re.compile(
    r"(健康|疾病|病历|诊断|医院|用药|药物|抑郁|焦虑症|"
    r"银行|信用卡|贷款|工资|收入|存款|密码|"
    r"住址|家庭地址|身份证|护照号)"
)

_SENTINEL = object()


class ProfileUnauthorized(Exception):
    """Raised when a profile write is not covered by the person's consent."""


class _TrustedConfirmationToken:
    """Opaque marker for the explicit, user-initiated confirmation path.

    A confirmation is not authorised by a caller-supplied string such as
    ``actor="user"``. The trusted CLI (or an administrator) must present this
    token, or an explicit admin identity. The background refinement path never
    has it, so it can never confirm its own pending proposal.
    """

    __slots__ = ()


TRUSTED_CONFIRMATION = _TrustedConfirmationToken()

# V0.9.9: the only fields a single turn may ask the person to confirm. All of
# them must originate in the person's own explicit statement; nothing here is
# ever inferred from context.
SELF_REPORT_FIELDS: frozenset[str] = frozenset(
    {
        "identity.name",
        "identity.preferred_name",
        "identity.nicknames",
        "communication.address_preference",
    }
)
_ENTRY_SELF_REPORT_FIELDS = frozenset(
    {
        "identity.name",
        "identity.preferred_name",
        "identity.nicknames",
    }
)


class ProfileProposalStale(ValueError):
    """A pending proposal no longer matches the profile it was based on.

    The candidate is left for re-review; no entry, revision or candidate is
    silently overwritten. Callers see an explicit ``STALE_PROPOSAL`` status.
    """

    status = "STALE_PROPOSAL"

    def __init__(
        self,
        candidate_id: str,
        expected_revision: Optional[int],
        current_revision: Optional[int],
        *,
        reason: str = "profile_revision_changed",
    ) -> None:
        super().__init__(reason)
        self.candidate_id = candidate_id
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        self.reason = reason


class ReviewExecutionLost(ValueError):
    """The review execution no longer owns the profile write slot."""

    def __init__(self, reason: str = "review_execution_lost") -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class ProfileInvalidationResult:
    person_id: str
    event_id: Optional[str] = None
    memory_id: Optional[str] = None
    deletion_mode: str = "RETRACTION"
    revision: Optional[int] = None
    entries_updated: int = 0
    entries_archived: int = 0
    entries_deleted: int = 0
    candidates_cleaned: int = 0
    receipts_cleaned: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.entries_updated
            or self.entries_archived
            or self.entries_deleted
            or self.candidates_cleaned
            or self.receipts_cleaned
        )


def _dedupe(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


class PersonProfileService:
    def __init__(
        self,
        repo: PersonProfileRepository,
        persons: PersonRepository,
        *,
        audit: Optional[AuditRepository] = None,
    ) -> None:
        self.repo = repo
        self.persons = persons
        self.audit = audit

    # -- authorisation -----------------------------------------------------

    async def can_persist(self, person_id: str) -> bool:
        person = await self.persons.get(person_id)
        if person is None or person.registration_status != RegistrationStatus.ACTIVE:
            return False
        consent = await self.persons.get_consent(person_id)
        return bool(consent and consent.allow_persistence)

    async def can_learn(self, person_id: str) -> bool:
        if not await self.can_persist(person_id):
            return False
        consent = await self.persons.get_consent(person_id)
        return bool(consent and consent.allow_memory_learning)

    async def can_use_external_ai(self, person_id: str) -> bool:
        """Whether this person's profile may be sent to an external model.

        Local learning is not permission to upload private context: the
        external-AI grant is checked independently and re-checked at apply
        time, after the model has already run.
        """
        if not await self.can_persist(person_id):
            return False
        consent = await self.persons.get_consent(person_id)
        return bool(consent and consent.allow_external_ai)

    # -- reads -------------------------------------------------------------

    async def get(self, person_id: str) -> Optional[PersonInteractionProfile]:
        row = await self.repo.get(person_id)
        if row is None:
            return None
        person = await self.persons.get(person_id)
        fields = {
            item["field_key"]: self._field_model(item)
            for item in await self.repo.list_fields(person_id)
        }
        entries = self._group_entries(await self.repo.list_entries(person_id))
        candidates = [
            self._candidate_model(item)
            for item in await self.repo.list_candidates(person_id)
        ]
        return PersonInteractionProfile(
            person_id=person_id,
            display_name=person.display_name if person is not None else "unknown",
            relationship_tier=row["relationship_tier"] or "STRANGER",
            fields=fields,
            entries=entries,
            identity=self._identity_view(entries),
            relationship=self._relationship_view(
                entries, row["relationship_tier"] or "STRANGER"
            ),
            candidates=candidates,
            compact_summary=row["compact_summary"] or "",
            metadata=ProfileMetadata(
                revision=int(row["revision"] or 1),
                created_at=row["created_at"] and _parse(row["created_at"]),
                updated_at=row["updated_at"] and _parse(row["updated_at"]),
                last_reviewed_at=row["last_reviewed_at"]
                and _parse(row["last_reviewed_at"]),
                updated_by=row["updated_by"] or "system",
            ),
        )

    async def get_full_snapshot(
        self, person_id: str
    ) -> Optional[FullProfileSnapshot]:
        """Read everything one review needs, from one consistent DB view.

        The whole profile is read at the program layer — the model is not given
        the Speaker's 600-character budget. The read transaction keeps the row,
        revision, entries, fields and pending candidates from the same instant.
        """
        if not await self.can_persist(person_id):
            return None
        async with self.repo.db.transaction():
            row = await self.repo.get(person_id)
            if row is None:
                return None
            person = await self.persons.get(person_id)
            fields = {
                item["field_key"]: item
                for item in await self.repo.list_fields(person_id)
            }
            entries = self._group_entries(
                await self.repo.list_entries(
                    person_id,
                    statuses=(
                        ProfileEntryStatus.ACTIVE.value,
                        ProfileEntryStatus.TENTATIVE.value,
                    ),
                )
            )
            candidates = await self.repo.list_candidates(person_id, status=None)
            pending = [
                {
                    "candidate_id": item["candidate_id"],
                    "field_key": item["field_key"],
                    "operation": item.get("operation") or "ADD",
                    "proposed_content": item["candidate_value"],
                    "target_entry_ids": item.get("target_entry_ids") or [],
                    "status": item["status"],
                    "review_status": item.get("review_status") or "PENDING",
                    "confidence": item["confidence"],
                    "expected_revision": item.get("expected_revision"),
                    "requires_confirmation": bool(
                        item.get("requires_confirmation")
                    ),
                }
                for item in candidates
                if item["status"] in (
                    CandidateStatus.PENDING.value,
                    CandidateStatus.NEEDS_REVIEW.value,
                )
            ]
            tier = row["relationship_tier"] or "STRANGER"
            return FullProfileSnapshot(
                person_id=person_id,
                revision=int(row["revision"] or 1),
                display_name=(
                    person.display_name if person is not None else "unknown"
                ),
                identity=self._identity_view(entries),
                relationship=self._relationship_view(entries, tier),
                communication=self._category_view(entries, fields, "communication"),
                personality=self._category_view(entries, fields, "personality"),
                interests=self._category_view(entries, fields, "interests"),
                knowledge=self._category_view(entries, fields, "knowledge"),
                background=self._category_view(entries, fields, "background"),
                projects=self._category_view(entries, fields, "projects"),
                goals=self._category_view(entries, fields, "goals"),
                habits=self._category_view(entries, fields, "habits"),
                values=self._category_view(entries, fields, "values"),
                interaction=self._category_view(entries, fields, "interaction"),
                entries={
                    key: [self._snapshot_entry(entry) for entry in items]
                    for key, items in entries.items()
                },
                legacy_fields={
                    key: value["value"] for key, value in fields.items()
                },
                pending_candidates=pending,
                compact_summary=row["compact_summary"] or "",
                updated_at=row["updated_at"],
                last_reviewed_at=row["last_reviewed_at"],
            )

    # -- lifecycle ---------------------------------------------------------

    async def ensure_profile(
        self,
        person_id: str,
        *,
        relationship_tier: str = "STRANGER",
        actor: str = "system",
        enforce_authorization: bool = True,
    ) -> bool:
        """Create the empty profile if persistence is authorised.

        Safe to call repeatedly (idempotent). Returns True when a row was
        created by this call.
        """
        if enforce_authorization and not await self.can_persist(person_id):
            return False
        created = await self.repo.ensure(
            person_id, relationship_tier=relationship_tier, actor=actor
        )
        if created and self.audit is not None:
            await self.audit.record(
                actor=actor,
                action="profile_create",
                entity_type="person_profile",
                entity_id=person_id,
                detail=f"tier={relationship_tier}",
            )
        return created

    async def on_consent_changed(self, person_id: str) -> None:
        """Keep the profile consistent after a consent update."""
        if not await self.can_persist(person_id):
            await self.purge(person_id, reason="persistence_withdrawn")
        else:
            await self.ensure_profile(person_id, actor="consent_change")

    async def on_status_changed(self, person_id: str) -> None:
        person = await self.persons.get(person_id)
        if person is None:
            return
        if person.registration_status == RegistrationStatus.REVOKED:
            await self.purge(person_id, reason="person_revoked")
        elif person.registration_status == RegistrationStatus.ACTIVE:
            await self.ensure_profile(person_id, actor="status_change")

    async def purge(self, person_id: str, *, reason: str) -> None:
        """Remove all long-term profile data for one person.

        Called on revocation or persistence withdrawal. Idempotent.
        """
        await self.repo.purge(person_id)
        if self.audit is not None:
            await self.audit.record(
                actor="system",
                action="profile_purge",
                entity_type="person_profile",
                entity_id=person_id,
                detail=reason,
            )

    async def note_reviewed(self, person_id: str) -> None:
        await self.repo.set_reviewed(person_id)

    # -- explicit updates --------------------------------------------------

    async def set_field(
        self,
        person_id: str,
        field_key: str,
        value: Any,
        *,
        source: ProfileSource | str = ProfileSource.DIRECT_SELF_REPORT,
        confidence: float = 1.0,
        event_id: Optional[str] = None,
        memory_ids: Iterable[str] = (),
        candidate_ids: Iterable[str] = (),
        reason: Optional[str] = None,
        merge: bool = False,
        require_learning: bool = True,
        actor: str = "learning",
    ) -> int:
        """Write an effective field value and record the revision.

        ``source`` must be an explicit-report source (direct self report, user
        correction or admin). Inferences go through :meth:`propose_candidate`.
        """
        source_value = ProfileSource(source).value
        if source_value == ProfileSource.INTERACTION_INFERENCE.value:
            raise ValueError("inference must use propose_candidate")
        if require_learning and not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)
        if not await self.can_persist(person_id):
            raise ProfileUnauthorized(person_id)

        normalised = validate_field_value(field_key, value)
        memory_ids = list(memory_ids)
        candidate_ids = list(candidate_ids)
        event_ids = [event_id] if event_id else []
        async with self.repo.db.transaction():
            row = await self.repo.get(person_id)
            if row is None:
                created = await self.repo.ensure(person_id, actor=actor)
                row = await self.repo.get(person_id)
                if row is None:
                    raise ProfileUnauthorized(person_id)
                if created and self.audit is not None:
                    await self.audit.record(
                        actor=actor,
                        action="profile_create",
                        entity_type="person_profile",
                        entity_id=person_id,
                        detail="implicit",
                    )

            old_field = await self.repo.get_field(person_id, field_key)
            if (
                old_field is not None
                and old_field["status"] == "ACTIVE"
                and old_field["value"] == normalised
                and old_field["source"] == source_value
                and not merge
            ):
                # Repeating the same explicit statement is not a new revision.
                return int(row["revision"] or 1)
            if merge and field_key in LIST_FIELDS:
                previous = []
                if old_field is not None and old_field["status"] == "ACTIVE":
                    previous = old_field["value"] or []
                normalised = validate_field_value(
                    field_key, list(previous) + list(normalised)
                )

            revision = int(row["revision"] or 1) + 1
            await self.repo.upsert_field(
                person_id,
                field_key,
                value=normalised,
                source=source_value,
                confidence=confidence,
                evidence_memory_ids=memory_ids,
                evidence_event_ids=event_ids,
                evidence_candidate_ids=candidate_ids,
                revision=revision,
            )
            await self.repo.touch(person_id, updated_by=actor)
            await self.repo.insert_revision(
                person_id=person_id,
                revision=revision,
                changes={
                    "field_key": field_key,
                    "old": old_field["value"] if old_field else None,
                    "new": normalised,
                    "source": source_value,
                },
                changed_by=actor,
                source_event_id=event_id,
                reason=reason,
            )
            await self._record_evidence(
                person_id,
                field_key,
                memory_ids=memory_ids,
                event_ids=event_ids,
            )
            await self._refresh_summary(person_id)
        await self._audit_field(
            actor, "profile_field_set", person_id, field_key, source_value
        )
        return revision

    async def correct_field(
        self,
        person_id: str,
        field_key: str,
        *,
        value: Any = _SENTINEL,
        revoke: bool = False,
        event_id: Optional[str] = None,
        reason: Optional[str] = None,
        actor: str = "user",
    ) -> int:
        """Apply an explicit correction, replacing or retiring the old value.

        Pending candidate values for the same field are rejected so an earlier
        inference cannot come back after the person corrected it.
        """
        if revoke and value is not _SENTINEL:
            raise ValueError("pass either value or revoke, not both")
        if not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)

        async with self.repo.db.transaction():
            row = await self.repo.get(person_id)
            if row is None:
                raise ProfileUnauthorized(person_id)
            old_field = await self.repo.get_field(person_id, field_key)
            for candidate in await self.repo.find_pending(person_id, field_key):
                await self.repo.set_candidate_status(
                    candidate["candidate_id"],
                    CandidateStatus.REJECTED.value,
                    reason="user_correction",
                )
            revision = int(row["revision"] or 1) + 1
            if revoke:
                if old_field is not None:
                    await self.repo.retire_field(
                        person_id, field_key, revision=revision
                    )
                changes = {
                    "field_key": field_key,
                    "old": old_field["value"] if old_field else None,
                    "new": None,
                    "retired": True,
                    "source": ProfileSource.USER_CORRECTION.value,
                }
            else:
                normalised = validate_field_value(field_key, value)
                await self.repo.upsert_field(
                    person_id,
                    field_key,
                    value=normalised,
                    source=ProfileSource.USER_CORRECTION.value,
                    confidence=1.0,
                    evidence_memory_ids=[],
                    evidence_event_ids=([event_id] if event_id else []),
                    evidence_candidate_ids=[],
                    revision=revision,
                )
                changes = {
                    "field_key": field_key,
                    "old": old_field["value"] if old_field else None,
                    "new": normalised,
                    "source": ProfileSource.USER_CORRECTION.value,
                    "corrects": old_field["source"] if old_field else None,
                }
            await self.repo.touch(person_id, updated_by=actor)
            await self.repo.insert_revision(
                person_id=person_id,
                revision=revision,
                changes=changes,
                changed_by=actor,
                source_event_id=event_id,
                reason=reason or "user_correction",
            )
            if event_id:
                await self._record_evidence(
                    person_id,
                    field_key,
                    memory_ids=[],
                    event_ids=[event_id],
                )
            await self._refresh_summary(person_id)
        await self._audit_field(
            actor,
            "profile_field_corrected" if not revoke else "profile_field_retired",
            person_id,
            field_key,
            ProfileSource.USER_CORRECTION.value,
        )
        return revision

    # -- candidates --------------------------------------------------------

    async def propose_candidate(
        self,
        person_id: str,
        field_key: str,
        value: Any,
        *,
        source: ProfileSource | str = ProfileSource.INTERACTION_INFERENCE,
        confidence: float = 0.5,
        event_id: Optional[str] = None,
        memory_ids: Iterable[str] = (),
        candidate_ids: Iterable[str] = (),
        reason: Optional[str] = None,
        require_learning: bool = True,
        actor: str = "learning",
    ) -> Optional[str]:
        """Record an unconfirmed inference. Never changes the effective field."""
        if require_learning and not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)
        if not await self.can_persist(person_id):
            raise ProfileUnauthorized(person_id)
        normalised = validate_field_value(field_key, value)
        memory_ids = list(memory_ids)
        candidate_ids = list(candidate_ids)
        event_ids = [event_id] if event_id else []

        current = await self.repo.get_field(person_id, field_key)
        if (
            current is not None
            and current["status"] == "ACTIVE"
            and current["value"] == normalised
        ):
            return None

        async with self.repo.db.transaction():
            await self.repo.ensure(person_id, actor=actor)
            pending = await self.repo.find_pending(person_id, field_key)
            same = next(
                (
                    item
                    for item in pending
                    if item["candidate_value"] == normalised
                ),
                None,
            )
            if same is not None:
                await self.repo.add_candidate_evidence(
                    same["candidate_id"],
                    memory_ids=memory_ids,
                    event_ids=event_ids,
                    candidate_ids=candidate_ids,
                )
                await self._record_evidence(
                    person_id,
                    field_key,
                    memory_ids=memory_ids,
                    event_ids=event_ids,
                    candidate_id=same["candidate_id"],
                )
                return same["candidate_id"]

            explicit = ProfileSource(source).value in (
                ProfileSource.DIRECT_SELF_REPORT.value,
                ProfileSource.USER_CORRECTION.value,
                ProfileSource.ADMIN.value,
            )
            for item in pending:
                if explicit:
                    # A new explicit statement genuinely replaces an earlier
                    # *pending inference*. This is the only superseding case.
                    await self.repo.set_candidate_status(
                        item["candidate_id"],
                        CandidateStatus.SUPERSEDED.value,
                        reason="superseded_by_explicit_report",
                    )
                elif field_key in LIST_FIELDS:
                    # Multi-value understanding coexists: "robot dog" and
                    # "jazz" are both true and neither replaces the other.
                    continue
                else:
                    # Two incompatible inferred values for one scalar field are
                    # a conflict for a person to review, not a silent winner.
                    await self.repo.set_candidate_status(
                        item["candidate_id"],
                        CandidateStatus.NEEDS_REVIEW.value,
                        reason="conflicts_with_pending_candidate",
                    )
            candidate_id = await self.repo.insert_candidate(
                person_id=person_id,
                field_key=field_key,
                value=normalised,
                source=ProfileSource(source).value,
                confidence=confidence,
                evidence_memory_ids=memory_ids,
                evidence_event_ids=event_ids,
                evidence_candidate_ids=candidate_ids,
                reason=reason,
            )
            await self._record_evidence(
                person_id,
                field_key,
                memory_ids=memory_ids,
                event_ids=event_ids,
                candidate_id=candidate_id,
            )
        return candidate_id

    async def confirm_candidate(
        self,
        candidate_id: str,
        *,
        reason: Optional[str] = None,
        actor: str = "learning",
    ) -> int:
        candidate = await self.repo.get_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate {candidate_id!r}")
        if candidate["status"] != CandidateStatus.PENDING.value:
            raise ValueError(
                f"candidate {candidate_id!r} is {candidate['status']}"
            )
        person_id = candidate["person_id"]
        if not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)

        async with self.repo.db.transaction():
            row = await self.repo.get(person_id)
            if row is None:
                raise ProfileUnauthorized(person_id)
            old_field = await self.repo.get_field(person_id, candidate["field_key"])
            value = candidate["candidate_value"]
            if candidate["field_key"] in LIST_FIELDS and old_field is not None:
                previous = (
                    old_field["value"]
                    if old_field["status"] == "ACTIVE"
                    else []
                ) or []
                value = validate_field_value(
                    candidate["field_key"], list(previous) + list(value)
                )
            revision = int(row["revision"] or 1) + 1
            await self.repo.upsert_field(
                person_id,
                candidate["field_key"],
                value=value,
                source=candidate["source"],
                confidence=candidate["confidence"],
                evidence_memory_ids=candidate["evidence_memory_ids"],
                evidence_event_ids=candidate["evidence_event_ids"],
                evidence_candidate_ids=(
                    candidate["evidence_candidate_ids"] + [candidate_id]
                ),
                revision=revision,
            )
            await self.repo.set_candidate_status(
                candidate_id,
                CandidateStatus.CONFIRMED.value,
                reason=reason,
            )
            await self.repo.touch(person_id, updated_by=actor)
            await self.repo.insert_revision(
                person_id=person_id,
                revision=revision,
                changes={
                    "field_key": candidate["field_key"],
                    "old": old_field["value"] if old_field else None,
                    "new": value,
                    "source": candidate["source"],
                    "candidate_id": candidate_id,
                },
                changed_by=actor,
                source_event_id=None,
                reason=reason or "candidate_confirmed",
            )
            await self._refresh_summary(person_id)
        await self._audit_field(
            actor,
            "profile_candidate_confirmed",
            person_id,
            candidate["field_key"],
            candidate["source"],
        )
        return revision

    async def reject_candidate(
        self,
        candidate_id: str,
        *,
        reason: str = "rejected",
        actor: str = "learning",
    ) -> None:
        candidate = await self.repo.get_candidate(candidate_id)
        if candidate is None:
            return
        if candidate["status"] != CandidateStatus.PENDING.value:
            return
        await self.repo.set_candidate_status(
            candidate_id, CandidateStatus.REJECTED.value, reason=reason
        )

    # -- V0.9.9 single-turn self-report confirmation ----------------------

    async def propose_self_report(
        self,
        person_id: str,
        field_key: str,
        value: Any,
        *,
        event_id: str,
        quote: str = "",
        reason: Optional[str] = None,
        actor: str = "profile_confirmation",
    ) -> Optional[str]:
        """Record one explicit self-report as a *pending* candidate.

        No effective profile field is written here. The candidate is marked
        ``DIRECT_SELF_REPORT`` so the later confirmation (or the silence rule)
        can apply it without pretending the person confirmed anything.
        """
        if field_key not in SELF_REPORT_FIELDS:
            raise ValueError("field_not_self_reportable")
        if not event_id:
            raise ValueError("no_evidence")
        if not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)
        normalised = self._normalise_self_report(field_key, value)
        if not normalised:
            return None

        async with self.repo.db.transaction():
            await self.repo.ensure(person_id, actor=actor)
            row = await self.repo.get(person_id)
            if row is None:
                raise ProfileUnauthorized(person_id)

            if field_key in _ENTRY_SELF_REPORT_FIELDS:
                effective = await self.repo.list_entries(
                    person_id,
                    statuses=(
                        ProfileEntryStatus.ACTIVE.value,
                        ProfileEntryStatus.TENTATIVE.value,
                    ),
                    field_key=field_key,
                )
                if any(item["content"] == normalised for item in effective):
                    return None
            else:
                old_field = await self.repo.get_field(person_id, field_key)
                if (
                    old_field is not None
                    and old_field["status"] == "ACTIVE"
                    and old_field["value"] == normalised
                ):
                    return None

            pending = await self.repo.find_pending(person_id, field_key)
            for item in pending:
                if (
                    item["candidate_value"] == normalised
                    and (item.get("operation") or "ADD") == "ADD"
                ):
                    # A background Profile Review may have proposed the same
                    # value first. One pending candidate is shared; the direct
                    # statement upgrades its source and joins its evidence.
                    if item["source"] != ProfileSource.DIRECT_SELF_REPORT.value:
                        await self.repo.set_candidate_source(
                            item["candidate_id"],
                            source=ProfileSource.DIRECT_SELF_REPORT.value,
                            confidence=1.0,
                        )
                    await self.repo.add_candidate_evidence(
                        item["candidate_id"],
                        memory_ids=[],
                        event_ids=[event_id],
                        candidate_ids=[],
                    )
                    return item["candidate_id"]
            for item in pending:
                # A new explicit statement replaces an earlier pending
                # proposal for the same field, whether inferred or explicit.
                await self.repo.set_candidate_status(
                    item["candidate_id"],
                    CandidateStatus.SUPERSEDED.value,
                    reason="superseded_by_self_report",
                )
            for item in await self.repo.list_candidates(
                person_id, status=CandidateStatus.NEEDS_REVIEW.value
            ):
                if item["field_key"] == field_key:
                    await self.repo.set_candidate_status(
                        item["candidate_id"],
                        CandidateStatus.SUPERSEDED.value,
                        reason="superseded_by_self_report",
                    )

            candidate_id = await self.repo.insert_candidate(
                person_id=person_id,
                field_key=field_key,
                value=normalised,
                source=ProfileSource.DIRECT_SELF_REPORT.value,
                confidence=1.0,
                evidence_memory_ids=[],
                evidence_event_ids=[event_id],
                evidence_candidate_ids=[],
                reason=reason or "direct_self_report",
                operation="ADD",
                target_entry_ids=[],
                # Not bound to a revision: the confirmation may be delivered
                # in a later reply, and silence is deliberately not treated as
                # a stale-proposal confirmation.
                expected_revision=None,
                requires_confirmation=True,
                review_status="PENDING",
            )
            await self._record_evidence(
                person_id,
                field_key,
                memory_ids=[],
                event_ids=[event_id],
                candidate_id=candidate_id,
            )
        return candidate_id

    async def accept_self_report(
        self,
        candidate_id: str,
        *,
        confirmed: bool,
        event_id: Optional[str] = None,
        actor: str = "profile_confirmation",
        reason: Optional[str] = None,
    ) -> Optional[int]:
        """Apply one validated ``DIRECT_SELF_REPORT`` candidate.

        ``confirmed=True`` is used only when the person explicitly agreed to
        the confirmation question. ``confirmed=False`` is the silent-accept
        rule: the candidate came from a clear first-person statement and the
        person did not answer, so it is applied with its real source and
        *without* being marked as an explicit confirmation.

        This is the trusted strict-field write boundary. It never presents the
        opaque ``TRUSTED_CONFIRMATION`` token and never fabricates an actor.
        """
        candidate = await self.repo.get_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate {candidate_id!r}")
        field_key = candidate["field_key"]
        if field_key not in SELF_REPORT_FIELDS:
            raise ValueError("field_not_self_reportable")
        if candidate["source"] != ProfileSource.DIRECT_SELF_REPORT.value:
            raise ValueError("not_a_direct_self_report")
        if candidate["status"] == CandidateStatus.CONFIRMED.value:
            return None
        if candidate["status"] != CandidateStatus.PENDING.value:
            raise ValueError(f"candidate {candidate_id!r} is {candidate['status']}")

        person_id = candidate["person_id"]
        if not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)
        value = candidate["candidate_value"]
        evidence_events = list(
            dict.fromkeys(
                list(candidate["evidence_event_ids"])
                + ([event_id] if event_id else [])
            )
        )

        async with self.repo.db.transaction():
            fresh = await self.repo.get_candidate(candidate_id)
            if fresh is None or fresh["status"] != CandidateStatus.PENDING.value:
                return None
            row = await self.repo.get(person_id)
            if row is None:
                raise ProfileUnauthorized(person_id)

            if field_key in _ENTRY_SELF_REPORT_FIELDS:
                effective = await self.repo.list_entries(
                    person_id,
                    statuses=(
                        ProfileEntryStatus.ACTIVE.value,
                        ProfileEntryStatus.TENTATIVE.value,
                    ),
                    field_key=field_key,
                )
                old_value = [item["content"] for item in effective]
                if not confirmed and any(
                    item["content"] != value for item in effective
                ):
                    # An existing explicitly confirmed identity is never
                    # overwritten by silence. An explicit confirmation may
                    # replace it; that is the only path here.
                    await self.repo.set_candidate_status(
                        candidate_id,
                        CandidateStatus.NEEDS_REVIEW.value,
                        reason="conflicts_with_confirmed_identity",
                    )
                    return None

            revision = int(row["revision"] or 1) + 1
            entry_id: Optional[str] = None
            if field_key in _ENTRY_SELF_REPORT_FIELDS:
                proposal = ProfileChangeProposal(
                    operation="ADD",
                    field_key=field_key,
                    target_entry_ids=[],
                    proposed_content=str(value),
                    evidence_event_ids=evidence_events,
                    confidence=1.0,
                    requires_confirmation=True,
                )
                change = ValidatedChange(proposal=proposal, auto_apply=False)
                applied = await self._apply_change_locked(
                    person_id,
                    change,
                    revision=revision,
                    source_event_id=event_id,
                    confirmed=confirmed,
                    actor=actor,
                    trusted=True,
                    source_override=ProfileSource.DIRECT_SELF_REPORT.value,
                )
                entry_id = applied.get("entry_id")
                new_value: Any = applied.get("after")
            else:
                validated = validate_field_value(field_key, value)
                old_field = await self.repo.get_field(person_id, field_key)
                old_value = old_field["value"] if old_field else None
                await self.repo.upsert_field(
                    person_id,
                    field_key,
                    value=validated,
                    source=ProfileSource.DIRECT_SELF_REPORT.value,
                    confidence=1.0,
                    evidence_memory_ids=[],
                    evidence_event_ids=evidence_events,
                    evidence_candidate_ids=[candidate_id],
                    revision=revision,
                )
                await self._record_evidence(
                    person_id,
                    field_key,
                    memory_ids=[],
                    event_ids=evidence_events,
                    candidate_id=candidate_id,
                )
                new_value = validated

            await self.repo.touch(person_id, updated_by=actor)
            await self.repo.insert_revision(
                person_id=person_id,
                revision=revision,
                changes={
                    "field_key": field_key,
                    "old": old_value,
                    "new": new_value,
                    "source": ProfileSource.DIRECT_SELF_REPORT.value,
                    "candidate_id": candidate_id,
                    "entry_id": entry_id,
                    "confirmed": bool(confirmed),
                },
                changed_by=actor,
                source_event_id=event_id,
                reason=reason
                or (
                    "confirmed_by_person"
                    if confirmed
                    else "accepted_direct_self_report"
                ),
            )
            await self.repo.set_candidate_status(
                candidate_id,
                CandidateStatus.CONFIRMED.value,
                reason=reason
                or (
                    "confirmed_by_person"
                    if confirmed
                    else "accepted_direct_self_report"
                ),
            )
            await self._refresh_summary(person_id)

        await self._audit_field(
            actor,
            "profile_self_report_confirmed"
            if confirmed
            else "profile_self_report_accepted",
            person_id,
            field_key,
            ProfileSource.DIRECT_SELF_REPORT.value,
        )
        return revision

    def _normalise_self_report(self, field_key: str, value: Any) -> str:
        text = " ".join(str(value or "").split()).strip()
        if field_key == "communication.address_preference":
            return validate_field_value(field_key, text)
        spec = profile_field_spec(field_key)
        limit = spec.max_chars if spec is not None else 80
        return text[:limit]

    # -- V0.9.7 refinement -------------------------------------------------

    async def apply_refinement(
        self,
        person_id: str,
        decision: Any,
        *,
        source_event_id: Optional[str] = None,
        actor: str = "profile_refinement",
        execution: Optional[ProfileReviewExecution] = None,
    ) -> ProfileApplyResult:
        """Validate and apply one review's proposals atomically.

        The model never reaches this method directly. ``decision`` carries
        proposals that the engine already checked against the *existing*
        snapshot; this method re-checks authorization, revision, ownership,
        evidence and limits, then commits everything in one SQLite
        transaction. A stale ``base_revision`` is reported, never overwritten.

        When ``execution`` is supplied (the background worker path), the final
        ``BEGIN IMMEDIATE`` transaction also proves that this exact execution
        still owns the person lease and the ``RUNNING`` receipt. The check is
        repeated just before the revision row is inserted, so a lease that
        expired — or a receipt taken over by a newer worker — aborts the whole
        transaction instead of committing a partial change.
        """
        status = str(getattr(decision, "status", "") or "")
        base_revision = int(getattr(decision, "base_revision", 0) or 0)
        source_event_id = source_event_id or getattr(
            decision, "source_event_id", None
        )
        reason = str(getattr(decision, "reason", "") or "")
        if status == "NO_PROFILE":
            return ProfileApplyResult(
                person_id=person_id,
                source_event_id=source_event_id,
                status="NO_PROFILE",
                base_revision=base_revision,
                reason=reason,
            )
        if not await self.can_learn(person_id):
            return ProfileApplyResult(
                person_id=person_id,
                source_event_id=source_event_id,
                status="UNAUTHORIZED",
                base_revision=base_revision,
                reason="learning_or_persistence_not_authorized",
            )
        if not await self.can_use_external_ai(person_id):
            return ProfileApplyResult(
                person_id=person_id,
                source_event_id=source_event_id,
                status="UNAUTHORIZED",
                base_revision=base_revision,
                reason="external_ai_not_authorized",
            )

        # Idempotency: a retried job whose first attempt committed but whose
        # receipt update failed must not add a second revision or duplicate an
        # ADD/MERGE.
        if source_event_id:
            for revision in await self.repo.list_revisions(person_id):
                if (
                    revision["source_event_id"] == source_event_id
                    and revision["changed_by"] == "profile_refinement"
                ):
                    return ProfileApplyResult(
                        person_id=person_id,
                        source_event_id=source_event_id,
                        status="ALREADY_APPLIED",
                        base_revision=base_revision,
                        new_revision=revision["revision"],
                        reason="same_event_already_applied",
                    )

        accepted, pending, rejected = self._classify_changes(decision)
        accepted, late_pending, late_rejected = await self._precheck_changes(
            person_id, accepted
        )
        pending.extend(late_pending)
        rejected.extend(late_rejected)

        if not accepted:
            async with self.repo.db.transaction(immediate=True):
                await self._assert_review_execution_locked(
                    person_id, source_event_id, execution
                )
                row = await self.repo.get(person_id)
                if row is None:
                    return ProfileApplyResult(
                        person_id=person_id,
                        source_event_id=source_event_id,
                        status="UNAUTHORIZED",
                        base_revision=base_revision,
                        reason="no_profile",
                    )
                if int(row["revision"] or 1) != base_revision:
                    return ProfileApplyResult(
                        person_id=person_id,
                        source_event_id=source_event_id,
                        status="REVISION_CONFLICT",
                        base_revision=base_revision,
                        reason="profile_revision_changed",
                    )
                for change in pending:
                    await self._store_pending_candidate(
                        person_id, change, base_revision
                    )
                await self.repo.set_reviewed(person_id)
            return ProfileApplyResult(
                person_id=person_id,
                source_event_id=source_event_id,
                status="PENDING_CONFIRMATION" if pending else "NO_CHANGE",
                base_revision=base_revision,
                pending=[self._pending_summary(change) for change in pending],
                rejected=rejected,
                reason=reason,
            )

        async with self.repo.db.transaction(immediate=True):
            # Believing we still hold the lease is not enough: prove it in the
            # same write-locked transaction that commits the change.
            await self._assert_review_execution_locked(
                person_id, source_event_id, execution
            )
            row = await self.repo.get(person_id)
            if row is None:
                return ProfileApplyResult(
                    person_id=person_id,
                    source_event_id=source_event_id,
                    status="UNAUTHORIZED",
                    base_revision=base_revision,
                    reason="no_profile",
                )
            if int(row["revision"] or 1) != base_revision:
                # Do not just swap in the latest revision: the caller reloads
                # and re-evaluates whether the old proposals still hold.
                return ProfileApplyResult(
                    person_id=person_id,
                    source_event_id=source_event_id,
                    status="REVISION_CONFLICT",
                    base_revision=base_revision,
                    reason="profile_revision_changed",
                )
            new_revision = await self.repo.bump_revision(
                person_id,
                expected_revision=base_revision,
                updated_by="profile_refinement",
                mark_reviewed=True,
            )
            if not new_revision:
                return ProfileApplyResult(
                    person_id=person_id,
                    source_event_id=source_event_id,
                    status="REVISION_CONFLICT",
                    base_revision=base_revision,
                    reason="profile_revision_changed",
                )
            applied: list[dict] = []
            for change in accepted:
                applied.append(
                    await self._apply_change_locked(
                        person_id,
                        change,
                        revision=new_revision,
                        source_event_id=source_event_id,
                        confirmed=False,
                        actor=actor,
                    )
                )
            stored_pending = []
            for change in pending:
                await self._store_pending_candidate(
                    person_id, change, base_revision
                )
                stored_pending.append(self._pending_summary(change))
            # Last check before the commit point: a lease that expired while
            # the local checks ran must roll back everything above.
            await self._assert_review_execution_locked(
                person_id, source_event_id, execution
            )
            await self.repo.insert_revision(
                person_id=person_id,
                revision=new_revision,
                changes={
                    "changes": applied,
                    "pending": stored_pending,
                    "rejected": rejected,
                },
                changed_by=actor,
                source_event_id=source_event_id,
                reason=reason or "profile_refinement",
            )
            await self._refresh_summary(person_id)

        await self._audit_field(
            actor,
            "profile_refinement_applied",
            person_id,
            ",".join(sorted({item["field_key"] for item in applied}))[:200],
            ProfileSource.INTERACTION_INFERENCE.value,
        )
        return ProfileApplyResult(
            person_id=person_id,
            source_event_id=source_event_id,
            status="APPLIED" if not stored_pending else "APPLIED_WITH_PENDING",
            base_revision=base_revision,
            new_revision=new_revision,
            applied=applied,
            pending=stored_pending,
            rejected=rejected,
            reason=reason,
        )

    async def _assert_review_execution_locked(
        self,
        person_id: str,
        source_event_id: Optional[str],
        execution: Optional[ProfileReviewExecution],
    ) -> None:
        """Prove this execution may commit, inside the write transaction."""
        if execution is None:
            return
        if (
            execution.person_id != person_id
            or execution.source_event_id != source_event_id
        ):
            raise ReviewExecutionLost("review_execution_scope_mismatch")
        lease = await self.repo.get_review_lease(person_id)
        if (
            lease is None
            or lease["owner_id"] != execution.lease_owner_id
            or lease["expires_at"] < utcnow()
        ):
            raise ReviewExecutionLost("review_lease_lost")
        receipt = await self.repo.get_review_receipt(
            person_id, execution.source_event_id
        )
        if (
            receipt is None
            or receipt["review_id"] != execution.review_id
            or receipt["status"] != "RUNNING"
            or receipt.get("execution_owner_id") != execution.lease_owner_id
        ):
            raise ReviewExecutionLost("review_receipt_not_owned")
        # A permanently deleted event must never reach a profile write.
        cur = await self.persons.db.execute(
            "SELECT 1 FROM utterances WHERE event_id = ?",
            (execution.source_event_id,),
        )
        if await cur.fetchone() is None:
            raise ReviewExecutionLost("source_event_deleted")

    async def confirm_profile_proposal(
        self,
        candidate_id: str,
        *,
        actor: str = "user",
        reason: Optional[str] = None,
        authority: Any = None,
        actor_person_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> int:
        """Apply a pending refinement proposal after explicit confirmation.

        This is the trusted path for strict identity (and other pending)
        changes. It requires either the opaque trusted-confirmation token held
        by the local CLI or an explicit administrator/person identity — a bare
        ``actor="user"`` string is not authorisation.

        The candidate is bound to the revision it was proposed against. If the
        profile has moved, or a target/evidence it cites is no longer valid,
        the result is an explicit ``STALE_PROPOSAL``: the candidate is left for
        re-review and nothing is overwritten, bumped or confirmed.
        """
        candidate = await self.repo.get_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate {candidate_id!r}")
        if candidate["status"] not in (
            CandidateStatus.PENDING.value,
            CandidateStatus.NEEDS_REVIEW.value,
        ):
            raise ValueError(f"candidate {candidate_id!r} is {candidate['status']}")
        spec = profile_field_spec(candidate["field_key"])
        if spec is None or spec.storage is not ProfileFieldStorage.ENTRY:
            raise ValueError("candidate is not an understanding entry proposal")
        person_id = candidate["person_id"]
        if authority is not TRUSTED_CONFIRMATION:
            if not is_admin and (
                not actor_person_id or actor_person_id != person_id
            ):
                raise ProfileUnauthorized(
                    f"{person_id}: confirmation requires a trusted actor"
                )
        if not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)

        expected_revision = candidate.get("expected_revision")
        row = await self.repo.get(person_id)
        if row is None:
            raise ProfileUnauthorized(person_id)
        current_revision = int(row["revision"] or 1)
        if (
            expected_revision is not None
            and int(expected_revision) != current_revision
        ):
            await self._mark_proposal_needs_review(
                candidate_id, "stale_expected_revision"
            )
            raise ProfileProposalStale(
                candidate_id, int(expected_revision), current_revision
            )

        proposal = ProfileChangeProposal(
            operation=candidate.get("operation") or "ADD",
            field_key=candidate["field_key"],
            target_entry_ids=list(candidate.get("target_entry_ids") or []),
            proposed_content=str(candidate["candidate_value"] or ""),
            evidence_event_ids=list(candidate["evidence_event_ids"]),
            confidence=float(candidate["confidence"] or 0.0),
            requires_confirmation=True,
        )
        change = ValidatedChange(proposal=proposal, auto_apply=False)

        try:
            async with self.repo.db.transaction():
                fresh = await self.repo.get(person_id)
                if fresh is None:
                    raise ProfileUnauthorized(person_id)
                fresh_revision = int(fresh["revision"] or 1)
                if fresh_revision != current_revision:
                    raise ProfileProposalStale(
                        candidate_id, expected_revision, fresh_revision
                    )
                # Re-check the entries this proposal points at *now*: the old
                # target may be archived, merged or replaced.
                targets = []
                for entry_id in dict.fromkeys(proposal.target_entry_ids):
                    entry = await self.repo.get_entry(entry_id)
                    if (
                        entry is None
                        or entry["person_id"] != person_id
                        or entry["status"]
                        not in (
                            ProfileEntryStatus.ACTIVE.value,
                            ProfileEntryStatus.TENTATIVE.value,
                        )
                    ):
                        raise ProfileProposalStale(
                            candidate_id,
                            expected_revision,
                            fresh_revision,
                            reason="target_entries_changed",
                        )
                    targets.append(entry)
                validate_change_targets(
                    proposal.operation, proposal.field_key, targets
                )
                await self._assert_evidence_locked(
                    person_id,
                    list(proposal.evidence_event_ids),
                    list(proposal.evidence_quotes),
                )
                revision = await self.repo.bump_revision(
                    person_id,
                    expected_revision=fresh_revision,
                    updated_by=actor,
                    mark_reviewed=True,
                )
                if not revision:
                    raise ProfileProposalStale(
                        candidate_id, expected_revision, fresh_revision
                    )
                applied = await self._apply_change_locked(
                    person_id,
                    change,
                    revision=revision,
                    source_event_id=None,
                    confirmed=True,
                    actor=actor,
                )
                await self.repo.set_candidate_status(
                    candidate_id,
                    CandidateStatus.CONFIRMED.value,
                    reason=reason or "confirmed_by_person",
                )
                await self.repo.insert_revision(
                    person_id=person_id,
                    revision=revision,
                    changes={
                        "changes": [applied],
                        "candidate_id": candidate_id,
                        "confirmed": True,
                    },
                    changed_by=actor,
                    source_event_id=None,
                    reason=reason or "profile_proposal_confirmed",
                )
                await self._refresh_summary(person_id)
        except ProfileProposalStale as stale:
            await self._mark_proposal_needs_review(
                candidate_id, stale.reason
            )
            raise
        except ValueError as exc:
            # Validation is part of the same all-or-nothing boundary: the
            # transaction rolled back, so mark the proposal for re-review in
            # a separate, committed statement.
            await self._mark_proposal_needs_review(candidate_id, str(exc))
            raise ProfileProposalStale(
                candidate_id,
                expected_revision,
                current_revision,
                reason=str(exc) or "invalid_proposal",
            ) from exc

        await self._audit_field(
            actor,
            "profile_proposal_confirmed",
            person_id,
            candidate["field_key"],
            candidate["source"],
        )
        return revision

    async def _mark_proposal_needs_review(
        self, candidate_id: str, reason: str
    ) -> None:
        candidate = await self.repo.get_candidate(candidate_id)
        if candidate is None:
            return
        if candidate["status"] == CandidateStatus.CONFIRMED.value:
            return
        async with self.repo.db.transaction():
            await self.repo.set_candidate_status(
                candidate_id,
                CandidateStatus.NEEDS_REVIEW.value,
                reason=(reason or "stale_proposal")[:200],
            )

    async def reject_profile_proposal(
        self, candidate_id: str, *, reason: str = "rejected_by_person"
    ) -> None:
        candidate = await self.repo.get_candidate(candidate_id)
        if candidate is None:
            return
        if candidate["status"] not in (
            CandidateStatus.PENDING.value,
            CandidateStatus.NEEDS_REVIEW.value,
        ):
            return
        await self.repo.set_candidate_status(
            candidate_id, CandidateStatus.REJECTED.value, reason=reason
        )

    # -- refinement internals ---------------------------------------------

    @staticmethod
    def _classify_changes(decision: Any):
        accepted: list[ValidatedChange] = []
        pending: list[ValidatedChange] = []
        rejected: list[dict] = []
        if isinstance(decision, ProfileReviewResult):
            for change in decision.changes:
                if not change.accepted:
                    continue
                if change.auto_apply:
                    accepted.append(change)
                else:
                    pending.append(change)
            rejected.extend(decision.rejected)
            return accepted, pending, rejected
        for item in getattr(decision, "changes", []) or []:
            if isinstance(item, ValidatedChange):
                (accepted if item.auto_apply else pending).append(item)
            elif isinstance(item, ProfileChangeProposal):
                pending.append(ValidatedChange(proposal=item, auto_apply=False))
            else:
                rejected.append({"reason": "unsupported_change"})
        return accepted, pending, rejected

    async def _precheck_changes(
        self, person_id: str, changes: list[ValidatedChange]
    ):
        """Defence in depth around the engine's validation.

        These checks are deliberately conservative: anything that cannot be
        proven safe becomes a pending proposal instead of an automatic write.
        """
        accepted: list[ValidatedChange] = []
        pending: list[ValidatedChange] = []
        rejected: list[dict] = []
        entries = {
            item["entry_id"]: item
            for item in await self.repo.list_entries(
                person_id,
                statuses=(
                    ProfileEntryStatus.ACTIVE.value,
                    ProfileEntryStatus.TENTATIVE.value,
                ),
            )
        }
        event_ids: set[str] = set()
        for change in changes[:MAX_REFINEMENT_CHANGES]:
            proposal = change.proposal
            spec = profile_field_spec(proposal.field_key)
            if spec is None or spec.storage is not ProfileFieldStorage.ENTRY:
                rejected.append(
                    {
                        "operation": proposal.operation,
                        "field_key": proposal.field_key,
                        "reason": "field_not_refinable",
                    }
                )
                continue
            if spec.strict_identity or not spec.model_writable:
                pending.append(
                    change.model_copy(
                        update={"auto_apply": False,
                                "pending_reason": "confirmed_identity_requires_confirmation"}
                    )
                )
                continue
            if _SENSITIVE_CONTENT.search(proposal.proposed_content or ""):
                pending.append(
                    change.model_copy(
                        update={"auto_apply": False,
                                "pending_reason": "sensitive_field_requires_review"}
                    )
                )
                continue
            targets = list(dict.fromkeys(proposal.target_entry_ids))
            if proposal.operation != "ADD" and not targets:
                rejected.append(
                    {
                        "operation": proposal.operation,
                        "field_key": proposal.field_key,
                        "reason": "missing_target_entry_ids",
                    }
                )
                continue
            found = [entries.get(entry_id) for entry_id in targets]
            if any(item is None for item in found):
                rejected.append(
                    {
                        "operation": proposal.operation,
                        "field_key": proposal.field_key,
                        "reason": "target_entry_not_in_this_profile",
                    }
                )
                continue
            try:
                validate_change_targets(
                    proposal.operation,
                    proposal.field_key,
                    [item for item in found if item is not None],
                )
            except ValueError as exc:
                rejected.append(
                    {
                        "operation": proposal.operation,
                        "field_key": proposal.field_key,
                        "reason": str(exc) or "invalid_change_target",
                    }
                )
                continue
            if spec.max_entries and proposal.operation == "ADD":
                effective = await self.repo.list_entries(
                    person_id,
                    statuses=(
                        ProfileEntryStatus.ACTIVE.value,
                        ProfileEntryStatus.TENTATIVE.value,
                    ),
                    field_key=proposal.field_key,
                )
                if len(effective) >= spec.max_entries:
                    pending.append(
                        change.model_copy(
                            update={"auto_apply": False,
                                    "pending_reason": "field_capacity_reached"}
                        )
                    )
                    continue
            evidence = list(dict.fromkeys(proposal.evidence_event_ids))
            user_texts: dict[str, str] = {}
            if evidence:
                placeholders = ",".join("?" for _ in evidence)
                cur = await self.persons.db.execute(
                    "SELECT event_id, text FROM utterances WHERE event_id IN"
                    f" ({placeholders}) AND speaker_person_id = ?",
                    tuple(evidence) + (person_id,),
                )
                rows = await cur.fetchall()
                user_texts = {row["event_id"]: row["text"] for row in rows}
                if not set(evidence) <= set(user_texts):
                    # The cited message no longer exists (privacy-deleted) or
                    # belongs to someone else. It can never become valid later,
                    # so reject the whole proposal instead of persisting a
                    # pending candidate that quotes a dead source.
                    rejected.append(
                        {
                            "operation": proposal.operation,
                            "field_key": proposal.field_key,
                            "reason": "evidence_event_not_accessible",
                        }
                    )
                    continue
                event_ids.update(user_texts)
            if proposal.evidence_quotes:
                if not evidence:
                    pending.append(
                        change.model_copy(
                            update={"auto_apply": False,
                                    "pending_reason": "evidence_quote_without_event"}
                        )
                    )
                    continue
                normalised_texts = [
                    " ".join(str(text or "").split())
                    for text in user_texts.values()
                ]
                missing = any(
                    not any(
                        " ".join(str(quote or "").split())
                        and " ".join(str(quote or "").split()) in text
                        for text in normalised_texts
                    )
                    for quote in proposal.evidence_quotes
                )
                if missing:
                    rejected.append(
                        {
                            "operation": proposal.operation,
                            "field_key": proposal.field_key,
                            "reason": "evidence_quote_not_found",
                        }
                    )
                    continue
            if not evidence and not proposal.evidence_quotes:
                pending.append(
                    change.model_copy(
                        update={"auto_apply": False, "pending_reason": "no_evidence"}
                    )
                )
                continue
            if proposal.confidence < 0.7:
                pending.append(
                    change.model_copy(
                        update={"auto_apply": False, "pending_reason": "low_confidence"}
                    )
                )
                continue
            accepted.append(change)
        for change in changes[MAX_REFINEMENT_CHANGES:]:
            pending.append(
                change.model_copy(
                    update={"auto_apply": False, "pending_reason": "change_limit_exceeded"}
                )
            )
        return accepted, pending, rejected

    async def _apply_change_locked(
        self,
        person_id: str,
        change: ValidatedChange,
        *,
        revision: int,
        source_event_id: Optional[str],
        confirmed: bool,
        actor: str,
        trusted: bool = False,
        source_override: Optional[str] = None,
    ) -> dict:
        proposal = change.proposal
        spec = profile_field_spec(proposal.field_key)
        if spec is None or spec.storage is not ProfileFieldStorage.ENTRY:
            raise ValueError("field_not_refinable")
        content = self._normalise_entry_content(proposal.proposed_content, spec)
        if not content and proposal.operation != "ARCHIVE":
            raise ValueError("missing_proposed_content")
        # Final write-boundary checks. Even a caller that constructs
        # ``ValidatedChange(auto_apply=True)`` by hand cannot bypass these.
        # ``trusted`` is reserved for the locally validated single-turn
        # self-report path; it still requires the person's explicit statement.
        if (
            not confirmed
            and not trusted
            and (spec.strict_identity or not spec.model_writable)
        ):
            raise ValueError("confirmed_identity_requires_confirmation")
        if not confirmed and _SENSITIVE_CONTENT.search(content):
            raise ValueError("sensitive_field_requires_review")
        targets: list[dict] = []
        for entry_id in dict.fromkeys(proposal.target_entry_ids):
            entry = await self.repo.get_entry(entry_id)
            if (
                entry is None
                or entry["person_id"] != person_id
                or entry["status"]
                not in (
                    ProfileEntryStatus.ACTIVE.value,
                    ProfileEntryStatus.TENTATIVE.value,
                )
            ):
                raise ValueError("target_entry_not_effective")
            targets.append(entry)
        validate_change_targets(proposal.operation, proposal.field_key, targets)
        if proposal.operation == "ADD" and spec.max_entries:
            existing = await self.repo.list_entries(
                person_id,
                statuses=(
                    ProfileEntryStatus.ACTIVE.value,
                    ProfileEntryStatus.TENTATIVE.value,
                ),
                field_key=proposal.field_key,
            )
            if len(existing) >= spec.max_entries:
                raise ValueError("field_capacity_reached")

        evidence_events = list(
            dict.fromkeys(
                list(proposal.evidence_event_ids)
                + ([source_event_id] if source_event_id else [])
            )
        )
        if not confirmed and not evidence_events and not proposal.evidence_quotes:
            raise ValueError("no_evidence")
        await self._assert_evidence_locked(
            person_id, evidence_events, proposal.evidence_quotes
        )
        source = source_override or (
            ProfileSource.USER_CORRECTION.value
            if confirmed
            else ProfileSource.INTERACTION_INFERENCE.value
        )
        operation = proposal.operation
        before = [item["content"] for item in targets]
        now = utcnow()

        if operation == "ADD":
            if spec.single_value:
                for existing in await self.repo.list_entries(
                    person_id,
                    statuses=(
                        ProfileEntryStatus.ACTIVE.value,
                        ProfileEntryStatus.TENTATIVE.value,
                    ),
                    field_key=proposal.field_key,
                ):
                    await self.repo.update_entry(
                        existing["entry_id"],
                        status=ProfileEntryStatus.ARCHIVED.value,
                        valid_until=now,
                    )
            entry_id = await self.repo.insert_entry(
                person_id=person_id,
                field_key=proposal.field_key,
                content=content,
                status=ProfileEntryStatus.ACTIVE.value,
                source=source,
                confidence=proposal.confidence,
                evidence_event_ids=evidence_events,
                evidence_memory_ids=[],
            )
        elif operation in ("UPDATE", "CORRECT"):
            if len(targets) != 1:
                raise ValueError("update_needs_one_target")
            target = targets[0]
            if target["field_key"] != proposal.field_key:
                # A project moving from active to completed keeps the old row
                # (archived, pointing at its successor) and adds a new one.
                entry_id = await self.repo.insert_entry(
                    person_id=person_id,
                    field_key=proposal.field_key,
                    content=content,
                    status=ProfileEntryStatus.ACTIVE.value,
                    source=source,
                    confidence=proposal.confidence,
                    evidence_event_ids=evidence_events,
                    evidence_memory_ids=target["evidence_memory_ids"],
                    merged_from=[target["entry_id"]],
                )
                await self.repo.update_entry(
                    target["entry_id"],
                    status=ProfileEntryStatus.ARCHIVED.value,
                    superseded_by=entry_id,
                    valid_until=now,
                )
            else:
                entry_id = target["entry_id"]
                await self.repo.update_entry(
                    entry_id,
                    content=content,
                    status=ProfileEntryStatus.ACTIVE.value,
                    confidence=proposal.confidence,
                    evidence_event_ids=evidence_events,
                )
        elif operation == "MERGE":
            if len(targets) < 2:
                raise ValueError("merge_needs_two_targets")
            if any(item["field_key"] != proposal.field_key for item in targets):
                raise ValueError("merge_targets_differ_in_field")
            memory_ids = sorted(
                {
                    memory_id
                    for item in targets
                    for memory_id in item["evidence_memory_ids"]
                }
            )
            entry_id = await self.repo.insert_entry(
                person_id=person_id,
                field_key=proposal.field_key,
                content=content,
                status=ProfileEntryStatus.ACTIVE.value,
                source=source,
                confidence=proposal.confidence,
                evidence_event_ids=evidence_events,
                evidence_memory_ids=memory_ids,
                merged_from=[item["entry_id"] for item in targets],
            )
            for item in targets:
                await self.repo.update_entry(
                    item["entry_id"],
                    status=ProfileEntryStatus.MERGED.value,
                    superseded_by=entry_id,
                    valid_until=now,
                )
        elif operation == "REFINE":
            if len(targets) != 1:
                raise ValueError("refine_needs_one_target")
            target = targets[0]
            if target["field_key"] != proposal.field_key:
                raise ValueError("refine_target_field_mismatch")
            entry_id = target["entry_id"]
            await self.repo.update_entry(
                entry_id,
                content=content,
                status=ProfileEntryStatus.ACTIVE.value,
                confidence=proposal.confidence,
                evidence_event_ids=evidence_events,
            )
        elif operation == "ARCHIVE":
            if not targets:
                raise ValueError("archive_needs_targets")
            for item in targets:
                if item["field_key"].startswith(
                    ("identity.", "relationship.")
                ):
                    raise ValueError("strict_field_cannot_be_archived")
                await self.repo.update_entry(
                    item["entry_id"],
                    status=ProfileEntryStatus.ARCHIVED.value,
                    valid_until=now,
                )
            entry_id = None
        else:
            raise ValueError(f"unsupported_operation:{operation}")

        for event_id in evidence_events:
            await self._record_evidence(
                person_id,
                proposal.field_key,
                memory_ids=[],
                event_ids=[event_id],
            )
        return {
            "operation": operation,
            "field_key": proposal.field_key,
            "target_entry_ids": [item["entry_id"] for item in targets],
            "before": before,
            "after": content if operation != "ARCHIVE" else None,
            "entry_id": entry_id,
            "status": "APPLIED",
            "reason": proposal.reason,
        }

    async def _store_pending_candidate(
        self, person_id: str, change: ValidatedChange, base_revision: int
    ) -> str:
        proposal = change.proposal
        content = self._normalise_entry_content(proposal.proposed_content)
        targets = sorted(set(proposal.target_entry_ids))
        for item in await self.repo.find_pending(person_id, proposal.field_key):
            # ADD("a new fact") and UPDATE("an old fact") may carry identical
            # text and target different entries. Only a genuinely identical
            # proposal may merge its evidence.
            same_proposal = (
                item["candidate_value"] == content
                and (item.get("operation") or "ADD") == proposal.operation
                and sorted(set(item.get("target_entry_ids") or [])) == targets
            )
            if not same_proposal:
                continue
            if item.get("expected_revision") not in (None, base_revision):
                # The old proposal was based on a profile that has since
                # moved. Keep it visible for re-review instead of piling new
                # evidence onto a stale base.
                await self.repo.set_candidate_status(
                    item["candidate_id"],
                    CandidateStatus.NEEDS_REVIEW.value,
                    reason="stale_base_revision",
                )
                continue
            await self.repo.add_candidate_evidence(
                item["candidate_id"],
                memory_ids=[],
                event_ids=list(proposal.evidence_event_ids),
                candidate_ids=[],
            )
            return item["candidate_id"]
        return await self.repo.insert_candidate(
            person_id=person_id,
            field_key=proposal.field_key,
            value=content,
            source=ProfileSource.INTERACTION_INFERENCE.value,
            confidence=proposal.confidence,
            evidence_memory_ids=[],
            evidence_event_ids=list(proposal.evidence_event_ids),
            evidence_candidate_ids=[],
            reason=change.pending_reason or proposal.reason or "pending_review",
            operation=proposal.operation,
            target_entry_ids=proposal.target_entry_ids,
            expected_revision=base_revision,
            requires_confirmation=True,
            review_status="PENDING",
        )

    @staticmethod
    def _normalise_entry_content(
        content: Any, spec: Any = None
    ) -> str:
        text = " ".join(str(content or "").split())
        limit = getattr(spec, "max_chars", 240) if spec is not None else 240
        return text[:limit]

    @staticmethod
    def _pending_summary(change: ValidatedChange) -> dict:
        return {
            "operation": change.proposal.operation,
            "field_key": change.proposal.field_key,
            "target_entry_ids": change.proposal.target_entry_ids,
            "proposed_content": change.proposal.proposed_content,
            "reason": change.pending_reason or "pending_review",
            "requires_confirmation": True,
        }

    # -- snapshot helpers --------------------------------------------------

    @staticmethod
    def _group_entries(items: list[dict]) -> dict[str, list[ProfileEntry]]:
        grouped: dict[str, list[ProfileEntry]] = {}
        for item in items:
            entry = ProfileEntry(
                entry_id=item["entry_id"],
                person_id=item["person_id"],
                field_key=item["field_key"],
                content=item["content"],
                status=item["status"],
                source=item["source"],
                confidence=item["confidence"],
                evidence_event_ids=item["evidence_event_ids"],
                evidence_memory_ids=item["evidence_memory_ids"],
                merged_from=item["merged_from"],
                valid_from=item["valid_from"],
                valid_until=item["valid_until"],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                superseded_by=item["superseded_by"],
            )
            grouped.setdefault(entry.field_key, []).append(entry)
        return grouped

    @staticmethod
    def _snapshot_entry(entry: ProfileEntry) -> SnapshotEntry:
        return SnapshotEntry(
            entry_id=entry.entry_id,
            field_key=entry.field_key,
            content=entry.content,
            status=entry.status,
            source=entry.source,
            confidence=entry.confidence,
            evidence_event_ids=entry.evidence_event_ids,
            evidence_memory_ids=entry.evidence_memory_ids,
            updated_at=entry.updated_at.isoformat() if entry.updated_at else None,
        )

    @staticmethod
    def _identity_view(entries: dict[str, list[ProfileEntry]]) -> dict[str, Any]:
        view: dict[str, Any] = {}
        for field_key, label in _IDENTITY_LABELS.items():
            texts = [
                entry.content
                for entry in entries.get(field_key, [])
                if entry.is_effective()
            ]
            view[label] = texts[0] if texts else "UNKNOWN"
        view["nicknames"] = [
            entry.content
            for entry in entries.get("identity.nicknames", [])
            if entry.is_effective()
        ]
        return view

    @staticmethod
    def _relationship_view(
        entries: dict[str, list[ProfileEntry]], tier: str
    ) -> dict[str, Any]:
        return {
            # The tier remains the single authoritative familiarity source.
            "familiarity": tier,
            "label": [
                entry.content
                for entry in entries.get("relationship.label", [])
                if entry.is_effective()
            ],
            "boundaries": [
                entry.content
                for entry in entries.get("relationship.boundaries", [])
                if entry.is_effective()
            ],
        }

    @staticmethod
    def _category_view(
        entries: dict[str, list[ProfileEntry]],
        fields: dict[str, dict],
        category: str,
    ) -> dict[str, Any]:
        view: dict[str, list[dict]] = {}
        for field_key, items in entries.items():
            spec = profile_field_spec(field_key)
            if spec is None or spec.category != category:
                continue
            effective = [
                {
                    "entry_id": entry.entry_id,
                    "content": entry.content,
                    "status": entry.status,
                    "source": entry.source,
                    "confidence": entry.confidence,
                }
                for entry in items
                if entry.is_effective()
            ]
            if effective:
                view[spec.label] = effective
        if category == "communication":
            view["settings"] = {
                key: value["value"] for key, value in fields.items()
            }
        return view

    # -- evidence / memory linkage ----------------------------------------

    async def invalidate_profile_sources(
        self,
        person_id: str,
        *,
        event_ids: Iterable[str] = (),
        memory_ids: Iterable[str] = (),
        deletion_mode: str = "RETRACTION",
    ) -> ProfileInvalidationResult:
        """Remove a *set* of sources from one person's profile in one commit.

        The caller must collect the complete source set before deleting
        anything: an entry that cites only a derived memory must still be found
        when its originating event is deleted. Entries, candidates, legacy
        fields, receipts, revisions and the summary are processed in one SQLite
        transaction with a single revision bump.
        """
        events = {str(item) for item in event_ids if item}
        memories = {str(item) for item in memory_ids if item}
        mode = str(deletion_mode or "RETRACTION").upper()
        if mode not in ("RETRACTION", "PRIVACY_DELETE"):
            raise ValueError("unsupported_deletion_mode")
        if not events and not memories:
            raise ValueError("event_id_or_memory_id_required")
        privacy = mode == "PRIVACY_DELETE"
        result = ProfileInvalidationResult(
            person_id=person_id,
            event_id=next(iter(events)) if len(events) == 1 else None,
            memory_id=next(iter(memories)) if len(memories) == 1 else None,
            deletion_mode=mode,
        )

        def touched(
            evidence_events: list[str], evidence_memories: list[str]
        ) -> bool:
            return bool(
                events.intersection(evidence_events or [])
                or memories.intersection(evidence_memories or [])
            )

        def remaining(values: list[str], removed: set[str]) -> list[str]:
            return [item for item in values or [] if item not in removed]

        async with self.repo.db.transaction(immediate=True):
            row = await self.repo.get(person_id)
            if row is None:
                return result
            current_revision = int(row["revision"] or 1)
            entries = [
                entry
                for entry in await self.repo.list_entries(person_id)
                if touched(
                    entry["evidence_event_ids"], entry["evidence_memory_ids"]
                )
            ]
            candidate_map: dict[str, dict] = {}
            for event_id in events:
                for candidate in await self.repo.list_candidates_referencing_source(
                    person_id, event_id=event_id
                ):
                    candidate_map[candidate["candidate_id"]] = candidate
            for memory_id in memories:
                for candidate in await self.repo.list_candidates_referencing_source(
                    person_id, memory_id=memory_id
                ):
                    candidate_map[candidate["candidate_id"]] = candidate
            candidates = list(candidate_map.values())
            fields = [
                field
                for field in await self.repo.list_fields(person_id)
                if touched(
                    field["evidence_event_ids"], field["evidence_memory_ids"]
                )
            ]
            affected_ids = [entry["entry_id"] for entry in entries]

            new_revision: Optional[int] = None
            if entries or fields:
                new_revision = await self.repo.bump_revision(
                    person_id,
                    expected_revision=current_revision,
                    updated_by=(
                        "privacy_delete" if privacy else "source_invalidation"
                    ),
                    mark_reviewed=False,
                )
                if not new_revision:
                    return result

            for entry in entries:
                if privacy:
                    await self.repo.delete_entry(entry["entry_id"])
                    result.entries_deleted += 1
                    continue
                remaining_events = remaining(entry["evidence_event_ids"], events)
                remaining_memories = remaining(
                    entry["evidence_memory_ids"], memories
                )
                effective = entry["status"] in (
                    ProfileEntryStatus.ACTIVE.value,
                    ProfileEntryStatus.TENTATIVE.value,
                )
                if not remaining_events and not remaining_memories:
                    if effective:
                        await self.repo.update_entry(
                            entry["entry_id"],
                            status=ProfileEntryStatus.ARCHIVED.value,
                            valid_until=utcnow(),
                        )
                        result.entries_archived += 1
                    # The retracted sources may no longer be cited as support.
                    await self.repo.replace_entry_evidence(
                        entry["entry_id"],
                        evidence_event_ids=remaining_events,
                        evidence_memory_ids=remaining_memories,
                    )
                elif entry["merged_from"]:
                    # A merged text combines several sources. Removing one does
                    # not prove the survivors still support every detail, so
                    # stop presenting it until it is reviewed.
                    if effective:
                        await self.repo.update_entry(
                            entry["entry_id"],
                            status=ProfileEntryStatus.PENDING_REVIEW.value,
                            valid_until=utcnow(),
                        )
                        result.entries_archived += 1
                    await self.repo.replace_entry_evidence(
                        entry["entry_id"],
                        evidence_event_ids=remaining_events,
                        evidence_memory_ids=remaining_memories,
                    )
                else:
                    await self.repo.replace_entry_evidence(
                        entry["entry_id"],
                        evidence_event_ids=remaining_events,
                        evidence_memory_ids=remaining_memories,
                    )
                    result.entries_updated += 1

            for field in fields:
                revision = new_revision or current_revision
                field_key = field["field_key"]
                if privacy:
                    await self.repo.redact_field(
                        person_id, field_key, revision=revision
                    )
                    result.entries_archived += 1
                    continue
                remaining_events = remaining(field["evidence_event_ids"], events)
                remaining_memories = remaining(
                    field["evidence_memory_ids"], memories
                )
                value = field["value"]
                if field_key in LIST_FIELDS:
                    old_items = list(value or [])
                    value = [
                        item
                        for item in old_items
                        if item.get("memory_id") not in memories
                    ]
                    removed_item = len(value) != len(old_items)
                else:
                    removed_item = False
                if not remaining_events and not remaining_memories:
                    await self.repo.retire_field(
                        person_id, field_key, revision=revision
                    )
                    result.entries_archived += 1
                elif removed_item or remaining_events != field["evidence_event_ids"]:
                    await self.repo.upsert_field(
                        person_id,
                        field_key,
                        value=value,
                        source=field["source"],
                        confidence=field["confidence"],
                        evidence_memory_ids=remaining_memories,
                        evidence_event_ids=remaining_events,
                        evidence_candidate_ids=field["evidence_candidate_ids"],
                        revision=revision,
                    )
                    result.entries_updated += 1

            for candidate in candidates:
                if privacy:
                    await self.repo.delete_candidate(candidate["candidate_id"])
                    result.candidates_cleaned += 1
                    continue
                remaining_events = remaining(
                    candidate["evidence_event_ids"], events
                )
                remaining_memories = remaining(
                    candidate["evidence_memory_ids"], memories
                )
                if not remaining_events and not remaining_memories:
                    await self.repo.set_candidate_status(
                        candidate["candidate_id"],
                        CandidateStatus.NEEDS_REVIEW.value,
                        reason="evidence_source_retracted",
                    )
                else:
                    await self.repo.replace_candidate_evidence(
                        candidate["candidate_id"],
                        evidence_event_ids=remaining_events,
                        evidence_memory_ids=remaining_memories,
                    )
                result.candidates_cleaned += 1

            if privacy:
                for event_id in events:
                    await self.repo.delete_evidence_for_source(
                        person_id, event_id=event_id
                    )
                    result.receipts_cleaned += (
                        await self.repo.cancel_review_receipts_for_source(
                            person_id, event_id
                        )
                    )
                    await self.repo.scrub_revisions_for_source(
                        person_id, event_id
                    )
                    await self.repo.scrub_receipts_for_source(
                        person_id, event_id
                    )
                for memory_id in memories:
                    await self.repo.delete_evidence_for_source(
                        person_id, memory_id=memory_id
                    )
                if affected_ids:
                    await self.repo.scrub_revisions_for_entries(
                        person_id, affected_ids
                    )
                    await self.repo.scrub_receipts_for_entries(
                        person_id, affected_ids
                    )

            if new_revision is not None:
                await self.repo.insert_revision(
                    person_id=person_id,
                    revision=new_revision,
                    changes={
                        "source_invalidated": {
                            "event_deleted": bool(events),
                            "memory_deleted": bool(memories),
                            "event_count": len(events),
                            "memory_count": len(memories),
                            "mode": mode,
                            "entries": len(entries),
                            "candidates": len(candidates),
                            "legacy_fields": len(fields),
                        }
                    },
                    changed_by=(
                        "privacy_delete" if privacy else "source_invalidation"
                    ),
                    source_event_id=None,
                    reason=(
                        "privacy_delete"
                        if privacy
                        else "evidence_source_retracted"
                    ),
                )
                await self._refresh_summary(person_id)
                result.revision = new_revision
        return result

    async def invalidate_profile_source(
        self,
        person_id: str,
        *,
        event_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        deletion_mode: str = "RETRACTION",
    ) -> ProfileInvalidationResult:
        """Single-source compatibility wrapper around the set-based path."""
        return await self.invalidate_profile_sources(
            person_id,
            event_ids=(event_id,) if event_id else (),
            memory_ids=(memory_id,) if memory_id else (),
            deletion_mode=deletion_mode,
        )

    async def propagate_memory_retraction(self, memory_id: str) -> int:
        """Stop referencing a revoked/deleted memory in every person profile.

        V0.9.7 entries and the pre-V0.9.7 field table are handled by
        :meth:`invalidate_profile_source`, one person at a time.
        """
        changed_people = 0
        for person_id in await self.repo.list_all_person_ids():
            result = await self.invalidate_profile_source(
                person_id, memory_id=memory_id, deletion_mode="RETRACTION"
            )
            if result.changed:
                changed_people += 1
        return changed_people

    # -- helpers -----------------------------------------------------------

    async def _refresh_summary(self, person_id: str) -> None:
        """Keep a bounded, human-readable summary on the main profile row.

        The summary is always written, including the empty string: after a
        deletion the compiled block may be empty, and leaving the previous
        value in place would keep deleted content alive in the profile row.
        """
        profile = await self.get(person_id)
        if profile is None:
            return
        block = compile_profile_context(profile) or ""
        await self.repo.set_summary(
            person_id, block, updated_by="profile_service"
        )

    async def _assert_evidence_locked(
        self,
        person_id: str,
        event_ids: list[str],
        quotes: list[str],
    ) -> None:
        """Re-check evidence inside the final write transaction.

        Evidence must still be a real message written by this person. A quote,
        when supplied, must occur in the cited message's own text — never in
        Ava's reply or another person's turn.
        """
        ids = [str(item) for item in dict.fromkeys(event_ids) if item]
        if not ids:
            if quotes:
                raise ValueError("evidence_quote_without_event")
            return
        placeholders = ",".join("?" for _ in ids)
        cur = await self.persons.db.execute(
            "SELECT event_id, text FROM utterances WHERE event_id IN"
            f" ({placeholders}) AND speaker_person_id = ?",
            tuple(ids) + (person_id,),
        )
        rows = await cur.fetchall()
        texts = {row["event_id"]: row["text"] for row in rows}
        if not set(ids) <= set(texts):
            raise ValueError("evidence_event_not_accessible")
        normalised_texts = [" ".join(str(text or "").split()) for text in texts.values()]
        for quote in quotes:
            normalised = " ".join(str(quote or "").split())
            if not any(normalised and normalised in text for text in normalised_texts):
                raise ValueError("evidence_quote_not_found")

    async def _record_evidence(
        self,
        person_id: str,
        field_key: str,
        *,
        memory_ids: Iterable[str],
        event_ids: Iterable[str],
        candidate_id: Optional[str] = None,
    ) -> None:
        for memory_id in _dedupe(memory_ids):
            await self.repo.insert_evidence(
                person_id=person_id,
                field_key=field_key,
                memory_id=memory_id,
                candidate_id=candidate_id,
                evidence_type="MEMORY",
            )
        for event_id in _dedupe(event_ids):
            await self.repo.insert_evidence(
                person_id=person_id,
                field_key=field_key,
                source_event_id=event_id,
                candidate_id=candidate_id,
                evidence_type="EVENT",
            )

    async def _audit_field(
        self,
        actor: str,
        action: str,
        person_id: str,
        field_key: str,
        source: str,
    ) -> None:
        if self.audit is None:
            return
        await self.audit.record(
            actor=actor,
            action=action,
            entity_type="person_profile",
            entity_id=person_id,
            detail=f"{field_key} source={source}",
        )

    @staticmethod
    def _field_model(item: dict) -> ProfileField:
        return ProfileField(
            field_key=item["field_key"],
            value=item["value"],
            source=item["source"],
            confidence=item["confidence"],
            status=item["status"],
            evidence_memory_ids=item["evidence_memory_ids"],
            evidence_event_ids=item["evidence_event_ids"],
            evidence_candidate_ids=item["evidence_candidate_ids"],
            revision=item["revision"],
            updated_at=item["updated_at"],
        )

    @staticmethod
    def _candidate_model(item: dict) -> ProfileCandidate:
        return ProfileCandidate(
            candidate_id=item["candidate_id"],
            person_id=item["person_id"],
            field_key=item["field_key"],
            candidate_value=item["candidate_value"],
            source=item["source"],
            confidence=item["confidence"],
            status=item["status"],
            evidence_memory_ids=item["evidence_memory_ids"],
            evidence_event_ids=item["evidence_event_ids"],
            evidence_candidate_ids=item["evidence_candidate_ids"],
            reason=item["reason"],
            created_at=item["created_at"],
            updated_at=item["updated_at"],
        )


def _parse(value: Any):
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


__all__ = ["PersonProfileService", "ProfileUnauthorized"]
