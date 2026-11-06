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
    ProfileReviewResult,
    SnapshotEntry,
    ValidatedChange,
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
            candidates = await self.repo.list_candidates(person_id)
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

    # -- V0.9.7 refinement -------------------------------------------------

    async def apply_refinement(
        self,
        person_id: str,
        decision: Any,
        *,
        source_event_id: Optional[str] = None,
        actor: str = "profile_refinement",
    ) -> ProfileApplyResult:
        """Validate and apply one review's proposals atomically.

        The model never reaches this method directly. ``decision`` carries
        proposals that the engine already checked against the *existing*
        snapshot; this method re-checks authorization, revision, ownership,
        evidence and limits, then commits everything in one SQLite
        transaction. A stale ``base_revision`` is reported, never overwritten.
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
            async with self.repo.db.transaction():
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

        async with self.repo.db.transaction():
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

    async def confirm_profile_proposal(
        self,
        candidate_id: str,
        *,
        actor: str = "user",
        reason: Optional[str] = None,
    ) -> int:
        """Apply a pending refinement proposal after explicit confirmation.

        This is the trusted path for strict identity (and other pending)
        changes. The candidate's content is treated as confirmed, but the
        person's permission and the profile's current revision are still
        checked; a stale candidate cannot be applied over a newer revision.
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
        if not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)
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
        async with self.repo.db.transaction():
            row = await self.repo.get(person_id)
            if row is None:
                raise ProfileUnauthorized(person_id)
            revision = await self.repo.bump_revision(
                person_id,
                expected_revision=int(row["revision"] or 1),
                updated_by=actor,
                mark_reviewed=True,
            )
            if not revision:
                raise ValueError("profile revision changed; confirm again")
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
        await self._audit_field(
            actor,
            "profile_proposal_confirmed",
            person_id,
            candidate["field_key"],
            candidate["source"],
        )
        return revision

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
            if proposal.operation == "ADD" and targets:
                rejected.append(
                    {
                        "operation": proposal.operation,
                        "field_key": proposal.field_key,
                        "reason": "add_must_not_target_entries",
                    }
                )
                continue
            if proposal.operation in (
                "UPDATE",
                "CORRECT",
                "MERGE",
                "REFINE",
                "ARCHIVE",
            ):
                if not targets:
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
                if any(
                    item["field_key"] != proposal.field_key for item in found
                ) and proposal.operation in ("MERGE", "REFINE"):
                    rejected.append(
                        {
                            "operation": proposal.operation,
                            "field_key": proposal.field_key,
                            "reason": "merge_targets_differ_in_field",
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
            if evidence:
                placeholders = ",".join("?" for _ in evidence)
                cur = await self.persons.db.execute(
                    "SELECT event_id FROM utterances WHERE event_id IN"
                    f" ({placeholders}) AND speaker_person_id = ?",
                    tuple(evidence) + (person_id,),
                )
                rows = await cur.fetchall()
                valid = {row["event_id"] for row in rows}
                if not set(evidence) <= valid:
                    pending.append(
                        change.model_copy(
                            update={"auto_apply": False,
                                    "pending_reason": "evidence_event_not_accessible"}
                        )
                    )
                    continue
                event_ids.update(valid)
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
    ) -> dict:
        proposal = change.proposal
        spec = profile_field_spec(proposal.field_key)
        if spec is None or spec.storage is not ProfileFieldStorage.ENTRY:
            raise ValueError("field_not_refinable")
        content = self._normalise_entry_content(proposal.proposed_content, spec)
        if not content and proposal.operation != "ARCHIVE":
            raise ValueError("missing_proposed_content")
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

        evidence_events = list(
            dict.fromkeys(
                list(proposal.evidence_event_ids)
                + ([source_event_id] if source_event_id else [])
            )
        )
        source = (
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
        for item in await self.repo.find_pending(person_id, proposal.field_key):
            if item["candidate_value"] == content:
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

    async def propagate_memory_retraction(self, memory_id: str) -> int:
        """Stop referencing a memory that was revoked or deleted.

        Returns the number of fields that changed. Scalar fields whose only
        remaining support was that memory are retired; list entries that point
        at it are removed.
        """
        references = await self.repo.fields_referencing_memory(memory_id)
        changed = 0
        seen: set[tuple[str, str]] = set()
        for reference in references:
            key = (reference["person_id"], reference["field_key"])
            if key in seen:
                continue
            seen.add(key)
            person_id, field_key = key
            async with self.repo.db.transaction():
                row = await self.repo.get(person_id)
                if row is None:
                    continue
                field = await self.repo.get_field(person_id, field_key)
                if field is None or field["status"] != "ACTIVE":
                    continue
                old_value = field["value"]
                memory_ids = [
                    item
                    for item in field["evidence_memory_ids"]
                    if item != memory_id
                ]
                if field_key in LIST_FIELDS:
                    value = [
                        entry
                        for entry in (old_value or [])
                        if entry.get("memory_id") != memory_id
                    ]
                    if len(value) == len(old_value or []):
                        continue
                    if not value:
                        revision = int(row["revision"] or 1) + 1
                        await self.repo.retire_field(
                            person_id, field_key, revision=revision
                        )
                        change = {"retired": True}
                    else:
                        revision = int(row["revision"] or 1) + 1
                        await self.repo.upsert_field(
                            person_id,
                            field_key,
                            value=validate_field_value(field_key, value),
                            source=field["source"],
                            confidence=field["confidence"],
                            evidence_memory_ids=memory_ids,
                            evidence_event_ids=field["evidence_event_ids"],
                            evidence_candidate_ids=field[
                                "evidence_candidate_ids"
                            ],
                            revision=revision,
                        )
                        change = {"old": old_value, "new": value}
                else:
                    if memory_id not in field["evidence_memory_ids"]:
                        continue
                    remaining_events = field["evidence_event_ids"]
                    remaining_candidates = field["evidence_candidate_ids"]
                    if (
                        not memory_ids
                        and not remaining_events
                        and not remaining_candidates
                    ):
                        revision = int(row["revision"] or 1) + 1
                        await self.repo.retire_field(
                            person_id, field_key, revision=revision
                        )
                        change = {"retired": True}
                    else:
                        if not memory_ids:
                            continue
                        revision = int(row["revision"] or 1) + 1
                        await self.repo.upsert_field(
                            person_id,
                            field_key,
                            value=old_value,
                            source=field["source"],
                            confidence=field["confidence"],
                            evidence_memory_ids=memory_ids,
                            evidence_event_ids=remaining_events,
                            evidence_candidate_ids=remaining_candidates,
                            revision=revision,
                        )
                        change = {"evidence_memory_removed": memory_id}
                await self.repo.touch(person_id, updated_by="memory_retraction")
                await self.repo.insert_revision(
                    person_id=person_id,
                    revision=revision,
                    changes={"field_key": field_key, **change},
                    changed_by="memory_retraction",
                    source_event_id=None,
                    reason=f"memory deleted: {memory_id}",
                )
                changed += 1
        return changed

    # -- helpers -----------------------------------------------------------

    async def _refresh_summary(self, person_id: str) -> None:
        """Keep a bounded, human-readable summary on the main profile row."""
        profile = await self.get(person_id)
        if profile is None:
            return
        block = compile_profile_context(profile)
        if block:
            await self.repo.set_summary(
                person_id, block, updated_by="profile_service"
            )

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
