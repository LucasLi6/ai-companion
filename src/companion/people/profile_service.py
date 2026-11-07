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
  immediately. Inferences are written only after the review engine and this
  service both prove them against the person's own persisted words.
- Every accepted change bumps the profile revision and appends a revision row.
  A user correction outranks an earlier inference, so the old wrong value is
  never resurrected.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from companion.domain.enums import RegistrationStatus
from companion.domain.models import utcnow
from companion.people.profile_compiler import compile_profile_context
from companion.people.profile_models import (
    LIST_FIELDS,
    MAX_LIST_ITEMS,
    PersonInteractionProfile,
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
    DirectSelfReportResult,
    FullProfileSnapshot,
    ProfileApplyResult,
    ProfileChangeProposal,
    ProfileReviewExecution,
    ProfileReviewResult,
    SnapshotEntry,
    ValidatedChange,
    entries_equivalent,
    is_human_confirmed_source,
    is_strict_field,
    requires_direct_quote,
    validate_change_targets,
)
from companion.people.profile_repository import PersonProfileRepository
from companion.people.profile_self_report import (
    has_explicit_self_report,
    normalise as normalise_self_report_text,
)
from companion.storage.repositories import AuditRepository, PersonRepository

MAX_REFINEMENT_CHANGES = 10
_LOGGER = logging.getLogger("companion.profile_service")
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


# Explicit first-person statements that may update strict identity through the
# trusted direct-write path. They are never inferred and never come from model
# output.
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
    receipts_cleaned: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.entries_updated
            or self.entries_archived
            or self.entries_deleted
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

    async def can_review(self, person_id: str) -> bool:
        """Whether a background review may run and may initialise a profile.

        All four grants are required: processing the turn, persisting the
        profile, learning from it, and sending it to the configured model.
        A missing profile row for such a person is a legal-init repair; a
        missing grant is not.
        """
        person = await self.persons.get(person_id)
        if person is None or person.registration_status != RegistrationStatus.ACTIVE:
            return False
        consent = await self.persons.get_consent(person_id)
        return bool(
            consent
            and consent.allow_processing
            and consent.allow_persistence
            and consent.allow_memory_learning
            and consent.allow_external_ai
        )

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
        revision, entries and legacy fields from the same instant.
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
        reason: Optional[str] = None,
        merge: bool = False,
        require_learning: bool = True,
        actor: str = "learning",
    ) -> int:
        """Write an effective field value and record the revision.

        ``source`` must be an explicit-report source (direct self report, user
        correction or admin). Inferences go through profile review, which writes
        only after local evidence validation.
        """
        source_value = ProfileSource(source).value
        if source_value == ProfileSource.INTERACTION_INFERENCE.value:
            raise ValueError("inference must use profile review")
        if require_learning and not await self.can_learn(person_id):
            raise ProfileUnauthorized(person_id)
        if not await self.can_persist(person_id):
            raise ProfileUnauthorized(person_id)

        normalised = validate_field_value(field_key, value)
        memory_ids = list(memory_ids)
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
                evidence_candidate_ids=[],
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

        An explicit correction outranks every earlier inference; because there
        is no candidate queue, there is nothing left to resurrect the old value.
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

    # -- trusted direct self-report ----------------------------------------

    async def apply_direct_self_report(
        self,
        person_id: str,
        field_key: str,
        value: Any,
        *,
        source_event_id: str,
        quote: Optional[str] = None,
        actor: str = "profile_confirmation",
    ) -> DirectSelfReportResult:
        """Write one explicit first-person statement directly.

        This is the only path allowed to write strict identity fields. It
        re-reads the original persisted utterance and requires the detector to
        find exactly this field/value as an explicit self-report. A model, an
        inference or a caller-supplied actor string cannot substitute for that
        evidence. Duplicates are a successful no-op, never a new revision.
        """
        field_key = str(field_key or "")
        if field_key not in SELF_REPORT_FIELDS:
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="REJECTED",
                reason="field_not_self_reportable",
            )
        if not source_event_id:
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="REJECTED",
                reason="no_source_event",
            )
        normalised = self._normalise_self_report(field_key, value)
        if not normalised:
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="REJECTED",
                reason="missing_value",
            )
        if not await self.can_learn(person_id):
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="UNAUTHORIZED",
                reason="learning_or_persistence_not_authorized",
            )

        cur = await self.persons.db.execute(
            "SELECT text, speaker_person_id FROM utterances WHERE event_id = ?",
            (source_event_id,),
        )
        event = await cur.fetchone()
        if event is None:
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="REJECTED",
                reason="source_event_not_found",
            )
        if event["speaker_person_id"] != person_id:
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="REJECTED",
                reason="source_event_not_owned_by_person",
            )
        text = normalise_self_report_text(event["text"])
        if quote and normalise_self_report_text(quote) not in text:
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="REJECTED",
                reason="evidence_quote_not_found",
            )
        if not has_explicit_self_report(text, field_key, normalised):
            # The statement, not the caller, is the authority. This blocks a
            # model or a buggy caller from writing a name the person never said.
            return DirectSelfReportResult(
                person_id=person_id,
                field_key=field_key,
                status="REJECTED",
                reason="not_an_explicit_self_report",
            )

        async with self.repo.db.transaction(immediate=True):
            row = await self.repo.get(person_id)
            if row is None:
                created = await self.repo.ensure(person_id, actor=actor)
                row = await self.repo.get(person_id)
                if row is None:
                    return DirectSelfReportResult(
                        person_id=person_id,
                        field_key=field_key,
                        status="UNAUTHORIZED",
                        reason="no_profile",
                    )
                if created and self.audit is not None:
                    await self.audit.record(
                        actor=actor,
                        action="profile_create",
                        entity_type="person_profile",
                        entity_id=person_id,
                        detail="direct_self_report",
                    )
            if not await self.can_learn(person_id):
                return DirectSelfReportResult(
                    person_id=person_id,
                    field_key=field_key,
                    status="UNAUTHORIZED",
                    reason="learning_or_persistence_not_authorized",
                )

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
                    await self.repo.set_reviewed(person_id)
                    return DirectSelfReportResult(
                        person_id=person_id,
                        field_key=field_key,
                        value=normalised,
                        status="NO_CHANGE",
                        revision=int(row["revision"] or 1),
                        reason="already_present",
                    )
                old_value: Any = [item["content"] for item in effective]
            else:
                old_field = await self.repo.get_field(person_id, field_key)
                if (
                    old_field is not None
                    and old_field["status"] == "ACTIVE"
                    and old_field["value"] == normalised
                ):
                    await self.repo.set_reviewed(person_id)
                    return DirectSelfReportResult(
                        person_id=person_id,
                        field_key=field_key,
                        value=normalised,
                        status="NO_CHANGE",
                        revision=int(row["revision"] or 1),
                        reason="already_present",
                    )
                old_value = old_field["value"] if old_field else None

            current_revision = int(row["revision"] or 1)
            new_revision = await self.repo.bump_revision(
                person_id,
                expected_revision=current_revision,
                updated_by=actor,
                mark_reviewed=True,
            )
            if not new_revision:
                return DirectSelfReportResult(
                    person_id=person_id,
                    field_key=field_key,
                    status="REVISION_CONFLICT",
                    revision=current_revision,
                    reason="profile_revision_changed",
                )

            entry_id: Optional[str] = None
            if field_key in _ENTRY_SELF_REPORT_FIELDS:
                spec = profile_field_spec(field_key)
                replace = bool(spec is not None and spec.single_value and effective)
                proposal = ProfileChangeProposal(
                    operation="CORRECT" if replace else "ADD",
                    field_key=field_key,
                    target_entry_ids=(
                        [str(effective[0]["entry_id"])] if replace else []
                    ),
                    proposed_content=normalised,
                    evidence_event_ids=[source_event_id],
                    confidence=1.0,
                )
                applied = await self._apply_change_locked(
                    person_id,
                    ValidatedChange(proposal=proposal),
                    revision=new_revision,
                    source_event_id=source_event_id,
                    actor=actor,
                    trusted=True,
                    source_override=ProfileSource.DIRECT_SELF_REPORT.value,
                )
                entry_id = applied.get("entry_id")
                new_value: Any = applied.get("after")
            else:
                await self.repo.upsert_field(
                    person_id,
                    field_key,
                    value=normalised,
                    source=ProfileSource.DIRECT_SELF_REPORT.value,
                    confidence=1.0,
                    evidence_memory_ids=[],
                    evidence_event_ids=[source_event_id],
                    evidence_candidate_ids=[],
                    revision=new_revision,
                )
                await self._record_evidence(
                    person_id,
                    field_key,
                    memory_ids=[],
                    event_ids=[source_event_id],
                )
                new_value = normalised

            await self.repo.insert_revision(
                person_id=person_id,
                revision=new_revision,
                changes={
                    "field_key": field_key,
                    "old": old_value,
                    "new": new_value,
                    "source": ProfileSource.DIRECT_SELF_REPORT.value,
                    "kind": "direct_self_report",
                    "entry_id": entry_id,
                },
                changed_by=actor,
                source_event_id=source_event_id,
                reason="explicit_self_report",
            )
            await self._refresh_summary(person_id)

        await self._audit_field(
            actor,
            "profile_direct_self_report",
            person_id,
            field_key,
            ProfileSource.DIRECT_SELF_REPORT.value,
        )
        return DirectSelfReportResult(
            person_id=person_id,
            field_key=field_key,
            value=normalised,
            status="APPLIED",
            revision=new_revision,
            reason="explicit_self_report",
            entry_id=entry_id,
        )

    async def apply_migrated_change(
        self,
        person_id: str,
        change: ValidatedChange,
        *,
        source_event_id: str,
        source: str = ProfileSource.INTERACTION_INFERENCE.value,
        trusted_identity: bool = False,
        actor: str = "profile_candidate_migration",
    ) -> dict:
        """Apply one candidate from the legacy table after local revalidation.

        The caller (the migration command) has already revalidated the value
        against the original event. This method still performs the normal
        precheck and final write checks, and leases no model output. Identity
        values are accepted only when the caller proves they come from the
        person's own explicit statement/alias inventory.
        """
        if not await self.can_learn(person_id):
            return {
                "status": "UNAUTHORIZED",
                "reason": "learning_or_persistence_not_authorized",
            }
        proposal = change.proposal
        cur = await self.persons.db.execute(
            "SELECT text, speaker_person_id FROM utterances WHERE event_id = ?",
            (source_event_id,),
        )
        event = await cur.fetchone()
        if event is None:
            return {"status": "REJECTED", "reason": "source_event_not_found"}
        if event["speaker_person_id"] != person_id:
            return {
                "status": "REJECTED",
                "reason": "source_event_not_owned_by_person",
            }
        if trusted_identity:
            if (
                proposal.field_key not in _ENTRY_SELF_REPORT_FIELDS
                or proposal.operation != "ADD"
            ):
                return {
                    "status": "REJECTED",
                    "reason": "identity_migration_not_explicit_add",
                }
            normalised = self._normalise_self_report(
                proposal.field_key, proposal.proposed_content
            )
            if not normalised or normalised != normalise_self_report_text(
                proposal.proposed_content
            ):
                return {
                    "status": "REJECTED",
                    "reason": "identity_value_not_source_text",
                }
            if normalised.casefold() not in normalise_self_report_text(
                event["text"]
            ).casefold():
                return {
                    "status": "REJECTED",
                    "reason": "identity_value_not_source_text",
                }

        async with self.repo.db.transaction(immediate=True):
            row = await self.repo.get(person_id)
            if row is None:
                return {"status": "UNAUTHORIZED", "reason": "no_profile"}
            accepted, rejected = await self._precheck_changes(
                person_id, [change], allow_strict=trusted_identity
            )
            if not accepted:
                reason = (
                    rejected[0].get("reason") if rejected else "not_applicable"
                )
                await self.repo.set_reviewed(person_id)
                return {
                    "status": (
                        "NO_CHANGE" if reason == "already_present" else "REJECTED"
                    ),
                    "reason": reason,
                }
            current_revision = int(row["revision"] or 1)
            new_revision = await self.repo.bump_revision(
                person_id,
                expected_revision=current_revision,
                updated_by=actor,
                mark_reviewed=True,
            )
            if not new_revision:
                return {
                    "status": "REVISION_CONFLICT",
                    "reason": "profile_revision_changed",
                }
            # A trusted identity value came from the person's own explicit
            # statement, so it keeps the same provenance as a live direct
            # self-report rather than being labelled as an inference.
            effective_source = (
                ProfileSource.DIRECT_SELF_REPORT.value
                if trusted_identity
                else source
            )
            applied = await self._apply_change_locked(
                person_id,
                accepted[0],
                revision=new_revision,
                source_event_id=source_event_id,
                actor=actor,
                trusted=trusted_identity,
                source_override=effective_source,
            )
            await self.repo.insert_revision(
                person_id=person_id,
                revision=new_revision,
                changes={
                    "changes": [applied],
                    "migration": {"source": effective_source},
                    "rejected": rejected,
                },
                changed_by=actor,
                source_event_id=source_event_id,
                reason="candidate_migration",
            )
            await self._refresh_summary(person_id)

        await self._audit_field(
            actor,
            "profile_candidate_migrated",
            person_id,
            proposal.field_key,
            effective_source,
        )
        return {
            "status": "APPLIED",
            "revision": new_revision,
            "applied": applied,
            "not_applied": rejected,
        }

    def _normalise_self_report(self, field_key: str, value: Any) -> str:
        """Normalise a self-reported value, or return "" when it is invalid.

        ``communication.address_preference`` is constrained to a small
        enumerated vocabulary; an arbitrary alias raises ``ValueError`` in
        ``validate_field_value``. Callers treat "" as a rejected value instead
        of letting the exception escape an otherwise routine turn.
        """
        text = normalise_self_report_text(value)
        if field_key == "communication.address_preference":
            try:
                return validate_field_value(field_key, text)
            except (TypeError, ValueError):
                return ""
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

        accepted, rejected = self._classify_changes(decision)
        accepted, late_rejected = await self._precheck_changes(
            person_id, accepted
        )
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
                await self.repo.set_reviewed(person_id)
            final_status = "NO_CHANGE" if status == "NO_CHANGE" else "REJECTED"
            return ProfileApplyResult(
                person_id=person_id,
                source_event_id=source_event_id,
                status=final_status,
                base_revision=base_revision,
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
                        actor=actor,
                    )
                )
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
            status="APPLIED",
            base_revision=base_revision,
            new_revision=new_revision,
            applied=applied,
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

    # -- refinement internals ---------------------------------------------

    @staticmethod
    def _classify_changes(decision: Any):
        """Split validated engine output into accepted and rejected.

        There is no pending bucket any more. A change is either proven safe to
        write by the engine and rechecked here, or it is rejected with a reason.
        """
        accepted: list[ValidatedChange] = []
        rejected: list[dict] = []
        if isinstance(decision, ProfileReviewResult):
            for change in decision.changes:
                if change.accepted:
                    accepted.append(change)
            rejected.extend(decision.rejected)
            return accepted, rejected
        for item in getattr(decision, "changes", []) or []:
            if isinstance(item, ValidatedChange) and item.accepted:
                accepted.append(item)
            elif isinstance(item, ProfileChangeProposal):
                rejected.append(
                    {
                        "operation": item.operation,
                        "field_key": item.field_key,
                        "reason": "proposal_not_validated",
                    }
                )
            else:
                rejected.append({"reason": "unsupported_change"})
        return accepted, rejected

    def _reject(
        self,
        proposal: ProfileChangeProposal,
        reason: str,
        **extra: Any,
    ) -> dict:
        payload = {
            "operation": proposal.operation,
            "field_key": proposal.field_key,
            "status": "REJECTED",
            "reason": reason,
        }
        payload.update(extra)
        return payload

    async def _precheck_changes(
        self,
        person_id: str,
        changes: list[ValidatedChange],
        *,
        allow_strict: bool = False,
    ):
        """Defence in depth around the engine's validation.

        Every failure here is a rejection with a reason. Nothing is parked in a
        queue for a second human confirmation. ``allow_strict`` exists only for
        the migration command, which proves identity values against the
        original event before calling; the normal review path never sets it.
        """
        accepted: list[ValidatedChange] = []
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
        for change in changes[:MAX_REFINEMENT_CHANGES]:
            proposal = change.proposal
            spec = profile_field_spec(proposal.field_key)
            if spec is None or spec.storage is not ProfileFieldStorage.ENTRY:
                rejected.append(self._reject(proposal, "field_not_refinable"))
                continue
            if (spec.strict_identity or not spec.model_writable) and not allow_strict:
                rejected.append(
                    self._reject(
                        proposal,
                        "strict_identity_requires_explicit_self_report",
                    )
                )
                continue
            if _SENSITIVE_CONTENT.search(proposal.proposed_content or ""):
                rejected.append(
                    self._reject(proposal, "sensitive_attribute_not_auto_saved")
                )
                continue
            targets = list(dict.fromkeys(proposal.target_entry_ids))
            if proposal.operation != "ADD" and not targets:
                rejected.append(
                    self._reject(proposal, "missing_target_entry_ids")
                )
                continue
            found = [entries.get(entry_id) for entry_id in targets]
            if any(item is None for item in found):
                rejected.append(
                    self._reject(proposal, "target_entry_not_in_this_profile")
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
                    self._reject(proposal, str(exc) or "invalid_change_target")
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
                    rejected.append(
                        self._reject(proposal, "field_capacity_reached")
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
                    # belongs to someone else. It can never become valid later.
                    rejected.append(
                        self._reject(proposal, "evidence_event_not_accessible")
                    )
                    continue
            if proposal.evidence_quotes:
                if not evidence:
                    rejected.append(
                        self._reject(proposal, "evidence_quote_without_event")
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
                        self._reject(proposal, "evidence_quote_not_found")
                    )
                    continue
            if not evidence and not proposal.evidence_quotes:
                rejected.append(self._reject(proposal, "no_evidence"))
                continue
            if requires_direct_quote(proposal.field_key) and not proposal.evidence_quotes:
                rejected.append(
                    self._reject(proposal, "trait_inference_requires_review")
                )
                continue
            if proposal.operation != "ADD" and any(
                is_human_confirmed_source(item.get("source"))
                for item in found
                if item is not None
            ):
                rejected.append(
                    self._reject(proposal, "human_confirmed_target_requires_review")
                )
                continue
            if proposal.operation == "ADD" and (
                any(
                    entries_equivalent(
                        item["content"], proposal.proposed_content or ""
                    )
                    for item in entries.values()
                    if item["field_key"] == proposal.field_key
                )
                or any(
                    item.proposal.field_key == proposal.field_key
                    and entries_equivalent(
                        item.proposal.proposed_content,
                        proposal.proposed_content or "",
                    )
                    for item in accepted
                )
            ):
                rejected.append(self._reject(proposal, "already_present"))
                continue
            accepted.append(change)
        omitted = changes[MAX_REFINEMENT_CHANGES:]
        if omitted:
            _LOGGER.warning(
                "profile_changes_omitted person_id=%s omitted=%s limit=%s keys=%s",
                person_id,
                len(omitted),
                MAX_REFINEMENT_CHANGES,
                ",".join(
                    f"{item.proposal.operation}:{item.proposal.field_key}"
                    for item in omitted
                ),
            )
        for change in omitted:
            rejected.append(
                self._reject(change.proposal, "change_limit_exceeded")
            )
        return accepted, rejected

    async def _apply_change_locked(
        self,
        person_id: str,
        change: ValidatedChange,
        *,
        revision: int,
        source_event_id: Optional[str],
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
        if not trusted and (spec.strict_identity or not spec.model_writable):
            raise ValueError("strict_identity_requires_explicit_self_report")
        if _SENSITIVE_CONTENT.search(content):
            raise ValueError("sensitive_attribute_not_auto_saved")
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
        if not trusted:
            if requires_direct_quote(proposal.field_key) and not proposal.evidence_quotes:
                raise ValueError("trait_inference_requires_review")
            if any(is_human_confirmed_source(item.get("source")) for item in targets):
                raise ValueError("human_confirmed_target_requires_review")
        if proposal.operation == "ADD":
            existing = await self.repo.list_entries(
                person_id,
                statuses=(
                    ProfileEntryStatus.ACTIVE.value,
                    ProfileEntryStatus.TENTATIVE.value,
                ),
                field_key=proposal.field_key,
            )
            if any(entries_equivalent(item["content"], content) for item in existing):
                raise ValueError("already_present")
            if spec.max_entries and len(existing) >= spec.max_entries:
                raise ValueError("field_capacity_reached")

        evidence_events = list(
            dict.fromkeys(
                list(proposal.evidence_event_ids)
                + ([source_event_id] if source_event_id else [])
            )
        )
        if not evidence_events and not proposal.evidence_quotes:
            raise ValueError("no_evidence")
        await self._assert_evidence_locked(
            person_id, evidence_events, proposal.evidence_quotes
        )
        source = source_override or ProfileSource.INTERACTION_INFERENCE.value
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

    @staticmethod
    def _normalise_entry_content(
        content: Any, spec: Any = None
    ) -> str:
        text = " ".join(str(content or "").split())
        limit = getattr(spec, "max_chars", 240) if spec is not None else 240
        return text[:limit]

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

            if privacy:
                for event_id in events:
                    await self.repo.delete_evidence_for_source(
                        person_id, event_id=event_id
                    )
                    await self.repo.delete_legacy_candidates_for_source(
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
                    await self.repo.delete_legacy_candidates_for_source(
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
    ) -> None:
        for memory_id in _dedupe(memory_ids):
            await self.repo.insert_evidence(
                person_id=person_id,
                field_key=field_key,
                memory_id=memory_id,
                evidence_type="MEMORY",
            )
        for event_id in _dedupe(event_ids):
            await self.repo.insert_evidence(
                person_id=person_id,
                field_key=field_key,
                source_event_id=event_id,
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


def _parse(value: Any):
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


__all__ = ["PersonProfileService", "ProfileUnauthorized"]
