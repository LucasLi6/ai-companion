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

from typing import Any, Iterable, Optional

from companion.domain.enums import RegistrationStatus
from companion.people.profile_compiler import compile_profile_context
from companion.people.profile_models import (
    CandidateStatus,
    LIST_FIELDS,
    MAX_LIST_ITEMS,
    PersonInteractionProfile,
    ProfileCandidate,
    ProfileField,
    ProfileFieldStatus,
    ProfileMetadata,
    ProfileSource,
    validate_field_value,
)
from companion.people.profile_repository import PersonProfileRepository
from companion.storage.repositories import AuditRepository, PersonRepository

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
        candidates = [
            self._candidate_model(item)
            for item in await self.repo.list_candidates(person_id)
        ]
        return PersonInteractionProfile(
            person_id=person_id,
            display_name=person.display_name if person is not None else "unknown",
            relationship_tier=row["relationship_tier"] or "STRANGER",
            fields=fields,
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

            for item in pending:
                await self.repo.set_candidate_status(
                    item["candidate_id"],
                    CandidateStatus.SUPERSEDED.value,
                    reason="superseded_by_new_evidence",
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
