"""Durable write-through for model-proposed memory candidates.

This is the layer that turns a candidate into a real, retrievable memory —
or decides, truthfully, that no memory should be written. It is deliberately
conservative and entirely local:

- **Trusted inputs only.** The speaker and subject come from the stored event,
  never from the model. A model-provided subject id can only *restrict* a
  write (an unknown/unauthorised third party is denied); it can never turn a
  third-party statement into a self-report.
- **Four categories.**
  A — direct, clear self-report: may become ``ACTIVE`` (self_reported, never
  human_confirmed). B — uncertain/inferred: ``CANDIDATE`` (or ``NEEDS_REVIEW``
  when it collides with a human-confirmed fact). C — third-party report:
  ``CANDIDATE`` with ``third_party_reported`` set, preserving the reporter.
  D — hypothetical, skipped or empty: no memory at all.
- **One write per candidate.** The candidate -> memory mapping
  (``memory_candidate_writes``) and the memory are committed in the same
  transaction, with ``memory_provenance.candidate_id`` uniquely indexed. A
  retry reads the recorded outcome instead of writing a second memory.
- **History is preserved.** A genuine correction updates the current-state
  memory in place (the old content stays in ``memory_revisions``); a conflict
  with a human-confirmed fact becomes ``NEEDS_REVIEW`` and never overwrites it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from companion.domain.enums import (
    AssertionType,
    CertaintyCategory,
    MemoryCandidateStatus,
    MemoryKind,
    MemoryOperation,
    MemoryStatus,
    MemoryWriteOutcome,
    RegistrationStatus,
)
from companion.domain.models import MemoryCandidate, UtteranceEvent
from companion.memory.candidates import normalize_candidate_content
from companion.memory.keys import ClaimKey, claim_is_grounded, extract_claim_key
from companion.memory.models import AccessContext, MemoryItem
from companion.memory.repository import MemoryRepository
from companion.memory.service import MemoryService
from companion.memory.topics import canonicalize_topic
from companion.memory.transience import is_temporary_emotion_statement
from companion.storage.repositories import Repositories

_CONFLICT_STATUSES = [
    MemoryStatus.ACTIVE.value,
    MemoryStatus.CANDIDATE.value,
    MemoryStatus.NEEDS_REVIEW.value,
    MemoryStatus.CONFLICTED.value,
]

_SELF_ASSERTIONS = (AssertionType.SELF_REPORT, AssertionType.CORRECTION)


@dataclass
class _Plan:
    """The local decision for one candidate, before any write happens."""

    outcome: MemoryWriteOutcome | None  # set for terminal no-write decisions
    content: str
    kind: MemoryKind
    status: MemoryStatus
    assertion: AssertionType = AssertionType.UNKNOWN
    certainty: CertaintyCategory = CertaintyCategory.UNKNOWN
    topic: Optional[str] = None
    key: Optional[ClaimKey] = None
    self_reported: bool = False
    third_party_reported: bool = False
    model_inferred: bool = False
    detail: str | None = None


class MemoryWriter:
    """Consolidate candidates into durable memories, idempotently."""

    def __init__(
        self,
        repos: Repositories,
        service: MemoryService,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.service = service
        self.mem_repo: MemoryRepository = service.repo
        self.logger = logger or logging.getLogger("companion.memory.consolidation")

    # -- entry points ------------------------------------------------------

    async def consolidate_event(
        self, event_id: str, *, only_explicit: bool = False
    ) -> list[dict]:
        """Process the candidates of one event and return truthful outcomes.

        ``only_explicit`` is used by the bounded synchronous write that runs
        before an explicit "remember this" acknowledgement; ordinary
        candidates stay with the background job.
        """
        event = await self.repos.utterance.get(event_id)
        if event is None:
            raise ConsolidationError(f"source event {event_id} not found")
        candidates = await self.repos.analysis.get_candidates(event_id)
        outcomes: list[dict] = []
        for candidate in candidates:
            if only_explicit and not candidate.explicit_remember_request:
                continue
            if candidate.status in (
                MemoryCandidateStatus.CONSOLIDATED,
                MemoryCandidateStatus.SKIPPED,
            ):
                # Already decided. Still surface the recorded outcome.
                mapping = await self.repos.analysis.get_candidate_write(
                    candidate.candidate_id
                )
                if mapping is not None:
                    outcomes.append(mapping)
                continue
            try:
                outcomes.append(await self.consolidate_candidate(candidate, event))
            except Exception as exc:
                await self._mark_failed(candidate, exc)
                raise
        return outcomes

    async def consolidate_candidate(
        self, candidate: MemoryCandidate, event: UtteranceEvent
    ) -> dict:
        # Idempotency anchor: a candidate already mapped is never re-written.
        mapping = await self.repos.analysis.get_candidate_write(candidate.candidate_id)
        if mapping is not None:
            return mapping

        # Crash-recovery fallback: provenance is committed with the memory, so
        # a committed memory is discoverable even if the mapping was lost.
        recovered = await self.mem_repo.find_memory_by_candidate(
            candidate.candidate_id
        )
        if recovered is not None:
            return await self._finish(
                candidate,
                MemoryWriteOutcome.SAVED,
                memory_id=recovered,
                detail="recovered_from_provenance",
            )

        plan = await self._classify(candidate, event)
        if plan.outcome is not None:
            return await self._finish(
                candidate,
                plan.outcome,
                detail=plan.detail,
                candidate_status=self._status_for(plan.outcome),
            )

        return await self._write(candidate, event, plan)

    # -- classification ----------------------------------------------------

    async def _classify(
        self, candidate: MemoryCandidate, event: UtteranceEvent
    ) -> _Plan:
        speaker_id = event.speaker_person_id
        content = " ".join(candidate.content.split()).strip()
        assertion = candidate.assertion_type
        certainty = candidate.certainty_category
        topic = canonicalize_topic(candidate.topic, content)
        key = extract_claim_key(content)
        kind = candidate.kind or (
            MemoryKind.PROJECT_STATE
            if topic is not None and topic.startswith("project")
            else MemoryKind.PERSONAL_FACT
        )

        denied = _Plan(
            outcome=MemoryWriteOutcome.DENIED,
            content=content,
            kind=kind,
            status=MemoryStatus.CANDIDATE,
        )

        # Category D: nothing durable to write.
        if not speaker_id:
            return _Plan(
                outcome=MemoryWriteOutcome.DENIED,
                content=content,
                kind=kind,
                status=MemoryStatus.CANDIDATE,
                detail="no_registered_speaker",
            )

        # Authorisation is the first gate: a revoked or suspended speaker (or
        # withdrawn learning consent) denies every write, even one that would
        # otherwise be discarded as empty or hypothetical.
        if not await self._learning_authorized(speaker_id):
            denied.detail = "source_authorization_revoked"
            return denied

        if len(content) < 2 or candidate.operation == MemoryOperation.SKIP:
            return _Plan(
                outcome=MemoryWriteOutcome.NO_MEMORY,
                content=content,
                kind=kind,
                status=MemoryStatus.CANDIDATE,
                detail="skipped_or_empty",
            )
        if assertion == AssertionType.HYPOTHETICAL:
            return _Plan(
                outcome=MemoryWriteOutcome.NO_MEMORY,
                content=content,
                kind=kind,
                status=MemoryStatus.CANDIDATE,
                detail="hypothetical",
            )
        # A passing emotional state is context for this conversation, not a
        # durable fact about the person. Store it in the emotion/scene layers,
        # not in long-term memory.
        if (
            candidate.subject_person_id in (None, speaker_id)
            and assertion in _SELF_ASSERTIONS
            and is_temporary_emotion_statement(content)
        ):
            return _Plan(
                outcome=MemoryWriteOutcome.NO_MEMORY,
                content=content,
                kind=kind,
                status=MemoryStatus.CANDIDATE,
                detail="temporary_emotion_not_durable",
            )

        hinted = candidate.subject_person_id
        third_party = hinted is not None and hinted != speaker_id
        if assertion == AssertionType.THIRD_PARTY_REPORT:
            third_party = True
        if third_party and hinted is not None and hinted != speaker_id:
            if not await self._learning_authorized(hinted):
                denied.detail = "subject_privacy_denied"
                return denied

        if third_party:
            # Category C: preserved with attribution, never authoritative.
            return _Plan(
                outcome=None,
                content=content,
                kind=kind,
                status=MemoryStatus.CANDIDATE,
                assertion=AssertionType.THIRD_PARTY_REPORT,
                certainty=certainty,
                topic=topic,
                key=key,
                third_party_reported=True,
                model_inferred=True,
            )

        evidence = self._direct_evidence(candidate, event, content)

        if assertion == AssertionType.INFERENCE or certainty == CertaintyCategory.INFERRED:
            return _Plan(
                outcome=None,
                content=content,
                kind=kind,
                status=MemoryStatus.CANDIDATE,
                assertion=assertion,
                certainty=certainty,
                topic=topic,
                key=key,
                self_reported=assertion in _SELF_ASSERTIONS,
                model_inferred=True,
            )

        is_self_report = assertion in _SELF_ASSERTIONS
        if (
            is_self_report
            and certainty in (CertaintyCategory.DIRECT, CertaintyCategory.UNKNOWN)
            and evidence
        ) or (candidate.explicit_remember_request and evidence):
            # Category A: direct, clear self-reported fact.
            return _Plan(
                outcome=None,
                content=content,
                kind=kind,
                status=MemoryStatus.ACTIVE,
                assertion=assertion if is_self_report else AssertionType.SELF_REPORT,
                certainty=(
                    certainty
                    if certainty != CertaintyCategory.UNKNOWN
                    else CertaintyCategory.DIRECT
                ),
                topic=topic,
                key=key,
                self_reported=True,
                model_inferred=False,
            )

        # Category B: uncertain or inferred — durable, but not authoritative.
        return _Plan(
            outcome=None,
            content=content,
            kind=kind,
            status=MemoryStatus.CANDIDATE,
            assertion=assertion,
            certainty=(
                certainty
                if certainty != CertaintyCategory.UNKNOWN
                else CertaintyCategory.UNCERTAIN
            ),
            topic=topic,
            key=key,
            self_reported=is_self_report,
            model_inferred=True,
        )

    def _direct_evidence(
        self, candidate: MemoryCandidate, event: UtteranceEvent, content: str
    ) -> bool:
        """Whether the utterance itself directly supports the candidate.

        A model-supplied evidence span is not sufficient by itself: quoting a
        real sentence does not prove the proposed claim follows from it. The
        claim's own wording (or, for preference/project/identity claims, the
        same entity plus the matching predicate) must be grounded in the
        utterance. This is what stops "今天写了代码" from becoming an
        authoritative "特别喜欢编程" memory.
        """
        return claim_is_grounded(
            content,
            event.text,
            evidence_span=candidate.source_evidence_span,
        )

    @staticmethod
    def _same_claim(key: ClaimKey, row: dict) -> bool:
        """Whether an existing memory describes the same attribute/entity.

        Stored provenance keys are authoritative when present. Rows written
        before the keys existed are parsed from their content on the fly, so a
        legacy memory participates correctly without a destructive backfill.
        """
        entity = row.get("entity_key")
        attribute = row.get("attribute_key")
        if entity is None or attribute is None:
            existing = extract_claim_key(row.get("content") or "")
            if existing is None:
                return False
            entity, attribute = existing.entity_key, existing.attribute_key
        return key.entity_key == entity and key.attribute_key == attribute

    # -- writes ------------------------------------------------------------

    async def _write(
        self, candidate: MemoryCandidate, event: UtteranceEvent, plan: _Plan
    ) -> dict:
        speaker_id = event.speaker_person_id
        assert speaker_id is not None
        access = AccessContext(
            actor_id=speaker_id, is_admin=False, role="cognitive"
        )
        norm = normalize_candidate_content(plan.content)

        # One transaction covers the lookups, the memory write, the provenance
        # and the candidate -> memory mapping: either all commit or none do.
        async with self.repos.db.transaction():
            existing = await self.mem_repo.list_for_subject(
                speaker_id, statuses=_CONFLICT_STATUSES, limit=50
            )
            same = next(
                (
                    row
                    for row in existing
                    if normalize_candidate_content(row["content"]) == norm
                ),
                None,
            )
            if same is not None:
                return await self._handle_existing(candidate, same, plan, access)

            conflict = None
            if plan.key is not None:
                conflict = next(
                    (
                        row
                        for row in existing
                        if normalize_candidate_content(row["content"]) != norm
                        and self._same_claim(plan.key, row)
                    ),
                    None,
                )
            if conflict is not None:
                outcome = await self._handle_conflict(
                    candidate, event, conflict, plan, access
                )
                if outcome is not None:
                    return outcome
            return await self._create(candidate, event, plan, access)

    async def _handle_existing(
        self, candidate: MemoryCandidate, row: dict, plan: _Plan, access: AccessContext
    ) -> dict:
        if (
            plan.status == MemoryStatus.ACTIVE
            and row["status"] == MemoryStatus.CANDIDATE.value
            and not row["human_confirmed"]
            and row["created_by"] == access.actor_id
        ):
            # The same fact is now directly supported: promote the candidate
            # rather than duplicating it.
            updated = await self.service.update(
                row["memory_id"],
                access=access,
                expected_version=row["version"],
                status=MemoryStatus.ACTIVE,
                reason="promoted_by_direct_self_report",
            )
            await self.mem_repo.add_provenance_event(
                row["memory_id"],
                event_type="promoted",
                source_event_id=candidate.source_event_id,
                candidate_id=candidate.candidate_id,
                speaker_person_id=None,
                subject_person_id=access.actor_id,
                old_version=row["version"],
                new_version=updated.version,
                reason="promoted_by_direct_self_report",
                payload={"from_status": row["status"], "to_status": "ACTIVE"},
            )
            return await self._finish(
                candidate,
                MemoryWriteOutcome.UPDATED,
                memory_id=row["memory_id"],
                detail="promoted_candidate_to_active",
            )
        return await self._finish(
            candidate,
            MemoryWriteOutcome.ALREADY_EXISTS,
            memory_id=row["memory_id"],
            detail="duplicate_content",
        )

    async def _handle_conflict(
        self,
        candidate: MemoryCandidate,
        event: UtteranceEvent,
        row: dict,
        plan: _Plan,
        access: AccessContext,
    ) -> dict | None:
        """Return a finished outcome, or ``None`` to create a new memory."""
        if row["human_confirmed"]:
            # Never overwrite a human-confirmed fact; surface a review item.
            item = await self._create_review(
                candidate, event, plan, access, row["memory_id"]
            )
            return await self._finish(
                candidate,
                MemoryWriteOutcome.PENDING_REVIEW,
                memory_id=item.memory_id,
                detail="conflicts_with_human_confirmed",
                candidate_status=MemoryCandidateStatus.NEEDS_REVIEW,
            )

        is_current_state = plan.kind in (
            MemoryKind.PERSONAL_FACT,
            MemoryKind.PROJECT_STATE,
        )
        explicit_update = (
            plan.assertion == AssertionType.CORRECTION
            or candidate.operation == MemoryOperation.UPDATE
        )
        genuine_self_update = (
            row["self_reported"]
            and row["status"] == MemoryStatus.ACTIVE.value
            and (explicit_update or (plan.status == MemoryStatus.ACTIVE and is_current_state))
        )
        if genuine_self_update:
            try:
                updated = await self.service.update(
                    row["memory_id"],
                    access=access,
                    expected_version=row["version"],
                    content=plan.content,
                    status=MemoryStatus.ACTIVE,
                    reason=(
                        "explicit_correction"
                        if plan.assertion == AssertionType.CORRECTION
                        else "self_report_update"
                    ),
                )
                await self.mem_repo.add_provenance_event(
                    row["memory_id"],
                    event_type=(
                        "corrected"
                        if plan.assertion == AssertionType.CORRECTION
                        else "updated"
                    ),
                    source_event_id=event.event_id,
                    candidate_id=candidate.candidate_id,
                    speaker_person_id=event.speaker_person_id,
                    subject_person_id=event.speaker_person_id,
                    old_version=row["version"],
                    new_version=updated.version,
                    reason=(
                        "explicit_correction"
                        if plan.assertion == AssertionType.CORRECTION
                        else "self_report_update"
                    ),
                    payload={
                        "old_content": row["content"],
                        "new_content": plan.content,
                        "entity_key": plan.key.entity_key if plan.key else None,
                        "attribute_key": plan.key.attribute_key if plan.key else None,
                    },
                )
            except Exception as exc:  # permission/version race: review instead
                self.logger.warning(
                    "memory_update_failed memory_id=%s error=%s", row["memory_id"], exc
                )
            else:
                return await self._finish(
                    candidate,
                    MemoryWriteOutcome.UPDATED,
                    memory_id=row["memory_id"],
                    detail="updated_current_state",
                )

        if plan.status == MemoryStatus.CANDIDATE:
            # Competing uncertain proposals coexist; no overwrite, no review.
            return None
        item = await self._create_review(
            candidate, event, plan, access, row["memory_id"]
        )
        return await self._finish(
            candidate,
            MemoryWriteOutcome.PENDING_REVIEW,
            memory_id=item.memory_id,
            detail="unresolved_conflict",
            candidate_status=MemoryCandidateStatus.NEEDS_REVIEW,
        )

    async def _create(
        self,
        candidate: MemoryCandidate,
        event: UtteranceEvent,
        plan: _Plan,
        access: AccessContext,
    ) -> dict:
        speaker_id = event.speaker_person_id
        assert speaker_id is not None
        item = await self.service.create(
            kind=plan.kind,
            content=plan.content,
            access=access,
            status=plan.status,
            visibility="PRIVATE",
            created_by=speaker_id,
            subjects=[("PERSON", speaker_id)] if not plan.third_party_reported else [],
            self_reported=plan.self_reported,
            third_party_reported=plan.third_party_reported,
            model_inferred=plan.model_inferred,
        )
        await self._record_provenance(
            item.memory_id, candidate, event, plan, new_version=item.version
        )
        if not plan.third_party_reported:
            await self._auto_assign(item.memory_id, speaker_id)
        return await self._finish(
            candidate,
            MemoryWriteOutcome.SAVED,
            memory_id=item.memory_id,
            detail=f"status={plan.status.value}",
        )

    async def _create_review(
        self,
        candidate: MemoryCandidate,
        event: UtteranceEvent,
        plan: _Plan,
        access: AccessContext,
        conflicts_with: str,
    ) -> MemoryItem:
        speaker_id = access.actor_id
        item = await self.service.create(
            kind=plan.kind,
            content=plan.content,
            access=access,
            status=MemoryStatus.NEEDS_REVIEW,
            visibility="PRIVATE",
            created_by=speaker_id,
            subjects=[("PERSON", speaker_id)],
            self_reported=plan.self_reported,
            third_party_reported=plan.third_party_reported,
            model_inferred=True,
        )
        await self.mem_repo.add_link(item.memory_id, conflicts_with, "conflicts_with")
        await self._record_provenance(
            item.memory_id, candidate, event, plan, new_version=item.version
        )
        return item

    async def _record_provenance(
        self,
        memory_id: str,
        candidate: MemoryCandidate,
        event: UtteranceEvent,
        plan: _Plan,
        *,
        new_version: int | None = None,
    ) -> None:
        await self.mem_repo.add_provenance(
            memory_id,
            source_event_id=event.event_id,
            candidate_id=candidate.candidate_id,
            speaker_person_id=event.speaker_person_id,
            subject_person_id=(
                None if plan.third_party_reported else event.speaker_person_id
            ),
            source_type="utterance",
            assertion_type=plan.assertion.value,
            certainty_category=plan.certainty.value,
            topic=plan.topic,
            entity_key=plan.key.entity_key if plan.key else None,
            attribute_key=plan.key.attribute_key if plan.key else None,
            explicit_remember_request=candidate.explicit_remember_request,
            self_reported=plan.self_reported,
            third_party_reported=plan.third_party_reported,
            model_inferred=plan.model_inferred,
            new_version=new_version,
        )

    async def _auto_assign(self, memory_id: str, person_id: str) -> None:
        created = await self.mem_repo.assign(
            memory_id, person_id, created_by="system"
        )
        if not created:
            return
        await self.repos.audit.record(
            actor="system",
            action="assign_memory",
            entity_type="memory",
            entity_id=memory_id,
            detail=f"person_id={person_id} auto=1",
        )

    # -- bookkeeping -------------------------------------------------------

    async def _finish(
        self,
        candidate: MemoryCandidate,
        outcome: MemoryWriteOutcome,
        *,
        memory_id: str | None = None,
        detail: str | None = None,
        candidate_status: MemoryCandidateStatus | None = None,
    ) -> dict:
        if candidate_status is None:
            candidate_status = self._status_for(outcome)
        async with self.repos.db.transaction():
            mapping = await self.repos.analysis.record_candidate_write(
                candidate.candidate_id,
                source_event_id=candidate.source_event_id,
                outcome=outcome.value,
                memory_id=memory_id,
                detail=detail,
            )
            await self.repos.analysis.update_candidate_status(
                candidate.candidate_id, candidate_status
            )
        self.logger.info(
            "memory_candidate_consolidated candidate_id=%s outcome=%s memory_id=%s",
            candidate.candidate_id,
            outcome.value,
            memory_id,
        )
        return mapping

    async def _mark_failed(self, candidate: MemoryCandidate, exc: Exception) -> None:
        self.logger.error(
            "memory_candidate_failed candidate_id=%s error=%s",
            candidate.candidate_id,
            type(exc).__name__,
        )
        try:
            await self.repos.analysis.update_candidate_status(
                candidate.candidate_id, MemoryCandidateStatus.FAILED
            )
        except Exception:
            # The database itself is failing; the job retry will try again.
            pass

    @staticmethod
    def _status_for(outcome: MemoryWriteOutcome) -> MemoryCandidateStatus:
        if outcome == MemoryWriteOutcome.DENIED:
            return MemoryCandidateStatus.SKIPPED
        if outcome == MemoryWriteOutcome.PENDING_REVIEW:
            return MemoryCandidateStatus.NEEDS_REVIEW
        if outcome == MemoryWriteOutcome.FAILED:
            return MemoryCandidateStatus.FAILED
        # SAVED / UPDATED / ALREADY_EXISTS / NO_MEMORY: the candidate has been
        # processed. "CONSOLIDATED" says only that; the mapping carries the
        # truthful durable result.
        return MemoryCandidateStatus.CONSOLIDATED

    async def _learning_authorized(self, person_id: str | None) -> bool:
        if person_id is None:
            return False
        person = await self.repos.person.get(person_id)
        if person is None or person.registration_status != RegistrationStatus.ACTIVE:
            return False
        consent = await self.repos.person.get_consent(person_id)
        return consent is not None and consent.allow_memory_learning


class ConsolidationError(Exception):
    pass
