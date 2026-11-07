"""Explicit memory saves and corrections requested by the current turn.

The semantic router may report ``MEMORY_SAVE`` / ``MEMORY_CORRECTION`` as a
processing intent. Those are *requests from the model*, not permission to
write: only this trusted service can turn one into a durable memory. It reuses
the exact same candidate pipeline as background learning — extraction,
trust-boundary sanitisation, consolidation — so an explicit save can never
create a parallel memory system or bypass consent.

Design rules:

- only the two explicit intents are handled here; everything else is
  ``NO_MEMORY`` and changes nothing;
- the speaker and their stored consent are re-checked here, not trusted from
  the route;
- candidates already produced for this event are reused instead of calling the
  model a second time;
- the write is committed before the caller may acknowledge it, and a retry
  reads the recorded candidate -> memory mapping rather than writing twice;
- no keyword is re-interpreted: if the router did not mark the intent, this
  service does nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from companion.domain.enums import (
    MemoryWriteOutcome,
    RegistrationStatus,
    MemoryStatus,
)
from companion.domain.models import UtteranceEvent
from companion.memory.candidates import prepare_candidates
from companion.memory.consolidation import MemoryWriter
from companion.memory.keys import claim_is_grounded
from companion.memory.models import AccessContext
from companion.routing.models import (
    ProcessingIntent,
    RoutingContext,
    ValidatedRouteDecision,
)
from companion.storage.repositories import Repositories

__all__ = [
    "ExplicitMemoryOperationService",
    "MemoryOperationResult",
]

_EXPLICIT_INTENTS = (
    ProcessingIntent.MEMORY_SAVE,
    ProcessingIntent.MEMORY_CORRECTION,
)

_OUTCOME_PRIORITY = (
    MemoryWriteOutcome.SAVED,
    MemoryWriteOutcome.UPDATED,
    MemoryWriteOutcome.PENDING_REVIEW,
    MemoryWriteOutcome.ALREADY_EXISTS,
    MemoryWriteOutcome.DENIED,
    MemoryWriteOutcome.NO_MEMORY,
)

_PRIOR_TURN_LIMIT = 6


@dataclass
class MemoryOperationResult:
    """Truthful, auditable result of one explicit memory operation."""

    event_id: str
    intent: Optional[str]
    status: str
    outcomes: list[dict] = field(default_factory=list)
    candidate_count: int = 0
    completed: bool = False
    error_code: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "intent": self.intent,
            "status": self.status,
            "outcomes": self.outcomes,
            "candidate_count": self.candidate_count,
            "completed": self.completed,
            "error_code": self.error_code,
        }


class ExplicitMemoryOperationService:
    def __init__(
        self,
        *,
        repos: Repositories,
        cognitive,
        memory_writer: MemoryWriter,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.cognitive = cognitive
        self.memory_writer = memory_writer
        self.logger = logger or logging.getLogger("companion.memory.explicit")

    async def execute(
        self,
        event: UtteranceEvent,
        route: ValidatedRouteDecision,
        consent,
        routing_context: Optional[RoutingContext] = None,
    ) -> MemoryOperationResult:
        intent = self._intent_of(route)
        if intent is None:
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=None,
                status=MemoryWriteOutcome.NO_MEMORY.value,
                completed=True,
            )

        speaker = event.speaker_person_id
        denied = await self._authorization_denied(speaker, consent)
        if denied is not None:
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=intent.value,
                status=MemoryWriteOutcome.DENIED.value,
                completed=True,
                error_code=denied,
            )

        # A follow-up such as "你写入记忆库啊" may contain no factual
        # statement, but can refer to a single earlier name self-report which
        # was stored as an unsearchable candidate by the older grounding code.
        # Recheck that original source locally rather than inventing a new
        # claim from the command sentence.
        recovered = await self._recover_direct_name(
            event, intent, routing_context
        )
        if recovered is not None:
            return recovered

        try:
            candidates = await self._candidates_for(event, routing_context)
        except Exception as exc:
            self.logger.warning(
                "explicit_memory_extraction_failed event_id=%s error=%s",
                event.event_id,
                type(exc).__name__,
            )
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=intent.value,
                status=MemoryWriteOutcome.FAILED.value,
                completed=False,
                error_code="extraction_failed",
            )

        if not candidates:
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=intent.value,
                status=MemoryWriteOutcome.NO_MEMORY.value,
                completed=True,
            )

        try:
            # Only the explicit pass runs here. A partial extraction failure
            # leaves the mapping rows committed and the retry reuses them.
            outcomes = await self.memory_writer.consolidate_event(
                event.event_id, only_explicit=True
            )
        except Exception as exc:
            self.logger.error(
                "explicit_memory_write_failed event_id=%s error=%s",
                event.event_id,
                type(exc).__name__,
            )
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=intent.value,
                status=MemoryWriteOutcome.FAILED.value,
                candidate_count=len(candidates),
                completed=False,
                error_code="write_failed",
            )

        return MemoryOperationResult(
            event_id=event.event_id,
            intent=intent.value,
            status=self._aggregate(outcomes),
            outcomes=outcomes,
            candidate_count=len(candidates),
            completed=True,
        )

    async def _recover_direct_name(
        self,
        event: UtteranceEvent,
        intent: ProcessingIntent,
        routing_context: Optional[RoutingContext],
    ) -> Optional[MemoryOperationResult]:
        """Promote exactly one legacy name candidate on a referential request.

        This is a narrow repair, not a generic write tool or an inference from
        chat history: the old candidate, source event and verbatim evidence
        must all be available and independently validate as a direct self-
        report. No ambiguous or cross-person candidate is promoted.
        """
        if intent != ProcessingIntent.MEMORY_SAVE:
            return None
        speaker = event.speaker_person_id
        if not speaker:
            return None
        # A retry of the same event returns its actual, previously committed
        # promotion, rather than incorrectly claiming that nothing was saved.
        cur = await self.repos.db.execute(
            "SELECT p.memory_id FROM memory_provenance_events p "
            "JOIN memory_subjects s ON s.memory_id = p.memory_id "
            "WHERE p.source_event_id = ? "
            "AND p.event_type = 'promoted_by_explicit_recheck' "
            "AND s.subject_type = 'PERSON' AND s.subject_id = ? LIMIT 1",
            (event.event_id, speaker),
        )
        replay = await cur.fetchone()
        if replay is not None:
            memory_id = replay["memory_id"]
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=intent.value,
                status=MemoryWriteOutcome.UPDATED.value,
                outcomes=[{
                    "outcome": MemoryWriteOutcome.UPDATED.value,
                    "memory_id": memory_id,
                    "detail": "verified_direct_name_promoted",
                }],
                completed=True,
            )

        text = event.text.strip()
        if not re.search(r"(记住|记下来|记忆库|保存|写入|存进去)", text):
            return None
        if re.search(r"(我叫|我是|我的名字|我名叫)", text):
            # A concrete, present-turn assertion follows the normal writer.
            return None
        if routing_context is None:
            return None
        turns = [
            item for item in (routing_context.recent_turns or [])
            if item.get("role") in ("human", "user")
            and item.get("speaker_person_id") == speaker
        ][-6:]
        if not any(
            re.search(r"(我是谁|我的名字|我叫什么|名字|姓名|我叫)", str(item.get("text") or ""))
            for item in turns
        ):
            return None

        repo = self.memory_writer.mem_repo
        rows = await repo.list_for_subject(
            speaker,
            statuses=[MemoryStatus.CANDIDATE.value],
            kinds=["PERSONAL_FACT"],
            limit=50,
        )
        matches = []
        for row in rows:
            if row.get("topic") != "name" or not row.get("self_reported"):
                continue
            if row.get("human_confirmed") or row.get("created_by") != speaker:
                continue
            provenance = await repo.get_provenance(row["memory_id"])
            if not provenance or provenance.get("speaker_person_id") != speaker:
                continue
            if (
                provenance.get("assertion_type") != "SELF_REPORT"
                or provenance.get("certainty_category") != "DIRECT"
            ):
                continue
            source = await self.repos.utterance.get(
                provenance.get("source_event_id") or ""
            )
            if (
                source is None
                or source.speaker_person_id != speaker
                or source.conversation_id != event.conversation_id
            ):
                continue
            candidates = await self.repos.analysis.get_candidates(source.event_id)
            original = next(
                (x for x in candidates
                 if x.candidate_id == provenance.get("candidate_id")),
                None,
            )
            if original is None or not claim_is_grounded(
                original.content,
                source.text,
                evidence_span=original.source_evidence_span,
            ):
                continue
            matches.append(row)
        if len(matches) != 1:
            return None

        row = matches[0]
        access = AccessContext(actor_id=speaker, is_admin=False, role="cognitive")
        try:
            async with self.repos.db.transaction():
                updated = await self.memory_writer.service.update(
                    row["memory_id"],
                    access=access,
                    expected_version=row["version"],
                    status=MemoryStatus.ACTIVE,
                    reason="promoted_by_explicit_recheck",
                )
                await repo.add_provenance_event(
                    row["memory_id"],
                    event_type="promoted_by_explicit_recheck",
                    source_event_id=event.event_id,
                    speaker_person_id=speaker,
                    subject_person_id=speaker,
                    old_version=row["version"],
                    new_version=updated.version,
                    reason="original_source_revalidated",
                )
        except Exception as exc:
            self.logger.error(
                "direct_name_recheck_failed event_id=%s error=%s",
                event.event_id,
                type(exc).__name__,
            )
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=intent.value,
                status=MemoryWriteOutcome.FAILED.value,
                completed=False,
                error_code="direct_name_recheck_failed",
            )
        return MemoryOperationResult(
            event_id=event.event_id,
            intent=intent.value,
            status=MemoryWriteOutcome.UPDATED.value,
            outcomes=[{
                "outcome": MemoryWriteOutcome.UPDATED.value,
                "memory_id": row["memory_id"],
                "detail": "verified_direct_name_promoted",
            }],
            completed=True,
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _intent_of(route: ValidatedRouteDecision) -> Optional[ProcessingIntent]:
        intents = set(getattr(route, "processing_intents", ()) or ())
        # A correction is more specific than a save; handle it as a correction
        # when the model reported both.
        if ProcessingIntent.MEMORY_CORRECTION in intents:
            return ProcessingIntent.MEMORY_CORRECTION
        if ProcessingIntent.MEMORY_SAVE in intents:
            return ProcessingIntent.MEMORY_SAVE
        return None

    async def _authorization_denied(self, speaker: Optional[str], consent) -> str | None:
        if not speaker:
            return "no_registered_speaker"
        person = await self.repos.person.get(speaker)
        if person is None:
            return "unknown_person"
        if person.registration_status != RegistrationStatus.ACTIVE:
            return f"person_{person.registration_status.value}"
        stored = await self.repos.person.get_consent(speaker)
        if stored is None or not stored.allow_memory_learning:
            return "memory_learning_not_granted"
        if not stored.allow_persistence:
            return "persistence_not_granted"
        # The caller's snapshot must not be more permissive than storage.
        if consent is not None and not bool(consent.allow_memory_learning):
            return "memory_learning_not_granted"
        return None

    async def _candidates_for(
        self, event: UtteranceEvent, routing_context: Optional[RoutingContext]
    ):
        existing = await self.repos.analysis.get_candidates(event.event_id)
        if existing:
            # A retry, or a background pass that already extracted this event.
            # The route intent makes these explicit; no second model call.
            await self.repos.analysis.mark_candidates_explicit(event.event_id)
            return existing

        if self.cognitive is None or not hasattr(self.cognitive, "extract_events"):
            raise RuntimeError("no cognitive provider supports extraction")

        prior_turns = None
        speaker_label = "unknown"
        if routing_context is not None:
            speaker_label = routing_context.current_speaker_label
            prior_turns = list(routing_context.recent_turns or [])[
                -_PRIOR_TURN_LIMIT:
            ]
        extracted = await self.cognitive.extract_events(
            [event], speaker_label=speaker_label, prior_turns=prior_turns
        )
        raw = (extracted or {}).get(event.event_id) or []
        prepared = await prepare_candidates(
            self.repos,
            event,
            raw,
            explicit_remember_request=True,
        )
        for candidate in prepared:
            await self.repos.analysis.save_candidate(candidate)
        return prepared

    @staticmethod
    def _aggregate(outcomes: list[dict]) -> str:
        found = {
            str(item.get("outcome") or "").upper()
            for item in outcomes
            if item.get("outcome")
        }
        for outcome in _OUTCOME_PRIORITY:
            if outcome.value in found:
                return outcome.value
        return MemoryWriteOutcome.NO_MEMORY.value
