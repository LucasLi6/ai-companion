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
from dataclasses import dataclass, field
from typing import Optional

from companion.domain.enums import (
    MemoryWriteOutcome,
    RegistrationStatus,
)
from companion.domain.models import UtteranceEvent
from companion.memory.candidates import prepare_candidates
from companion.memory.consolidation import MemoryWriter
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
