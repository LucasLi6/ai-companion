"""The Orchestrator: the single ordered pipeline for one utterance.

The pipeline is: validate person & consent -> deduplicate -> persist -> load
conversation context -> retrieve permitted memories -> call the cognitive
provider -> validate the analysis -> apply policy -> persist analysis/candidates
-> enqueue a memory-consolidation job -> update conversation state -> generate a
response when policy says RESPOND -> persist the final action -> report.

V0.0 processes each conversation's events with a single ordered consumer (one
``process_utterance`` call at a time); there is deliberately no distributed
queue.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from companion.domain.enums import (
    CognitiveAction,
    JobType,
    ProcessingStatus,
    ReasonCode,
    ReportStatus,
)
from companion.domain.models import (
    AgentAction,
    CognitiveResult,
    ConversationContext,
    ProcessingReport,
    UtteranceEvent,
    new_id,
    utcnow,
)
from companion.people.registry import PersonRegistry
from companion.storage.repositories import Repositories

from .conversation import ConversationManager
from .policy import Policy


class Orchestrator:
    def __init__(
        self,
        *,
        repos: Repositories,
        registry: PersonRegistry,
        conversations: ConversationManager,
        policy: Policy,
        cognitive,
        response,
        memory,
        jobs=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.registry = registry
        self.conversations = conversations
        self.policy = policy
        self.cognitive = cognitive
        self.response = response
        self.memory = memory
        self.jobs = jobs
        self.logger = logger or logging.getLogger("companion.orchestrator")

    async def process_utterance(self, event: UtteranceEvent) -> ProcessingReport:
        t0 = time.monotonic()
        self.logger.info(
            "event_received event_id=%s conversation_id=%s speaker_person_id=%s addressee_hint=%s",
            event.event_id,
            event.conversation_id,
            event.speaker_person_id,
            event.addressee_hint.value,
        )

        # 1. Validate person and consent.
        person_id = event.speaker_person_id
        if person_id is None:
            return self._denied(event, "unregistered_speaker")
        person = await self.registry.get(person_id)
        if person is None:
            return self._denied(event, "unknown_person")
        consent = await self.registry.get_consent(person_id)
        if consent is None or not consent.allow_processing:
            return self._denied(event, "consent_withheld")

        persist = consent.allow_persistence
        learn = consent.allow_memory_learning

        # 2. Ensure the conversation exists (the utterance row has an FK to it)
        # and load its persistent state (quiet mode, recent utterance ids).
        context = await self.conversations.get_or_create(event.conversation_id)

        # 3. Deduplicate against the same event id.
        existing = None
        if persist:
            existing = await self.repos.utterance.get(event.event_id)
            if existing is not None:
                if not self._same_content(existing, event):
                    return self._conflict(event, existing)
                if existing.processing_status == ProcessingStatus.COMPLETED:
                    return await self._duplicate_report(event.event_id)
                # Retry of a previously failed/partial event: clear stale
                # cognitive outputs and re-run. Never duplicates side effects
                # because analysis/action/candidate writes are keyed by event.
                await self.repos.analysis.clear_outputs(event.event_id)
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.PROCESSING
                )
            else:
                await self._persist_input(event)

        # 4. Retrieve permitted memories (hand-authored facts only).
        memories = await self.memory.retrieve(event.text, person_id)
        self.logger.info(
            "context_loaded conversation_id=%s quiet_mode=%s memories=%d",
            event.conversation_id,
            context.quiet_mode,
            len(memories),
        )

        # 6-12. Cognitive analysis, policy, persistence, response and action.
        steps = [
            f"[EVENT] Saved: {event.event_id}",
            f"[SPEAKER] {person_id}",
            "[CONTEXT] Conversation loaded",
            f"[MEMORY] Retrieved {len(memories)} items",
        ]
        try:
            analysis = await self.cognitive.analyze(event, context, memories)
            if analysis.event_id != event.event_id:
                raise ValueError(
                    f"analysis event_id mismatch: {analysis.event_id!r} != {event.event_id!r}"
                )
            self.logger.info(
                "cognitive_analysis_completed event_id=%s action=%s reason_code=%s provider=%s",
                event.event_id,
                analysis.action.value,
                analysis.reason_code.value,
                analysis.provider,
            )

            final_action, final_reason = self.policy.decide(analysis, context)
            steps.append("[COGNITION] Mock analysis completed")
            steps.append(f"[DECISION] {final_action.value}")

            analysis_id: Optional[str] = None
            job_id: Optional[str] = None
            if persist:
                analysis_id = new_id("ana")
                await self.repos.analysis.save_analysis(analysis, analysis_id)
                if learn:
                    for cand in analysis.memory_candidates:
                        await self.repos.analysis.save_candidate(cand)
                    if self.jobs is not None:
                        job_id = await self.jobs.enqueue(
                            JobType.MEMORY_CONSOLIDATION,
                            f"memory:{event.event_id}",
                            {"event_id": event.event_id},
                        )

            # 10. Update conversation state (survives restarts).
            await self.conversations.record_last_action(
                event.conversation_id, final_action.value
            )

            # 11. Generate a response only when policy allows RESPOND.
            response_text: Optional[str] = None
            if final_action == CognitiveAction.RESPOND:
                response_text = await self.response.generate(event, analysis, context)
                steps.append(f"[RESPONSE] {response_text}")

            # 12. Persist the final action.
            if persist:
                await self.repos.analysis.save_action(
                    AgentAction(
                        action_id=new_id("act"),
                        event_id=event.event_id,
                        action_type=final_action,
                        response_text=response_text,
                        reason_code=final_reason,
                        status="COMPLETED",
                        created_at=utcnow(),
                    )
                )

            if job_id is not None:
                steps.append("[MEMORY_JOB] Queued")

            if persist:
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.COMPLETED
                )

            steps.append("[STATUS] Completed")
            self.logger.info(
                "action_saved event_id=%s action=%s reason_code=%s duration_ms=%.2f status=COMPLETED",
                event.event_id,
                final_action.value,
                final_reason.value,
                (time.monotonic() - t0) * 1000,
            )
            return ProcessingReport(
                event_id=event.event_id,
                status=ReportStatus.COMPLETED.value,
                action=final_action,
                response_text=response_text,
                reason_code=final_reason,
                analysis_id=analysis_id,
                job_id=job_id,
                steps=steps,
            )
        except Exception as exc:  # provider failure — keep the event retryable.
            if persist:
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.FAILED
                )
            self.logger.error(
                "processing_failed event_id=%s error=%s duration_ms=%.2f",
                event.event_id,
                exc,
                (time.monotonic() - t0) * 1000,
            )
            steps.append(f"[STATUS] FAILED: {exc}")
            return ProcessingReport(
                event_id=event.event_id,
                status=ReportStatus.FAILED.value,
                error=str(exc),
                steps=steps,
            )

    # -- helpers -----------------------------------------------------------

    async def _persist_input(self, event: UtteranceEvent) -> None:
        to_save = event.model_copy(update={"processing_status": ProcessingStatus.PROCESSING})
        await self.repos.utterance.save(to_save)
        self.logger.info(
            "event_persisted event_id=%s conversation_id=%s speaker_person_id=%s",
            event.event_id,
            event.conversation_id,
            event.speaker_person_id,
        )

    def _same_content(self, stored: UtteranceEvent, incoming: UtteranceEvent) -> bool:
        return (
            stored.text == incoming.text
            and stored.conversation_id == incoming.conversation_id
            and stored.speaker_person_id == incoming.speaker_person_id
            and stored.addressee_hint == incoming.addressee_hint
            and stored.source == incoming.source
        )

    def _denied(self, event: UtteranceEvent, reason: str) -> ProcessingReport:
        self.logger.warning(
            "permission_denied event_id=%s reason=%s", event.event_id, reason
        )
        return ProcessingReport(
            event_id=event.event_id,
            status=ReportStatus.PERMISSION_DENIED.value,
            reason_code=ReasonCode.PERMISSION_DENIED,
            steps=[
                "[STATUS] PERMISSION_DENIED",
                f"[REASON] {reason}",
            ],
        )

    def _conflict(
        self, incoming: UtteranceEvent, stored: UtteranceEvent
    ) -> ProcessingReport:
        self.logger.warning(
            "event_conflict event_id=%s stored_text_mismatch", incoming.event_id
        )
        return ProcessingReport(
            event_id=incoming.event_id,
            status=ReportStatus.CONFLICT.value,
            steps=[
                f"[EVENT] Conflict: event_id {incoming.event_id} already exists with different content",
                "[STATUS] CONFLICT",
            ],
        )

    async def _duplicate_report(self, event_id: str) -> ProcessingReport:
        action = await self.repos.analysis.get_action(event_id)
        return ProcessingReport(
            event_id=event_id,
            status=ReportStatus.DUPLICATE.value,
            action=action.action_type if action else None,
            response_text=action.response_text if action else None,
            reason_code=action.reason_code if action else None,
            steps=["[STATUS] Duplicate (already completed)"],
        )
