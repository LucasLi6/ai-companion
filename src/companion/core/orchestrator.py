"""The Orchestrator: the single ordered pipeline for one utterance.

The pipeline is: validate person & consent -> deduplicate -> persist -> load
conversation context -> retrieve permitted memories -> call the cognitive
provider -> validate the analysis -> apply policy -> persist analysis/candidates
and enqueue a memory-consolidation job (atomically) -> update conversation
state -> generate a response when policy says RESPOND -> persist the final
action -> report.

The pipeline is *resumable*: each durable stage leaves a database record
(utterance, analysis, candidates, job, action) and a retry re-uses whatever
already succeeded instead of clearing and re-running. A ``ResponseProvider``
failure therefore never discards a valid analysis or strands a memory job, and
an already-persisted action is never re-emitted.

Each conversation is processed by a single ordered consumer (one
``process_utterance`` call at a time, enforced by a per-conversation
``asyncio.Lock``). Different conversations run concurrently. Events within a
conversation are handled in *submission order* (the order in which
``process_utterance`` acquires the conversation lock), which is the system's
ordering guarantee; ``occurred_at`` is caller timing, not a scheduling
guarantee.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from companion.domain.enums import (
    CognitiveAction,
    ProcessingStatus,
    ReasonCode,
    RegistrationStatus,
    ReportStatus,
)
from companion.domain.models import (
    AgentAction,
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
        self._conversation_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        lock = self._conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_id] = lock
        return lock

    async def process_utterance(self, event: UtteranceEvent) -> ProcessingReport:
        # Per-conversation serialization: events in the same conversation run
        # one at a time, in submission order.
        async with self._lock_for(event.conversation_id):
            return await self._process(event)

    async def _process(self, event: UtteranceEvent) -> ProcessingReport:
        t0 = time.monotonic()
        self.logger.info(
            "event_received event_id=%s conversation_id=%s speaker_person_id=%s addressee_hint=%s",
            event.event_id,
            event.conversation_id,
            event.speaker_person_id,
            event.addressee_hint.value,
        )

        # 1. Validate person, registration status and consent.
        person_id = event.speaker_person_id
        if person_id is None:
            return self._denied(event, "unregistered_speaker")
        person = await self.registry.get(person_id)
        if person is None:
            return self._denied(event, "unknown_person")
        if person.registration_status != RegistrationStatus.ACTIVE:
            return self._denied(event, f"person_{person.registration_status.value}")
        consent = await self.registry.get_consent(person_id)
        if consent is None or not consent.allow_processing:
            return self._denied(event, "consent_withheld")

        persist = consent.allow_persistence
        learn = consent.allow_memory_learning

        # 2. Load conversation context. With persistence the conversation is
        # created/updated; without persistence it is *read-only* (an existing
        # conversation's quiet mode is still honoured, but nothing is created
        # or written for this interaction).
        if persist:
            context = await self.conversations.get_or_create(event.conversation_id)
        else:
            context = await self.repos.conversation.get(event.conversation_id)
            if context is None:
                context = ConversationContext(conversation_id=event.conversation_id)

        # 3. Deduplicate / conflict / resume against the stored event.
        existing = None
        if persist:
            existing = await self.repos.utterance.get(event.event_id)
            if existing is not None:
                if existing.identity_fingerprint() != event.identity_fingerprint():
                    return self._conflict(event, existing)
                if existing.processing_status == ProcessingStatus.COMPLETED:
                    return await self._duplicate_report(event.event_id)
                # Otherwise resume: earlier stages that already succeeded are
                # re-used, never cleared.
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

        steps = [
            f"[EVENT] Saved: {event.event_id}",
            f"[SPEAKER] {person_id}",
            "[CONTEXT] Conversation loaded",
            f"[MEMORY] Retrieved {len(memories)} items",
        ]

        try:
            # 5. Cognitive analysis: reuse a durable analysis, else compute.
            analysis = None
            analysis_id: Optional[str] = None
            if persist:
                analysis = await self.repos.analysis.get_analysis(event.event_id)
                analysis_id = await self.repos.analysis.get_analysis_id(event.event_id)

            job_id: Optional[str] = None
            if analysis is None:
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
                steps.append(f"[COGNITION] Analysis completed by {analysis.provider}")
                if persist:
                    analysis_id = new_id("ana")
                    candidates = analysis.memory_candidates if learn else []
                    job_key = (
                        f"memory:{event.event_id}"
                        if (learn and self.jobs is not None)
                        else None
                    )
                    job_id = await self.repos.persist_cognitive_outcome(
                        analysis,
                        analysis_id,
                        candidates=candidates,
                        job_key=job_key,
                        job_payload={"event_id": event.event_id},
                    )
            else:
                self.logger.info(
                    "cognitive_analysis_reused event_id=%s provider=%s",
                    event.event_id,
                    analysis.provider,
                )
                steps.append("[COGNITION] Reused existing analysis")
                if learn and self.jobs is not None:
                    job = await self.repos.job.get_by_key(f"memory:{event.event_id}")
                    if job is not None:
                        job_id = job.job_id

            # 6. Policy decides the final action from the (reused or fresh)
            # analysis.
            final_action, final_reason = self.policy.decide(analysis, context)
            steps.append(f"[DECISION] {final_action.value}")

            # 7. Reuse the durable action if it already exists, else generate a
            # response and persist the action.
            existing_action = (
                await self.repos.analysis.get_action(event.event_id)
                if persist
                else None
            )
            response_text: Optional[str] = None
            if existing_action is not None:
                final_action = existing_action.action_type
                final_reason = existing_action.reason_code
                response_text = existing_action.response_text
                steps.append("[ACTION] Reused existing action")
            else:
                if final_action == CognitiveAction.RESPOND:
                    response_text = await self.response.generate(
                        event, analysis, context
                    )
                    steps.append(f"[RESPONSE] {response_text}")
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

            # 8. Finalise the durable conversation + event state.
            if persist:
                await self.conversations.record_last_action(
                    event.conversation_id, final_action.value
                )
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.COMPLETED
                )

            if job_id is not None:
                steps.append("[MEMORY_JOB] Queued")
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
            "event_conflict event_id=%s stored_content_mismatch", incoming.event_id
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
