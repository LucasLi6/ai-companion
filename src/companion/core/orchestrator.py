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

from companion.cognition.models import CognitiveContext
from companion.cognition.silence import (
    detect_explicit_remember,
    detect_explicit_silence,
)
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
from companion.memory.candidates import prepare_candidates
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
        context_builder=None,
        memory_writer=None,
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
        self.memory_writer = memory_writer
        # When set (real-DeepSeek mode), the orchestrator builds a privacy-
        # filtered CognitiveContext and hands the *same* context to both the
        # cognitive and response providers. In mock mode it stays None and the
        # providers receive the lightweight ConversationContext as before.
        self.context_builder = context_builder
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
        # Local, trusted turn signals. The detector distinguishes a direct
        # request from quoted/reported speech, so neither is delegated to the
        # model.
        explicit_silence = detect_explicit_silence(event.text)
        explicit_remember = detect_explicit_remember(event.text)
        self.logger.info(
            "context_loaded conversation_id=%s quiet_mode=%s memories=%d"
            " explicit_silence=%s explicit_remember=%s",
            event.conversation_id,
            context.quiet_mode,
            len(memories),
            explicit_silence,
            explicit_remember,
        )

        steps = [
            f"[EVENT] Saved: {event.event_id}",
            f"[SPEAKER] {person_id}",
            "[CONTEXT] Conversation loaded",
            f"[MEMORY] Retrieved {len(memories)} items",
        ]

        try:
            # 4b. Build the external-AI context (privacy-filtered history,
            # labels, memories) when the real provider is wired in. Mock mode
            # keeps the lightweight ConversationContext exactly as before.
            cognitive_context, provider_memories = await self._prepare_context(
                event, context, memories, steps
            )

            # 5. Cognitive analysis: reuse a durable analysis, else compute.
            analysis = None
            analysis_id: Optional[str] = None
            if persist:
                analysis = await self.repos.analysis.get_analysis(event.event_id)
                analysis_id = await self.repos.analysis.get_analysis_id(event.event_id)

            job_id: Optional[str] = None
            if analysis is None:
                analysis = await self.cognitive.analyze(
                    event, cognitive_context, provider_memories
                )
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
                    # Model output is untrusted: rebuild ids/subjects on the
                    # trusted side, dedupe, and drop every candidate when
                    # learning is not authorised.
                    candidates = (
                        await prepare_candidates(
                            self.repos,
                            event,
                            analysis.memory_candidates,
                            explicit_remember_request=explicit_remember,
                        )
                        if learn
                        else []
                    )
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

            # 6. Bounded synchronous write-through for an explicit "remember
            # this" request: the durable write must finish *before* the reply
            # that acknowledges it. Ordinary candidates stay with the job.
            memory_outcomes: list[dict] = []
            if persist and learn and self.memory_writer is not None:
                if explicit_remember:
                    memory_outcomes = await self.memory_writer.consolidate_event(
                        event.event_id, only_explicit=True
                    )
                    steps.append(
                        "[MEMORY_WRITE] "
                        + _summarize_outcomes(memory_outcomes)
                    )
                else:
                    memory_outcomes = (
                        await self.repos.analysis.list_candidate_writes(event.event_id)
                    )

            # 7. Policy decides the final action from the (reused or fresh)
            # analysis. Hard local rules (permission, explicit silence, quiet
            # mode, explicit human addressee) outrank the model suggestion.
            await self.policy.refresh(self.repos.settings)
            decision = self.policy.decide(
                analysis,
                context,
                addressee_hint=event.addressee_hint.value,
                explicit_silence=explicit_silence,
            )
            final_action, final_reason = decision.action, decision.reason_code
            steps.append(f"[DECISION] {final_action.value}")
            if decision.overrode_model:
                steps.append(f"[POLICY] {decision.override_reason}")

            # 8. Reuse the durable action if it already exists, else generate a
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
                    response_context = cognitive_context
                    if isinstance(response_context, CognitiveContext):
                        response_context = response_context.model_copy(
                            update={"memory_write_outcomes": memory_outcomes}
                        )
                    response_text = await self.response.generate(
                        event, analysis, response_context
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
                            model_suggested_action=decision.model_action,
                            policy_final_action=decision.action,
                            override_reason=decision.override_reason,
                        )
                    )

            # 9. Finalise the durable conversation + event state.
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
                memory_write_outcomes=memory_outcomes,
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

    async def _prepare_context(self, event, context, memories, steps):
        """Return (provider_context, provider_memories) for this utterance.

        Without a builder this is exactly the old behaviour: the lightweight
        ConversationContext and the retrieved memories. With a builder (real
        DeepSeek mode) the context is privacy-filtered before it can reach the
        network, and the providers get the same filtered memory list that the
        CognitiveContext carries.
        """
        if self.context_builder is None:
            return context, memories
        cognitive_context = await self.context_builder.build(event, memories)
        cognitive_context = cognitive_context.model_copy(
            update={"quiet_mode": context.quiet_mode}
        )
        provider_memories = cognitive_context.memories
        if len(provider_memories) != len(memories):
            steps.append(
                f"[PRIVACY] Filtered {len(memories) - len(provider_memories)} memories"
            )
        return cognitive_context, provider_memories

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


def _summarize_outcomes(outcomes: list[dict]) -> str:
    """Render truthful write outcomes without private content."""
    if not outcomes:
        return "no candidates"
    return ", ".join(
        f"{o.get('candidate_id', '?')}={o.get('outcome', '?')}" for o in outcomes
    )
