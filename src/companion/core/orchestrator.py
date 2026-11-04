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

from companion.agent.dialogue_pattern_monitor import DialoguePatternMonitor
from companion.agent.dialogue_style import compile_dialogue_style
from companion.agent.models import EffectiveAgentContext
from companion.agent.persona_core import sanitise_prompt_core
from companion.cognition.models import CognitiveContext
from companion.cognition.nonresponse import NonresponseTrace, derive_nonresponse_trace
from companion.cognition.scene_repository import SceneRevisionConflict
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
from companion.emotion.expression_policy import compile_expression_influence
from companion.emotion.transition import build_conversational_affect
from companion.memory.candidates import prepare_candidates
from companion.people.registry import PersonRegistry
from companion.storage.repositories import Repositories

from .conversation import ConversationManager
from .policy import Policy, PolicyDecision
from .safety import assess_safety, detect_language, safety_response


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
        emotion_service=None,
        agent_service=None,
        scene_manager=None,
        personality_compiler=None,
        style_repo=None,
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
        # Optional V0.3 emotion framework. When wired, the orchestrator computes
        # the companion's own simulated affect after memory retrieval and hands
        # the resulting (decayed) state + delivery plan to the response provider.
        # It never changes permission, privacy, silence or whether a reply is
        # sent. ``agent_service`` resolves the stable agent_id the state is
        # keyed by (works in mock and real-provider modes alike).
        self.emotion_service = emotion_service
        self.agent_service = agent_service
        # V0.4: optional, local-only scene and personality layer. When absent
        # (old mock wiring) the pipeline is byte-for-byte the V0.3.2 flow.
        self.scene_manager = scene_manager
        self.personality_compiler = personality_compiler
        self.style_repo = style_repo
        self.dialogue_patterns = DialoguePatternMonitor()
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
        # Local safety assessment is computed once and reused below: it both
        # selects the independent reply path and keeps crisis content out of
        # durable memory.
        safety = assess_safety(event.text)
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

            # 4c. Load the short-horizon scene and compile this turn's local
            # style/personality. Scene data is only attached to the rich
            # CognitiveContext that the real providers receive; the mock path
            # keeps its lightweight context unchanged. Any failure here is
            # stylistic and must never fail the turn.
            scene_enabled = await self._scene_enabled()
            personality_enabled = await self._personality_enabled()
            scene_record = None
            if not persist and self.scene_manager is not None:
                # A non-persistence turn cannot reuse or write scene state. It
                # is still the right moment to enforce a withdrawn grant: load
                # revalidates consent and deletes any derived snapshot without
                # sending it anywhere.
                try:
                    await self.scene_manager.load(
                        event.conversation_id, event.speaker_person_id
                    )
                except Exception as exc:
                    self.logger.error(
                        "scene_invalidation_failed event_id=%s error=%s",
                        event.event_id,
                        exc,
                    )
            if persist:
                cognitive_context, scene_record = await self._prepare_dialogue(
                    event,
                    cognitive_context,
                    provider_memories,
                    steps,
                    scene_enabled=scene_enabled,
                    personality_enabled=personality_enabled,
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
                        if learn and not safety.triggered
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

            # 6b. Emotion runs only after the analysis exists (fresh or reused)
            # and before any response is generated. It reads the model's
            # locally-validated affect hint and the *privacy-filtered* provider
            # memories, so a retry reuses the persisted analysis without another
            # model call. Purely presentational; it can never change the action.
            cognitive_context = await self._apply_emotion(
                event,
                person_id,
                provider_memories,
                cognitive_context,
                analysis,
                steps,
                persist,
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
            # 7b. Independent safety floor. An explicit high-risk statement is
            # answered by a fixed local reply, not by persona improvisation and
            # not by a model deferral. Only an explicit human addressee or a
            # denied permission can suppress it; quiet mode, a model WAIT and
            # an explicit "don't reply" cannot.
            if (
                safety.triggered
                and event.addressee_hint.value != "human"
                and analysis.reason_code != ReasonCode.PERMISSION_DENIED
            ):
                decision = PolicyDecision(
                    action=CognitiveAction.RESPOND,
                    reason_code=ReasonCode.SAFETY_RESPONSE,
                    model_action=decision.model_action,
                    override_reason=f"safety_{safety.category.lower()}",
                    rule="safety_response",
                )
                steps.append(f"[SAFETY] local_{safety.category.lower()}")

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
            trace = None
            action_obj = None
            if existing_action is not None:
                final_action = existing_action.action_type
                final_reason = existing_action.reason_code
                response_text = existing_action.response_text
                action_obj = existing_action
                trace = self._stored_trace(existing_action)
                steps.append("[ACTION] Reused existing action")
            else:
                # Derive the final, explainable silence reason before any
                # response. The scene uses it as the only local signal that a
                # WAIT may represent an unfinished thread. A retry reuses the
                # existing action (and therefore the original trace) above; it
                # never invents a new reason for a past WAIT/IGNORE.
                trace = derive_nonresponse_trace(
                    event=event,
                    analysis=analysis,
                    policy_decision=decision,
                    conversation_context=cognitive_context
                    if isinstance(cognitive_context, CognitiveContext)
                    else context,
                )
                if final_action == CognitiveAction.RESPOND:
                    if safety.triggered:
                        # Fixed local text: no model call, no persona/emotion
                        # shaping, no scene preview. The reason is auditable.
                        response_text = safety_response(
                            safety.category, detect_language(event.text)
                        )
                        steps.append("[RESPONSE][SAFETY] local safety reply")
                    else:
                        response_context = cognitive_context
                        if isinstance(response_context, CognitiveContext):
                            response_context = response_context.model_copy(
                                update={"memory_write_outcomes": memory_outcomes}
                            )
                        response_context = self._scene_preview_for_response(
                            event=event,
                            analysis=analysis,
                            previous_scene=scene_record,
                            scene_enabled=scene_enabled,
                            context=response_context,
                            trace=trace,
                            steps=steps,
                        )
                        response_text = await self.response.generate(
                            event, analysis, response_context
                        )
                        steps.append(f"[RESPONSE] {response_text}")
                action_obj = AgentAction(
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
                    nonresponse_trace=(
                        trace.model_dump() if trace is not None else None
                    ),
                )

            # 9. Persist the action, scene transition/receipt, conversation
            # state and event status in one transaction. The scene has a CAS
            # revision and one receipt per source event, so a retry or a crash
            # boundary cannot double-apply it or overwrite newer state.
            if persist:
                await self._commit_turn(
                    event=event,
                    action=action_obj,
                    existing_action=existing_action is not None,
                    trace=trace,
                    scene_enabled=scene_enabled,
                    previous_scene=scene_record,
                    analysis=analysis,
                    cognitive_context=cognitive_context,
                )
                if scene_enabled and action_obj is not None:
                    steps.append("[SCENE] Transition committed")
                # Stance continuity may only cite a response that was actually
                # emitted (or admin config). A retry reuses the existing action
                # and must not duplicate the record.
                if existing_action is None:
                    await self._record_stances(event, action_obj, cognitive_context)

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

    def _scene_preview_for_response(
        self,
        *,
        event,
        analysis,
        previous_scene,
        scene_enabled: bool,
        context,
        trace=None,
        steps=None,
    ):
        """Attach this turn's *candidate* scene to the response context.

        The scene is not persisted here: ``_commit_turn`` owns the CAS write.
        The response only needs the validated candidate so a topic shift or a
        new environment detail from the current turn is reflected in the reply
        instead of always seeing the previous snapshot. Failure is non-fatal.
        """
        if (
            not scene_enabled
            or self.scene_manager is None
            or not isinstance(context, CognitiveContext)
            or (
                analysis.scene_proposal is None
                and getattr(analysis, "pragmatic", None) is None
            )
        ):
            return context
        try:
            candidate = self.scene_manager.preview(
                previous_scene,
                analysis.scene_proposal,
                event,
                context=context,
                action_history=context.action_history,
                pending_followup=bool(
                    getattr(trace, "pending_followup", False)
                ),
                pragmatic=getattr(analysis, "pragmatic", None),
            )
            if steps is not None:
                steps.append("[SCENE] Candidate scene for response")
            return context.model_copy(
                update={"scene_state": candidate.prompt_dict()}
            )
        except Exception as exc:
            self.logger.warning(
                "scene_preview_failed event_id=%s error=%s",
                event.event_id,
                exc,
            )
            return context

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

    async def _apply_emotion(
        self,
        event,
        person_id,
        provider_memories,
        cognitive_context,
        analysis,
        steps,
        persist,
    ):
        """Compute the companion's own affect and attach it to the context.

        Called only after a *persisted* analysis exists. ``provider_memories``
        is the privacy-filtered list that already reached the model, never the
        raw retrieval set. ``process_turn`` appraises, applies and records the
        turn in one transaction, so a retry reuses the analysis and the receipt
        rather than double-applying the shift.

        Emotion is presentation-only and must never fail the turn, so any error
        is logged and swallowed — a reply still happens, just without an
        emotional delivery plan. ``persist`` gates only the event/snapshot
        writes (honouring the consent persistence flag); the in-memory state and
        speech plan are computed regardless.
        """
        if self.emotion_service is None:
            return cognitive_context
        try:
            agent_id = await self._resolve_agent_id()
            if agent_id is None:
                self.logger.warning(
                    "emotion_skipped event_id=%s reason=no_agent_id", event.event_id
                )
                return cognitive_context

            agent_context = getattr(cognitive_context, "agent_context", None)
            # The *previous* committed frame, before this turn's pragmatic
            # reading, lets the transition distinguish a real frame change
            # from a frame that simply stayed the same.
            previous_frame = "UNKNOWN"
            scene_view = getattr(cognitive_context, "scene_state", None)
            if isinstance(scene_view, dict):
                interaction = scene_view.get("interaction")
                if isinstance(interaction, dict):
                    previous_frame = interaction.get("frame") or "UNKNOWN"
            conversational_affect = build_conversational_affect(
                pragmatic=getattr(analysis, "pragmatic", None),
                signal=getattr(analysis, "affective_signal", None),
                previous_frame=previous_frame,
                actor_person_id=event.speaker_person_id,
            )
            state = await self.emotion_service.process_turn(
                agent_id,
                person_id,
                text=event.text,
                signal=getattr(analysis, "affective_signal", None),
                memories=provider_memories,
                agent_context=agent_context,
                conversation_id=event.conversation_id,
                source_event_id=event.event_id,
                persist=persist,
                addressee_hint=event.addressee_hint.value,
                actor_person_id=event.speaker_person_id,
                conversational_affect=conversational_affect,
            )
            speech_plan = await self.emotion_service.plan_speech(
                agent_id, state=state, text=""
            )
            expression = compile_expression_influence(
                state,
                config=getattr(self.emotion_service, "config", None),
            )
            agent_name = None
            if isinstance(cognitive_context, CognitiveContext):
                agent_name = (cognitive_context.agent_context or {}).get(
                    "display_name"
                )
                assistant_replies = [
                    turn.text
                    for turn in cognitive_context.turns
                    if turn.role == "assistant"
                ]
            else:
                assistant_replies = []
            pattern_report = self.dialogue_patterns.analyze(
                assistant_replies,
                agent_name=agent_name,
                user_name=(
                    cognitive_context.current_speaker_label
                    if isinstance(cognitive_context, CognitiveContext)
                    else None
                ),
            )

            if isinstance(cognitive_context, CognitiveContext):
                cognitive_context = cognitive_context.model_copy(
                    update={
                        "emotion_state": state.as_dict(),
                        "speech_plan": speech_plan.as_dict(),
                        "emotion_expression": expression.as_dict(),
                        "dialogue_pattern_guidance": pattern_report.guidance,
                    }
                )
            self.logger.info(
                "emotion_applied event_id=%s emotion=%s intensity=%.2f"
                " valence=%.2f mood=%s tension=%.2f expression=%s",
                event.event_id,
                state.emotion,
                state.intensity,
                state.valence,
                state.mood_emotion,
                state.relationship_tension,
                expression.explanation,
            )
            steps.append(
                f"[EMOTION] {state.emotion} (intensity={state.intensity:.2f})"
            )
        except Exception as exc:  # emotion is never allowed to fail the turn.
            self.logger.error(
                "emotion_failed event_id=%s error=%s", event.event_id, exc
            )
        return cognitive_context

    async def _record_stances(self, event, action_obj, cognitive_context) -> None:
        """Persist public stances from an emitted response. Never fatal."""
        if (
            self.agent_service is None
            or action_obj is None
            or action_obj.action_type != CognitiveAction.RESPOND
            or action_obj.reason_code == ReasonCode.SAFETY_RESPONSE
            or not action_obj.response_text
        ):
            return
        try:
            agent_id = await self._resolve_agent_id()
            if not agent_id:
                return
            topic = "UNKNOWN"
            scene_view = getattr(cognitive_context, "scene_state", None)
            if isinstance(scene_view, dict):
                topic = scene_view.get("topic") or "UNKNOWN"
            await self.agent_service.record_response_stances(
                agent_id,
                person_id=event.speaker_person_id,
                conversation_id=event.conversation_id,
                source_event_id=event.event_id,
                response_text=action_obj.response_text,
                topic=topic,
            )
        except Exception as exc:
            self.logger.warning(
                "stance_record_failed event_id=%s error=%s",
                event.event_id,
                exc,
            )

    async def _resolve_agent_id(self) -> Optional[str]:
        if self.agent_service is None:
            return None
        manifest = await self.agent_service.repo.get_manifest()
        return manifest.get("agent_id") if manifest else None

    async def _scene_enabled(self) -> bool:
        if self.scene_manager is None:
            return False
        return await self.repos.settings.get_bool("dialogue.scene.enabled", True)

    async def _personality_enabled(self) -> bool:
        if self.personality_compiler is None:
            return False
        return await self.repos.settings.get_bool(
            "dialogue.personality_generation.enabled", True
        )

    async def _prepare_dialogue(
        self,
        event,
        cognitive_context,
        provider_memories,
        steps,
        *,
        scene_enabled: bool,
        personality_enabled: bool,
    ):
        """Load scene state and compile local style/personality for this turn.

        This is intentionally non-fatal: a style or scene-loading error must
        not lose an otherwise valid turn. The scene *write* is separate and is
        transactional with the action.
        """
        scene_record = None
        if scene_enabled and self.scene_manager is not None:
            try:
                scene_record = await self.scene_manager.load(
                    event.conversation_id, event.speaker_person_id
                )
                if isinstance(cognitive_context, CognitiveContext):
                    cognitive_context = cognitive_context.model_copy(
                        update={
                            "scene_state": (
                                scene_record.state.prompt_dict()
                                if scene_record is not None
                                else None
                            )
                        }
                    )
                steps.append(
                    "[SCENE] Loaded" if scene_record is not None else "[SCENE] New"
                )
            except Exception as exc:
                self.logger.error(
                    "scene_load_failed event_id=%s error=%s", event.event_id, exc
                )

        if not personality_enabled or not isinstance(cognitive_context, CognitiveContext):
            return cognitive_context, scene_record

        try:
            effective = await self._effective_context(
                event.speaker_person_id, cognitive_context.agent_context
            )
            if effective is None:
                return cognitive_context, scene_record
            style = compile_dialogue_style(
                effective,
                memories=provider_memories,
                current_text=event.text,
            )
            examples: list[dict] = []
            if self.style_repo is not None and effective.agent_id:
                examples = await self.style_repo.list_for_prompt(
                    agent_id=effective.agent_id,
                    person_id=event.speaker_person_id,
                )
            scene_view = getattr(cognitive_context, "scene_state", None)
            stances = []
            if self.agent_service is not None and effective.agent_id:
                try:
                    stances = await self.agent_service.list_stances(
                        effective.agent_id,
                        person_id=event.speaker_person_id,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "stance_load_failed event_id=%s error=%s",
                        event.event_id,
                        exc,
                    )
            personality = (
                self.personality_compiler.compile(
                    effective,
                    style=style,
                    scene=scene_view,
                    style_examples=examples,
                    current_text=event.text,
                    stances=stances,
                )
                if self.personality_compiler is not None
                else None
            )
            guidance = (
                self.personality_compiler.pragmatic_guidance(
                    scene=scene_view,
                    action_history=cognitive_context.action_history,
                    current_addressee_hint=event.addressee_hint.value,
                    quiet_mode=cognitive_context.quiet_mode,
                )
                if self.personality_compiler is not None
                else []
            )
            cognitive_context = cognitive_context.model_copy(
                update={
                    "dialogue_style": style.as_dict(),
                    "personality": personality,
                    "style_examples": examples,
                    "pragmatic_guidance": guidance,
                }
            )
            steps.append("[PERSONALITY] Style compiled")
        except Exception as exc:
            self.logger.error(
                "personality_compile_failed event_id=%s error=%s",
                event.event_id,
                exc,
            )
        return cognitive_context, scene_record

    async def _effective_context(self, person_id: str, agent_context):
        data = agent_context if isinstance(agent_context, dict) else None
        if data is not None:
            try:
                agent_id = str(data.get("agent_id") or "")
                persona_core = await self._trusted_persona_core(
                    data.get("persona_core"), agent_id=agent_id
                )
                return EffectiveAgentContext(
                    agent_id=agent_id,
                    display_name=str(data.get("display_name") or "Companion"),
                    pronouns=data.get("pronouns"),
                    description=data.get("description"),
                    person_id=data.get("person_id") or person_id,
                    familiarity=str(data.get("familiarity") or "STRANGER"),
                    traits={
                        str(k): int(v)
                        for k, v in (data.get("traits") or {}).items()
                    },
                    overrides={
                        str(k): int(v)
                        for k, v in (data.get("overrides") or {}).items()
                    },
                    persona_core=persona_core,
                    response_conservatism=str(
                        data.get("response_conservatism") or "balanced"
                    ),
                    interruption_threshold=str(
                        data.get("interruption_threshold") or "medium"
                    ),
                    curiosity_level=str(data.get("curiosity_level") or "low"),
                    revision=int(data.get("revision") or 1),
                )
            except (TypeError, ValueError):
                pass
        if self.agent_service is not None:
            return await self.agent_service.effective(person_id)
        return None

    async def _trusted_persona_core(self, candidate, *, agent_id: str) -> dict:
        """Re-read the admin-maintained character core for this turn.

        The context builder already put a copy on ``agent_context``, but a
        transient dict is not authority: anything that reconstructs the
        effective context (including a future caller) could drop or forge it.
        When the trusted service is wired we always use its current revision;
        otherwise we whitelist the local copy through the same validators.
        """
        if self.agent_service is not None:
            try:
                core = await self.agent_service.show_persona_core()
            except Exception:  # pragma: no cover - defensive, logged below
                self.logger.warning(
                    "persona_core_trusted_read_failed agent_id=%s", agent_id
                )
                return sanitise_prompt_core(candidate, agent_id)
            supplied = None
            if isinstance(candidate, dict):
                try:
                    supplied = int(candidate.get("revision") or 0)
                except (TypeError, ValueError):
                    supplied = None
            if supplied is not None and supplied != core.revision:
                self.logger.info(
                    "persona_core_revision_refreshed agent_id=%s supplied=%s trusted=%d",
                    agent_id,
                    supplied,
                    core.revision,
                )
            return core.as_prompt_dict()
        return sanitise_prompt_core(candidate, agent_id)

    @staticmethod
    def _stored_trace(action):
        raw = getattr(action, "nonresponse_trace", None)
        if not isinstance(raw, dict):
            return None
        try:
            return NonresponseTrace.model_validate(raw)
        except (ValueError, TypeError):
            return None

    async def _commit_turn(
        self,
        *,
        event,
        action,
        existing_action: bool,
        trace,
        scene_enabled: bool,
        previous_scene,
        analysis,
        cognitive_context,
    ) -> None:
        """Atomic finalization of action + scene + conversation + event.

        The whole unit commits or rolls back together. A scene revision
        conflict rolls back any newly written action too, then reloads the
        newer scene and retries the *same* already-generated action (no model
        call). After a bounded retry the conflict is surfaced and the event is
        left FAILED for a later retry; newer scene state is never overwritten.
        """
        manager = self.scene_manager
        if not scene_enabled or manager is None:
            async with self.repos.db.transaction(immediate=True):
                if not existing_action:
                    await self.repos.analysis.save_action(action)
                await self.conversations.record_last_action(
                    event.conversation_id, action.action_type.value
                )
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.COMPLETED
                )
            return

        scene_repo = manager.scene_repo
        previous = previous_scene

        # Crash/legacy recovery: an action exists but the scene receipt does
        # not. If the snapshot already includes this event, record the missing
        # receipt without touching the scene; otherwise rebuild the transition
        # from the persisted analysis proposal.
        if existing_action:
            receipt = await scene_repo.get_receipt(event.event_id)
            if receipt is not None:
                async with self.repos.db.transaction(immediate=True):
                    await self.conversations.record_last_action(
                        event.conversation_id, action.action_type.value
                    )
                    await self.repos.utterance.update_status(
                        event.event_id, ProcessingStatus.COMPLETED
                    )
                return
            try:
                fresh = await manager.load(
                    event.conversation_id, event.speaker_person_id
                )
            except Exception:
                fresh = None
            previous = fresh
            if (
                fresh is not None
                and fresh.last_committed_event_id == event.event_id
            ):
                async with self.repos.db.transaction(immediate=True):
                    await scene_repo.save_receipt_only(
                        source_event_id=event.event_id,
                        conversation_id=event.conversation_id,
                        person_id=event.speaker_person_id,
                        result_revision=fresh.revision,
                        transition={
                            "reason": "snapshot_already_advanced",
                            "action": action.action_type.value,
                        },
                    )
                    await self.conversations.record_last_action(
                        event.conversation_id, action.action_type.value
                    )
                    await self.repos.utterance.update_status(
                        event.event_id, ProcessingStatus.COMPLETED
                    )
                return

        for attempt in range(3):
            candidate = manager.preview(
                previous,
                getattr(analysis, "scene_proposal", None),
                event,
                context=cognitive_context,
                action_history=getattr(cognitive_context, "action_history", None),
                pending_followup=bool(
                    trace is not None and getattr(trace, "pending_followup", False)
                ),
                pragmatic=getattr(analysis, "pragmatic", None),
            )
            candidate = manager.finalize(
                candidate,
                event=event,
                action=action.action_type,
                response_text=action.response_text,
                trace=trace,
            )
            expected_revision = previous.revision if previous is not None else 0
            try:
                async with self.repos.db.transaction(immediate=True):
                    if not existing_action:
                        await self.repos.analysis.save_action(action)
                    else:
                        # Another worker may have completed the receipt while
                        # we were preparing; re-check under the write lock.
                        current_receipt = await scene_repo.get_receipt(
                            event.event_id
                        )
                        if current_receipt is not None:
                            candidate = None
                    if candidate is not None:
                        await scene_repo.apply_transition(
                            source_event_id=event.event_id,
                            conversation_id=event.conversation_id,
                            person_id=event.speaker_person_id,
                            expected_revision=expected_revision,
                            state=candidate,
                            transition={
                                "action": action.action_type.value,
                                "topic": candidate.topic,
                                "discourse_phase": candidate.discourse_phase,
                                "tone": candidate.tone,
                                "revision": candidate.revision,
                            },
                        )
                    await self.conversations.record_last_action(
                        event.conversation_id, action.action_type.value
                    )
                    await self.repos.utterance.update_status(
                        event.event_id, ProcessingStatus.COMPLETED
                    )
                return
            except SceneRevisionConflict as exc:
                if attempt >= 2:
                    raise
                self.logger.warning(
                    "scene_revision_conflict event_id=%s attempt=%d error=%s",
                    event.event_id,
                    attempt + 1,
                    exc,
                )
                previous = await manager.load(
                    event.conversation_id, event.speaker_person_id
                )
                if (
                    existing_action
                    and previous is not None
                    and previous.last_committed_event_id == event.event_id
                ):
                    async with self.repos.db.transaction(immediate=True):
                        await scene_repo.save_receipt_only(
                            source_event_id=event.event_id,
                            conversation_id=event.conversation_id,
                            person_id=event.speaker_person_id,
                            result_revision=previous.revision,
                            transition={
                                "reason": "conflict_recovered_snapshot",
                                "action": action.action_type.value,
                            },
                        )
                        await self.conversations.record_last_action(
                            event.conversation_id, action.action_type.value
                        )
                        await self.repos.utterance.update_status(
                            event.event_id, ProcessingStatus.COMPLETED
                        )
                    return

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
