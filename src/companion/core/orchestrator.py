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
import contextlib
import logging
import time
from datetime import timedelta
from typing import Optional

from companion.agent.character import (
    CharacterSnapshotValidationError,
    build_character_snapshot,
    character_snapshot_fingerprint,
    load_emotional_dispositions,
    validate_stored_snapshot,
)
from companion.agent.dialogue_style import compile_dialogue_style
from companion.agent.models import EffectiveAgentContext
from companion.agent.persona_core import sanitise_prompt_core
from companion.cognition.models import ROLE_ASSISTANT, CognitiveContext
from companion.cognition.nonresponse import NonresponseTrace, derive_nonresponse_trace
from companion.cognition.scene_repository import SceneRevisionConflict
from companion.cognition.silence import (
    detect_explicit_remember,
    detect_explicit_silence,
)
from companion.domain.enums import (
    CognitiveAction,
    JobType,
    ProcessingStatus,
    ReasonCode,
    RegistrationStatus,
    ReportStatus,
    ReplyNecessity,
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
from companion.emotion.expression_policy import compile_affect_context
from companion.emotion.transition import build_conversational_affect
from companion.memory.auto_recall import (
    build_recall_query,
    router_requested_memory,
    select_auto_recalled,
    should_auto_recall,
)
from companion.memory.explicit_operations import MemoryOperationResult
from companion.tasks.models import TaskPriority, TaskRejected, TaskRequest, TaskSource
from companion.memory.candidates import prepare_candidates
from companion.memory.requests import (
    is_conversation_count_request,
    is_earliest_history_request,
    is_memory_sample_request,
)
from companion.people.registry import PersonRegistry
from companion.proactive.departure import detect_explicit_departure
from companion.prompts.layout import (
    StreamingTextNormalizer,
    normalize_response_text,
    response_layout_for,
)
from companion.runtime_context.connection import ConnectionInfo
from companion.runtime_context.snapshot import RuntimeSnapshotBuilder
from companion.storage.repositories import Repositories

from .conversation import ConversationManager
from .fast_path import LOCAL_PROVIDER, TurnAssessment, assess_turn, decide_route
from .metrics import LatencyMetrics
from .policy import Policy, PolicyDecision, TrustedTurnContext
from .response_events import ResponseEventSink
from .safety import assess_safety, detect_language, safety_response

from companion.routing.adapter import route_to_cognitive_result
from companion.routing.context import RoutingContextBuilder, build_speaker_context
from companion.routing.context_adapter import (
    attach_semantic_tool_signals,
    tool_results_to_memory_context,
)
from companion.routing.errors import RouterError, RouterUnavailable
from companion.routing.models import InputSemanticStatus, ProcessingIntent, RouteAction
from companion.tools.models import ToolExecutionContext


class PersonaRevisionConflict(Exception):
    """A retried analysis cannot be paired with the persona it was made under.

    Raised instead of silently generating a reply with a newer character
    revision than the persisted analysis. The caller reports an explicit
    conflict; the user can submit the utterance under a new event id.
    """


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
        runtime_snapshot_builder: Optional[RuntimeSnapshotBuilder] = None,
        metrics: Optional[LatencyMetrics] = None,
        profile_learning=None,
        profile_review=None,
        fast_path_enabled: bool = True,
        router_mode: str = "legacy",
        routing_context_builder: Optional[RoutingContextBuilder] = None,
        routing_service=None,
        tool_executor=None,
        web_search=None,
        explicit_memory_operations=None,
        data_control=None,
        task_service=None,
        profile_confirmation=None,
        turn_analysis_delay_seconds: float = 60.0,
        turn_analysis_batch_size: int = 5,
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
        # Retained for the admin data interface (style references are managed
        # through ``rt.style_repo``); the default generation chain no longer
        # retrieves or injects them per turn.
        self.style_repo = style_repo
        # When set (real-DeepSeek mode), the orchestrator builds a privacy-
        # filtered CognitiveContext and hands the *same* context to both the
        # cognitive and response providers. In mock mode it stays None and the
        # providers receive the lightweight ConversationContext as before.
        self.context_builder = context_builder
        # V0.7: deterministic, local-only person-profile learning. It never
        # makes a model call and never decides whether Ava replies; it only
        # records explicit preferences and unconfirmed candidate facts within
        # the speaker's standing consent.
        self.profile_learning = profile_learning
        # V0.9.7: background profile review request. Called inside the turn
        # commit transaction (receipt + job), so the conversation and its
        # review request commit or roll back together. It is never awaited for
        # the reply.
        self.profile_review = profile_review
        # V0.6: one immutable runtime snapshot per event (time, zone, gaps,
        # optional consented location). Missing builder keeps older wirings
        # working unchanged.
        self.runtime_snapshot_builder = runtime_snapshot_builder
        self.metrics = metrics or LatencyMetrics()
        # Operational switch. On by default: a clearly addressed direct turn
        # makes one model call. An operator (or a test targeting the richer
        # deliberation pipeline) can turn it off; routing still never looks at
        # message text.
        self.fast_path_enabled = fast_path_enabled
        # V0.8: "semantic" routes every eligible turn through the LLM router;
        # "legacy" keeps the previous local Fast Path / Deliberation pipeline.
        # In semantic mode the old text classifiers below are never consulted.
        mode = (router_mode or "legacy").strip().lower()
        if mode not in ("semantic", "legacy"):
            raise ValueError(f"unknown router_mode {router_mode!r}")
        self.router_mode = mode
        self.routing_context_builder = routing_context_builder
        self.routing_service = routing_service
        self.tool_executor = tool_executor
        self.web_search = web_search
        # V0.8 repair: explicit MEMORY_SAVE / MEMORY_CORRECTION intents use the
        # existing extraction -> candidates -> MemoryWriter chain. Privacy
        # intents go through a separate, confirmation-gated trusted service.
        self.explicit_memory_operations = explicit_memory_operations
        self.data_control = data_control
        # V0.9: task execution is Off-task. The orchestrator only publishes a
        # foreground activity signal; it never runs background work inline.
        self.task_service = task_service
        # V0.9.9: one-turn confirmation of an explicit identity self-report.
        # Detection runs before Speaker generation; the question is appended to
        # a deliverable reply and the next real user message resolves it. The
        # service (not the Speaker) owns every profile write.
        self.profile_confirmation = profile_confirmation
        # V0.6.1: a Fast Path turn only queues lightweight extraction. The job
        # waits up to ``delay`` seconds so several turns can share one small
        # request; once ``batch_size`` events are waiting it runs immediately.
        self.turn_analysis_delay_seconds = max(0.0, float(turn_analysis_delay_seconds))
        self.turn_analysis_batch_size = max(1, int(turn_analysis_batch_size))
        self.logger = logger or logging.getLogger("companion.orchestrator")
        self._conversation_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        lock = self._conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_id] = lock
        return lock

    def is_processing(self, conversation_id: str) -> bool:
        """Whether a real foreground turn currently owns this conversation.

        The proactive manager uses this only as a read-only staleness check;
        it never acquires the conversation lock for its own generation, so a
        user turn is never blocked behind proactive work.
        """
        lock = self._conversation_locks.get(conversation_id)
        return bool(lock is not None and lock.locked())

    def _foreground_scope(self):
        """Signal that a real foreground turn is active.

        Background jobs observe this only at safe checkpoints. When no task
        service is wired (legacy tests), this is a no-op context manager.
        """
        if self.task_service is None:
            return contextlib.nullcontext()
        return self.task_service.foreground.scope()

    async def process_utterance(
        self,
        event: UtteranceEvent,
        *,
        connection: Optional[ConnectionInfo] = None,
    ) -> ProcessingReport:
        # Per-conversation serialization: events in the same conversation run
        # one at a time, in submission order.
        async with self._lock_for(event.conversation_id):
            with self._foreground_scope():
                return await self._process(event, connection=connection)

    async def process_utterance_stream(
        self,
        event: UtteranceEvent,
        *,
        connection: Optional[ConnectionInfo] = None,
    ):
        """Process one event, yielding true incremental :class:`ResponseEvent`s.

        The turn runs in a background task so deltas can be yielded while the
        database work continues. Only a short local transaction is open at a
        time; no DB transaction spans the model stream. If the consumer stops
        early, the task is cancelled and any text already shown is persisted
        with an explicit ``CANCELLED`` action status, never as a complete reply.
        """
        queue: asyncio.Queue = asyncio.Queue()
        sink = ResponseEventSink(
            event.event_id, event.conversation_id, queue.put_nowait
        )

        async def run() -> ProcessingReport:
            try:
                async with self._lock_for(event.conversation_id):
                    with self._foreground_scope():
                        return await self._process(
                            event, connection=connection, stream=sink
                        )
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _process(
        self,
        event: UtteranceEvent,
        *,
        connection: Optional[ConnectionInfo] = None,
        stream: Optional[ResponseEventSink] = None,
    ) -> ProcessingReport:
        t0 = time.monotonic()
        timings: dict[str, float] = {}
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

        # 4. Retrieve permitted memories (hand-authored facts only). In
        # semantic mode retrieval is deferred until after the router has decided
        # *whether* and *what* this turn needs; the router itself never sees
        # long-term memories.
        semantic_mode = self.router_mode == "semantic"
        memories = [] if semantic_mode else await self.memory.retrieve(
            event.text, person_id
        )
        # Local, trusted turn signals. These are legacy routing helpers: in
        # semantic mode the router owns the semantic decision and they are not
        # consulted for routing at all.
        explicit_silence = (
            False if semantic_mode else detect_explicit_silence(event.text)
        )
        explicit_remember = (
            False if semantic_mode else detect_explicit_remember(event.text)
        )
        # Local safety assessment is computed once and reused below: it both
        # selects the independent reply path and keeps crisis content out of
        # durable memory. This is a fixed local floor, not text-based routing.
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

        # 3b. One runtime snapshot per event. It is created after the event is
        # durable (FK) and reused verbatim on a retry: received_at and every
        # temporal gap stay fixed. Location is only resolved when GeoIP is
        # enabled and this speaker allows external AI use.
        runtime_snapshot = None
        if self.runtime_snapshot_builder is not None:
            try:
                runtime_snapshot = await self.runtime_snapshot_builder.build(
                    event,
                    connection=connection,
                    allow_external_ai=bool(consent.allow_external_ai),
                    persist=persist,
                )
                steps.append(
                    f"[RUNTIME] {runtime_snapshot.timezone}"
                    f" ({runtime_snapshot.timezone_source}),"
                    f" reentry={runtime_snapshot.reentry_state}"
                )
            except Exception as exc:  # runtime context is never fatal
                self.logger.warning(
                    "runtime_snapshot_failed event_id=%s error=%s",
                    event.event_id,
                    exc,
                )
        timings["context_ms"] = (time.monotonic() - t0) * 1000

        # Response lifecycle state for the streaming report. Initialised before
        # the pipeline so a failure in any earlier stage can still be reported
        # truthfully (nothing was streamed).
        response_id = new_id("resp")
        response_text: Optional[str] = None
        stream_emitted = False

        try:
            # 4a. Semantic mode: the LLM router runs *before* any long-term
            # memory is retrieved and before the full speaker context exists.
            # Hard local gates (external-AI consent, trusted human addressee,
            # safety floor) can skip the router entirely; they never fall back
            # to keyword routing.
            routing_outcome = None
            route = None
            routing_view = None
            if semantic_mode:
                routing_view, routing_outcome, route = await self._route_semantic(
                    event,
                    context,
                    consent,
                    runtime_snapshot,
                    steps,
                    safety_triggered=safety.triggered,
                    persist=persist,
                    timings=timings,
                )
                # Long-term memory reaches this turn only through the
                # read-only tools the router requested. There is deliberately
                # no unconditional ``memory.retrieve`` in semantic mode.
                route_tool_results = (
                    routing_outcome.tool_results
                    if routing_outcome is not None
                    else []
                )
                memories = tool_results_to_memory_context(route_tool_results)
                self.logger.info(
                    "semantic_route_ready event_id=%s routed=%s memories=%d",
                    event.event_id,
                    route is not None,
                    len(memories),
                )
                # 4a-bis. A validated non-communicative turn with no internal
                # operation is finalized without assembling a speaker context,
                # compiling persona/scene or calling emotion and response
                # providers. The analysis, final action, conversation state and
                # event lifecycle still commit atomically through `_commit_turn`.
                if self._is_noncommunicative_silent(route):
                    return await self._finalize_silent_turn(
                        event=event,
                        context=context,
                        route=route,
                        routing_outcome=routing_outcome,
                        steps=steps,
                        timings=timings,
                        t0=t0,
                        persist=persist,
                    )

                # 4a-ter. Conditional auto-recall (V0.9.9.3). Only for a
                # validated RESPOND turn whose wording clearly continues an
                # earlier experience, and only when the router did *not*
                # involve a memory tool. This never changes whether a reply
                # happens, never invents a tool result, and never bypasses the
                # store's access checks: it just gives the speaker the same
                # authorized background records it would have had.
                if (
                    self.memory is not None
                    and person_id is not None
                    and not router_requested_memory(route, route_tool_results)
                    and should_auto_recall(route, event.text, routing_view)
                ):
                    recall_query = build_recall_query(event.text, routing_view)
                    if recall_query:
                        try:
                            recalled = await self.memory.retrieve(
                                recall_query, person_id
                            )
                        except Exception as exc:  # recall is never fatal
                            self.logger.warning(
                                "auto_recall_failed event_id=%s error=%s",
                                event.event_id,
                                type(exc).__name__,
                            )
                            recalled = []
                        selected = select_auto_recalled(
                            recalled,
                            recall_query,
                            exclude_ids={
                                str(item.get("memory_id"))
                                for item in memories
                                if item.get("memory_id")
                            },
                        )
                        if selected:
                            memories = [*memories, *selected]
                            steps.append(
                                f"[MEMORY] auto_recall items={len(selected)}"
                            )
                        else:
                            self.logger.info(
                                "auto_recall_empty event_id=%s", event.event_id
                            )

            # 4b. Build the external-AI context (privacy-filtered history,
            # labels, memories) when the real provider is wired in. Mock mode
            # keeps the lightweight ConversationContext exactly as before.
            t_speaker_context = time.monotonic()
            cognitive_context, provider_memories = await self._prepare_context(
                event, context, memories, steps, routing_view=routing_view
            )
            if semantic_mode:
                if (
                    routing_view is not None
                    and isinstance(cognitive_context, CognitiveContext)
                ):
                    cognitive_context = attach_semantic_tool_signals(
                        cognitive_context,
                        routing_view,
                        route_tool_results,
                    )
            else:
                cognitive_context = await self._attach_memory_signals(
                    event, person_id, memories, cognitive_context, steps
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
            else:
                # A non-persistence turn still needs the same trusted character
                # view for the two provider calls; only style compilation and
                # scene writes are skipped.
                cognitive_context = await self._attach_character_snapshot(
                    event, cognitive_context
                )

            cognitive_context = self._attach_runtime_snapshot(
                cognitive_context, runtime_snapshot
            )

            # 5. Routing: a clearly addressed, ordinary direct turn takes the
            # Fast Path (no cognitive call). Ambiguous addressees, explicit
            # memory requests and domains the local metadata cannot settle use
            # the Deliberation Path. The decision never comes from message text.
            t_analysis = time.monotonic()
            analysis = None
            analysis_id: Optional[str] = None
            if persist:
                analysis = await self.repos.analysis.get_analysis(event.event_id)
                analysis_id = await self.repos.analysis.get_analysis_id(event.event_id)

            fast_path = False
            assessment: Optional[TurnAssessment] = None
            job_id: Optional[str] = None
            if semantic_mode:
                # The router already produced (or deliberately skipped) the
                # semantic decision. No old classifier runs here: not
                # ``assess_turn``, not ``decide_route`` and none of the text
                # detectors used by the legacy path.
                if analysis is not None:
                    steps.append("[COGNITION] Reused existing analysis")
                    if learn and self.jobs is not None:
                        job = await self.repos.job.get_by_key(
                            f"memory:{event.event_id}"
                        )
                        if job is not None:
                            job_id = job.job_id
                else:
                    if route is not None:
                        analysis = route_to_cognitive_result(
                            event,
                            route,
                            routing_outcome.tool_results
                            if routing_outcome is not None
                            else [],
                        )
                    else:
                        analysis = self._local_semantic_analysis(
                            event, consent, safety.triggered
                        )
                    steps.append(
                        "[COGNITION] Semantic router: "
                        f"{analysis.action.value}, necessity="
                        f"{analysis.reply_necessity.value}"
                    )
                    if persist:
                        analysis_id = new_id("ana")
                        job_id = await self.repos.persist_cognitive_outcome(
                            analysis,
                            analysis_id,
                            candidates=[],
                            job_key=None,
                            job_payload={},
                            persona=self._persona_record(cognitive_context),
                        )
                        # Background extraction still runs, but never as a
                        # second full cognitive call on the critical path. An
                        # explicit memory intent is extracted and written
                        # synchronously below, so queueing a racing background
                        # pass would only duplicate candidates.
                        route_intents = (
                            set(route.processing_intents)
                            if route is not None
                            else set()
                        )
                        explicit_memory = bool(
                            route_intents
                            & {
                                ProcessingIntent.MEMORY_SAVE,
                                ProcessingIntent.MEMORY_CORRECTION,
                            }
                        )
                        if (
                            learn
                            and self.jobs is not None
                            and not safety.triggered
                            and not self._is_noncommunicative_silent(route)
                            and not (
                                explicit_memory
                                and self.explicit_memory_operations is not None
                            )
                        ):
                            job_id = await self._enqueue_turn_analysis(event)
                            steps.append("[MEMORY_JOB] Background analysis queued")
            else:
              local_capable = (
                self.fast_path_enabled
                and self.context_builder is not None
                and bool(consent.allow_external_ai)
              )
              if analysis is None and local_capable:
                # V0.7: addressee and reply necessity are assessed separately.
                # Explicit silence, a confirmed human addressee, quiet mode and
                # a wait request are decided locally; an unclear addressee or a
                # memory request still uses the existing Deliberation Path.
                # Legacy/mock wirings keep their original Deliberation Path, so
                # no old behaviour is silently rewritten.
                assessment = assess_turn(
                    addressee_hint=event.addressee_hint.value,
                    text=event.text,
                    explicit_silence=explicit_silence,
                    quiet_mode=bool(context.quiet_mode),
                    explicit_remember=explicit_remember,
                    fast_path_enabled=local_capable,
                    model_available=local_capable,
                )
                if assessment.is_local:
                    fast_path = True
                    steps.append(f"[ROUTE] local:{assessment.rule}")
              if analysis is None and not fast_path:
                route = decide_route(
                    addressee_hint=event.addressee_hint.value,
                    explicit_remember=explicit_remember,
                    # Fast Path still obeys the external-AI gate: without
                    # an explicit grant the turn goes through the standard
                    # path, where the privacy gateway returns a local
                    # denial without any network call.
                    fast_path_enabled=local_capable,
                )
                fast_path = route.is_fast
                steps.append(f"[ROUTE] {route.route}:{route.reason}")
              if analysis is not None and str(
                getattr(analysis, "provider", "")
              ) == LOCAL_PROVIDER:
                fast_path = True

              if analysis is None and fast_path:
                # A local decision record, explicitly marked as local. It is not
                # a model analysis and is never presented as one. RESPOND,
                # WAIT and IGNORE all reach here; only the action differs.
                analysis = self._local_turn_analysis(event, assessment)
                steps.append(
                    "[COGNITION] Skipped (local decision: "
                    f"{analysis.action.value}, necessity="
                    f"{analysis.reply_necessity.value})"
                )
                self.logger.info(
                    "local_assessment_selected event_id=%s action=%s rule=%s",
                    event.event_id,
                    analysis.action.value,
                    assessment.rule if assessment is not None else "reused",
                )
                if persist:
                    analysis_id = new_id("ana")
                    job_id = await self.repos.persist_cognitive_outcome(
                        analysis,
                        analysis_id,
                        candidates=[],
                        job_key=None,
                        job_payload={},
                        persona=self._persona_record(cognitive_context),
                    )
                    # Memory learning still happens, but off the critical path:
                    # a durable, idempotent job extracts candidates after the
                    # reply. Nothing is acknowledged before a real write.
                    if learn and self.jobs is not None and not safety.triggered:
                        job_id = await self._enqueue_turn_analysis(event)
                        steps.append("[MEMORY_JOB] Background analysis queued")
              elif analysis is None:
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
                        persona=self._persona_record(cognitive_context),
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
                    if str(getattr(analysis, "provider", "")) == LOCAL_PROVIDER:
                        job_id = await self._enqueue_turn_analysis(event) or job_id
            if (
                semantic_mode
                and route is not None
                and routing_view is not None
                and isinstance(cognitive_context, CognitiveContext)
            ):
                cognitive_context = build_speaker_context(
                    event,
                    routing_view,
                    route,
                    routing_outcome.tool_results
                    if routing_outcome is not None
                    else [],
                    base=cognitive_context,
                )
            timings["speaker_context_ms"] = (
                time.monotonic() - t_speaker_context
            ) * 1000
            timings["analysis_ms"] = (time.monotonic() - t_analysis) * 1000

            # 5b. A retried analysis must be answered by the same trusted
            # character revision that produced it. If the persona changed
            # between the failed response and this retry, restore the recorded
            # snapshot; if that is impossible, fail explicitly rather than
            # silently mixing two versions.
            try:
                cognitive_context = await self._restore_analysis_persona(
                    event, cognitive_context
                )
            except PersonaRevisionConflict as exc:
                if persist:
                    await self.repos.utterance.update_status(
                        event.event_id, ProcessingStatus.FAILED
                    )
                self.logger.warning(
                    "persona_revision_conflict event_id=%s error=%s",
                    event.event_id,
                    exc,
                )
                steps.append(f"[CONFLICT] {exc}")
                return ProcessingReport(
                    event_id=event.event_id,
                    status=ReportStatus.CONFLICT.value,
                    error=str(exc),
                    steps=steps,
                )

            # 6. Local policy decides the final action from the (reused or
            # fresh) analysis before any further side effects. Hard local rules
            # (permission, explicit silence, quiet mode, explicit human
            # addressee) outrank the model suggestion.
            t_policy = time.monotonic()
            await self.policy.refresh(self.repos.settings)
            if semantic_mode and route is not None:
                decision = self.policy.decide_semantic(
                    route,
                    TrustedTurnContext(
                        addressee_hint=event.addressee_hint.value,
                        quiet_mode=bool(context.quiet_mode),
                        permission_denied=False,
                        safety_triggered=False,
                    ),
                )
            else:
                # Legacy path, plus semantic-mode local gates (no external-AI
                # consent, trusted human addressee, safety floor) which produce
                # a local analysis without calling the router.
                decision = self.policy.decide(
                    analysis,
                    context,
                    addressee_hint=event.addressee_hint.value,
                    explicit_silence=explicit_silence,
                )
            timings["policy_ms"] = (time.monotonic() - t_policy) * 1000
            # 6b. Independent safety floor. An explicit high-risk statement is
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

            # 7. Bounded synchronous write-through for an explicit "remember
            # this" request: the durable write must finish *before* the reply
            # that acknowledges it. Ordinary candidates stay with the job.
            memory_outcomes: list[dict] = []
            data_control_result = None
            if semantic_mode:
                intents = (
                    set(route.processing_intents)
                    if route is not None
                    else set()
                )
                explicit_intent = bool(
                    intents
                    & {
                        ProcessingIntent.MEMORY_SAVE,
                        ProcessingIntent.MEMORY_CORRECTION,
                    }
                )
                if (
                    persist
                    and learn
                    and route is not None
                    and explicit_intent
                    and self.explicit_memory_operations is not None
                ):
                    # WAIT/IGNORE may still perform this authorized internal
                    # write; the action only decides whether a reply follows.
                    memory_operation = await self._execute_explicit_memory(
                        event, route, consent, routing_view
                    )
                    memory_outcomes = memory_operation.outcomes
                    if not memory_outcomes:
                        # A routed SAVE with zero extracted candidates, DENIED
                        # permission or failed write is a real operation result.
                        # Pass it to Speaker; otherwise it may infer a false
                        # lack of write capability from the read-only tool list.
                        memory_outcomes = [{
                            "outcome": (
                                "EXPLICIT_NO_MEMORY"
                                if memory_operation.status == "NO_MEMORY"
                                else memory_operation.status
                            ),
                            "detail": memory_operation.error_code or (
                                "no_supported_fact_extracted"
                                if memory_operation.status == "NO_MEMORY"
                                else ""
                            ),
                        }]
                    steps.append(
                        "[MEMORY_WRITE] "
                        f"intent={memory_operation.intent} "
                        f"status={memory_operation.status} "
                        f"candidates={memory_operation.candidate_count}"
                    )
                    self.logger.info(
                        "explicit_memory_operation event_id=%s intent=%s"
                        " status=%s candidates=%d",
                        event.event_id,
                        memory_operation.intent,
                        memory_operation.status,
                        memory_operation.candidate_count,
                    )
                if (
                    persist
                    and ProcessingIntent.PRIVACY_REQUEST in intents
                    and self.data_control is not None
                ):
                    # A privacy request is a *request*: this only validates it
                    # and records a pending, confirmation-gated operation. It
                    # never deletes anything and never treats the audit trail
                    # as evidence that data was removed.
                    data_control_result = await self.data_control.resolve_request(
                        route.data_control_request, event, consent
                    )
                    steps.append(
                        "[PRIVACY] "
                        f"{data_control_result.status}"
                        + (
                            f":{data_control_result.detail}"
                            if data_control_result.detail
                            else ""
                        )
                    )
                    self.logger.info(
                        "privacy_request_resolved event_id=%s status=%s"
                        " operation_id=%s",
                        event.event_id,
                        data_control_result.status,
                        data_control_result.operation_id,
                    )
                if persist and not memory_outcomes:
                    memory_outcomes = (
                        await self.repos.analysis.list_candidate_writes(
                            event.event_id
                        )
                    )
            elif persist and learn and self.memory_writer is not None:
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

            # 8. Emotion runs only after the analysis exists (fresh or reused),
            # after the local policy decision and memory outcomes, and before
            # any response is generated. It reads the model's locally-validated
            # affect hint and the *privacy-filtered* provider memories, so a
            # retry reuses the persisted analysis without another model call.
            # Purely presentational; it can never change the action.
            cognitive_context = await self._apply_emotion(
                event,
                person_id,
                provider_memories,
                cognitive_context,
                analysis,
                steps,
                persist,
            )
            runtime_snapshot = await self._update_runtime_emotion(
                event, runtime_snapshot, cognitive_context
            )
            cognitive_context = self._attach_runtime_snapshot(
                cognitive_context, runtime_snapshot
            )

            # 8b. V0.9.9.6: an explicit first-person statement is written
            # directly here, before Speaker generation. There is no question to
            # prepare or append, and a failed write never affects the turn.
            await self._apply_direct_profile_self_reports(
                event, person_id, persist=persist
            )

            # 9. Reuse the durable action if it already exists, else generate a
            # response and persist the action.
            existing_action = (
                await self.repos.analysis.get_action(event.event_id)
                if persist
                else None
            )
            if existing_action is not None and existing_action.status not in (
                "COMPLETED",
                "TRUNCATED",
            ):
                # A previous attempt was interrupted mid-stream. Its partial
                # text is retained only long enough to be truthful; the retry
                # must not reuse it as a finished reply.
                if persist:
                    await self.repos.analysis.delete_action(event.event_id)
                existing_action = None
                steps.append("[ACTION] Discarded interrupted partial response")
            response_text: Optional[str] = None
            trace = None
            action_obj = None
            response_id = new_id("resp")
            action_status = "COMPLETED"
            stream_meta: dict = {}
            stream_emitted = False
            if existing_action is not None:
                final_action = existing_action.action_type
                final_reason = existing_action.reason_code
                response_text = existing_action.response_text
                action_obj = existing_action
                action_status = existing_action.status
                trace = self._stored_trace(existing_action)
                if stream is not None and final_action == CognitiveAction.RESPOND:
                    stream.started(response_id)
                    stream_emitted = True
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
                t_generation = time.monotonic()
                if final_action == CognitiveAction.RESPOND:
                    if safety.triggered:
                        # Fixed local text: no model call, no persona/emotion
                        # shaping, no scene preview. The reason is auditable.
                        response_text = safety_response(
                            safety.category, detect_language(event.text)
                        )
                        if stream is not None:
                            stream.started(response_id)
                            stream_emitted = True
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
                        if stream is not None and hasattr(
                            self.response, "generate_stream"
                        ):
                            (
                                response_text,
                                action_status,
                                response_id,
                                stream_meta,
                            ) = await self._generate_streaming(
                                event=event,
                                analysis=analysis,
                                response_context=response_context,
                                stream=stream,
                                started_at=t_generation,
                                persist=persist,
                                trace=trace,
                            )
                            stream_emitted = True
                        else:
                            generated = await self.response.generate(
                                event, analysis, response_context
                            )
                            response_text = str(generated)
                            # A provider may report that it stopped at the
                            # token limit; the reply is real but incomplete, so
                            # it must not be persisted as COMPLETED.
                            if (
                                getattr(generated, "finish_reason", None)
                                == "length"
                            ):
                                action_status = "TRUNCATED"
                        steps.append(f"[RESPONSE] {response_text}")
                timings["generation_ms"] = (time.monotonic() - t_generation) * 1000
                if stream_meta.get("first_token_ms") is not None:
                    timings["time_to_first_token_ms"] = stream_meta["first_token_ms"]
                    timings["speaker_first_token_ms"] = stream_meta[
                        "first_token_ms"
                    ]
            action_obj = AgentAction(
                    action_id=new_id("act"),
                    event_id=event.event_id,
                    action_type=final_action,
                    response_text=response_text,
                    reason_code=final_reason,
                    status=action_status,
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
            t_commit = time.monotonic()
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
                    learn_profile=bool(persist and learn and not safety.triggered),
                )
                if scene_enabled and action_obj is not None:
                    steps.append("[SCENE] Transition committed")
                # Stance continuity may only cite a response that was actually
                # emitted (or admin config). A retry reuses the existing action
                # and must not duplicate the record.
                if existing_action is None:
                    await self._record_stances(event, action_obj, cognitive_context)
            timings["commit_ms"] = (time.monotonic() - t_commit) * 1000

            if job_id is not None:
                steps.append("[MEMORY_JOB] Queued")
            steps.append("[STATUS] Completed")
            if (
                stream is not None
                and final_action == CognitiveAction.RESPOND
            ):
                if not stream_emitted:
                    stream.started(response_id)
                stream.completed(
                    response_id,
                    text=response_text or "",
                    finish_reason=stream_meta.get("finish_reason"),
                    usage=stream_meta.get("usage"),
                    report_status=ReportStatus.COMPLETED.value,
                    action=final_action.value,
                )
            timings["total_ms"] = (time.monotonic() - t0) * 1000
            self.metrics.record_many(timings)
            self.logger.info(
                "action_saved event_id=%s router_mode=%s route_action=%s"
                " input_semantics=%s router_provider_calls=%d"
                " router_followup_calls=%d tool_calls=%d router_llm_ms=%.2f"
                " tool_execution_ms=%.2f speaker_context_ms=%.2f"
                " speaker_first_token_ms=%.2f generation_ms=%.2f"
                " commit_ms=%.2f total_ms=%.2f action=%s reason_code=%s"
                " status=COMPLETED",
                event.event_id,
                self.router_mode,
                route.action.value
                if semantic_mode and route is not None
                else "NONE",
                route.input_semantics.value
                if semantic_mode and route is not None
                else "NONE",
                routing_outcome.provider_calls if routing_outcome is not None else 0,
                routing_outcome.planning_steps if routing_outcome is not None else 0,
                len(routing_outcome.tool_results)
                if routing_outcome is not None
                else 0,
                timings.get("router_llm_ms", 0.0),
                timings.get("tool_execution_ms", 0.0),
                timings.get("speaker_context_ms", 0.0),
                timings.get("speaker_first_token_ms", -1.0),
                timings.get("generation_ms", 0.0),
                timings.get("commit_ms", 0.0),
                timings["total_ms"],
                final_action.value,
                final_reason.value,
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
                timings=timings,
                steps=steps,
            )
        except asyncio.CancelledError:
            # Natural cancellation (consumer stopped). A partial, if any, was
            # already persisted by ``_generate_streaming`` with CANCELLED.
            if persist:
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.FAILED
                )
            timings["total_ms"] = (time.monotonic() - t0) * 1000
            self.metrics.record_many(timings)
            self.logger.info(
                "processing_cancelled event_id=%s duration_ms=%.2f",
                event.event_id,
                timings["total_ms"],
            )
            raise
        except Exception as exc:  # provider failure — keep the event retryable.
            if stream is not None and stream_emitted:
                # The stream already began (a partial or full text may have
                # been shown) but the turn did not commit. Report it as failed
                # rather than leaving the consumer waiting for a completion.
                stream.failed(
                    response_id, error=str(exc), partial_text=response_text
                )
            if persist:
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.FAILED
                )
            timings["total_ms"] = (time.monotonic() - t0) * 1000
            self.metrics.record_many(timings)
            self.logger.error(
                "processing_failed event_id=%s error=%s duration_ms=%.2f",
                event.event_id,
                exc,
                timings["total_ms"],
            )
            steps.append(f"[STATUS] FAILED: {exc}")
            return ProcessingReport(
                event_id=event.event_id,
                status=ReportStatus.FAILED.value,
                error=str(exc),
                timings=timings,
                steps=steps,
            )

    # -- helpers -----------------------------------------------------------

    # -- semantic routing --------------------------------------------------

    async def _route_semantic(
        self,
        event: UtteranceEvent,
        conversation: ConversationContext,
        consent,
        runtime_snapshot,
        steps: list[str],
        *,
        safety_triggered: bool,
        persist: bool,
        timings: Optional[dict] = None,
    ):
        """Run the LLM router or apply a hard local gate.

        The gates here are *trusted metadata*, not text classification: a
        withdrawn external-AI grant, an explicit human addressee or the
        independent safety floor. None of them falls back to keyword routing.
        """
        if event.addressee_hint.value == "human":
            steps.append("[ROUTE] local_gate:trusted_human_addressee")
            return None, None, None
        if safety_triggered:
            steps.append("[ROUTE] local_gate:safety")
            return None, None, None
        if not bool(consent.allow_external_ai):
            steps.append("[ROUTE] local_gate:external_ai_not_allowed")
            return None, None, None
        if self.routing_service is None or self.routing_context_builder is None:
            raise RouterUnavailable(
                "semantic routing is enabled but no router service is configured"
            )

        t_context = time.monotonic()
        routing_view = await self.routing_context_builder.build(event, conversation)
        if timings is not None:
            timings["routing_context_ms"] = (time.monotonic() - t_context) * 1000
        execution_context = ToolExecutionContext(
            event=event,
            person_id=event.speaker_person_id,
            repos=self.repos,
            memory=self.memory,
            registry=self.registry,
            privacy_gateway=getattr(
                self.routing_context_builder, "privacy_gateway", None
            ),
            runtime_snapshot=runtime_snapshot,
            allow_external_ai=True,
            web_search=self.web_search,
            logger=self.logger,
        )
        outcome = await self.routing_service.execute(
            event, routing_view, execution_context, persist=persist
        )
        if timings is not None:
            timings["router_llm_ms"] = outcome.router_llm_ms
            timings["router_followup_ms"] = outcome.router_followup_ms
            timings["tool_execution_ms"] = outcome.tool_execution_ms
        steps.append(
            "[ROUTE] semantic:"
            f"{outcome.route.action.value}, input="
            f"{outcome.route.input_semantics.value}, necessity="
            f"{outcome.route.reply_necessity.value},"
            f" intents={[item.value for item in outcome.route.processing_intents]}"
        )
        if outcome.tool_results:
            statuses = ", ".join(
                f"{item.tool_name}={item.status.value}"
                for item in outcome.tool_results
            )
            steps.append(f"[TOOLS] {statuses}")
        if outcome.planning_steps:
            steps.append("[ROUTE] follow-up plan applied")
        return routing_view, outcome, outcome.route

    @staticmethod
    def _is_noncommunicative_silent(route) -> bool:
        """A validated route that needs no reply and no internal operation.

        This reads only validated enums and lists. It never inspects the input
        text, so unusual spelling, symbols or language cannot be classified
        here; that judgment belongs to the semantic router.
        """
        return (
            route is not None
            and route.input_semantics == InputSemanticStatus.NON_COMMUNICATIVE
            and not route.processing_intents
            and not route.tool_requests
        )

    async def _finalize_silent_turn(
        self,
        *,
        event: UtteranceEvent,
        context: ConversationContext,
        route,
        routing_outcome,
        steps: list[str],
        timings: dict[str, float],
        t0: float,
        persist: bool,
    ) -> ProcessingReport:
        """Commit a non-communicative turn without building a speaker context.

        It is not an early ``return`` that skips persistence: the analysis is
        stored, the final policy action and non-response trace are recorded,
        and the conversation/event lifecycle commits through the same atomic
        `_commit_turn` used by the normal path. Scene transitions and profile
        observation are deliberately not advanced for this input.
        """
        analysis = None
        analysis_id = None
        if persist:
            analysis = await self.repos.analysis.get_analysis(event.event_id)
            analysis_id = await self.repos.analysis.get_analysis_id(event.event_id)

        if analysis is None:
            analysis = route_to_cognitive_result(
                event,
                route,
                routing_outcome.tool_results if routing_outcome is not None else [],
            )
            steps.append(
                "[COGNITION] Semantic router: "
                f"{analysis.action.value}, input=NON_COMMUNICATIVE"
            )
            if persist:
                analysis_id = new_id("ana")
                await self.repos.persist_cognitive_outcome(
                    analysis,
                    analysis_id,
                    candidates=[],
                    job_key=None,
                    job_payload={},
                    persona=None,
                )
        else:
            steps.append("[COGNITION] Reused existing analysis")

        t_policy = time.monotonic()
        await self.policy.refresh(self.repos.settings)
        decision = self.policy.decide_semantic(
            route,
            TrustedTurnContext(
                addressee_hint=event.addressee_hint.value,
                quiet_mode=bool(context.quiet_mode),
            ),
        )
        timings["policy_ms"] = (time.monotonic() - t_policy) * 1000
        final_action, final_reason = decision.action, decision.reason_code
        steps.append(f"[DECISION] {final_action.value}")
        if decision.overrode_model:
            steps.append(f"[POLICY] {decision.override_reason}")

        existing_action = (
            await self.repos.analysis.get_action(event.event_id)
            if persist
            else None
        )
        if existing_action is not None and existing_action.status not in (
            "COMPLETED",
            "TRUNCATED",
        ):
            if persist:
                await self.repos.analysis.delete_action(event.event_id)
            existing_action = None
            steps.append("[ACTION] Discarded interrupted partial response")

        if existing_action is not None:
            final_action = existing_action.action_type
            final_reason = existing_action.reason_code
            action_obj = existing_action
            trace = self._stored_trace(existing_action)
            steps.append("[ACTION] Reused existing action")
        else:
            trace = derive_nonresponse_trace(
                event=event,
                analysis=analysis,
                policy_decision=decision,
                conversation_context=context,
            )
            action_obj = AgentAction(
                action_id=new_id("act"),
                event_id=event.event_id,
                action_type=final_action,
                response_text=None,
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

        t_commit = time.monotonic()
        if persist:
            # No scene transition and no profile learning: the input was not a
            # communicative act, so it must not advance either state.
            await self._commit_turn(
                event=event,
                action=action_obj,
                existing_action=existing_action is not None,
                trace=trace,
                scene_enabled=False,
                previous_scene=None,
                analysis=analysis,
                cognitive_context=context,
                learn_profile=False,
            )
        timings["commit_ms"] = (time.monotonic() - t_commit) * 1000
        timings.setdefault("speaker_context_ms", 0.0)
        timings["total_ms"] = (time.monotonic() - t0) * 1000
        self.metrics.record_many(timings)
        self.logger.info(
            "silent_turn_committed event_id=%s router_mode=semantic"
            " route_action=%s input_semantics=%s router_provider_calls=%d"
            " router_followup_calls=%d tool_calls=%d router_llm_ms=%.2f"
            " tool_execution_ms=%.2f speaker_context_ms=%.2f total_ms=%.2f",
            event.event_id,
            RouteAction(route.action.value).value,
            route.input_semantics.value,
            routing_outcome.provider_calls if routing_outcome is not None else 0,
            routing_outcome.planning_steps if routing_outcome is not None else 0,
            len(routing_outcome.tool_results) if routing_outcome is not None else 0,
            timings.get("router_llm_ms", 0.0),
            timings.get("tool_execution_ms", 0.0),
            timings.get("speaker_context_ms", 0.0),
            timings["total_ms"],
        )
        steps.append("[STATUS] Completed")
        return ProcessingReport(
            event_id=event.event_id,
            status=ReportStatus.COMPLETED.value,
            action=final_action,
            response_text=None,
            reason_code=final_reason,
            analysis_id=analysis_id,
            timings=timings,
            steps=steps,
        )

    def _local_semantic_analysis(
        self, event: UtteranceEvent, consent, safety_triggered: bool
    ) -> CognitiveResult:
        """A trusted, non-router decision for a gated semantic turn.

        It is explicitly labelled as local policy; it is never presented as a
        model analysis and it never fabricates an LLM route.
        """
        if event.addressee_hint.value == "human":
            return CognitiveResult(
                event_id=event.event_id,
                meaning="trusted_human_addressee",
                action=CognitiveAction.IGNORE,
                reason_code=ReasonCode.DIRECT_HUMAN_ADDRESS,
                response_style="local_policy",
                provider="local_policy",
                reply_necessity=ReplyNecessity.UNNECESSARY,
            )
        if safety_triggered:
            return CognitiveResult(
                event_id=event.event_id,
                meaning="local_safety_floor",
                action=CognitiveAction.RESPOND,
                reason_code=ReasonCode.SAFETY_RESPONSE,
                response_style="local_policy",
                provider="local_policy",
                reply_necessity=ReplyNecessity.REQUIRED,
            )
        if not bool(consent.allow_external_ai):
            return CognitiveResult(
                event_id=event.event_id,
                meaning="external_ai_not_allowed",
                action=CognitiveAction.WAIT,
                reason_code=ReasonCode.PERMISSION_DENIED,
                response_style="local_policy",
                provider="local_policy",
                reply_necessity=ReplyNecessity.UNNECESSARY,
            )
        return CognitiveResult(
            event_id=event.event_id,
            meaning="semantic_router_unavailable",
            action=CognitiveAction.WAIT,
            reason_code=ReasonCode.UNKNOWN,
            response_style="local_policy",
            provider="local_policy",
            reply_necessity=ReplyNecessity.UNNECESSARY,
        )

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

    async def _apply_direct_profile_self_reports(
        self, event, person_id: str, *, persist: bool
    ) -> None:
        """Write explicit self-reports directly; never affect the turn."""
        service = self.profile_confirmation
        if service is None or not persist:
            return None
        try:
            await service.begin_turn(event=event, person_id=person_id)
        except Exception as exc:
            self.logger.warning(
                "profile_self_report_failed conversation_id=%s error=%s",
                event.conversation_id,
                type(exc).__name__,
            )
        return None

    async def build_proactive_judge_context(
        self, *, source_event, person_id: str, mode: str = "IMMEDIATE"
    ) -> Optional[dict]:
        """Assemble the bounded, committed view the Proactive Judge may see.

        It contains recent real turns (including already delivered proactive
        messages), the last delivered Ava reply, the committed scene and the
        elapsed time since that reply. No full memory retrieval and no full
        profile are performed here.

        ``mode`` is ``IMMEDIATE`` (right after a delivered reply) or
        ``LONG_IDLE`` (after a 2-4 minute silence). Only ``LONG_IDLE`` carries
        the structurally validated explicit-departure floor.
        """
        if self.context_builder is None:
            return None
        context = await self.context_builder.build_proactive(
            source_event, [], history_limit=12
        )
        kept = [
            turn
            for turn in context.turns
            if turn.role != ROLE_ASSISTANT and turn.event_id == source_event.event_id
        ]
        if not kept:
            # The triggering turn did not survive privacy filtering.
            return None

        turns = []
        for turn in context.turns[-12:]:
            if not (turn.text or "").strip():
                continue
            item = {
                "role": "assistant" if turn.role == ROLE_ASSISTANT else "user",
                "text": turn.text,
            }
            if item["role"] == "user":
                item["speaker"] = turn.speaker_label or "unknown"
            turns.append(item)
        last_ava = ""
        for turn in reversed(context.turns):
            if turn.role == ROLE_ASSISTANT and (turn.text or "").strip():
                last_ava = turn.text
                break

        scene_context = None
        if self.scene_manager is not None:
            try:
                record = await self.scene_manager.load(
                    source_event.conversation_id, person_id
                )
                if record is not None:
                    scene_context = record.state.prompt_dict()
            except Exception as exc:  # a judge without scene is still valid
                self.logger.warning(
                    "proactive_scene_read_failed conversation_id=%s error=%s",
                    source_event.conversation_id,
                    exc,
                )

        seconds_since_last_response = 0.0
        action = await self.repos.analysis.get_action(source_event.event_id)
        if action is not None and action.created_at is not None:
            seconds_since_last_response = max(
                0.0, (utcnow() - action.created_at).total_seconds()
            )

        return {
            "conversation_id": source_event.conversation_id,
            "person_id": person_id,
            "recent_turns": turns,
            "last_ava_response": last_ava,
            "scene_context": scene_context,
            "seconds_since_last_response": seconds_since_last_response,
            "mode": "LONG_IDLE" if str(mode).upper() == "LONG_IDLE" else "IMMEDIATE",
            "explicit_departure": (
                detect_explicit_departure(getattr(source_event, "text", ""))
                if str(mode).upper() == "LONG_IDLE"
                else False
            ),
        }

    async def build_proactive_context(self, *, source_event, person_id: str):
        """Build the full Speaker context for one Ava-initiated turn.

        Persona, profile, memory retrieval, scene, emotion and runtime context
        all come from the same services as a normal reply. This method never
        holds the conversation lock and never writes conversation state.
        """
        if self.context_builder is None:
            return None
        try:
            memories = await self.memory.retrieve(source_event.text, person_id)
        except Exception as exc:
            self.logger.warning(
                "proactive_memory_retrieval_failed conversation_id=%s error=%s",
                source_event.conversation_id,
                exc,
            )
            memories = []
        context = await self.context_builder.build_proactive(
            source_event, memories, history_limit=12
        )
        if not any(
            turn.role != ROLE_ASSISTANT and turn.event_id == source_event.event_id
            for turn in context.turns
        ):
            return None

        scene_enabled = await self._scene_enabled()
        personality_enabled = await self._personality_enabled()
        context, _scene_record = await self._prepare_dialogue(
            source_event,
            context,
            list(context.memories or []),
            [],
            scene_enabled=scene_enabled,
            personality_enabled=personality_enabled,
        )
        if self.runtime_snapshot_builder is not None:
            try:
                snapshot = await self.runtime_snapshot_builder.build(
                    source_event,
                    allow_external_ai=True,
                    persist=False,
                )
                context = self._attach_runtime_snapshot(context, snapshot)
            except Exception as exc:
                self.logger.warning(
                    "proactive_runtime_snapshot_failed conversation_id=%s error=%s",
                    source_event.conversation_id,
                    exc,
                )
        return await self._attach_live_affect(context, person_id)

    async def _attach_live_affect(self, context, person_id: str):
        """Reuse the current decayed emotion state without writing a new turn."""
        if self.emotion_service is None or not isinstance(context, CognitiveContext):
            return context
        try:
            agent_id = await self._resolve_agent_id()
            if not agent_id:
                return context
            state = await self.emotion_service.get_state(agent_id, person_id)
            affect_context = compile_affect_context(
                state,
                interaction_frame="UNKNOWN",
                frame_changed=False,
                config=getattr(self.emotion_service, "config", None),
            )
            return context.model_copy(
                update={
                    "emotion_state": state.as_dict(),
                    "affect_context": affect_context,
                }
            )
        except Exception as exc:
            self.logger.warning(
                "proactive_affect_failed person_id=%s error=%s", person_id, exc
            )
            return context

    async def generate_proactive_reply(
        self, *, source_event, person_id: str, mode: str = "IMMEDIATE"
    ) -> dict:
        """Generate one proactive reply through the existing Speaker path."""
        context = await self.build_proactive_context(
            source_event=source_event, person_id=person_id
        )
        if context is None:
            raise RuntimeError("proactive context unavailable")
        analysis = CognitiveResult(
            event_id=source_event.event_id,
            meaning="local proactive generation (no new user message)",
            action=CognitiveAction.RESPOND,
            reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
            response_style="proactive",
            provider="local_proactive",
            intent="proactive",
            addressed_to="agent",
            reply_necessity=ReplyNecessity.USEFUL,
            conversation_state="proactive",
        )
        generated = await self.response.generate_proactive(
            source_event, analysis, context, mode=mode
        )
        return {
            "text": str(generated).strip(),
            "finish_reason": getattr(generated, "finish_reason", None),
            "model": getattr(self.response, "last_model", "") or "",
            "context": context,
        }

    async def _prepare_context(
        self, event, context, memories, steps, routing_view=None
    ):
        """Return (provider_context, provider_memories) for this utterance.

        Without a builder this is exactly the old behaviour: the lightweight
        ConversationContext and the retrieved memories. With a builder (real
        DeepSeek mode) the context is privacy-filtered before it can reach the
        network, and the providers get the same filtered memory list that the
        CognitiveContext carries.

        Semantic mode already built and privacy-filtered a minimal history
        window for the router. Reusing those authorized turns avoids a second
        query that could observe a different permission state mid-turn.
        """
        if self.context_builder is None:
            return context, memories
        if routing_view is not None:
            cognitive_context = (
                await self.context_builder.build_from_authorized_turns(
                    event, memories, routing_view
                )
            )
            cognitive_context = cognitive_context.model_copy(
                update={"quiet_mode": context.quiet_mode}
            )
            provider_memories = list(cognitive_context.memories or [])
            gateway = getattr(self.context_builder, "privacy_gateway", None)
            if gateway is not None:
                provider_memories, _dropped = await gateway.filter_memories(
                    provider_memories, self._subject_ids_of
                )
                if len(provider_memories) != len(memories):
                    steps.append(
                        "[PRIVACY] Filtered "
                        f"{len(memories) - len(provider_memories)} memories"
                    )
                cognitive_context = cognitive_context.model_copy(
                    update={"memories": provider_memories}
                )
            return cognitive_context, provider_memories
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

    async def _attach_memory_signals(
        self, event, person_id, memories, context, steps
    ):
        """Tell the model what the local lookup actually did.

        Three different situations must not collapse into one silence:

        - no long-term memory exists for this person;
        - the store has memories but this query matched none;
        - history was not available to this request.

        Only the rich provider context can carry these facts; the mock path is
        unchanged. Explicit sample/history requests use the same access-checked
        service the admin UI uses, so no cross-person leak is possible.
        """
        if not isinstance(context, CognitiveContext):
            return context
        if self.context_builder is None:
            return context

        # Retrieval is already privacy-filtered in the provider context. The
        # raw list may contain memories the gateway dropped; using it would
        # tell the model "found relevant memories" when the background is
        # actually empty.
        visible_memories = list(context.memories or [])
        status = "hits" if visible_memories else "empty"
        samples: list[dict] = []
        history_excerpt: list[dict] = []
        history_loaded = len(getattr(context, "turns", []) or [])

        if not visible_memories:
            visible_any = await self.memory.browse(person_id, limit=1)
            status = "empty" if visible_any else "no_memories"

        if is_memory_sample_request(event.text):
            samples = await self._visible_memory_samples(person_id)
            if samples:
                status = "hits"
            elif status == "hits":
                # Keyword hits exist but none survived the explicit sample
                # lookup; report it truthfully as an empty retrieval.
                status = "empty"
            steps.append(f"[MEMORY] Explicit sample requested ({len(samples)})")

        if is_earliest_history_request(event.text):
            history_excerpt = await self._earliest_history(event, person_id)
            if history_excerpt:
                history_loaded = max(history_loaded, len(history_excerpt))
            steps.append(f"[HISTORY] Earliest excerpt ({len(history_excerpt)})")

        # Exact counts are local facts. The model must cite these numbers
        # rather than estimate them from the last chat time or the visible
        # history window. Only this speaker's own records are counted.
        conversation_stats = None
        if is_conversation_count_request(event.text):
            conversation_stats = await self.repos.utterance.conversation_stats(
                person_id
            )
            steps.append(
                f"[STATS] messages={conversation_stats['user_messages']}"
                f" replies={conversation_stats['assistant_replies']}"
                f" sessions={conversation_stats['sessions']}"
            )

        return context.model_copy(
            update={
                "memory_retrieval_status": status,
                "memory_samples": samples,
                "history_excerpt": history_excerpt,
                "history_loaded": history_loaded,
                "conversation_stats": conversation_stats,
            }
        )

    async def _visible_memory_samples(self, person_id: str) -> list[dict]:
        samples = await self.memory.browse(person_id, limit=3)
        gateway = getattr(self.context_builder, "privacy_gateway", None)
        if gateway is not None:
            samples, _dropped = await gateway.filter_memories(
                samples, self._subject_ids_of
            )
        return samples

    async def _earliest_history(self, event, person_id: str) -> list[dict]:
        turns = await self.repos.utterance.earliest_for_speaker(
            person_id,
            event.conversation_id,
            exclude_event_id=event.event_id,
            limit=3,
        )
        gateway = getattr(self.context_builder, "privacy_gateway", None)
        if gateway is not None:
            turns = await gateway.filter_utterances(turns)
        return [
            {
                "occurred_at": turn.occurred_at.isoformat(),
                "speaker": turn.speaker_label or "unknown",
                "text": turn.text,
            }
            for turn in turns
        ]

    @staticmethod
    async def _subject_ids_of(memory: dict) -> list[str]:
        """PERSON subject ids, matching the privacy gateway's async contract."""
        return [
            subject["subject_id"]
            for subject in memory.get("subjects", [])
            if subject.get("subject_type") == "PERSON"
        ]

    @staticmethod
    def _attach_runtime_snapshot(context, snapshot):
        """Attach the per-event runtime data to the provider context.

        It is prompt *data* (local, trusted state), never an instruction and
        never a place for model output. Only the rich CognitiveContext reaches
        the real provider; the mock path keeps its lightweight context.
        """
        if snapshot is None or not isinstance(context, CognitiveContext):
            return context
        return context.model_copy(update={"runtime_snapshot": snapshot.as_dict()})

    async def _execute_explicit_memory(
        self, event, route, consent, routing_view
    ) -> MemoryOperationResult:
        """Run an explicit memory intent as a foreground off-task task.

        The router only *proposes* the intent; this trusted local code chooses
        the task type, foreground mode and ROUTER source. When no full task
        service is wired (legacy unit wirings), the same service is called
        directly, preserving behaviour.
        """
        intent = None
        if getattr(route, "processing_intents", None):
            intents = set(route.processing_intents)
            if "MEMORY_CORRECTION" in {getattr(i, "value", i) for i in intents}:
                intent = JobType.MEMORY_CORRECTION.value
            elif "MEMORY_SAVE" in {getattr(i, "value", i) for i in intents}:
                intent = JobType.MEMORY_SAVE.value
        task_service = self.task_service
        if (
            task_service is None
            or intent is None
            or intent not in task_service.supported_types
        ):
            return await self.explicit_memory_operations.execute(
                event, route, consent, routing_view
            )
        payload = {
            # Trusted reference for permission re-reads. The event snapshot is
            # historical proposal data and is never used as a grant.
            "event_id": event.event_id,
            "event": event.model_dump(mode="json"),
            "route": route.model_dump(mode="json"),
            "consent": consent.model_dump(mode="json") if consent is not None else None,
            "routing_context": (
                routing_view.model_dump(mode="json")
                if routing_view is not None
                and hasattr(routing_view, "model_dump")
                else None
            ),
        }
        try:
            result = await task_service.execute_now(
                TaskRequest(
                    type=intent,
                    payload=payload,
                    mode="FOREGROUND",
                    source=TaskSource.ROUTER,
                    priority=TaskPriority.HIGH,
                    source_event_id=event.event_id,
                    job_key=f"explicit_memory:{event.event_id}:{intent}",
                )
            )
        except TaskRejected as exc:
            return MemoryOperationResult(
                event_id=event.event_id,
                intent=intent,
                status="DENIED",
                completed=True,
                error_code=exc.reason.lower(),
            )
        if isinstance(result.result, dict):
            try:
                return MemoryOperationResult(**result.result)
            except TypeError:
                pass
        return MemoryOperationResult(
            event_id=event.event_id,
            intent=intent,
            status="FAILED",
            completed=False,
            error_code=result.error or "task_failed",
        )

    async def _enqueue_turn_analysis(self, event: UtteranceEvent) -> Optional[str]:
        """Queue the batched extraction job for an already-answered turn.

        The event's local analysis is already persisted at this point, so it is
        visible to ``pending_events``. A small number of waiting events is
        delayed so they can share one request; once the batch is full the job
        is due immediately. The job key keeps this idempotent.
        """
        if self.jobs is None:
            return None
        next_run_at = None
        if event.speaker_person_id is not None and self.turn_analysis_delay_seconds:
            waiting = await self.repos.extraction.pending_events(
                event.conversation_id,
                event.speaker_person_id,
                limit=self.turn_analysis_batch_size,
            )
            if len(waiting) < self.turn_analysis_batch_size:
                next_run_at = utcnow() + timedelta(
                    seconds=self.turn_analysis_delay_seconds
                )
        job_key = f"turn_analysis:{event.event_id}"
        if (
            self.task_service is not None
            and JobType.TURN_ANALYSIS.value in self.task_service.supported_types
        ):
            # Ordinary extraction is off-task work too: route it through the
            # same local checks, priority order and durable queue. The legacy
            # direct enqueue remains for wirings without a task service.
            try:
                return await self.task_service.enqueue(
                    TaskRequest(
                        type=JobType.TURN_ANALYSIS.value,
                        payload={"event_id": event.event_id},
                        source=TaskSource.SYSTEM,
                        source_ref=event.event_id,
                        job_key=job_key,
                        next_run_at=next_run_at,
                    )
                )
            except TaskRejected as exc:
                # A full autonomous queue may defer learning; it never blocks
                # or rewrites the user-visible reply.
                self.logger.info(
                    "turn_analysis_not_queued event_id=%s reason=%s",
                    event.event_id,
                    exc.reason,
                )
                return None
        return await self.repos.job.enqueue(
            JobType.TURN_ANALYSIS,
            job_key,
            {"event_id": event.event_id},
            next_run_at=next_run_at,
        )

    @staticmethod
    def _local_turn_analysis(
        event, assessment: Optional[TurnAssessment]
    ) -> CognitiveResult:
        """A local decision record for a turn settled by trusted metadata.

        This is deliberately *not* presented as a model analysis: ``provider``
        is ``local_fast_path`` and the meaning string says so. The recorded
        action and reply necessity come from the local assessment, so a WAIT
        or IGNORE is stored as such instead of being disguised as a RESPOND.
        The policy still applies every local rule on top of it.
        """
        if assessment is None:
            # Reused local analysis (retry of a Fast Path turn).
            assessment = TurnAssessment(
                action=CognitiveAction.RESPOND,
                reason=ReasonCode.DIRECT_AGENT_ADDRESS,
                reply_necessity=ReplyNecessity.USEFUL,
                rule="local_direct_address",
                addressee="agent",
            )
        return CognitiveResult(
            event_id=event.event_id,
            meaning=(
                f"local selective response ({assessment.rule}); "
                "no model analysis"
            ),
            action=assessment.action,
            reason_code=assessment.reason,
            response_style="fast_path",
            provider=LOCAL_PROVIDER,
            intent="local_assessment",
            addressed_to=assessment.addressee,
            reply_necessity=assessment.reply_necessity,
            conversation_state=assessment.rule,
        )

    async def _update_runtime_emotion(self, event, snapshot, cognitive_context):
        """Record the companion's own state on the runtime snapshot.

        The snapshot is created before emotion is appraised; the emotion value
        is therefore attached afterwards, in memory for the prompt and (when
        persistence is allowed) on the stored row. A retry reuses the stored
        snapshot and this updates it again with the freshly reused state.
        """
        if snapshot is None:
            return snapshot
        state = getattr(cognitive_context, "emotion_state", None)
        if not isinstance(state, dict):
            return snapshot
        emotion = state.get("emotion")
        if not emotion:
            return snapshot
        updated = snapshot.model_copy(update={"ava_emotion": str(emotion)})
        try:
            await self.repos.runtime_snapshot.set_emotion(
                event.event_id, updated.ava_emotion
            )
        except Exception as exc:  # runtime context is never fatal
            self.logger.warning(
                "runtime_emotion_update_failed event_id=%s error=%s",
                event.event_id,
                exc,
            )
        return updated

    async def _generate_streaming(
        self,
        *,
        event,
        analysis,
        response_context,
        stream,
        started_at: float,
        persist: bool,
        trace=None,
    ):
        """Stream a response, yielding deltas through ``stream``.

        Returns ``(text, action_status, response_id, meta)``. A partial reply
        is persisted with an explicit ``CANCELLED``/``FAILED`` status and is
        never reported as complete. Only after the provider reports the full
        completion is the status ``COMPLETED`` (or ``TRUNCATED`` when the
        provider stopped because of the token limit).
        """
        response_id = new_id("resp")
        chunks: list[str] = []
        emitted_chunks: list[str] = []
        finish_reason: Optional[str] = None
        usage: Optional[dict] = None
        first_token = False
        first_token_ms: Optional[float] = None
        layout = response_layout_for(event.text)
        # Emit already-normalised deltas so the text a surface displays is the
        # same string the database stores. A trailing whitespace run is held
        # back until the next real character, which makes the concatenation of
        # every emitted delta exactly ``normalize_response_text(raw, layout)``.
        normalizer = StreamingTextNormalizer(layout)
        stream.started(response_id)
        try:
            async for delta in self.response.generate_stream(
                event, analysis, response_context
            ):
                if delta.content:
                    if not first_token:
                        first_token = True
                        first_token_ms = (time.monotonic() - started_at) * 1000
                    chunks.append(delta.content)
                    visible = normalizer.feed(delta.content)
                    if visible:
                        emitted_chunks.append(visible)
                        stream.delta(response_id, visible)
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                if delta.usage_reported and delta.total_tokens is not None:
                    usage = {
                        "prompt_tokens": delta.prompt_tokens,
                        "completion_tokens": delta.completion_tokens,
                        "total_tokens": delta.total_tokens,
                    }
        except asyncio.CancelledError:
            partial = "".join(chunks)
            await self._persist_partial(
                event=event,
                analysis=analysis,
                text=partial,
                status="CANCELLED",
                trace=trace,
                persist=persist,
            )
            stream.cancelled(response_id, partial_text=partial or None)
            raise
        except Exception as exc:
            partial = "".join(chunks)
            await self._persist_partial(
                event=event,
                analysis=analysis,
                text=partial,
                status="FAILED",
                trace=trace,
                persist=persist,
            )
            stream.failed(response_id, error=str(exc), partial_text=partial or None)
            raise
        text = "".join(chunks).strip()
        # Second-layer layout guarantee: for ordinary chat, collapse any stray
        # paragraph breaks into one natural paragraph so the *persisted* text
        # matches what the chat surface displays. This operates on the whole
        # accumulated text (never per-delta) and never runs for structured
        # requests (code / tutorials / formulas keep their layout verbatim).
        text = normalize_response_text(text, layout)
        status = "TRUNCATED" if finish_reason == "length" else "COMPLETED"
        # ``emitted_text`` is what the surface has actually displayed. Because
        # the deltas were normalised incrementally it is exactly ``text`` (the
        # trailing held-back whitespace is trimmed by both layouts), so the
        # caller can send only the remaining suffix — an appended confirmation
        # question — without re-printing the reply.
        return (
            text,
            status,
            response_id,
            {
                "finish_reason": finish_reason,
                "usage": usage,
                "first_token_ms": first_token_ms,
                "emitted_text": "".join(emitted_chunks),
            },
        )

    async def _persist_partial(
        self,
        *,
        event,
        analysis,
        text: str,
        status: str,
        trace=None,
        persist: bool,
    ) -> None:
        """Persist interrupted output in a short, separate transaction.

        No DB transaction is held while the model streams; this only runs once
        the stream has already ended (cancelled or failed). The event stays
        retryable: a later retry deletes this partial action and regenerates.
        """
        if not persist:
            return
        action = AgentAction(
            action_id=new_id("act"),
            event_id=event.event_id,
            action_type=CognitiveAction.RESPOND,
            response_text=text or None,
            reason_code=getattr(analysis, "reason_code", ReasonCode.UNKNOWN),
            status=status,
            created_at=utcnow(),
            model_suggested_action=getattr(analysis, "action", None),
            nonresponse_trace=(
                trace.model_dump() if trace is not None else None
            ),
        )
        try:
            async with self.repos.db.transaction(immediate=True):
                await self.repos.analysis.delete_action(event.event_id)
                await self.repos.analysis.save_action(action)
                await self.repos.utterance.update_status(
                    event.event_id, ProcessingStatus.FAILED
                )
        except Exception as exc:  # never mask the original interruption
            self.logger.warning(
                "partial_persist_failed event_id=%s status=%s error=%s",
                event.event_id,
                status,
                exc,
            )

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
        """Appraise this turn's emotion and attach a short state context.

        The collapsed V0.5 flow is exactly: emotional event → state update →
        a short, human-readable affect context when (and only when) the state
        is meaningful. ``process_turn`` appraises, applies and records the turn
        in one transaction, so a retry reuses the analysis and the receipt
        rather than double-applying the shift.

        What is *not* computed here any more: a per-turn speech plan (the
        independent TTS capability still exists on the service), a numeric
        directness/warmth stance, or dialogue-pattern feedback. Those duplicated
        the response prompt's own guidance and added serial work per turn.
        Weak or irrelevant affect yields an empty ``affect_context`` instead of
        a filled-in placeholder.

        Emotion is presentation-only and must never fail the turn, so any error
        is logged and swallowed — a reply still happens, just without a state
        line. ``persist`` gates only the event/snapshot writes (honouring the
        consent persistence flag).
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
            pragmatic = getattr(analysis, "pragmatic", None)
            current_frame = "UNKNOWN"
            if pragmatic is not None:
                current_frame = getattr(pragmatic, "interaction_frame", "UNKNOWN")
            frame_changed = (
                str(current_frame or "UNKNOWN").upper()
                not in ("UNKNOWN", "")
                and str(current_frame).upper() != str(previous_frame).upper()
            )
            conversational_affect = build_conversational_affect(
                pragmatic=pragmatic,
                signal=getattr(analysis, "affective_signal", None),
                previous_frame=previous_frame,
                actor_person_id=event.speaker_person_id,
            )
            snapshot = getattr(cognitive_context, "character_snapshot", None)
            dispositions = None
            if isinstance(snapshot, dict):
                raw_dispositions = snapshot.get("emotional_dispositions")
                if isinstance(raw_dispositions, dict):
                    dispositions = raw_dispositions
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
                dispositions=dispositions,
            )
            emotion_config = getattr(self.emotion_service, "config", None)
            affect_context = compile_affect_context(
                state,
                interaction_frame=current_frame,
                frame_changed=frame_changed,
                config=emotion_config,
            )

            if isinstance(cognitive_context, CognitiveContext):
                cognitive_context = cognitive_context.model_copy(
                    update={
                        "emotion_state": state.as_dict(),
                        "affect_context": affect_context,
                    }
                )
            self.logger.info(
                "emotion_applied event_id=%s emotion=%s intensity=%.2f"
                " valence=%.2f mood=%s tension=%.2f affect_context=%s",
                event.event_id,
                state.emotion,
                state.intensity,
                state.valence,
                state.mood_emotion,
                state.relationship_tension,
                bool(affect_context),
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

    @staticmethod
    def _persona_record(context) -> Optional[dict]:
        """The trusted persona revision to persist next to an analysis."""
        if not isinstance(context, CognitiveContext):
            return None
        snapshot = context.character_snapshot
        if not isinstance(snapshot, dict):
            return None
        agent_id = snapshot.get("agent_id")
        try:
            revision = int(snapshot.get("revision"))
        except (TypeError, ValueError):
            revision = None
        if not agent_id or revision is None:
            return None
        return {
            "agent_id": str(agent_id),
            "revision": revision,
            "snapshot": snapshot,
        }

    async def _restore_analysis_persona(self, event, cognitive_context):
        """Keep a retried analysis and its reply on one character snapshot.

        The Persona Core revision alone is not enough: traits, description,
        relationship tier, stances and emotional dispositions can change while
        the revision stays the same. Consistency is therefore decided by the
        SHA-256 fingerprint of the complete snapshot:

        - same fingerprint: the current snapshot is exactly the one the
          analysis saw;
        - different fingerprint with a valid stored snapshot: restore that
          snapshot for this response only (it is never written back over the
          current Persona Core);
        - missing/corrupt/inconsistent metadata or snapshot: an explicit
          conflict. A legacy row (schema 16) has NULL metadata and is
          "unknown", never assumed to match.
        """
        if not isinstance(cognitive_context, CognitiveContext):
            return cognitive_context
        current = cognitive_context.character_snapshot
        if not isinstance(current, dict):
            # Mock/lightweight mode has no persona content in the providers, so
            # there is no character version that could be mixed.
            return cognitive_context

        stored = await self.repos.analysis.get_analysis_persona(event.event_id)
        if stored is None:
            return cognitive_context

        try:
            saved = validate_stored_snapshot(stored)
            current_agent = str(current.get("agent_id") or "").strip()
            if not current_agent:
                raise CharacterSnapshotValidationError(
                    "the current character snapshot has no agent_id"
                )
            if saved["agent_id"] != current_agent:
                raise CharacterSnapshotValidationError(
                    "agent identity mismatch: the analysis was produced by"
                    f" {saved['agent_id']}, current is {current_agent}"
                )
            saved_fingerprint = character_snapshot_fingerprint(saved)
            current_fingerprint = character_snapshot_fingerprint(current)
        except CharacterSnapshotValidationError as exc:
            raise PersonaRevisionConflict(
                f"{exc}; submit the utterance with a new event_id"
            ) from exc

        if saved_fingerprint == current_fingerprint:
            return cognitive_context

        # Restoring a snapshot can re-expose stances or other user-derived
        # material recorded under an authorization that has since been
        # withdrawn. The current grant, not the old snapshot, decides.
        if not await self._external_ai_authorized(event.speaker_person_id):
            raise PersonaRevisionConflict(
                "external AI authorization was withdrawn after this analysis"
                " was saved; its character snapshot cannot be resent"
                " (submit the utterance with a new event_id)"
            )

        self.logger.info(
            "persona_snapshot_restored event_id=%s analysis_fingerprint=%s"
            " current_fingerprint=%s",
            event.event_id,
            saved_fingerprint[:12],
            current_fingerprint[:12],
        )
        return cognitive_context.model_copy(
            update={"character_snapshot": saved}
        )

    async def _external_ai_authorized(self, person_id) -> bool:
        """The same trusted consent check the privacy gateway applies."""
        if person_id is None:
            return False
        person = await self.registry.get(person_id)
        if person is None or person.registration_status != RegistrationStatus.ACTIVE:
            return False
        consent = await self.registry.get_consent(person_id)
        return bool(
            consent is not None
            and consent.allow_processing
            and consent.allow_external_ai
        )

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

    async def _attach_character_snapshot(
        self,
        event,
        cognitive_context,
        *,
        effective=None,
        stances=None,
    ):
        """Attach the one trusted CharacterSnapshot both calls will share.

        Non-fatal by design: without a snapshot both prompts fall back to the
        older agent-context rendering rather than failing the turn.
        """
        if not isinstance(cognitive_context, CognitiveContext):
            return cognitive_context
        if cognitive_context.character_snapshot is not None:
            return cognitive_context
        try:
            if effective is None:
                effective = await self._effective_context(
                    event.speaker_person_id,
                    cognitive_context.agent_context,
                )
            if effective is None:
                return cognitive_context
            scene_view = getattr(cognitive_context, "scene_state", None)
            topic = (
                scene_view.get("topic") if isinstance(scene_view, dict) else None
            )
            if stances is None:
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
            dispositions = await load_emotional_dispositions(self.repos.settings)
            snapshot = build_character_snapshot(
                effective,
                stances=stances,
                query_text=event.text,
                topic=topic,
                dispositions=dispositions,
            )
            return cognitive_context.model_copy(
                update={"character_snapshot": snapshot.as_dict()}
            )
        except Exception as exc:
            self.logger.warning(
                "character_snapshot_failed event_id=%s error=%s",
                event.event_id,
                exc,
            )
            return cognitive_context

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
        """Load scene state and compile this turn's character view/style.

        The trusted :class:`CharacterSnapshot` is built once here and attached
        to the shared context, so the cognitive and response calls see the same
        persona revision and relevant stances. Style compilation is optional
        and only adds legacy prompt fields; it never controls identity.

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

        if not isinstance(cognitive_context, CognitiveContext):
            return cognitive_context, scene_record

        try:
            effective = await self._effective_context(
                event.speaker_person_id, cognitive_context.agent_context
            )
            if effective is None:
                return cognitive_context, scene_record

            scene_view = getattr(cognitive_context, "scene_state", None)
            stances: list = []
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

            # One trusted character view per turn, shared by the cognitive and
            # response calls (same revision, same relevance-filtered stances).
            # It is built even when the optional style compiler is disabled.
            cognitive_context = await self._attach_character_snapshot(
                event,
                cognitive_context,
                effective=effective,
                stances=stances,
            )
            snapshot = getattr(cognitive_context, "character_snapshot", None)
            if isinstance(snapshot, dict):
                steps.append(
                    f"[CHARACTER] Snapshot revision={snapshot.get('revision')}"
                )

            if not personality_enabled:
                return cognitive_context, scene_record

            style = compile_dialogue_style(
                effective,
                memories=provider_memories,
                current_text=event.text,
            )
            personality = (
                self.personality_compiler.compile(
                    effective,
                    style=style,
                    scene=scene_view,
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
        learn_profile: bool = False,
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
                await self._observe_profile(event, learn_profile)
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
                    await self._observe_profile(event, learn_profile)
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

    async def _observe_profile(self, event, enabled: bool) -> None:
        """Request a background review; never write the profile locally.

        A conversation turn may only *register* the review request, in the same
        transaction as the turn. The single authoritative path for changing the
        profile is: read the complete old profile -> model proposes -> local
        rules validate -> atomic commit. The old regex learner is kept for
        explicit callers but is no longer allowed to mutate the profile from
        ordinary conversation text.
        """
        if not enabled:
            return
        person_id = event.speaker_person_id
        if person_id is None:
            return
        if self.profile_learning is not None:
            signals = self.profile_learning.observe_signals(event.text)
            if signals:
                self.logger.info(
                    "profile_signals_observed event_id=%s fields=%s",
                    event.event_id,
                    ",".join(signal.field_key for signal in signals),
                )
        if self.profile_review is not None:
            # Deliberately not swallowed: losing this insert would leave a
            # committed turn with no recoverable review request.
            await self.profile_review.request(person_id, event.event_id)

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
