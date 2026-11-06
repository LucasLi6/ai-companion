"""Composition root: wire every component for the CLI and tests.

This is the single place that knows how the concrete pieces fit together. The
Orchestrator itself only depends on the provider/memory *protocols*, so the
mock and real-DeepSeek backends are swapped here without touching the pipeline.

``provider="mock"`` (the default) needs no key and makes no network call.
``provider="deepseek"`` builds the real client from ``DEEPSEEK_*`` environment
variables; a missing key raises :class:`DeepSeekConfigError` (never a silent
fallback to mock).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from companion.agent.personality_compiler import PersonalityCompiler
from companion.agent.service import AgentService
from companion.agent.style_examples import StyleExampleRepository
from companion.cognition.context_builder import ContextBuilder
from companion.cognition.scene_manager import SceneManager
from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.core.conversation import ConversationManager
from companion.core.metrics import LatencyMetrics
from companion.core.orchestrator import Orchestrator
from companion.core.policy import Policy
from companion.domain.enums import JobType
from companion.emotion.parameters import load_emotion_config
from companion.emotion.repository import EmotionRepository
from companion.emotion.service import EmotionService
from companion.jobs.embedding_handler import build_embedding_handler
from companion.jobs.handlers import build_turn_analysis_handler, memory_consolidation
from companion.jobs.profile_review import (
    ProfileReviewScheduler,
    build_profile_review_handler,
)
from companion.jobs.worker import JobWorker
from companion.memory.consolidation import MemoryWriter
from companion.memory.embedding_repository import EmbeddingRepository
from companion.memory.embedding_service import EmbeddingService
from companion.memory.explicit_operations import ExplicitMemoryOperationService
from companion.memory.hybrid_retriever import HybridRetriever
from companion.memory.service import MemoryService, build_memory_service
from companion.memory.store import SqliteMemoryStore
from companion.people.registry import PersonRegistry
from companion.people.profile_learning import PersonProfileLearningService
from companion.people.profile_refinement import (
    ProfileRefinementEngine,
    ProfileReviewContextSource,
)
from companion.people.profile_service import PersonProfileService
from companion.privacy.data_control import DataControlService
from companion.providers.deepseek_client import (
    DeepSeekConfigError,
    build_deepseek_client,
    load_deepseek_config,
)
from companion.providers.deepseek_cognitive import DeepSeekCognitiveProvider
from companion.providers.deepseek_profile import (
    DeepSeekProfileRefinementProvider,
    MockProfileRefinementProvider,
)
from companion.providers.deepseek_proactive import (
    DeepSeekProactiveJudgeProvider,
)
from companion.proactive.judge import ProactiveJudge
from companion.proactive.manager import ProactiveManager
from companion.providers.deepseek_response import DeepSeekResponseProvider
from companion.providers.deepseek_router import DeepSeekRouterProvider
from companion.providers.mock import MockCognitiveProvider, MockResponseProvider
from companion.providers.websearch import SettingsWebSearchProvider
from companion.routing.context import RoutingContextBuilder
from companion.routing.service import RoutingService
from companion.routing.validator import RoutingValidator
from companion.tools.executor import ToolExecutor
from companion.tools.registry import build_default_tool_registry
from companion.runtime_context.geolocation import GeoLocationService
from companion.runtime_context.snapshot import RuntimeSnapshotBuilder
from companion.runtime_context.timezone import TimeZoneService
from companion.storage.database import Database
from companion.storage.repositories import Repositories, make_repositories
from companion.tasks.handlers import (
    build_explicit_memory_task_handler,
    build_research_task_handler,
)
from companion.tasks.service import TaskService

PROVIDERS = ("mock", "deepseek")


@dataclass
class Runtime:
    db: Database
    repos: Repositories
    registry: PersonRegistry
    conversations: ConversationManager
    policy: Policy
    cognitive: object
    response: object
    memory: SqliteMemoryStore
    memory_service: MemoryService
    agent_service: AgentService
    worker: Optional[JobWorker]
    orchestrator: Orchestrator
    provider: str = "mock"
    context_builder: Optional[ContextBuilder] = None
    privacy_gateway: Optional[ForegroundPrivacyGateway] = None
    embedding_service: Optional[EmbeddingService] = None
    emotion_service: Optional[EmotionService] = None
    hybrid_retriever: Optional[HybridRetriever] = None
    scene_manager: Optional[SceneManager] = None
    style_repo: Optional[StyleExampleRepository] = None
    personality_compiler: Optional[PersonalityCompiler] = None
    runtime_snapshot_builder: Optional[RuntimeSnapshotBuilder] = None
    geo_service: Optional[GeoLocationService] = None
    metrics: Optional[LatencyMetrics] = None
    profile_service: Optional[PersonProfileService] = None
    router_mode: str = "legacy"
    router_provider: object = None
    routing_service: object = None
    tool_registry: object = None
    tool_executor: object = None
    web_search: object = None
    explicit_memory_operations: object = None
    data_control: object = None
    task_service: Optional[TaskService] = None
    research_engine: object = None
    profile_refinement_engine: object = None
    proactive_manager: Optional[ProactiveManager] = None


def build_runtime(
    db: Database,
    *,
    with_worker: bool = True,
    cognitive=None,
    response=None,
    logger: Optional[logging.Logger] = None,
    provider: str = "mock",
    deepseek_client=None,
    deepseek_config=None,
    allow_mixed_providers: bool = False,
    embedding_provider=None,
    scene_enabled: Optional[bool] = None,
    personality_generation_enabled: Optional[bool] = None,
    cognitive_repair_enabled: Optional[bool] = None,
    fast_path_enabled: Optional[bool] = None,
    router_mode: Optional[str] = None,
    router_provider=None,
    tool_registry=None,
    web_search_provider=None,
    turn_analysis_delay_seconds: Optional[float] = None,
    real_idle_provider: Optional[bool] = None,
    profile_refinement_provider=None,
    profile_review_delay_seconds: float = 30.0,
    proactive_judge=None,
    proactive_delay_seconds: float = 2.0,
    proactive_short_task_timeout: float = 1.0,
) -> Runtime:
    """Build a fully wired runtime over an already-connected database.

    ``provider="mock"`` preserves the V0.0 behaviour exactly (and accepts
    injected ``cognitive``/``response`` for tests). ``provider="deepseek"``
    requires both providers to come from DeepSeek: injecting only one is
    rejected unless ``allow_mixed_providers=True`` is set explicitly, so a
    real-cognitive + mock-response misconfiguration cannot happen by accident.
    """
    provider = (provider or "mock").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")

    repos = make_repositories(db)
    profile_service = PersonProfileService(
        repos.profile, repos.person, audit=repos.audit
    )
    profile_learning = PersonProfileLearningService(profile_service)
    registry = PersonRegistry(
        repos.person, audit=repos.audit, profiles=profile_service
    )
    conversations = ConversationManager(repos.conversation, repos.utterance)
    policy = Policy()
    memory_service = build_memory_service(repos)
    memory_writer = MemoryWriter(repos, memory_service, logger=logger)

    # Optional hybrid vector retrieval. When no embedding provider is wired in
    # (the default), retrieval stays keyword/identity-only and nothing here
    # loads a model or makes a network call. A provider is explicit opt-in.
    embedding_service: Optional[EmbeddingService] = None
    hybrid_retriever = None
    if embedding_provider is not None:
        embedding_service = EmbeddingService(
            EmbeddingRepository(db), embedding_provider, logger=logger
        )

        async def _load_visible(memory_id: str, access, _service=memory_service) -> dict | None:
            item = await _service.get_item(memory_id, access)
            if item is None:
                return None
            subjects = await _service.repo.get_subjects(memory_id)
            sources = await _service.repo.get_sources(memory_id)
            return SqliteMemoryStore._result(item, subjects, sources)

        hybrid_retriever = HybridRetriever(
            embedding_service, load_memory=_load_visible, logger=logger
        )

        async def _index_on_change(
            action: str, memory_id: str, version: int, status: str,
            _service=embedding_service,
        ) -> None:
            """Enqueue vector indexing after a memory write (automatic lifecycle).

            Only ACTIVE memories are worth embedding; a retract/delete is
            handled by retrieval (which requires ACTIVE) and by the embedding
            table's cascade, so it enqueues nothing. The idempotency key keeps
            repeated writes from creating duplicate work.
            """
            if status != "ACTIVE" or action == "deleted":
                return
            key = f"embed:{memory_id}:{_service.model_id}:{version}"
            await repos.job.enqueue(
                JobType.EMBEDDING,
                key,
                {"memory_id": memory_id, "memory_version": version},
            )

        memory_service.on_change = _index_on_change

    memory = SqliteMemoryStore(memory_service, hybrid=hybrid_retriever)
    agent_service = AgentService(
        repos=repos, registry=registry, audit=repos.audit, logger=logger
    )

    # Persistent emotion framework (V0.3). Local and deterministic — no model,
    # no network, no real TTS. It rides on the same SQLite file and is always
    # built so emotion works identically in mock and real-provider modes.
    emotion_service = EmotionService(
        EmotionRepository(db),
        audit=repos.audit,
        logger=logger,
        config=load_emotion_config(),
    )

    # V0.4 scene/personality is enabled by default for the real DeepSeek path,
    # and stays off for the plain mock path unless a caller opts in. The
    # per-turn app_settings flags can disable it later without a code change.
    if scene_enabled is None:
        scene_enabled = provider == "deepseek"
    if personality_generation_enabled is None:
        personality_generation_enabled = provider == "deepseek"
    style_repo = StyleExampleRepository(db)
    scene_manager = (
        SceneManager(
            repos=repos,
            registry=registry,
            require_external_ai=(provider == "deepseek"),
            logger=logger,
        )
        if scene_enabled
        else None
    )
    personality_compiler = (
        PersonalityCompiler() if personality_generation_enabled else None
    )

    context_builder: Optional[ContextBuilder] = None
    gateway: Optional[ForegroundPrivacyGateway] = None

    if provider == "deepseek":
        if (cognitive is not None or response is not None) and not allow_mixed_providers:
            raise ValueError(
                "provider='deepseek' requires both cognitive and response to come"
                " from DeepSeek; mixed providers need allow_mixed_providers=True"
            )
        if deepseek_client is None:
            config = deepseek_config or load_deepseek_config()
            deepseek_client = build_deepseek_client(config, logger=logger)

        gateway = ForegroundPrivacyGateway(registry, logger=logger)
        context_builder = ContextBuilder(
            repos=repos,
            registry=registry,
            privacy_gateway=gateway,
            agent_service=agent_service,
            profile_service=profile_service,
            logger=logger,
        )
        if cognitive_repair_enabled is None:
            cognitive_repair_enabled = os.environ.get(
                "DEEPSEEK_COGNITIVE_REPAIR_ENABLED", ""
            ).strip().lower() in ("1", "true", "yes", "on")
        cognitive = cognitive or DeepSeekCognitiveProvider(
            deepseek_client,
            privacy_gateway=gateway,
            logger=logger,
            repair_enabled=bool(cognitive_repair_enabled),
        )
        response = response or DeepSeekResponseProvider(
            deepseek_client, privacy_gateway=gateway, logger=logger
        )
    else:
        cognitive = cognitive or MockCognitiveProvider()
        response = response or MockResponseProvider()

    # V0.8 router mode. A real DeepSeek wiring always defaults to semantic
    # routing, even when a client was injected by a test: an injected client is
    # the model boundary, not evidence that keyword routing is wanted. Only an
    # explicit argument or AVA_ROUTER_MODE=legacy opts out. Mock stays legacy.
    resolved_mode = (router_mode or os.environ.get("AVA_ROUTER_MODE") or "").strip().lower()
    if resolved_mode and resolved_mode not in ("semantic", "legacy"):
        raise ValueError(
            f"unknown router mode {resolved_mode!r}; expected 'semantic' or 'legacy'"
        )
    if not resolved_mode:
        resolved_mode = "semantic" if provider == "deepseek" else "legacy"

    routing_builder = None
    routing_service = None
    tool_executor = None
    tool_registry = tool_registry
    web_search = web_search_provider
    # Research keeps an explicit injection, but otherwise uses its own
    # mock-capable wrapper: offline background research must not inherit the
    # Router's "real results only" provider.
    research_search_provider = web_search_provider
    if resolved_mode == "semantic":
        if router_provider is None:
            if provider != "deepseek" or deepseek_client is None:
                raise ValueError(
                    "semantic router mode requires a real DeepSeek client or an"
                    " explicit router_provider; refusing to fall back to keyword"
                    " routing"
                )
            router_provider = DeepSeekRouterProvider(deepseek_client, logger=logger)
        if tool_registry is None:
            tool_registry = build_default_tool_registry()
        if web_search is None:
            # Resolve provider/enabled/key at execution time. The handler owns
            # the ``web_search.enabled`` gate; this wrapper never substitutes
            # mock results for the Router (``allow_mock=False``), so a missing
            # credential is reported truthfully instead of faked.
            web_search = SettingsWebSearchProvider(
                repos.settings, allow_mock=False, logger=logger
            )
        if gateway is None:
            gateway = ForegroundPrivacyGateway(registry, logger=logger)
        routing_builder = RoutingContextBuilder(
            repos=repos,
            registry=registry,
            privacy_gateway=gateway,
            profile_service=profile_service,
            scene_manager=scene_manager,
            tool_registry=tool_registry,
            logger=logger,
        )
        tool_executor = ToolExecutor(tool_registry, logger=logger)
        routing_service = RoutingService(
            provider=router_provider,
            validator=RoutingValidator(),
            registry=tool_registry,
            executor=tool_executor,
            repository=repos.routing,
            logger=logger,
        )

    explicit_memory_operations = ExplicitMemoryOperationService(
        repos=repos,
        cognitive=cognitive,
        memory_writer=memory_writer,
        logger=logger,
    )
    data_control = DataControlService(
        repos=repos,
        memory_service=memory_service,
        profile_service=profile_service,
        logger=logger,
    )

    # V0.9.2: one research engine belongs to the runtime itself, and its
    # RESEARCH_TASK handler is registered with the normal worker. Curiosity
    # and the idle scheduler enqueue through TaskService; they never start a
    # second research scheduler. Search and synthesis each resolve their
    # provider at call time:
    #   - research search defaults to offline mock when network search is off;
    #   - real Brave search requires web_search.enabled plus a key;
    #   - synthesis uses mock unless idle.real_deepseek.enabled is on, in which
    #     case a missing key is a clear configuration error (never a silent
    #     mock reported as real research).
    from companion.idle.research import ResearchEngine
    from companion.idle.sources import ResearchSourceGuard
    from companion.providers.deepseek_idle import (
        DeepSeekIdleProvider,
        GatedDeepSeekIdleProvider,
    )

    research_search = research_search_provider or SettingsWebSearchProvider(
        repos.settings, allow_mock=True, logger=logger
    )
    if real_idle_provider:
        idle_client = deepseek_client
        if idle_client is None:
            idle_client = build_deepseek_client(
                deepseek_config or load_deepseek_config(), logger=logger
            )
        research_synth = DeepSeekIdleProvider(idle_client, gate=repos.settings)
    else:
        research_synth = GatedDeepSeekIdleProvider(repos.settings, logger=logger)
    research_engine = ResearchEngine(
        memory_service,
        repos.research,
        research_search,
        research_synth,
        settings=repos.settings,
        state=repos.research_state,
        source_guard=ResearchSourceGuard(
            memory_service, registry, logger=logger
        ),
        logger=logger,
    )

    # V0.9.7: background profile refinement. The engine is only wired when a
    # worker can process its durable jobs. It reuses the same DeepSeek client
    # and the same privacy gateway as the foreground path; it never runs
    # inside the reply path and never writes the database itself.
    profile_refinement_engine = None
    profile_review_scheduler = None
    if with_worker:
        if profile_refinement_provider is None:
            if provider == "deepseek" and deepseek_client is not None:
                profile_refinement_provider = DeepSeekProfileRefinementProvider(
                    deepseek_client, logger=logger
                )
            else:
                profile_refinement_provider = MockProfileRefinementProvider()
        review_gateway = gateway or ForegroundPrivacyGateway(
            registry, logger=logger
        )
        profile_refinement_engine = ProfileRefinementEngine(
            profile_service,
            repos.profile,
            profile_refinement_provider,
            ProfileReviewContextSource(
                repos, privacy_gateway=review_gateway, logger=logger
            ),
            logger=logger,
        )
        profile_review_scheduler = ProfileReviewScheduler(
            repos.profile,
            repos.job,
            persons=repos.person,
            delay_seconds=profile_review_delay_seconds,
            logger=logger,
        )

    worker = None
    if with_worker:
        handlers = {"MEMORY_CONSOLIDATION": memory_consolidation}
        if context_builder is not None:
            handlers[JobType.TURN_ANALYSIS.value] = build_turn_analysis_handler(
                cognitive=cognitive,
                context_builder=context_builder,
                memory=memory,
            )
        if embedding_service is not None:
            handlers["EMBEDDING"] = build_embedding_handler(embedding_service)
        explicit_task_handler = build_explicit_memory_task_handler(
            explicit_memory_operations
        )
        handlers[JobType.MEMORY_SAVE.value] = explicit_task_handler
        handlers[JobType.MEMORY_CORRECTION.value] = explicit_task_handler
        handlers[JobType.RESEARCH_TASK.value] = build_research_task_handler(
            research_engine
        )
        if profile_refinement_engine is not None:
            handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
                profile_refinement_engine, logger=logger
            )
        worker = JobWorker(repos, handlers, logger=logger)

    # V0.9: one entry point for foreground and background off-task work. The
    # worker remains the single processor; TaskService owns local checks
    # (supported type, permission, dedupe, autonomous budget) and the
    # foreground execution path.
    permissioned_types = {
        JobType.TURN_ANALYSIS.value,
        JobType.MEMORY_SAVE.value,
        JobType.MEMORY_CORRECTION.value,
        JobType.PROFILE_REVIEW.value,
    }

    def _task_event_id(request) -> Optional[str]:
        """Resolve the source event from trusted task metadata only.

        Explicit-memory payloads carry ``{"event": ..., "route": ...}`` rather
        than a top-level ``event_id``. ``source_event_id``/``source_ref`` are
        written by local code and are authoritative; the payload snapshot is
        only a fallback for rows queued by older builds. A consent snapshot in
        the payload is deliberately ignored.
        """
        event_id = getattr(request, "source_event_id", None) or getattr(
            request, "source_ref", None
        )
        if event_id:
            return str(event_id)
        payload = request.payload or {}
        event_id = payload.get("event_id") or payload.get("source_event_id")
        if not event_id and isinstance(payload.get("event"), dict):
            event_id = payload["event"].get("event_id")
        return str(event_id) if event_id else None

    async def task_permission(request) -> bool:
        """Re-read live consent at enqueue, claim and commit time.

        A job approved when it was queued must not keep running after the
        speaker withdraws processing or persistence consent. Explicit memory
        tasks also require the current learning grant, a real source event and
        a registered speaker: an unverifiable source is rejected rather than
        default-allowed. The ordinary extraction handler remains a completed
        no-op when only learning is withdrawn (nothing it writes depends on
        that grant).
        """
        if request.type not in permissioned_types:
            return True
        explicit = request.type in {
            JobType.MEMORY_SAVE.value,
            JobType.MEMORY_CORRECTION.value,
        }
        profile_review = request.type == JobType.PROFILE_REVIEW.value
        event_id = _task_event_id(request)
        if not event_id:
            # An explicit memory write always names its source event. A missing
            # reference is unverifiable, not a default grant.
            return not explicit and not profile_review
        event = await repos.utterance.get(event_id)
        if event is None or event.speaker_person_id is None:
            # Extraction of synthetic/legacy rows keeps the pre-V0.9 behaviour.
            return not explicit and not profile_review
        consent = await repos.person.get_consent(event.speaker_person_id)
        if consent is None:
            return False
        allowed = bool(consent.allow_processing and consent.allow_persistence)
        if explicit:
            allowed = allowed and bool(consent.allow_memory_learning)
        if profile_review:
            # A profile review uploads the person's context and existing
            # understanding to an external model, so it needs both the
            # learning grant and the external-AI grant, re-read live.
            allowed = (
                allowed
                and bool(consent.allow_memory_learning)
                and bool(consent.allow_external_ai)
            )
        return allowed

    async def task_claim_gate(request):
        """Defer research while disabled instead of claiming/cancelling it.

        The check reads the setting at claim time, so re-enabling resumes the
        existing rows and checkpoints. A withdrawal of permission is a
        different, terminal decision made by ``task_permission`` above.
        """
        if request.type == JobType.RESEARCH_TASK.value:
            if not await repos.settings.get_bool("research.enabled", False):
                return "RESEARCH_DISABLED"
        return None

    task_service = TaskService(
        repos,
        handlers=worker.handlers if worker is not None else {},
        worker=worker,
        logger=logger,
        permission_checker=task_permission,
        claim_gate=task_claim_gate,
    )

    # V0.6 runtime context. Timezone resolution is always on (it needs no
    # consent and no network). GeoIP is strictly opt-in: it requires
    # GEOIP_ENABLED and a locally licensed GeoLite2 City database; no
    # credentials are read, stored or committed here.
    geoip_enabled = os.environ.get("GEOIP_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    geo_service = GeoLocationService(
        db_path=os.environ.get("GEOIP_DB_PATH"),
        enabled=geoip_enabled,
        logger=logger,
    )
    runtime_snapshot_builder = RuntimeSnapshotBuilder(
        repos,
        timezone_service=TimeZoneService(),
        geo_service=geo_service,
        logger=logger,
    )
    metrics = LatencyMetrics()
    if fast_path_enabled is None:
        fast_path_enabled = os.environ.get(
            "AVA_FAST_PATH_ENABLED", "true"
        ).strip().lower() in ("1", "true", "yes", "on")
    if turn_analysis_delay_seconds is None:
        try:
            turn_analysis_delay_seconds = float(
                os.environ.get("AVA_TURN_ANALYSIS_DELAY_SECONDS", "60")
            )
        except ValueError:
            turn_analysis_delay_seconds = 60.0

    orchestrator = Orchestrator(
        repos=repos,
        registry=registry,
        conversations=conversations,
        policy=policy,
        cognitive=cognitive,
        response=response,
        memory=memory,
        jobs=worker,
        context_builder=context_builder,
        memory_writer=memory_writer,
        emotion_service=emotion_service,
        agent_service=agent_service,
        scene_manager=scene_manager,
        personality_compiler=personality_compiler,
        style_repo=style_repo,
        runtime_snapshot_builder=runtime_snapshot_builder,
        metrics=metrics,
        profile_learning=profile_learning,
        profile_review=profile_review_scheduler,
        fast_path_enabled=bool(fast_path_enabled),
        router_mode=resolved_mode,
        routing_context_builder=routing_builder,
        routing_service=routing_service,
        tool_executor=tool_executor,
        web_search=web_search,
        explicit_memory_operations=explicit_memory_operations,
        data_control=data_control,
        task_service=task_service,
        turn_analysis_delay_seconds=float(turn_analysis_delay_seconds),
        logger=logger,
    )

    # V0.9.8: one lightweight proactive check after a delivered normal reply.
    # It is wired only for the real external-AI path (the Judge is a model
    # call); the manager itself is provider-agnostic and can be injected in
    # tests. It never holds the conversation lock and never creates a second
    # agent or task queue.
    proactive_manager = None
    if provider == "deepseek":
        judge = proactive_judge
        if judge is None:
            judge = ProactiveJudge(
                DeepSeekProactiveJudgeProvider(deepseek_client, logger=logger),
                logger=logger,
            )
        proactive_manager = ProactiveManager(
            repos=repos,
            registry=registry,
            conversations=conversations,
            orchestrator=orchestrator,
            judge=judge,
            delay_seconds=proactive_delay_seconds,
            short_task_timeout=proactive_short_task_timeout,
            logger=logger,
        )

    return Runtime(
        db=db,
        repos=repos,
        registry=registry,
        conversations=conversations,
        policy=policy,
        cognitive=cognitive,
        response=response,
        memory=memory,
        memory_service=memory_service,
        agent_service=agent_service,
        worker=worker,
        orchestrator=orchestrator,
        provider=provider,
        context_builder=context_builder,
        privacy_gateway=gateway,
        embedding_service=embedding_service,
        emotion_service=emotion_service,
        hybrid_retriever=hybrid_retriever,
        scene_manager=scene_manager,
        style_repo=style_repo,
        personality_compiler=personality_compiler,
        runtime_snapshot_builder=runtime_snapshot_builder,
        geo_service=geo_service,
        metrics=metrics,
        profile_service=profile_service,
        router_mode=resolved_mode,
        router_provider=router_provider,
        routing_service=routing_service,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        web_search=web_search,
        explicit_memory_operations=explicit_memory_operations,
        data_control=data_control,
        task_service=task_service,
        research_engine=research_engine,
        profile_refinement_engine=profile_refinement_engine,
        proactive_manager=proactive_manager,
    )
