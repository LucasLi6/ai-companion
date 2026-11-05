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
from companion.jobs.worker import JobWorker
from companion.memory.consolidation import MemoryWriter
from companion.memory.embedding_repository import EmbeddingRepository
from companion.memory.embedding_service import EmbeddingService
from companion.memory.hybrid_retriever import HybridRetriever
from companion.memory.service import MemoryService, build_memory_service
from companion.memory.store import SqliteMemoryStore
from companion.people.registry import PersonRegistry
from companion.people.profile_learning import PersonProfileLearningService
from companion.people.profile_service import PersonProfileService
from companion.providers.deepseek_client import (
    DeepSeekConfigError,
    build_deepseek_client,
    load_deepseek_config,
)
from companion.providers.deepseek_cognitive import DeepSeekCognitiveProvider
from companion.providers.deepseek_response import DeepSeekResponseProvider
from companion.providers.mock import MockCognitiveProvider, MockResponseProvider
from companion.runtime_context.geolocation import GeoLocationService
from companion.runtime_context.snapshot import RuntimeSnapshotBuilder
from companion.runtime_context.timezone import TimeZoneService
from companion.storage.database import Database
from companion.storage.repositories import Repositories, make_repositories

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
    turn_analysis_delay_seconds: Optional[float] = None,
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
        worker = JobWorker(repos, handlers, logger=logger)

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
        fast_path_enabled=bool(fast_path_enabled),
        turn_analysis_delay_seconds=float(turn_analysis_delay_seconds),
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
    )
