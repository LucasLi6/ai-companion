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
from dataclasses import dataclass
from typing import Optional

from companion.agent.service import AgentService
from companion.cognition.context_builder import ContextBuilder
from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.core.conversation import ConversationManager
from companion.core.orchestrator import Orchestrator
from companion.core.policy import Policy
from companion.jobs.handlers import memory_consolidation
from companion.jobs.worker import JobWorker
from companion.memory.consolidation import MemoryWriter
from companion.memory.service import MemoryService, build_memory_service
from companion.memory.store import SqliteMemoryStore
from companion.people.registry import PersonRegistry
from companion.providers.deepseek_client import (
    DeepSeekConfigError,
    build_deepseek_client,
    load_deepseek_config,
)
from companion.providers.deepseek_cognitive import DeepSeekCognitiveProvider
from companion.providers.deepseek_response import DeepSeekResponseProvider
from companion.providers.mock import MockCognitiveProvider, MockResponseProvider
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
    registry = PersonRegistry(repos.person, audit=repos.audit)
    conversations = ConversationManager(repos.conversation, repos.utterance)
    policy = Policy()
    memory_service = build_memory_service(repos)
    memory_writer = MemoryWriter(repos, memory_service, logger=logger)
    memory = SqliteMemoryStore(memory_service)
    agent_service = AgentService(
        repos=repos, registry=registry, audit=repos.audit, logger=logger
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
            logger=logger,
        )
        cognitive = cognitive or DeepSeekCognitiveProvider(
            deepseek_client, privacy_gateway=gateway, logger=logger
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
        worker = JobWorker(repos, handlers, logger=logger)

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
    )
