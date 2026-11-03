"""Composition root: wire every component for the CLI and tests.

This is the single place that knows how the concrete pieces fit together. The
Orchestrator itself only depends on the provider/memory *protocols*, so V0.1
can swap the mock providers here without touching the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from companion.core.conversation import ConversationManager
from companion.core.orchestrator import Orchestrator
from companion.core.policy import Policy
from companion.jobs.handlers import memory_consolidation
from companion.jobs.worker import JobWorker
from companion.memory.repository import MemoryRepository
from companion.memory.service import MemoryService
from companion.memory.store import SqliteMemoryStore
from companion.people.registry import PersonRegistry
from companion.providers.mock import MockCognitiveProvider, MockResponseProvider
from companion.storage.database import Database
from companion.storage.repositories import Repositories, make_repositories


@dataclass
class Runtime:
    db: Database
    repos: Repositories
    registry: PersonRegistry
    conversations: ConversationManager
    policy: Policy
    cognitive: MockCognitiveProvider
    response: MockResponseProvider
    memory: SqliteMemoryStore
    memory_service: MemoryService
    worker: Optional[JobWorker]
    orchestrator: Orchestrator


def build_runtime(
    db: Database,
    *,
    with_worker: bool = True,
    cognitive=None,
    response=None,
    logger: Optional[logging.Logger] = None,
) -> Runtime:
    """Build a fully wired runtime over an already-connected database.

    ``cognitive``/``response`` let tests inject failing or custom providers.
    """
    repos = make_repositories(db)
    registry = PersonRegistry(repos.person, audit=repos.audit)
    conversations = ConversationManager(repos.conversation, repos.utterance)
    policy = Policy()
    memory_repo = MemoryRepository(db)
    memory_service = MemoryService(
        memory_repo,
        authorize_subject=registry.is_learning_authorized,
        audit=repos.audit,
    )
    memory = SqliteMemoryStore(memory_service)
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
        worker=worker,
        orchestrator=orchestrator,
    )
