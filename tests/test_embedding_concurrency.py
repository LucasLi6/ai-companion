"""V0.3.1 embedding runtime: off-loop calls, stale guards, rollback.

The provider is deliberately slow and blocking so these tests fail if it is
ever called on the event loop, and concurrent writes are exercised through the
same single SQLite connection the runtime uses.
"""

from __future__ import annotations

import asyncio
import threading
import time

from companion.domain.enums import MemoryKind
from companion.jobs.embedding_handler import build_embedding_handler
from companion.memory.embeddings import (
    FakeEmbeddingProvider,
    content_hash,
    unpack_vector,
)
from companion.memory.embedding_repository import EmbeddingRepository
from companion.memory.embedding_service import EmbeddingService
from companion.memory.models import AccessContext
from companion.memory.vector_index import VectorIndex

ADMIN = AccessContext("admin", is_admin=True)


class _BlockingProvider(FakeEmbeddingProvider):
    """Deterministic fake that blocks its worker thread and records overlap."""

    def __init__(self, delay: float = 0.05):
        super().__init__()
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.threads: set[str] = set()
        self._lock = threading.Lock()

    def embed_documents(self, texts):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.threads.add(threading.current_thread().name)
        try:
            time.sleep(self.delay)
            return super().embed_documents(texts)
        finally:
            with self._lock:
                self.active -= 1


def _service(db, provider, **kwargs):
    return EmbeddingService(EmbeddingRepository(db), provider, **kwargs)


async def test_provider_runs_off_loop_and_is_serialised(rt):
    provider = _BlockingProvider()
    service = _service(rt.db, provider, max_concurrent=1)
    items = [
        await rt.memory_service.create(
            kind=MemoryKind.PERSONAL_FACT,
            content=f"并发记忆 {i}",
            access=ADMIN,
        )
        for i in range(3)
    ]

    ticks = 0
    done = False

    async def ticker():
        nonlocal ticks
        while not done:
            ticks += 1
            await asyncio.sleep(0.001)

    task = asyncio.create_task(ticker())
    try:
        results = await asyncio.gather(
            *[
                service.embed_memory(
                    item.memory_id,
                    kind="PERSONAL_FACT",
                    content=f"并发记忆 {i}",
                    memory_version=1,
                )
                for i, item in enumerate(items)
            ]
        )
    finally:
        done = True
        await task

    assert all(r["status"] == "READY" for r in results)
    # The loop kept running while the blocking provider slept in a thread.
    assert ticks > 5
    assert all(name != threading.main_thread().name for name in provider.threads)
    # A semaphore of 1 means no two embeddings overlapped.
    assert provider.max_active == 1


async def test_stale_job_never_overwrites_newer_vector(rt):
    provider = FakeEmbeddingProvider()
    service = _service(rt.db, provider)
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="v1", access=ADMIN
    )
    await service.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="v1", memory_version=1
    )
    await rt.memory_service.update(
        item.memory_id, access=ADMIN, expected_version=1, content="v2"
    )
    await service.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="v2", memory_version=2
    )

    stale = await service.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="v1", memory_version=1
    )
    assert stale["status"] == "STALE_SKIPPED"
    stored = await service.repo.get_embedding(item.memory_id, service.model_id)
    assert stored["memory_version"] == 2


async def test_old_failure_cannot_replace_newer_ready(rt):
    service = _service(rt.db, FakeEmbeddingProvider())
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="v1", access=ADMIN
    )
    await service.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="v1", memory_version=1
    )
    await rt.memory_service.update(
        item.memory_id, access=ADMIN, expected_version=1, content="v2"
    )
    await service.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="v2", memory_version=2
    )

    # A late failure from the v1 attempt must not demote the v2 READY vector.
    await service.repo.record_failure(
        item.memory_id,
        service.model_id,
        "late failure",
        memory_version=1,
        content_hash="whatever",
    )
    stored = await service.repo.get_embedding(item.memory_id, service.model_id)
    assert stored["embedding_status"] == "READY"
    assert stored["memory_version"] == 2


async def test_stale_vector_is_not_searchable(rt):
    service = _service(rt.db, FakeEmbeddingProvider())
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="旧内容", access=ADMIN
    )
    await service.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="旧内容", memory_version=1
    )
    await rt.memory_service.update(
        item.memory_id, access=ADMIN, expected_version=1, content="新内容"
    )
    # The v1 vector still exists on disk, but it no longer describes the
    # current ACTIVE memory, so it must be excluded from ranking.
    ready = await service.repo.list_ready(service.model_id)
    assert item.memory_id not in {r["memory_id"] for r in ready}
    hits = await VectorIndex(service.repo).search(
        service.model_id, [1.0] * service.provider.dimensions
    )
    assert item.memory_id not in {mid for mid, _ in hits}


async def test_handler_uses_current_row_not_stale_payload(rt):
    service = _service(rt.db, FakeEmbeddingProvider())
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="v1", access=ADMIN
    )
    await rt.memory_service.update(
        item.memory_id, access=ADMIN, expected_version=1, content="v2"
    )
    handler = build_embedding_handler(service)
    # Deliberately stale payload version: the handler must re-read the row.
    await handler(
        {"memory_id": item.memory_id, "memory_version": 1},
        rt.repos,
        rt.orchestrator.logger,
    )
    stored = await service.repo.get_embedding(item.memory_id, service.model_id)
    assert stored["memory_version"] == 2
    assert stored["content_hash"] == content_hash("PERSONAL_FACT", "v2")


async def test_write_failure_rolls_back_vector(rt):
    service = _service(rt.db, FakeEmbeddingProvider())
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="回滚", access=ADMIN
    )
    repo = service.repo
    original = repo.upsert_embedding

    async def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    repo.upsert_embedding = boom
    try:
        try:
            await service.embed_memory(
                item.memory_id, kind="PERSONAL_FACT", content="回滚", memory_version=1
            )
        except RuntimeError:
            pass
    finally:
        repo.upsert_embedding = original
    assert await repo.get_embedding(item.memory_id, service.model_id) is None


async def test_concurrent_embedding_same_memory_is_safe(rt):
    provider = _BlockingProvider(delay=0.02)
    service = _service(rt.db, provider, max_concurrent=1)
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="并发同一条", access=ADMIN
    )
    results = await asyncio.gather(
        *[
            service.embed_memory(
                item.memory_id,
                kind="PERSONAL_FACT",
                content="并发同一条",
                memory_version=1,
            )
            for _ in range(3)
        ]
    )
    assert {r["status"] for r in results} <= {"READY", "UNCHANGED"}
    stored = await service.repo.get_embedding(item.memory_id, service.model_id)
    assert stored["embedding_status"] == "READY"
    assert len(unpack_vector(stored["vector"])) == service.provider.dimensions
    assert provider.max_active == 1
