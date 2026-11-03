"""Phase 4: research-job state machine, claim, crash recovery and idempotency.

These tests prove the reliability properties of :class:`ResearchEngine`:

- atomic claim: N competing workers run a task exactly once,
- crash recovery: an expired ``RUNNING`` lease is reclaimed and completed,
- idempotency: a task whose consolidated memory already exists is resumed
  without creating a duplicate,
- gating: ``research.enabled`` / ``web_search.enabled`` are checked before any
  network call, and a real (HTTP) provider refuses without authorisation,
- budget: the search limit is capped.
"""

from __future__ import annotations

import asyncio

from companion.domain.enums import MemoryKind, MemoryStatus, MemoryVisibility
from companion.idle import build_idle_scheduler
from companion.idle.research import ResearchEngine
from companion.memory.models import AccessContext
from companion.providers.deepseek import DeepSeekProvider
from companion.providers.websearch import HttpWebSearchProvider, SearchResult

ADMIN = AccessContext("admin", is_admin=True)


async def _world_knowledge(rt, task_id: str) -> list[dict]:
    found = await rt.memory_service.search(access=ADMIN, kinds=["WORLD_KNOWLEDGE"])
    return [m for m in found if m["memory_id"] == f"mem:{task_id}"]


# -- atomic claim under contention ---------------------------------------


async def test_two_workers_run_a_task_exactly_once(rt):
    task_id = await rt.repos.research.create("并发下只跑一次？", source_kind="manual")

    class SlowSearch:
        name = "slow-mock"

        async def search(self, query, limit=5):
            await asyncio.sleep(0.05)
            return [SearchResult(title="t", url="https://example.com/1", snippet="s")]

    a = ResearchEngine(
        rt.memory_service, rt.repos.research, SlowSearch(), DeepSeekProvider(), owner_id="a"
    )
    b = ResearchEngine(
        rt.memory_service, rt.repos.research, SlowSearch(), DeepSeekProvider(), owner_id="b"
    )
    results = await asyncio.gather(a.run_one(), b.run_one())

    completed = [r for r in results if r is not None and r["status"] == "COMPLETED"]
    assert len(completed) == 1, results
    assert completed[0]["task_id"] == task_id
    # Exactly one consolidated memory, never a duplicate.
    assert len(await _world_knowledge(rt, task_id)) == 1


# -- crash recovery ------------------------------------------------------


async def test_expired_lease_is_reclaimed_and_completed_once(rt):
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("崩溃后恢复？", source_kind="manual")
    # A worker claimed the task, then crashed without a live lease.
    assert await rt.repos.research.claim(task_id, "worker1", 300)
    await rt.repos.research.db.execute(
        "UPDATE research_tasks SET lease_expires_at = NULL WHERE task_id = ?",
        (task_id,),
    )

    scheduler = build_idle_scheduler(rt)
    results = await scheduler.research.run_all(limit=1)

    assert [r["task_id"] for r in results] == [task_id]
    assert results[0]["status"] == "COMPLETED"
    assert len(await _world_knowledge(rt, task_id)) == 1


async def test_crash_after_consolidate_resumes_without_duplicate(rt):
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("重复知识？", source_kind="manual")
    memory_id = f"mem:{task_id}"
    # The consolidation wrote a memory, but the crash happened before the task
    # was marked COMPLETED.
    await rt.memory_service.create(
        kind=MemoryKind.WORLD_KNOWLEDGE,
        content="已存在的知识",
        access=ADMIN,
        visibility=MemoryVisibility.PUBLIC,
        status=MemoryStatus.CANDIDATE,
        model_inferred=True,
        memory_id=memory_id,
    )
    await rt.repos.research.db.execute(
        "UPDATE research_tasks SET status = 'RUNNING', memory_id = ?,"
        " lease_expires_at = NULL WHERE task_id = ?",
        (memory_id, task_id),
    )

    scheduler = build_idle_scheduler(rt)
    results = await scheduler.research.run_all(limit=1)

    assert results[0]["status"] == "COMPLETED"
    assert results[0]["memory_id"] == memory_id
    matching = await _world_knowledge(rt, task_id)
    assert len(matching) == 1
    # The pre-existing memory is reused, its content untouched.
    assert matching[0]["content"] == "已存在的知识"


# -- gating & budget -----------------------------------------------------


async def test_research_disabled_runs_no_work_and_no_search(rt):
    await rt.repos.settings.set("research.enabled", "false")
    calls: list[str] = []

    class CountingSearch:
        name = "counting"

        async def search(self, query, limit=5):
            calls.append(query)
            return []

    engine = ResearchEngine(
        rt.memory_service,
        rt.repos.research,
        CountingSearch(),
        DeepSeekProvider(),
        settings=rt.repos.settings,
    )
    assert await engine.run_all(limit=5) == []
    assert calls == []


async def test_real_web_search_refuses_without_authorisation(rt):
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "false")
    task_id = await rt.repos.research.create("联网问题", source_kind="manual")
    # An explicitly disabled real provider must be refused before it is called.
    engine = ResearchEngine(
        rt.memory_service,
        rt.repos.research,
        HttpWebSearchProvider(enabled=False),
        DeepSeekProvider(),
        settings=rt.repos.settings,
    )
    for _ in range(3):
        await engine.run_one()
    task = await rt.repos.research.get(task_id)
    assert task["status"] == "FAILED"
    # The pre-network authorisation check fired (not the provider's own
    # "disabled" message), so ``search`` was never reached.
    assert "not authorised" in (task["error"] or "")


async def test_search_limit_is_capped(rt):
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("预算", source_kind="manual")
    seen: dict[str, int] = {}

    class RecordingSearch:
        name = "recording"

        async def search(self, query, limit=5):
            seen["limit"] = limit
            return [SearchResult(title="t", url="https://example.com/1", snippet="s")]

    engine = ResearchEngine(
        rt.memory_service,
        rt.repos.research,
        RecordingSearch(),
        DeepSeekProvider(),
        settings=rt.repos.settings,
        search_limit=7,
    )
    await engine.run_one()
    assert seen["limit"] == 7
    assert (await rt.repos.research.get(task_id))["status"] == "COMPLETED"
