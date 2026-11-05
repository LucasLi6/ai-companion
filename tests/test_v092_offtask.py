"""V0.9.2: off-task reliability regressions.

These exercise the real Runtime -> TaskService -> SQLite job -> JobWorker ->
ResearchEngine chain. They cover dedupe, disable/resume, provenance privacy,
cancellation/revocation, checkpoint resume after a search, and the CLI's
obligation to execute through the durable worker.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from companion.cli import main
from companion.domain.enums import (
    JobStatus,
    MemoryKind,
    MemoryVisibility,
)
from companion.domain.models import Consent, utcnow
from companion.idle import build_idle_scheduler
from companion.idle.research import ResearchEngine
from companion.idle.scheduling import (
    enqueue_existing_research,
    schedule_research,
)
from companion.idle.sources import ResearchSourceGuard
from companion.memory.models import AccessContext
from companion.providers.deepseek import DeepSeekProvider
from companion.providers.websearch import BraveWebSearchProvider, SearchResult
from companion.runtime import build_runtime
from companion.storage.database import Database
from companion.storage.repositories import ResearchCapacityError

ADMIN = AccessContext("admin", is_admin=True)


def _consent(person_id: str, *, external_ai: bool) -> Consent:
    return Consent(
        person_id=person_id,
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
        updated_at=utcnow(),
    )


async def _memory_count(rt, task_id: str) -> int:
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS n FROM memory_items WHERE memory_id = ?",
        (f"mem:{task_id}",),
    )
    return int((await cur.fetchone())["n"])


def _brave_results(question: str = "x", *, calls: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Real source",
                            "url": "https://real.example/article",
                            "description": "A real Brave snippet.",
                        }
                    ]
                }
            },
        )

    return httpx.MockTransport(handler)


def _exploding_transport(calls: list) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("the network must not be reached")

    return httpx.MockTransport(handler)


# -- dedupe & capacity ----------------------------------------------------


@pytest.mark.asyncio
async def test_curiosity_question_is_not_scheduled_twice(rt):
    await rt.memory_service.create(
        kind=MemoryKind.PROJECT_STATE,
        content="机械设计进行中",
        access=ADMIN,
        visibility=MemoryVisibility.PUBLIC,
    )
    scheduler = build_idle_scheduler(rt)

    first = await scheduler.curiosity.run()
    second = await scheduler.curiosity.run()

    assert len(first) == 1
    assert second == []
    rows = await rt.repos.research.list()
    assert len(rows) == 1
    assert rows[0]["task_id"] == first[0]
    assert json.loads(rows[0]["source_memory_ids"])
    job = await rt.repos.job.get_by_key(f"research:{first[0]}")
    assert job is not None
    assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_dedupe_normalizes_question_punctuation_and_source_order(rt):
    first, created, reason = await rt.repos.research.create_unique(
        "  OTA 为什么 不稳定？？",
        origin="curiosity",
        source_memory_ids=["mem_b", "mem_a"],
    )
    assert created is True and reason == "created"

    duplicate, created, reason = await rt.repos.research.create_unique(
        "ota 为什么 不稳定",
        origin="curiosity",
        source_memory_ids=["mem_a", "mem_b"],
    )
    assert duplicate == first
    assert created is False
    assert reason == "active_duplicate"

    # A different source scope is a different need.
    other, created, reason = await rt.repos.research.create_unique(
        "ota 为什么 不稳定",
        origin="curiosity",
        source_memory_ids=["mem_c"],
    )
    assert created is True
    assert other != first


@pytest.mark.asyncio
async def test_capacity_counts_unfinished_research_not_jobs(rt):
    await rt.repos.research.create_unique(
        "第一个未完成的问题",
        origin="curiosity",
        source_memory_ids=["mem_1"],
        max_active=1,
    )
    # No job was ever created for that row, yet it still occupies the slot.
    with pytest.raises(ResearchCapacityError) as exc:
        await rt.repos.research.create_unique(
            "第二个问题",
            origin="curiosity",
            source_memory_ids=["mem_2"],
            max_active=1,
        )
    assert exc.value.active == 1
    assert exc.value.limit == 1

    report = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "第二个问题",
        origin="curiosity",
        source_memory_ids=["mem_2"],
        max_active=1,
    )
    assert report["created"] is False
    assert report["reason"] == "AUTONOMOUS_RESEARCH_FULL"

    # User-submitted research is authoritative and not blocked by the
    # autonomous limit.
    user = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "用户主动提交的问题",
        origin="user",
        max_active=1,
    )
    assert user["created"] is True
    assert user["job_id"]


@pytest.mark.asyncio
async def test_completed_research_can_be_researched_after_the_recheck_window(rt):
    first, created, _ = await rt.repos.research.create_unique(
        "OTA 更新了吗",
        origin="curiosity",
        source_memory_ids=["mem_1"],
        recheck_seconds=3600,
    )
    assert created is True
    await rt.repos.research.complete(first, "done")

    duplicate, created, reason = await rt.repos.research.create_unique(
        "OTA 更新了吗",
        origin="curiosity",
        source_memory_ids=["mem_1"],
        recheck_seconds=3600,
    )
    assert duplicate == first
    assert created is False
    assert reason == "recently_completed"

    # After the knowledge-expiry window, the same need may be researched again.
    await rt.db.execute(
        "UPDATE research_tasks SET completed_at = ? WHERE task_id = ?",
        ("2000-01-01T00:00:00+00:00", first),
    )
    again, created, reason = await rt.repos.research.create_unique(
        "OTA 更新了吗",
        origin="curiosity",
        source_memory_ids=["mem_1"],
        recheck_seconds=3600,
    )
    assert created is True and reason == "created"
    assert again != first


@pytest.mark.asyncio
async def test_schedule_research_creates_the_record_and_job_together(rt):
    report = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "事务性研究",
        origin="curiosity",
        source_memory_ids=["mem_x"],
    )

    assert report["created"] is True
    assert report["job_id"]
    task = await rt.repos.research.get(report["task_id"])
    job = await rt.repos.job.get(report["job_id"])
    assert task["status"] == "PENDING"
    assert job.status == JobStatus.PENDING
    assert job.source_ref == report["task_id"]
    assert job.payload["research_task_id"] == report["task_id"]


# -- provenance / source privacy -----------------------------------------


@pytest.mark.asyncio
async def test_multi_person_source_requires_every_subject_external_ai(rt, register):
    alice = await register("Alice", external_ai=True)
    bob = await register("Bob", external_ai=False)
    memory = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="Alice 与 Bob 的私人安排",
        access=ADMIN,
        visibility=MemoryVisibility.PRIVATE,
        subjects=[("PERSON", alice.person_id), ("PERSON", bob.person_id)],
    )
    guard = ResearchSourceGuard(rt.memory_service, rt.registry)
    task = {
        "origin": "curiosity",
        "source_memory_ids": json.dumps([memory.memory_id]),
    }

    denied = await guard.check(task, network_required=True)
    assert denied.allowed is False
    assert denied.reason == "permission_revoked"

    await rt.registry.set_consent(_consent(bob.person_id, external_ai=True))
    allowed = await guard.check(task, network_required=True)
    assert allowed.allowed is True
    assert allowed.mode == "validated"


@pytest.mark.asyncio
async def test_private_unauthorized_memory_never_becomes_a_question(rt, register):
    person = await register("Alice", external_ai=False)
    await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="Alice 的私人计划",
        access=ADMIN,
        visibility=MemoryVisibility.PRIVATE,
        subjects=[("PERSON", person.person_id)],
    )
    scheduler = build_idle_scheduler(rt)

    assert await scheduler.curiosity.run() == []
    assert await rt.repos.research.list() == []


@pytest.mark.asyncio
async def test_legacy_task_without_sources_never_reaches_real_search(rt):
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "true")
    calls: list[httpx.Request] = []
    rt.research_engine.web_search = BraveWebSearchProvider(
        api_key="test-key", transport=_exploding_transport(calls)
    )
    task_id = await rt.repos.research.create(
        "可能包含私人内容的查询", source_kind="manual"
    )  # origin defaults to "legacy"
    await enqueue_existing_research(rt.task_service, task_id)

    await rt.worker.run_until_idle(max_iterations=10)

    task = await rt.repos.research.get(task_id)
    assert task["status"] == "FAILED"
    assert "legacy_source_unknown" in (task["error"] or "")
    assert calls == []
    assert await _memory_count(rt, task_id) == 0


@pytest.mark.asyncio
async def test_revoked_external_ai_cancels_before_the_network(rt, register):
    person = await register("Alice", external_ai=True)
    memory = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="Alice 的私人计划",
        access=ADMIN,
        visibility=MemoryVisibility.PRIVATE,
        subjects=[("PERSON", person.person_id)],
    )
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "true")
    report = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "Alice 的计划有新进展吗",
        origin="curiosity",
        source_memory_ids=[memory.memory_id],
    )
    task_id = report["task_id"]
    calls: list[httpx.Request] = []
    rt.research_engine.web_search = BraveWebSearchProvider(
        api_key="test-key", transport=_exploding_transport(calls)
    )
    await rt.registry.set_consent(_consent(person.person_id, external_ai=False))

    await rt.worker.run_until_idle(max_iterations=5)

    job = await rt.repos.job.get_by_key(f"research:{task_id}")
    assert job.status == JobStatus.CANCELLED
    assert (await rt.repos.research.get(task_id))["status"] == "CANCELLED"
    assert calls == []
    assert await _memory_count(rt, task_id) == 0


# -- disable / resume -----------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_research_is_deferred_and_reenabling_resumes(rt):
    report = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "禁用后恢复",
        origin="user",
    )
    task_id = report["task_id"]
    job_id = report["job_id"]

    await rt.worker.run_once()

    job = await rt.repos.job.get(job_id)
    assert job.status == JobStatus.PENDING
    assert job.attempts == 0  # deferred, never claimed or cancelled
    assert (await rt.repos.research.get(task_id))["status"] == "PENDING"
    assert await _memory_count(rt, task_id) == 0

    await rt.repos.settings.set("research.enabled", "true")
    await rt.worker.run_once()

    assert (await rt.repos.job.get(job_id)).status == JobStatus.COMPLETED
    assert (await rt.repos.research.get(task_id))["status"] == "COMPLETED"
    assert await _memory_count(rt, task_id) == 1


# -- checkpoint resume ----------------------------------------------------


@pytest.mark.asyncio
async def test_interrupted_after_search_resumes_without_a_second_search(rt):
    await rt.repos.settings.set("research.enabled", "true")
    report = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "搜索后中断",
        origin="user",
    )
    task_id = report["task_id"]
    job_id = report["job_id"]

    searches = 0
    original_search = rt.research_engine.web_search.search

    async def counted_search(query, limit=5):
        nonlocal searches
        searches += 1
        return await original_search(query, limit=limit)

    syntheses = 0
    original_synthesize = rt.research_engine.deepseek.synthesize

    async def flaky_synthesize(question, results):
        nonlocal syntheses
        syntheses += 1
        if syntheses == 1:
            raise RuntimeError("synthesis interrupted")
        return await original_synthesize(question, results)

    rt.research_engine.web_search.search = counted_search
    rt.research_engine.deepseek.synthesize = flaky_synthesize

    await rt.worker.run_once()
    assert (await rt.repos.job.get(job_id)).status == JobStatus.PENDING
    assert searches == 1
    assert await _memory_count(rt, task_id) == 0

    await rt.worker.run_once()
    assert (await rt.repos.job.get(job_id)).status == JobStatus.COMPLETED
    assert searches == 1  # the checkpointed search was not repeated
    assert syntheses == 2  # the failed synthesis was retried once
    assert await _memory_count(rt, task_id) == 1


# -- real search research -------------------------------------------------


@pytest.mark.asyncio
async def test_real_brave_sources_enter_candidate_memory_without_claiming_real_research(
    rt,
):
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "true")
    calls: list[httpx.Request] = []
    rt.research_engine.web_search = BraveWebSearchProvider(
        api_key="test-key", transport=_brave_results(calls=calls)
    )
    report = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "Brave 真实来源？",
        origin="user",
        source_kind="manual",
    )
    task_id = report["task_id"]
    job_id = report["job_id"]

    await rt.worker.run_once()

    task = await rt.repos.research.get(task_id)
    job = await rt.repos.job.get(job_id)
    assert task["status"] == "COMPLETED"
    assert job.status == JobStatus.COMPLETED
    assert calls and calls[0].url.params["q"] == "Brave 真实来源？"

    view = await rt.memory_service.get_view(f"mem:{task_id}", ADMIN)
    assert view["item"]["status"] == "CANDIDATE"
    assert view["item"]["model_inferred"] is True
    assert view["item"]["human_confirmed"] is False
    assert view["item"]["source_supported"] is False
    assert any(
        source["url"] == "https://real.example/article" for source in view["sources"]
    )
    assert any(
        "brave" in (source.get("provider") or "") for source in view["sources"]
    )

    # Real search + mock synthesis is not "real research".
    assert job.result["search_mode"] == "real"
    assert job.result["synthesis_mode"] == "mock"
    assert job.result["real_research"] is False
    assert (await rt.research_engine.provider_report())["real_research"] is False


@pytest.mark.asyncio
async def test_research_with_missing_brave_key_fails_truthfully(rt, monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "true")
    await rt.repos.settings.set("web_search.provider", "brave")
    report = await schedule_research(
        rt.db,
        rt.repos.research,
        rt.task_service,
        "缺少密钥时不得假装成功",
        origin="user",
    )
    task_id = report["task_id"]

    await rt.worker.run_until_idle(max_iterations=10)

    task = await rt.repos.research.get(task_id)
    assert task["status"] == "FAILED"
    assert "brave_api_key_missing" in (task["error"] or "")
    assert await _memory_count(rt, task_id) == 0
    assert (await rt.repos.job.get(report["job_id"])).status == JobStatus.FAILED


# -- CLI ------------------------------------------------------------------


def _seed_research(db_path: str, question: str) -> str:
    async def _run() -> str:
        database = await Database(db_path).connect()
        await database.initialize()
        rt = build_runtime(database)
        task_id = await rt.repos.research.create(question, source_kind="manual")
        await database.close()
        return task_id

    return asyncio.run(_run())


def _inspect(db_path: str, task_id: str) -> tuple:
    async def _run() -> tuple:
        database = await Database(db_path).connect()
        await database.initialize()
        rt = build_runtime(database)
        task = await rt.repos.research.get(task_id)
        job = await rt.repos.job.get_by_key(f"research:{task_id}")
        memory = await rt.memory_service.get_item(f"mem:{task_id}", ADMIN)
        await database.close()
        return task, job, memory

    return asyncio.run(_run())


def test_research_run_executes_through_the_durable_worker(tmp_path, capsys):
    db_path = str(tmp_path / "cli-worker.db")
    assert main(["--db", db_path, "init"]) == 0
    task_id = _seed_research(db_path, "CLI 必须走 JobWorker")
    capsys.readouterr()

    async def _enable() -> None:
        database = await Database(db_path).connect()
        await database.initialize()
        rt = build_runtime(database)
        await rt.repos.settings.set("research.enabled", "true")
        await database.close()

    asyncio.run(_enable())

    assert main(["--db", db_path, "research", "run", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert task_id in out
    assert "COMPLETED" in out

    task, job, memory = _inspect(db_path, task_id)
    assert task["status"] == "COMPLETED"
    assert job is not None  # the CLI created a durable job ...
    assert job.status == JobStatus.COMPLETED  # ... and the worker ran it
    assert memory is not None


def test_research_run_disabled_leaves_the_task_untouched(tmp_path, capsys):
    db_path = str(tmp_path / "cli-disabled.db")
    assert main(["--db", db_path, "init"]) == 0
    task_id = _seed_research(db_path, "禁用时不得执行")
    capsys.readouterr()

    assert main(["--db", db_path, "research", "run", "--limit", "1"]) == 0
    assert "disabled" in capsys.readouterr().out

    task, job, memory = _inspect(db_path, task_id)
    assert task["status"] == "PENDING"
    assert job is None
    assert memory is None
