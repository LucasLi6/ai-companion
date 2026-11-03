"""Phase C: durable research, learning and goal state.

Everything here runs offline: the mock search and mock DeepSeek providers are
used throughout, and the real idle adapter is exercised only through a fake
client behind its explicit gate.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import MemoryKind, MemoryVisibility
from companion.idle import build_idle_scheduler
from companion.idle.research import ResearchEngine
from companion.memory.models import AccessContext
from companion.providers.deepseek import DeepSeekProvider
from companion.providers.deepseek_idle import (
    DeepSeekIdleProvider,
    IdleProviderNotAuthorised,
)
from companion.providers.websearch import MockWebSearchProvider

ADMIN = AccessContext("admin", is_admin=True)


async def _seed(rt, content="机械设计进行中", kind=MemoryKind.PROJECT_STATE):
    return await rt.memory_service.create(
        kind=kind, content=content, access=ADMIN, visibility=MemoryVisibility.PUBLIC
    )


async def test_research_persists_projects_runs_sources_claims(rt):
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("OTA 为什么不稳定？", source_kind="manual")

    scheduler = build_idle_scheduler(rt)
    results = await scheduler.research.run_all(limit=1)
    assert [r["status"] for r in results] == ["COMPLETED"]

    task = await rt.repos.research.get(task_id)
    project_id = task["project_id"]
    run_id = task["run_id"]
    assert project_id and run_id

    project = await rt.repos.research_state.get_project(project_id)
    assert project["question"] == "OTA 为什么不稳定？"

    runs = await rt.repos.research_state.list_runs(project_id)
    assert len(runs) == 1
    assert runs[0]["status"] == "COMPLETED"
    assert runs[0]["provider"] == "mock-web-search+mock-deepseek"

    queries = await rt.repos.research_state.list_queries(run_id)
    assert len(queries) == 1 and queries[0]["result_count"] == 3

    sources = await rt.repos.research_state.list_sources()
    assert len(sources) == 3
    assert all(s["trust_level"] == "UNVERIFIED" for s in sources)
    assert all(s["retrieved_at"] for s in sources)

    for source in sources:
        excerpts = await rt.repos.research_state.list_excerpts(source["source_id"])
        assert len(excerpts) == 1
        assert 0 < excerpts[0]["char_count"] <= 500
        rules = await rt.repos.research_state.list_refresh_rules()
        assert any(r["source_id"] == source["source_id"] for r in rules)

    claims = await rt.repos.research_state.list_claims(project_id)
    assert len(claims) == 1
    # A synthesized summary is a proposal, never trusted knowledge.
    assert claims[0]["status"] == "PROPOSED"
    assert claims[0]["verification_level"] == "UNVERIFIED"
    evidence = await rt.repos.research_state.list_claim_evidence(claims[0]["claim_id"])
    assert len(evidence) == 3

    findings = await rt.repos.research_state.list_findings(project_id)
    assert len(findings) == 1 and findings[0]["status"] == "DRAFT"


async def test_research_retry_is_idempotent_and_marks_failed_run(rt):
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("重试问题", source_kind="manual")

    class FlakySearch(MockWebSearchProvider):
        name = "mock-web-search"

        def __init__(self):
            self.calls = 0

        async def search(self, query, limit=5):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return await super().search(query, limit)

    engine = ResearchEngine(
        rt.memory_service,
        rt.repos.research,
        FlakySearch(),
        DeepSeekProvider(),
        settings=rt.repos.settings,
    )
    first = await engine.run_one()
    assert first["status"] == "PENDING"

    task = await rt.repos.research.get(task_id)
    project_id = task["project_id"]
    runs = await rt.repos.research_state.list_runs(project_id)
    assert [r["status"] for r in runs] == ["FAILED"]
    assert runs[0]["error"] == "temporary failure"

    second = await engine.run_one()
    assert second["status"] == "COMPLETED"
    runs = await rt.repos.research_state.list_runs(project_id)
    assert [r["status"] for r in runs] == ["FAILED", "COMPLETED"]
    # Durable ids are deterministic, so the retry did not duplicate sources.
    sources = await rt.repos.research_state.list_sources()
    assert len(sources) == 3


async def test_idle_cycle_records_learning_run_and_queue(rt):
    await _seed(rt)
    await rt.repos.settings.set("idle.enabled", "true")
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("idle.research_limit", "2")

    scheduler = build_idle_scheduler(rt)
    result = await scheduler.run_cycle()
    assert result["skipped"] is False
    assert result["reflections"] == 1
    assert result["questions"] >= 1

    runs = await rt.repos.db.execute("SELECT * FROM learning_runs")
    rows = await runs.fetchall()
    assert len(rows) == 1
    assert rows[0]["kind"] == "idle_cycle"
    assert rows[0]["status"] == "COMPLETED"
    assert rows[0]["provider"] == "mock-deepseek"

    checkpoints = await rt.repos.learning.list_checkpoints(rows[0]["run_id"])
    assert [c["stage"] for c in checkpoints] == ["reflection", "curiosity", "research"]

    queue = await rt.repos.learning.list_curiosity("QUEUED")
    assert len(queue) >= 1
    reflections = await rt.repos.learning.list_reflections()
    assert len(reflections) == 1 and reflections[0]["status"] == "PROPOSED"


async def test_idle_cycle_failure_is_recorded(rt):
    await rt.repos.settings.set("idle.enabled", "true")

    class Broken:
        provider = type("P", (), {"name": "mock-deepseek"})()

        async def run(self, limit=50):
            raise RuntimeError("provider down")

    scheduler = build_idle_scheduler(rt)
    scheduler.reflection = Broken()
    with pytest.raises(RuntimeError):
        await scheduler.run_cycle()
    rows = await (await rt.repos.db.execute("SELECT * FROM learning_runs")).fetchall()
    assert len(rows) == 1 and rows[0]["status"] == "FAILED"
    assert rows[0]["error"] == "provider down"


async def test_goal_dependencies_and_checkpoints(rt):
    goals = rt.repos.goals
    a = await goals.create("设计底盘", next_action="画图")
    b = await goals.create("组装原型", next_action="等待底盘")
    await goals.add_dependency(b, a)

    ready = await goals.ready()
    assert [g["goal_id"] for g in ready] == [a]

    await goals.add_checkpoint(a, "IN_PROGRESS", summary="初稿完成")
    await goals.set_status(a, "DONE")
    ready = await goals.ready()
    assert [g["goal_id"] for g in ready] == [b]
    assert len(await goals.list_checkpoints(a)) == 1

    with pytest.raises(ValueError):
        await goals.add_dependency(a, b)
    with pytest.raises(ValueError):
        await goals.add_dependency(a, a)


class FakeClient:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))

        class Resp:
            content = self.payload

        return Resp()


async def test_real_idle_provider_requires_gate(rt):
    client = FakeClient('{"reflections": []}')
    provider = DeepSeekIdleProvider(client, gate=rt.repos.settings)
    with pytest.raises(IdleProviderNotAuthorised):
        await provider.reflect([{"content": "x"}])
    assert client.calls == []

    await rt.repos.settings.set("idle.real_deepseek.enabled", "true")
    assert await provider.reflect([{"content": "x"}]) == []


async def test_real_idle_provider_parses_and_labels(rt):
    await rt.repos.settings.set("idle.real_deepseek.enabled", "true")
    client = FakeClient('{"reflections": [{"content": "值得核对"}]}')
    provider = DeepSeekIdleProvider(client, gate=rt.repos.settings)
    out = await provider.reflect([{"kind": "PROJECT_STATE", "content": "机械设计"}])
    assert out == [
        {"content": "值得核对", "kind": "REFLECTION", "conflicts_with": None}
    ]
    assert client.calls and client.calls[0][1]["json_mode"] is True
