"""Idle cognitive & research engines (Phase D): mock providers, reflection,
curiosity, research execution, consolidation and the scheduler gate.
"""

from __future__ import annotations

from companion.domain.enums import MemoryKind, MemoryVisibility
from companion.idle import build_idle_scheduler
from companion.idle.research import ResearchEngine
from companion.memory.models import AccessContext
from companion.providers.deepseek import DeepSeekProvider
from companion.providers.websearch import (
    HttpWebSearchProvider,
    MockWebSearchProvider,
    make_web_search,
)

ADMIN = AccessContext("admin", is_admin=True)


async def _seed_active_memory(rt, kind=MemoryKind.PROJECT_STATE, content="机械设计进行中"):
    return await rt.memory_service.create(
        kind=kind, content=content, access=ADMIN, visibility=MemoryVisibility.PUBLIC
    )


# -- mock providers ------------------------------------------------------


async def test_mock_deepseek_reflect():
    provider = DeepSeekProvider()
    assert await provider.reflect([]) == []
    proposals = await provider.reflect(
        [{"content": "A", "kind": "PROJECT_STATE"}, {"content": "B", "kind": "PROJECT_STATE"}]
    )
    assert len(proposals) == 1
    assert proposals[0]["content"].startswith("[mock]")
    assert proposals[0]["conflicts_with"] is None


async def test_mock_deepseek_ask_questions():
    provider = DeepSeekProvider()
    memories = [
        {"kind": "EPISODE", "content": "过去的一次对话"},
        {"kind": "PROJECT_STATE", "content": "软件尚未完成"},
        {"kind": "WORLD_KNOWLEDGE", "content": "OTA 采用双 bank"},
    ]
    questions = await provider.ask_questions(memories)
    assert len(questions) == 2  # only PROJECT_STATE and WORLD_KNOWLEDGE qualify
    assert all(q.startswith("[mock]") for q in questions)


async def test_mock_deepseek_synthesize():
    provider = DeepSeekProvider()
    sources = [{"title": "Doc A"}, {"title": "Doc B"}]
    summary = await provider.synthesize("如何烧录？", sources)
    assert "如何烧录？" in summary
    assert "Doc A" in summary


async def test_mock_web_search_is_offline_and_deterministic():
    provider = MockWebSearchProvider()
    results = await provider.search("ota")
    assert len(results) == 3
    assert all(r.url.startswith("https://example.com/") for r in results)


async def test_real_web_search_disabled_by_default(rt):
    # Default settings => mock provider.
    provider = await make_web_search(rt.repos.settings)
    assert isinstance(provider, MockWebSearchProvider)

    # A real provider with enabled=False refuses to hit the network.
    real = HttpWebSearchProvider(enabled=False)
    try:
        await real.search("ota")
        assert False, "expected a refusal"
    except RuntimeError as exc:
        assert "disabled" in str(exc)


# -- engines -------------------------------------------------------------


async def test_reflection_engine_proposes_candidates(rt):
    await _seed_active_memory(rt)
    scheduler = build_idle_scheduler(rt)
    results = await scheduler.reflection.run()
    assert len(results) == 1
    assert results[0].status.value == "CANDIDATE"
    assert results[0].human_confirmed is False
    assert results[0].model_inferred is True


async def test_curiosity_engine_queues_questions(rt):
    await _seed_active_memory(rt, kind=MemoryKind.PROJECT_STATE)
    scheduler = build_idle_scheduler(rt)
    ids = await scheduler.curiosity.run()
    assert len(ids) == 1
    tasks = await rt.repos.research.list(status="PENDING")
    assert any(t["task_id"] in ids and t["source_kind"] == "curiosity" for t in tasks)


async def test_research_engine_completes_and_consolidates(rt):
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("OTA 为什么不稳定？", source_kind="manual")
    scheduler = build_idle_scheduler(rt)
    results = await scheduler.research.run_all(limit=1)

    assert len(results) == 1
    assert results[0]["status"] == "COMPLETED"

    task = await rt.repos.research.get(task_id)
    assert task["result"] and task["completed_at"]

    # Consolidation wrote a WORLD_KNOWLEDGE memory with web sources attached.
    found = await rt.memory_service.search(
        access=ADMIN, kinds=["WORLD_KNOWLEDGE"], query="OTA"
    )
    assert found, "no consolidated world knowledge written"
    view = await rt.memory_service.get_view(found[0]["memory_id"], ADMIN)
    assert len(view["sources"]) == 3
    assert view["item"]["model_inferred"] is True
    assert view["item"]["human_confirmed"] is False
    # Mock research must never publish trusted knowledge.
    assert view["item"]["status"] == "CANDIDATE"
    assert view["item"]["source_supported"] is False


async def test_research_engine_failure_requeues_then_fails(rt):
    task_id = await rt.repos.research.create("会失败的问题", source_kind="manual")

    class FailingSearch:
        name = "failing"

        async def search(self, query, limit=5):
            raise RuntimeError("boom")

    engine = ResearchEngine(
        rt.memory_service, rt.repos.research, FailingSearch(), DeepSeekProvider()
    )
    # A retryable failure requeues while attempts < max_attempts, then fails.
    first = await engine.run_one()
    assert first["status"] == "PENDING"
    assert first["attempts"] == 1

    second = await engine.run_one()
    assert second["status"] == "PENDING"
    assert second["attempts"] == 2

    third = await engine.run_one()
    assert third["task_id"] == task_id
    assert third["status"] == "FAILED"
    assert "boom" in (third["error"] or "")


async def test_idle_scheduler_respects_disabled(rt):
    await _seed_active_memory(rt)
    await rt.repos.settings.set("idle.enabled", "false")
    scheduler = build_idle_scheduler(rt)
    result = await scheduler.run_cycle()
    assert result["skipped"] is True
    assert result["reflections"] == 0


async def test_idle_scheduler_full_cycle(rt):
    await _seed_active_memory(rt, kind=MemoryKind.PROJECT_STATE)
    await rt.repos.settings.set("idle.enabled", "true")
    await rt.repos.settings.set("research.enabled", "true")
    scheduler = build_idle_scheduler(rt)
    result = await scheduler.run_cycle()

    assert result["skipped"] is False
    assert result["reflections"] == 1
    assert result["questions"] == 1
    # The curiosity-queued question is picked up by the research phase.
    assert result["research"] == 1
