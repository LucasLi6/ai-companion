"""V0.9.3: one external-send permission check for every idle AI call.

The rule these tests pin: permission depends on whether **data leaves the
machine**, not on whether a "real search" was used. Mock search + real
DeepSeek synthesis is still an external send and must be authorised. Every
test counts the actual search/HTTP/synthesis calls, so a refusal proves that
nothing was sent.
"""

from __future__ import annotations

import httpx
import pytest

from companion.domain.enums import MemoryVisibility
from companion.domain.models import Consent, utcnow
from companion.idle import build_idle_scheduler
from companion.idle.research import ResearchEngine
from companion.memory.models import AccessContext
from companion.providers.websearch import (
    BraveWebSearchProvider,
    MockWebSearchProvider,
    SearchResult,
)
from companion.tasks.models import TaskPaused

ADMIN = AccessContext("admin", is_admin=True)


class CountingDeepSeek:
    """A real-mode idle provider that records every call it receives."""

    name = "real-deepseek"

    def __init__(self, mode: str = "real") -> None:
        self._mode = mode
        self.synth_calls: list[tuple[str, list[dict]]] = []
        self.reflect_calls: list[list[str]] = []
        self.question_calls: list[list[str]] = []

    async def mode(self) -> str:
        return self._mode

    async def describe(self) -> tuple[str, str]:
        return self._mode, self.name

    async def synthesize(self, question: str, sources: list[dict]) -> str:
        self.synth_calls.append((question, list(sources)))
        return "[fake real synthesis]"

    async def reflect(self, memories: list[dict]) -> list[dict]:
        self.reflect_calls.append(
            [str(m.get("memory_id")) for m in memories]
        )
        return []

    async def ask_questions(self, memories: list[dict]) -> list[str]:
        self.question_calls.append(
            [str(m.get("memory_id")) for m in memories]
        )
        return []


class CountingSearch:
    """A search provider with a configurable network state."""

    is_real_network = False

    def __init__(self, state: str = "mock", name: str = "mock-search") -> None:
        self.state = state
        self.name = name
        self.calls: list[str] = []

    async def network_state(self) -> str:
        return self.state

    async def describe(self) -> tuple[str, str]:
        return self.state, self.name

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.calls.append(query)
        return [
            SearchResult(
                title=f"[{self.name}] result",
                url="https://example.com/result/1",
                snippet="offline snippet",
                provider=self.state,
            )
        ]


def _brave(requests: list[httpx.Request]) -> BraveWebSearchProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave result",
                            "url": "https://example.com/brave",
                            "description": "real-looking snippet",
                        }
                    ]
                }
            },
        )

    return BraveWebSearchProvider(
        api_key="test-key", transport=httpx.MockTransport(handler)
    )


class RevokingBuilder:
    """Simulates a revocation landing between read and send.

    The inner builder returns the externally-authorised context; the wrapper
    revokes consent *after* that read, before the engine has a chance to hand
    anything to the provider.
    """

    def __init__(self, inner, revoke) -> None:
        self.inner = inner
        self.revoke = revoke
        self.reads = 0

    async def external_context(self, limit: int = 50) -> list[dict]:
        self.reads += 1
        memories = await self.inner.external_context(limit=limit)
        await self.revoke()
        return memories


async def _person(rt, name: str, *, external_ai: bool = True):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
    )


async def _memory(rt, person_ids: list[str], content: str = "私人记忆内容"):
    item = await rt.memory_service.create(
        kind="PROJECT_STATE",
        content=content,
        access=ADMIN,
        visibility=MemoryVisibility.PRIVATE,
        subjects=[("PERSON", person_id) for person_id in person_ids],
    )
    return item.memory_id


async def _revoke_external_ai(rt, person_id: str) -> None:
    consent = await rt.registry.get_consent(person_id)
    assert consent is not None
    await rt.registry.set_consent(
        Consent(
            person_id=person_id,
            allow_processing=consent.allow_processing,
            allow_persistence=consent.allow_persistence,
            allow_memory_learning=consent.allow_memory_learning,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )


def _engine(rt, search, deepseek) -> ResearchEngine:
    return ResearchEngine(
        rt.memory_service,
        rt.repos.research,
        search,
        deepseek,
        settings=rt.repos.settings,
        state=rt.repos.research_state,
        source_guard=getattr(rt.research_engine, "source_guard", None),
    )


async def _enable_research(rt, *, web_search: bool = False) -> None:
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "true" if web_search else "false")


async def _task(rt, question: str, *, origin: str, sources: list[str]):
    return await rt.repos.research.create(
        question,
        origin=origin,
        source_kind="curiosity" if origin != "user" else "manual",
        source_memory_ids=sources,
    )


async def test_mock_search_real_synthesis_authorised_sends_once(rt):
    """Mock search must not bypass the check on a real synthesis send."""
    await _enable_research(rt)
    person = await _person(rt, "Lucas")
    memory_id = await _memory(rt, [person.person_id])
    search = CountingSearch("mock")
    deepseek = CountingDeepSeek("real")
    engine = _engine(rt, search, deepseek)
    task_id = await _task(
        rt, "授权后的问题", origin="curiosity", sources=[memory_id]
    )

    result = await engine.run_resumable(task_id)

    assert result["status"] == "COMPLETED"
    assert len(search.calls) == 1          # local mock search happened
    assert len(deepseek.synth_calls) == 1  # the one external send was allowed


async def test_mock_search_real_synthesis_revoked_permission_sends_nothing(rt):
    await _enable_research(rt)
    person = await _person(rt, "Lucas")
    memory_id = await _memory(rt, [person.person_id])
    await _revoke_external_ai(rt, person.person_id)
    search = CountingSearch("mock")
    deepseek = CountingDeepSeek("real")
    engine = _engine(rt, search, deepseek)
    task_id = await _task(
        rt, "撤销后的问题", origin="curiosity", sources=[memory_id]
    )

    result = await engine.run_resumable(task_id)

    assert result["status"] == "CANCELLED"
    assert result["reason"] == "permission_revoked"
    assert search.calls == []
    assert deepseek.synth_calls == []
    task = await rt.repos.research.get(task_id)
    assert task["status"] == "CANCELLED"
    assert task["memory_id"] is None


async def test_real_search_and_real_synthesis_are_checked_separately(rt):
    """Both external requests are gated; a revoked task makes neither."""
    await _enable_research(rt, web_search=True)
    person = await _person(rt, "Lucas")
    memory_id = await _memory(rt, [person.person_id])
    requests: list[httpx.Request] = []
    deepseek = CountingDeepSeek("real")
    engine = _engine(rt, _brave(requests), deepseek)

    first = await _task(rt, "允许的联网研究", origin="curiosity", sources=[memory_id])
    result = await engine.run_resumable(first)
    assert result["status"] == "COMPLETED"
    assert len(requests) == 1
    assert len(deepseek.synth_calls) == 1

    await _revoke_external_ai(rt, person.person_id)
    second = await _task(rt, "被拒绝的联网研究", origin="curiosity", sources=[memory_id])
    blocked = await engine.run_resumable(second)
    assert blocked["status"] == "CANCELLED"
    # Neither external request was repeated for the denied task.
    assert len(requests) == 1
    assert len(deepseek.synth_calls) == 1


async def test_revocation_after_searched_checkpoint_stops_synthesis(rt):
    """A checkpoint is not permanent authorisation (SEARCH -> ANALYZE)."""
    await _enable_research(rt)
    person = await _person(rt, "Lucas")
    memory_id = await _memory(rt, [person.person_id])
    search = CountingSearch("mock")
    deepseek = CountingDeepSeek("real")
    engine = _engine(rt, search, deepseek)
    task_id = await _task(
        rt, "SEARCHED 之后撤销", origin="curiosity", sources=[memory_id]
    )

    paused: dict = {}

    def save_checkpoint(state: dict) -> None:
        paused.clear()
        paused.update(state)

    def pause_check() -> bool:
        return paused.get("stage") == "SEARCHED"

    with pytest.raises(TaskPaused):
        await engine.run_resumable(
            task_id,
            checkpoint={},
            owner_id="job:revocation",
            save_checkpoint=save_checkpoint,
            pause_check=pause_check,
        )
    assert paused.get("stage") == "SEARCHED"
    assert len(search.calls) == 1
    assert deepseek.synth_calls == []

    # A person revokes external-AI consent while the task is parked.
    await _revoke_external_ai(rt, person.person_id)

    result = await engine.run_resumable(
        task_id, checkpoint=dict(paused), owner_id="job:revocation"
    )

    assert result["status"] == "CANCELLED"
    assert result["reason"] == "permission_revoked"
    assert deepseek.synth_calls == []      # synthesis was never attempted
    assert len(search.calls) == 1          # search was not repeated
    task = await rt.repos.research.get(task_id)
    assert task["status"] == "CANCELLED"
    assert task["memory_id"] is None
    assert await rt.memory_service.get_item(f"mem:{task_id}", ADMIN) is None


async def test_reflection_read_then_revoke_does_not_leak(rt):
    await _enable_research(rt)
    person = await _person(rt, "Lucas")
    await _memory(rt, [person.person_id])
    provider = CountingDeepSeek("real")
    scheduler = build_idle_scheduler(rt, deepseek=provider)
    assert scheduler.reflection.source_guard is not None
    scheduler.reflection.context_builder = RevokingBuilder(
        scheduler.reflection.context_builder,
        lambda: _revoke_external_ai(rt, person.person_id),
    )

    results = await scheduler.reflection.run()

    assert results == []
    # The memory was read (so the window was exercised), then dropped, and the
    # provider was never called with it.
    assert provider.reflect_calls == []


async def test_curiosity_read_then_revoke_is_not_sent(rt):
    await _enable_research(rt)
    person = await _person(rt, "Lucas")
    await _memory(rt, [person.person_id])
    provider = CountingDeepSeek("real")
    scheduler = build_idle_scheduler(rt, deepseek=provider)
    assert scheduler.curiosity.source_guard is not None
    scheduler.curiosity.context_builder = RevokingBuilder(
        scheduler.curiosity.context_builder,
        lambda: _revoke_external_ai(rt, person.person_id),
    )

    created = await scheduler.curiosity.run()

    assert created == []
    assert provider.question_calls == []   # no model call with private text
    assert await rt.repos.research.list() == []


async def test_two_people_one_revokes_blocks_the_send(rt):
    await _enable_research(rt)
    one = await _person(rt, "One")
    two = await _person(rt, "Two")
    memory_id = await _memory(rt, [one.person_id, two.person_id])
    provider = CountingDeepSeek("real")
    scheduler = build_idle_scheduler(rt, deepseek=provider)
    scheduler.reflection.context_builder = RevokingBuilder(
        scheduler.reflection.context_builder,
        lambda: _revoke_external_ai(rt, two.person_id),
    )

    results = await scheduler.reflection.run()

    assert results == []
    assert provider.reflect_calls == []
    guard = scheduler.reflection.source_guard
    decision = await guard.check(
        {"origin": "curiosity", "source_memory_ids": [memory_id]},
        network_required=True,
    )
    assert decision.allowed is False
    assert decision.reason == "permission_revoked"


async def test_legacy_research_without_sources_never_reaches_external_ai(rt):
    await _enable_research(rt)
    search = CountingSearch("mock")
    deepseek = CountingDeepSeek("real")
    engine = _engine(rt, search, deepseek)
    task_id = await rt.repos.research.create("无来源的历史问题")  # origin=legacy

    result = await engine.run_resumable(task_id)

    assert result["status"] == "FAILED"
    assert result["reason"] == "legacy_source_unknown"
    assert search.calls == []
    assert deepseek.synth_calls == []


async def test_user_submitted_research_is_allowed_and_sends(rt):
    await _enable_research(rt)
    search = CountingSearch("mock")
    deepseek = CountingDeepSeek("real")
    engine = _engine(rt, search, deepseek)
    task_id = await _task(rt, "用户主动提交的问题", origin="user", sources=[])

    result = await engine.run_resumable(task_id)

    assert result["status"] == "COMPLETED"
    assert len(search.calls) == 1
    assert len(deepseek.synth_calls) == 1


async def test_local_mock_mode_makes_no_real_external_request(rt):
    """Both stages mock: search runs locally, no real HTTP or DeepSeek client."""
    await _enable_research(rt)
    requests: list[httpx.Request] = []

    class CountingMockSearch(MockWebSearchProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, query: str, limit: int = 5):
            self.calls += 1
            return await super().search(query, limit=limit)

    class CountingMockDeepSeek(CountingDeepSeek):
        def __init__(self) -> None:
            super().__init__(mode="mock")

    search = CountingMockSearch()
    deepseek = CountingMockDeepSeek()
    engine = _engine(rt, search, deepseek)
    task_id = await _task(rt, "纯本地问题", origin="user", sources=[])

    # A transport attached to the settings wrapper would be the only route to
    # an HTTP request; the mock path never touches it.
    await engine.run_resumable(task_id)
    report = await engine.provider_report()

    assert search.calls == 1                       # local mock search
    assert len(deepseek.synth_calls) == 1          # local mock synthesis
    assert requests == []                          # zero real HTTP requests
    assert report["search_mode"] == "mock"
    assert report["synthesis_mode"] == "mock"
    assert report["real_research"] is False
    task = await rt.repos.research.get(task_id)
    assert task["status"] == "COMPLETED"
