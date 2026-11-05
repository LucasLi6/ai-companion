"""V0.9.3: one real-or-mock idle provider for every learning stage.

Reflection, curiosity and research synthesis must agree on the provider in
use. The default is all-mock; an explicitly authorised real run is all-real;
and an unauthorised or unconfigured real provider fails loudly instead of
quietly falling back to mock while still being reported as real.
"""

from __future__ import annotations

import pytest

from companion.cli import _idle_provider_status
from companion.domain.enums import JobType, MemoryStatus
from companion.idle import build_idle_scheduler
from companion.idle.research import ResearchEngine
from companion.memory.models import AccessContext
from companion.providers.deepseek_client import DeepSeekConfigError
from companion.providers.deepseek_idle import (
    DeepSeekIdleProvider,
    GatedDeepSeekIdleProvider,
    IdleProviderNotAuthorised,
)
from companion.runtime import build_runtime
from companion.tasks.models import TaskPaused

ADMIN = AccessContext("admin", is_admin=True)


class FakeClient:
    def __init__(self, payload: str = "{}") -> None:
        self.payload = payload
        self.calls: list[tuple[list[dict], dict]] = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))

        class Response:
            content = self.payload

        return Response()


class CountingDeepSeek:
    name = "real-deepseek"

    def __init__(self, mode: str = "real") -> None:
        self._mode = mode
        self.synth_calls: list[tuple[str, list[dict]]] = []

    async def mode(self) -> str:
        return self._mode

    async def describe(self) -> tuple[str, str]:
        return self._mode, self.name

    async def synthesize(self, question: str, sources: list[dict]) -> str:
        self.synth_calls.append((question, list(sources)))
        return "[fake]"

    async def reflect(self, memories: list[dict]) -> list[dict]:
        return []

    async def ask_questions(self, memories: list[dict]) -> list[str]:
        return []


class CountingMockSearch:
    name = "mock-search"
    is_real_network = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def network_state(self) -> str:
        return "mock"

    async def describe(self) -> tuple[str, str]:
        return "mock", self.name

    async def search(self, query: str, limit: int = 5) -> list:
        from companion.providers.websearch import SearchResult

        self.calls.append(query)
        return [
            SearchResult(
                title="mock result",
                url="https://example.com/mock",
                snippet="mock snippet",
                provider="mock",
            )
        ]


async def test_default_runtime_uses_one_mock_provider_for_all_stages(rt):
    scheduler = build_idle_scheduler(rt)
    provider = rt.research_engine.deepseek

    assert scheduler.reflection.provider is provider
    assert scheduler.curiosity.provider is provider
    assert scheduler.research.deepseek is provider
    assert await provider.describe() == ("mock", "mock-deepseek")
    assert scheduler.reflection.source_guard is rt.research_engine.source_guard


async def test_authorised_real_runtime_uses_one_real_provider_for_all_stages(db):
    client = FakeClient('{"reflections": []}')
    rt = build_runtime(db, real_idle_provider=True, deepseek_client=client)
    await rt.repos.settings.set("idle.real_deepseek.enabled", "true")
    scheduler = build_idle_scheduler(rt)
    provider = rt.research_engine.deepseek

    assert isinstance(provider, DeepSeekIdleProvider)
    assert scheduler.reflection.provider is provider
    assert scheduler.curiosity.provider is provider
    assert scheduler.research.deepseek is provider
    assert await provider.describe() == ("real", "deepseek-idle")

    report = await rt.research_engine.provider_report()
    assert report["synthesis_mode"] == "real"
    assert report["synthesis_provider"] == "deepseek-idle"


async def test_missing_key_fails_every_real_stage_explicitly(db, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    gate = build_runtime(db).repos.settings
    await gate.set("idle.real_deepseek.enabled", "true")
    provider = GatedDeepSeekIdleProvider(gate)

    with pytest.raises(DeepSeekConfigError):
        await provider.reflect([{"memory_id": "mem_1", "content": "x"}])
    with pytest.raises(DeepSeekConfigError):
        await provider.ask_questions([{"memory_id": "mem_1", "content": "x"}])
    with pytest.raises(DeepSeekConfigError):
        await provider.synthesize("q", [])


async def test_real_adapter_gate_is_checked_on_every_call(rt):
    client = FakeClient("answer")
    provider = DeepSeekIdleProvider(client, gate=rt.repos.settings)
    memory = [{"memory_id": "mem_1", "kind": "PROJECT_STATE", "content": "x"}]

    with pytest.raises(IdleProviderNotAuthorised):
        await provider.reflect(memory)
    assert await provider.mode() == "unauthorised"
    assert client.calls == []

    await rt.repos.settings.set("idle.real_deepseek.enabled", "true")
    client.payload = '{"reflections": []}'
    assert await provider.reflect(memory) == []
    assert len(client.calls) == 1

    # Turning the switch off stops the next call; no cached real result and no
    # silent mock answer are used instead.
    await rt.repos.settings.set("idle.real_deepseek.enabled", "false")
    with pytest.raises(IdleProviderNotAuthorised):
        await provider.synthesize("q", [])
    assert len(client.calls) == 1
    assert await provider.mode() == "unauthorised"


async def test_idle_scheduler_reuses_the_runtime_research_executor(db):
    rt = build_runtime(db, real_idle_provider=True, deepseek_client=FakeClient())
    handler_before = rt.worker.handlers[JobType.RESEARCH_TASK.value]

    scheduler = build_idle_scheduler(rt)

    assert scheduler.research is rt.research_engine
    assert rt.worker.handlers[JobType.RESEARCH_TASK.value] is handler_before


async def test_cli_status_reads_the_live_providers(db):
    rt = build_runtime(db, real_idle_provider=True, deepseek_client=FakeClient())
    await rt.repos.settings.set("idle.real_deepseek.enabled", "true")

    lines = await _idle_provider_status(rt)

    assert "Idle Provider: deepseek-idle (real)" in lines
    assert "Reflection: real" in lines
    assert "Curiosity: real" in lines
    assert any("Research Search:" in line and "(mock)" in line for line in lines)
    assert "Research Synthesis: deepseek-idle (real)" in lines


async def test_cli_status_is_truthful_before_real_mode_is_enabled(rt):
    lines = await _idle_provider_status(rt)
    assert lines[0] == "Idle Provider: mock-deepseek (mock)"
    assert "Reflection: mock" in lines
    assert "Curiosity: mock" in lines


async def test_resumed_task_uses_current_config_but_keeps_old_evidence_labels(rt):
    """Mock search evidence is never relabelled real by a later config change."""
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "false")
    task_id = await rt.repos.research.create(
        "混合模式恢复", origin="user", source_kind="manual"
    )
    mock_search = CountingMockSearch()
    mock_deepseek = CountingDeepSeek("mock")
    engine = ResearchEngine(
        rt.memory_service,
        rt.repos.research,
        mock_search,
        mock_deepseek,
        settings=rt.repos.settings,
        state=rt.repos.research_state,
        source_guard=rt.research_engine.source_guard,
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
            owner_id="job:mixed",
            save_checkpoint=save_checkpoint,
            pause_check=pause_check,
        )
    assert paused.get("search_mode_used") == "mock"

    # The provider configuration changes while the task is parked.
    real_deepseek = CountingDeepSeek("real")
    engine.deepseek = real_deepseek
    await rt.repos.settings.set("web_search.enabled", "true")

    payloads: list[dict] = []

    async def commit(write, *, result=None):
        await write()
        payloads.append(dict(result or {}))

    result = await engine.run_resumable(
        task_id,
        checkpoint=dict(paused),
        owner_id="job:mixed",
        commit=commit,
    )

    assert result["status"] == "COMPLETED"
    assert len(real_deepseek.synth_calls) == 1  # current config is used
    assert len(mock_search.calls) == 1          # old search is not repeated
    payload = payloads[-1]
    assert payload["search_mode"] == "mock"     # old evidence keeps its label
    assert payload["search_provider"] == "mock-search"
    assert payload["synthesis_mode"] == "real"
    assert payload["real_research"] is False

    sources = await rt.memory_service.repo.get_sources(f"mem:{task_id}")
    assert sources and all(s["provider"] != "brave-web-search" for s in sources)


async def test_mock_results_are_never_marked_real_research(rt):
    await rt.repos.settings.set("research.enabled", "true")
    await rt.repos.settings.set("web_search.enabled", "false")
    task_id = await rt.repos.research.create(
        "离线研究", origin="user", source_kind="manual"
    )
    mock_search = CountingMockSearch()
    mock_deepseek = CountingDeepSeek("mock")
    engine = ResearchEngine(
        rt.memory_service,
        rt.repos.research,
        mock_search,
        mock_deepseek,
        settings=rt.repos.settings,
        state=rt.repos.research_state,
        source_guard=rt.research_engine.source_guard,
    )

    result = await engine.run_resumable(task_id)

    assert result["status"] == "COMPLETED"
    assert len(mock_deepseek.synth_calls) == 1
    report = await engine.provider_report()
    assert report["real_research"] is False
    memory = await rt.memory_service.get_item(f"mem:{task_id}", ADMIN)
    assert memory is not None
    assert memory.status == MemoryStatus.CANDIDATE
    assert memory.model_inferred is True
    assert memory.source_supported is False
