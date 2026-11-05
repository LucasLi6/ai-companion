"""Tool registry/executor: closed allowlist, permissions, truthful statuses."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from companion.providers.websearch import HttpWebSearchProvider
from companion.routing.models import ToolRequest, ToolStatus
from companion.tools.executor import ToolExecutor
from companion.tools.models import ToolExecutionContext, ToolHandlerResult, ToolSpec
from companion.tools.registry import ToolRegistry, build_default_tool_registry


class FakeMemory:
    def __init__(self):
        self.calls = 0

    async def retrieve(self, query, person_id):
        self.calls += 1
        return [
            {
                "memory_id": "mem1",
                "kind": "PREFERENCE",
                "content": "likes tea",
                "updated_at": "2026-01-01",
            }
        ]

    async def browse(self, person_id, limit=3):
        return []


def context(memory=None, *, person_id="p1", web_search=None, allow=True):
    return ToolExecutionContext(
        event=SimpleNamespace(event_id="evt", conversation_id="conv"),
        person_id=person_id,
        repos=SimpleNamespace(),
        memory=memory or FakeMemory(),
        web_search=web_search,
        allow_external_ai=allow,
    )


@pytest.mark.asyncio
async def test_memory_tool_is_scoped_and_returns_source_refs():
    memory = FakeMemory()
    executor = ToolExecutor(build_default_tool_registry(include_web=False))
    results = await executor.execute_many(
        [ToolRequest(tool_name="memory.search", arguments={"query": "tea"})],
        context(memory),
    )
    assert results[0].status == ToolStatus.SUCCESS
    assert results[0].data[0]["memory_id"] == "mem1"
    assert results[0].source_refs == ["mem1"]
    assert memory.calls == 1


@pytest.mark.asyncio
async def test_tool_denied_without_a_speaker():
    executor = ToolExecutor(build_default_tool_registry(include_web=False))
    results = await executor.execute_many(
        [ToolRequest(tool_name="memory.search", arguments={"query": "tea"})],
        context(FakeMemory(), person_id=None),
    )
    assert results[0].status == ToolStatus.DENIED


@pytest.mark.asyncio
async def test_unknown_tool_and_invalid_arguments_are_never_success():
    executor = ToolExecutor(build_default_tool_registry(include_web=False))
    unknown, invalid = await executor.execute_many(
        [
            ToolRequest(tool_name="sql.execute", arguments={"sql": "drop"}),
            ToolRequest(tool_name="memory.search", arguments={"query": ""}),
        ],
        context(),
    )
    assert unknown.status == ToolStatus.DENIED
    assert unknown.error_code == "unknown_tool"
    assert invalid.status == ToolStatus.ERROR
    assert invalid.error_code == "invalid_arguments"


@pytest.mark.asyncio
async def test_web_is_disabled_without_a_real_provider_and_untrusted_with_one():
    executor = ToolExecutor(build_default_tool_registry())
    disabled = await executor.execute_many(
        [ToolRequest(tool_name="web.search", arguments={"query": "news"})],
        context(web_search=HttpWebSearchProvider(enabled=False)),
    )
    assert disabled[0].status == ToolStatus.DISABLED

    class Realish:
        async def search(self, query, limit=5):
            return [
                SimpleNamespace(
                    title="t", url="https://example.invalid/a", snippet="s"
                )
            ]

    allowed = await executor.execute_many(
        [ToolRequest(tool_name="web.search", arguments={"query": "news"})],
        context(web_search=Realish()),
    )
    assert allowed[0].status == ToolStatus.SUCCESS
    assert allowed[0].untrusted is True


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _SlowHandler:
    async def execute(self, arguments, context):
        await asyncio.sleep(0.2)
        return ToolHandlerResult(status=ToolStatus.SUCCESS, data="late")


class _BoomHandler:
    async def execute(self, arguments, context):
        raise RuntimeError("boom")


class _HugeHandler:
    async def execute(self, arguments, context):
        return ToolHandlerResult(status=ToolStatus.EMPTY, data="x" * 5000)


def _custom_registry():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="test.slow",
            description="slow",
            argument_model=_Args,
            permission="runtime.read",
            timeout_seconds=0.01,
        ),
        _SlowHandler(),
    )
    registry.register(
        ToolSpec(
            name="test.boom",
            description="boom",
            argument_model=_Args,
            permission="runtime.read",
        ),
        _BoomHandler(),
    )
    registry.register(
        ToolSpec(
            name="test.huge",
            description="huge",
            argument_model=_Args,
            permission="runtime.read",
            max_result_chars=100,
        ),
        _HugeHandler(),
    )
    return registry


@pytest.mark.asyncio
async def test_timeout_error_and_oversize_are_truthful_statuses():
    executor = ToolExecutor(_custom_registry())
    results = await executor.execute_many(
        [
            ToolRequest(tool_name="test.slow", arguments={}),
            ToolRequest(tool_name="test.boom", arguments={}),
            ToolRequest(tool_name="test.huge", arguments={}),
        ],
        context(),
    )
    assert results[0].status == ToolStatus.TIMEOUT
    assert results[1].status == ToolStatus.ERROR
    assert results[2].status == ToolStatus.ERROR


def test_registry_is_a_closed_allowlist():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register(
            ToolSpec(
                name="bad.write",
                description="x",
                argument_model=_Args,
                has_side_effects=True,
            ),
            _BoomHandler(),
        )
    spec = ToolSpec(name="ok.read", description="x", argument_model=_Args)
    registry.register(spec, _BoomHandler())
    with pytest.raises(ValueError):
        registry.register(spec, _BoomHandler())
