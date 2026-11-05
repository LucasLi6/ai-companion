"""V0.9.2: Router-driven real web search.

The Router keeps its existing read-only ``web.search`` tool. These tests prove
that the tool now reaches a real provider when one is configured, that no
network call happens when search is unnecessary or disabled, and that the
executed (untrusted) snippets reach the Speaker prompt with their URLs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from companion.cognition.models import CognitiveContext
from companion.domain.enums import EventSource, ReportStatus
from companion.domain.models import make_utterance_event
from companion.providers.websearch import (
    BraveWebSearchProvider,
    MockWebSearchProvider,
    SettingsWebSearchProvider,
)
from companion.prompts.response import build_response_messages
from companion.routing.context_adapter import attach_semantic_tool_signals
from companion.routing.models import RawRouteDecision, ToolRequest, ToolStatus
from companion.runtime import build_runtime
from companion.storage.repositories import SettingsRepository
from companion.tools.executor import ToolExecutor
from companion.tools.models import ToolExecutionContext
from companion.tools.registry import build_default_tool_registry


class _FakeRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self, *, tools=None):
        self.calls = 0
        self._tools = tools or []

    async def route(self, context):
        self.calls += 1
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": "RESPOND",
                "reply_necessity": "REQUIRED",
                "addressee_assessment": "AGENT",
                "input_semantics": "COMMUNICATIVE",
                "turn_state": "CONTINUING",
                "tool_requests": self._tools,
                "needs_followup_planning": False,
                "reason": "live information may be required",
            }
        )

    async def replan(self, context, previous, results):
        return await self.route(context)


class _ExplodingCognitive:
    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call the cognitive provider")


class _RecordingResponse:
    def __init__(self):
        self.calls = []

    async def generate(self, event, analysis, context):
        self.calls.append(context)
        return "semantic reply"


def _semantic_runtime(db, router, web_search):
    response = _RecordingResponse()
    rt = build_runtime(
        db=db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=router,
        deepseek_client=object(),
        cognitive=_ExplodingCognitive(),
        response=response,
        allow_mixed_providers=True,
        web_search_provider=web_search,
    )
    return rt, response


async def _say(rt, person_id, text):
    event = make_utterance_event(
        conversation_id="conv",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )
    return event, await rt.orchestrator.process_utterance(event)


def _brave_transport(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "北京今日新闻",
                            "url": "https://example.com/live",
                            "description": "今天北京有一条新的交通消息。",
                        }
                    ]
                }
            },
        )

    return httpx.MockTransport(handler)


def _execution_context(rt, person_id, provider, *, allow=True, registry=True):
    return ToolExecutionContext(
        event=SimpleNamespace(event_id="evt-web", conversation_id="conv"),
        person_id=person_id,
        repos=rt.repos,
        memory=SimpleNamespace(),
        registry=rt.registry if registry else None,
        allow_external_ai=allow,
        web_search=provider,
    )


@pytest.mark.asyncio
async def test_router_live_info_search_reaches_speaker_as_untrusted_sources(
    db, monkeypatch
):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    settings = SettingsRepository(db)
    calls: list[httpx.Request] = []
    provider = SettingsWebSearchProvider(
        settings, allow_mock=False, transport=_brave_transport(calls)
    )
    router = _FakeRouter(
        tools=[
            {
                "tool_name": "web.search",
                "arguments": {"query": "今天 北京 新闻", "limit": 3},
                "purpose": "look up today's news",
                "required": True,
            }
        ]
    )
    rt, response = _semantic_runtime(db, router, provider)
    await settings.set("web_search.enabled", "true")
    await settings.set("web_search.provider", "brave")
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    _, report = await _say(rt, person.person_id, "今天北京有什么新闻？")

    assert report.status == ReportStatus.COMPLETED.value
    assert len(calls) == 1
    assert calls[0].url.params["q"] == "今天 北京 新闻"
    assert calls[0].url.params["count"] == "3"

    context = response.calls[0]
    tool = context.tool_results[0]
    assert tool["tool_name"] == "web.search"
    assert tool["status"] == "SUCCESS"
    assert tool["untrusted"] is True
    assert tool["data"][0]["url"] == "https://example.com/live"
    assert tool["source_refs"] == ["https://example.com/live"]

    prompt = json.dumps(build_response_messages(context, None), ensure_ascii=False)
    assert "https://example.com/live" in prompt
    assert "北京今日新闻" in prompt
    assert "UNTRUSTED_EXTERNAL" in prompt


@pytest.mark.asyncio
async def test_router_does_not_search_when_no_tool_is_requested(db, monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    settings = SettingsRepository(db)
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("web.search must not run for an unnecessary turn")

    provider = SettingsWebSearchProvider(
        settings, allow_mock=False, transport=httpx.MockTransport(exploding)
    )
    router = _FakeRouter(tools=[])
    rt, response = _semantic_runtime(db, router, provider)
    await settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_external_ai=True,
    )

    _, report = await _say(rt, person.person_id, "你好呀，今天过得怎么样？")

    assert report.status == ReportStatus.COMPLETED.value
    assert calls == []
    assert response.calls[0].tool_results == []


@pytest.mark.asyncio
async def test_router_disabled_web_search_makes_no_request_and_is_reported(db):
    settings = SettingsRepository(db)
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("network call while web_search.enabled=false")

    provider = SettingsWebSearchProvider(
        settings, allow_mock=False, transport=httpx.MockTransport(exploding)
    )
    router = _FakeRouter(
        tools=[
            {
                "tool_name": "web.search",
                "arguments": {"query": "news"},
                "purpose": "look something up",
                "required": True,
            }
        ]
    )
    rt, response = _semantic_runtime(db, router, provider)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_external_ai=True,
    )

    _, report = await _say(rt, person.person_id, "search the news")

    assert calls == []
    tool = response.calls[0].tool_results[0]
    assert tool["tool_name"] == "web.search"
    assert tool["status"] == "DISABLED"
    assert tool["error_code"] == "web_search_disabled"
    rows = await rt.repos.routing.list_tool_calls(report.event_id)
    assert rows[0]["error_code"] == "web_search_disabled"


@pytest.mark.asyncio
async def test_missing_brave_key_is_reported_as_config_error_not_mock(rt, monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    await rt.repos.settings.set("web_search.enabled", "true")
    await rt.repos.settings.set("web_search.provider", "brave")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    provider = SettingsWebSearchProvider(rt.repos.settings, allow_mock=False)
    executor = ToolExecutor(build_default_tool_registry())

    results = await executor.execute_many(
        [ToolRequest(tool_name="web.search", arguments={"query": "news"})],
        _execution_context(rt, person.person_id, provider),
    )

    assert results[0].status == ToolStatus.ERROR
    assert results[0].error_code == "web_search_config:brave_api_key_missing"
    assert results[0].data is None


@pytest.mark.asyncio
async def test_mock_provider_can_never_answer_a_router_search(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    executor = ToolExecutor(build_default_tool_registry())

    results = await executor.execute_many(
        [ToolRequest(tool_name="web.search", arguments={"query": "news"})],
        _execution_context(rt, person.person_id, MockWebSearchProvider()),
    )

    assert results[0].status == ToolStatus.DISABLED
    assert results[0].error_code == "web_search_disabled"
    assert results[0].data is None


@pytest.mark.asyncio
async def test_live_external_ai_revocation_blocks_the_network_call(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("permission was revoked before execution")

    provider = BraveWebSearchProvider(
        api_key="test-key", transport=httpx.MockTransport(exploding)
    )
    # The router's trusted snapshot says "allowed"; the live consent does not.
    context = _execution_context(rt, person.person_id, provider, allow=True)
    await rt.registry.set_consent(
        context_consent(person.person_id, external_ai=False)
    )
    executor = ToolExecutor(build_default_tool_registry())

    results = await executor.execute_many(
        [ToolRequest(tool_name="web.search", arguments={"query": "news"})],
        context,
    )

    assert results[0].status == ToolStatus.DISABLED
    assert results[0].error_code == "external_ai_permission_revoked"
    assert calls == []


def context_consent(person_id: str, *, external_ai: bool):
    from companion.domain.models import Consent, utcnow

    return Consent(
        person_id=person_id,
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
        updated_at=utcnow(),
    )


@pytest.mark.asyncio
async def test_provider_failure_reaches_the_speaker_as_an_error(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    provider = BraveWebSearchProvider(
        api_key="test-key",
        retries=0,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "down"})
        ),
    )
    executor = ToolExecutor(build_default_tool_registry())

    results = await executor.execute_many(
        [ToolRequest(tool_name="web.search", arguments={"query": "news"})],
        _execution_context(rt, person.person_id, provider),
    )

    assert results[0].status == ToolStatus.ERROR
    assert results[0].error_code == "web_search_failed:brave_server_error"
    assert results[0].data is None

    base = CognitiveContext(
        conversation_id="conv",
        current_event_id="evt-web",
        current_text="news?",
    )
    context = attach_semantic_tool_signals(
        base, SimpleNamespace(recent_turns=[]), results
    )
    payload = context.tool_results[0]
    assert payload["status"] == "ERROR"
    assert payload["data"] is None
    assert payload["error_code"] == "web_search_failed:brave_server_error"
    prompt = json.dumps(build_response_messages(context, None), ensure_ascii=False)
    assert "ERROR" in prompt
    assert "web.search" in prompt
