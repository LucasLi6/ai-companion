"""V0.9.4: fixed DuckDuckGo provider and its Router integration.

The DuckDuckGo HTML endpoint serves an "anomaly" challenge (HTTP 202) to
bot-like clients; a browser ``User-Agent`` avoids it. The provider must:

- request with GET and a browser ``User-Agent``,
- decode ``//duckduckgo.com/l/?uddg=…`` redirect links to their real target,
- extract ``result__snippet`` text,
- treat the challenge page as a *retryable error*, never an empty success,
- report a genuine "no results" page as an empty lookup (honest, and distinct
  from SUCCESS).

Every request uses ``httpx.MockTransport``; no real network is touched.
"""

from __future__ import annotations

import json

import httpx
import pytest

from companion.domain.enums import EventSource, ReportStatus
from companion.domain.models import make_utterance_event
from companion.providers.websearch import HttpWebSearchProvider, WebSearchError
from companion.prompts.response import build_response_messages
from companion.routing.models import RawRouteDecision
from companion.runtime import build_runtime
from companion.storage.repositories import SettingsRepository

# Realistic DuckDuckGo HTML: the first result is a ``uddg`` redirect link, the
# second is a plain direct link. Snippets live in ``result__snippet``.
DDG_HTML = """\
<html><body>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fzh.wikipedia.org%2Fwiki%2F%E9%9D%92%E9%9C%89%E7%B4%A0&amp;rut=abc123">青霉素 - 维基百科</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fzh.wikipedia.org%2Fwiki%2F%E9%9D%92%E9%9C%89%E7%B4%A0&amp;rut=abc123">青霉素是一种<b>抗生素</b>，由弗莱明发现。</a>
</div>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://example.com/direct">直接链接标题</a>
  </h2>
  <a class="result__snippet" href="https://example.com/direct">这是一个直接链接的摘要。</a>
</div>
</body></html>
"""


def _ddg(html: str, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=html)

    return httpx.MockTransport(handler)


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


def _semantic_runtime(db, router, provider):
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
        web_search_provider=provider,
    )
    return rt, response


# -- A: direct DuckDuckGo search -----------------------------------------


@pytest.mark.asyncio
async def test_duckduckgo_decodes_uddg_and_extracts_snippets():
    provider = HttpWebSearchProvider(
        enabled=True, retries=0, transport=_ddg(DDG_HTML)
    )
    results = await provider.search("青霉素", limit=5)

    assert [r.url for r in results] == [
        "https://zh.wikipedia.org/wiki/青霉素",
        "https://example.com/direct",
    ]
    assert results[0].title == "青霉素 - 维基百科"
    assert "抗生素" in results[0].snippet
    assert results[1].title == "直接链接标题"
    assert results[1].snippet == "这是一个直接链接的摘要。"
    assert all(r.provider == "duckduckgo" for r in results)


@pytest.mark.asyncio
async def test_duckduckgo_uses_get_with_browser_user_agent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["ua"] = request.headers.get("User-Agent", "")
        return httpx.Response(200, text=DDG_HTML)

    provider = HttpWebSearchProvider(
        enabled=True, retries=0, transport=httpx.MockTransport(handler)
    )
    await provider.search("青霉素", limit=1)

    assert seen["method"] == "GET"
    assert "Mozilla" in seen["ua"]


@pytest.mark.asyncio
async def test_duckduckgo_anomaly_challenge_is_retryable_error_not_empty():
    provider = HttpWebSearchProvider(
        enabled=True,
        retries=0,
        transport=_ddg("<html>anomaly challenge bot puzzle</html>", status=202),
    )
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "duckduckgo_anomaly"
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_duckduckgo_genuine_no_results_is_an_empty_lookup():
    provider = HttpWebSearchProvider(
        enabled=True,
        retries=0,
        transport=_ddg("<html><body>no results</body></html>"),
    )
    assert await provider.search("zzzz-nothing") == []


@pytest.mark.asyncio
async def test_duckduckgo_disabled_refuses_to_search():
    provider = HttpWebSearchProvider(enabled=False, transport=_ddg(DDG_HTML))
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "web_search_disabled"


# -- B: full Ava dialogue search (Router -> DuckDuckGo -> Speaker) ---------


@pytest.mark.asyncio
async def test_duckduckgo_full_dialogue_search_reaches_speaker(db):
    settings = SettingsRepository(db)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text=DDG_HTML)

    provider = HttpWebSearchProvider(
        enabled=True, retries=0, transport=httpx.MockTransport(handler)
    )
    router = _FakeRouter(
        tools=[
            {
                "tool_name": "web.search",
                "arguments": {"query": "青霉素", "limit": 3},
                "purpose": "look up 青霉素",
                "required": True,
            }
        ]
    )
    rt, response = _semantic_runtime(db, router, provider)
    await settings.set("web_search.enabled", "true")
    await settings.set("web_search.provider", "duckduckgo")
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    event = make_utterance_event(
        conversation_id="conv",
        text="搜索一下青霉素是什么",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)

    assert report.status == ReportStatus.COMPLETED.value
    assert len(calls) == 1
    assert calls[0].method == "GET"

    context = response.calls[0]
    tool = context.tool_results[0]
    assert tool["tool_name"] == "web.search"
    assert tool["status"] == "SUCCESS"
    assert tool["untrusted"] is True
    assert tool["data"][0]["url"] == "https://zh.wikipedia.org/wiki/青霉素"
    assert "抗生素" in tool["data"][0]["snippet"]
    assert tool["source_refs"] == [
        "https://zh.wikipedia.org/wiki/青霉素",
        "https://example.com/direct",
    ]

    prompt = json.dumps(build_response_messages(context, None), ensure_ascii=False)
    assert "https://zh.wikipedia.org/wiki/青霉素" in prompt
    assert "UNTRUSTED_EXTERNAL" in prompt


# -- C: normal chat does not search --------------------------------------


@pytest.mark.asyncio
async def test_duckduckgo_normal_chat_makes_no_request(db):
    settings = SettingsRepository(db)
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("web.search must not run for casual chat")

    provider = HttpWebSearchProvider(
        enabled=True, retries=0, transport=httpx.MockTransport(exploding)
    )
    router = _FakeRouter(tools=[])
    rt, response = _semantic_runtime(db, router, provider)
    await settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    event = make_utterance_event(
        conversation_id="conv",
        text="你好呀，今天过得怎么样？",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)

    assert report.status == ReportStatus.COMPLETED.value
    assert calls == []
    assert response.calls[0].tool_results == []


# -- D: disabled search gives a clear reason, never a fake result ---------


@pytest.mark.asyncio
async def test_duckduckgo_disabled_reports_clear_reason_no_fake_results(db):
    settings = SettingsRepository(db)
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("network call while web_search.enabled=false")

    provider = HttpWebSearchProvider(
        enabled=True, retries=0, transport=httpx.MockTransport(exploding)
    )
    router = _FakeRouter(
        tools=[
            {
                "tool_name": "web.search",
                "arguments": {"query": "青霉素"},
                "purpose": "look up",
                "required": True,
            }
        ]
    )
    rt, response = _semantic_runtime(db, router, provider)
    # Intentionally do NOT set web_search.enabled=true.
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    event = make_utterance_event(
        conversation_id="conv",
        text="搜索一下青霉素",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)

    assert calls == []
    tool = response.calls[0].tool_results[0]
    assert tool["tool_name"] == "web.search"
    assert tool["status"] == "DISABLED"
    assert tool["error_code"] == "web_search_disabled"
    assert tool["data"] is None


# -- E: error recovery -----------------------------------------------------


@pytest.mark.asyncio
async def test_duckduckgo_timeout_is_retried_then_recovers():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, text=DDG_HTML)

    provider = HttpWebSearchProvider(
        enabled=True, retries=2, backoff_base=0, transport=httpx.MockTransport(handler)
    )
    results = await provider.search("青霉素", limit=2)

    assert results[0].url == "https://zh.wikipedia.org/wiki/青霉素"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_duckduckgo_anomaly_is_retried_then_recovers():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(202, text="anomaly challenge")
        return httpx.Response(200, text=DDG_HTML)

    provider = HttpWebSearchProvider(
        enabled=True, retries=2, backoff_base=0, transport=httpx.MockTransport(handler)
    )
    results = await provider.search("青霉素", limit=1)

    assert results[0].url == "https://zh.wikipedia.org/wiki/青霉素"
    assert len(calls) == 2
