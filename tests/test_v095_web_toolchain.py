"""V0.9.5: general internet tool chain.

Covers the weather provider/handler, the SSRF-hardened page reader, the
search-url allowlist that connects them, and the Speaker-side projection and
budgeting that must never turn a failure into a guess.

Every request uses ``httpx.MockTransport`` and every DNS lookup is stubbed, so
the suite stays offline.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from companion.providers.weather import (
    FORECAST_ENDPOINT,
    GEOCODING_ENDPOINT,
    OpenMeteoProvider,
    WeatherError,
)
from companion.providers.webfetch import (
    HttpWebFetchProvider,
    WebFetchError,
    _is_public_ip,
)
from companion.prompts.response import (
    MAX_TOOL_CONTEXT_CHARS,
    _project_web_fetch_result,
    _project_web_result,
    _project_weather_result,
    _tool_payload,
)
from companion.routing.models import RawRouteDecision, ToolRequest, ToolStatus
from companion.runtime import build_runtime
from companion.tools.executor import ToolExecutor
from companion.tools.handlers.webfetch import MAX_PAGES_PER_TURN, WebFetchHandler
from companion.tools.handlers.weather import WeatherForecastHandler
from companion.tools.handlers.web import RESULT_BUDGET_CHARS, WebSearchHandler
from companion.tools.models import (
    ToolExecutionContext,
    ToolHandlerResult,
    ToolSpec,
)
from companion.tools.registry import ToolRegistry, build_default_tool_registry

GEO_PAYLOAD = {
    "results": [
        {
            "name": "Houston",
            "admin1": "Texas",
            "country": "United States",
            "latitude": 29.76328,
            "longitude": -95.36327,
            "timezone": "America/Chicago",
            "population": 2314157,
        },
        {
            "name": "Houston",
            "admin1": "Missouri",
            "country": "United States",
            "latitude": 37.32616,
            "longitude": -91.95599,
            "timezone": "America/Chicago",
            "population": 68000,
        },
    ]
}

FORECAST_PAYLOAD = {
    "timezone": "America/Chicago",
    "current": {
        "time": "2026-11-05T10:00",
        "temperature_2m": 25.5,
        "apparent_temperature": 26.1,
        "relative_humidity_2m": 60,
        "weather_code": 1,
        "wind_speed_10m": 12.3,
    },
    "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
    "daily": {
        "time": ["2026-11-05"],
        "temperature_2m_max": [29.0],
        "temperature_2m_min": [18.0],
        "precipitation_probability_max": [20],
    },
}


def _weather_transport(calls=None, *, geo=None, forecast=None, geo_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if "geocoding" in request.url.host:
            return httpx.Response(geo_status, json=geo or GEO_PAYLOAD)
        return httpx.Response(200, json=forecast or FORECAST_PAYLOAD)

    return httpx.MockTransport(handler)


def _context(rt, person_id, *, provider=None, extras=None, allow=True):
    return ToolExecutionContext(
        event=SimpleNamespace(event_id="evt-095", conversation_id="conv"),
        person_id=person_id,
        repos=rt.repos,
        memory=SimpleNamespace(),
        registry=rt.registry,
        allow_external_ai=allow,
        web_search=provider,
        extras=dict(extras or {}),
    )


# -- weather provider ------------------------------------------------------


@pytest.mark.asyncio
async def test_weather_provider_normalises_geocoding_and_forecast():
    calls: list[httpx.Request] = []
    provider = OpenMeteoProvider(
        transport=_weather_transport(calls), timeout=3.0
    )

    data = await provider.forecast("Houston")

    assert len(calls) == 2
    assert calls[0].url.host == "geocoding-api.open-meteo.com"
    assert calls[1].url.host == "api.open-meteo.com"
    assert calls[1].url.path == "/v1/forecast"
    assert data["location"] == "Houston, Texas, United States"
    assert data["timezone"] == "America/Chicago"
    assert data["observed_at"] == "2026-11-05T10:00"
    assert data["temperature"] == 25.5
    assert data["temperature_unit"] == "°C"
    assert data["weather"] == "大部晴朗"
    assert data["daily_max"] == 29.0
    assert data["daily_min"] == 18.0
    assert data["precipitation_probability"] == 20.0
    assert data["source"] == "Open-Meteo"
    assert data["alternatives"] == ["Houston, Missouri, United States"]


@pytest.mark.asyncio
async def test_chinese_city_name_is_geocoded_through_an_explicit_alias():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding" in request.url.host:
            seen.append(request.url.params.get("name"))
            return httpx.Response(200, json=GEO_PAYLOAD)
        return httpx.Response(200, json=FORECAST_PAYLOAD)

    provider = OpenMeteoProvider(transport=httpx.MockTransport(handler))
    data = await provider.forecast("休斯顿")

    # The index does not know the Chinese name; it knows "Houston".
    assert seen == ["Houston"]
    assert data["location"].startswith("Houston")


@pytest.mark.asyncio
async def test_a_weak_chinese_prefix_match_is_not_reported_as_the_city():
    geo = {
        "results": [
            {
                "name": "休斯顿威尔",
                "admin1": "肯塔基州",
                "country": "美国",
                "latitude": 37.0,
                "longitude": -86.0,
                "population": 399,
            }
        ]
    }
    provider = OpenMeteoProvider(transport=_weather_transport(geo=geo))

    with pytest.raises(WeatherError) as exc:
        await provider.forecast("休斯顿维尔")

    assert exc.value.code == "weather_location_not_found"


@pytest.mark.asyncio
async def test_the_largest_same_name_place_wins():
    geo = {
        "results": [
            {
                "name": "Houston",
                "admin1": "Missouri",
                "country": "United States",
                "latitude": 37.3,
                "longitude": -91.9,
                "population": 68000,
            },
            {
                "name": "Houston",
                "admin1": "Texas",
                "country": "United States",
                "latitude": 29.7,
                "longitude": -95.3,
                "population": 2314157,
                "timezone": "America/Chicago",
            },
        ]
    }
    provider = OpenMeteoProvider(transport=_weather_transport(geo=geo))

    data = await provider.forecast("Houston")

    assert data["location"] == "Houston, Texas, United States"
    assert "Houston, Missouri, United States" in data["alternatives"]


@pytest.mark.asyncio
async def test_weather_provider_unknown_place_is_not_found_not_weather():
    provider = OpenMeteoProvider(transport=_weather_transport(geo={"results": []}))

    with pytest.raises(WeatherError) as exc:
        await provider.forecast("zzzz-not-a-place")

    assert exc.value.code == "weather_location_not_found"


@pytest.mark.asyncio
async def test_weather_provider_invalid_json_is_a_truthful_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    provider = OpenMeteoProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(WeatherError) as exc:
        await provider.forecast("Houston")
    assert exc.value.code == "weather_invalid_response"


@pytest.mark.asyncio
async def test_weather_provider_timeout_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    provider = OpenMeteoProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(WeatherError) as exc:
        await provider.forecast("Houston")
    assert exc.value.code == "weather_timeout"
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_weather_handler_disabled_without_search_setting(rt):
    handler = WeatherForecastHandler(
        OpenMeteoProvider(transport=_weather_transport())
    )
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )

    outcome = await handler.execute(
        {"location": "Houston"}, _context(rt, person.person_id)
    )

    assert outcome.status == ToolStatus.DISABLED
    assert outcome.error_code == "web_search_disabled"


@pytest.mark.asyncio
async def test_weather_handler_denies_after_live_consent_withdrawal(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    handler = WeatherForecastHandler(
        OpenMeteoProvider(transport=_weather_transport())
    )
    from companion.domain.models import Consent, utcnow

    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    context = _context(rt, person.person_id, allow=True)

    outcome = await handler.execute({"location": "Houston"}, context)

    assert outcome.status == ToolStatus.DENIED
    assert outcome.error_code == "external_ai_permission_revoked"


@pytest.mark.asyncio
async def test_weather_handler_success_is_untrusted_and_empty_is_honest(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )

    ok = await WeatherForecastHandler(
        OpenMeteoProvider(transport=_weather_transport())
    ).execute({"location": "Houston"}, _context(rt, person.person_id))
    assert ok.status == ToolStatus.SUCCESS
    assert ok.untrusted is True
    assert ok.data["temperature"] == 25.5

    missing = await WeatherForecastHandler(
        OpenMeteoProvider(transport=_weather_transport(geo={"results": []}))
    ).execute({"location": "Nowhere"}, _context(rt, person.person_id))
    assert missing.status == ToolStatus.EMPTY
    assert missing.error_code == "weather_location_not_found"


@pytest.mark.asyncio
async def test_weather_is_registered_with_an_injectable_provider():
    fake = OpenMeteoProvider(transport=_weather_transport())
    registry = build_default_tool_registry(weather_provider=fake)

    spec = registry.get("weather.forecast").spec
    assert spec.permission == "web.read"
    assert spec.timeout_seconds == 15.0
    assert spec.max_result_chars == 3000
    assert registry.get("weather.forecast").handler.provider is fake
    assert registry.get("web.search").spec.timeout_seconds == 40.0


# -- webfetch: address checks ---------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.9",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
    ],
)
def test_non_public_addresses_are_refused(address):
    assert _is_public_ip(address) is False


def test_public_addresses_are_allowed():
    assert _is_public_ip("93.184.216.34") is True
    assert _is_public_ip("2606:2800:220:1:248:1893:25c8:1946") is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://localhost:8080/admin",
        "http://127.0.0.1/",
        "http://user:pass@example.com/",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
async def test_dangerous_urls_are_refused_before_any_request(url):
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("must not connect")

    provider = HttpWebFetchProvider(transport=httpx.MockTransport(exploding))
    with pytest.raises(WebFetchError):
        await provider.fetch(url)
    assert calls == []


@pytest.mark.asyncio
async def test_a_hostname_resolving_to_a_private_address_is_refused():
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("must not connect")

    provider = HttpWebFetchProvider(transport=httpx.MockTransport(exploding))

    async def fake_resolve(host, port):
        return ["10.0.0.7"]

    provider._resolve = fake_resolve
    with pytest.raises(WebFetchError) as exc:
        await provider.fetch("https://evil.example.com/page")
    assert exc.value.code == "web_fetch_private_address"
    assert calls == []


@pytest.mark.asyncio
async def test_redirect_targets_are_validated_too():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    provider = HttpWebFetchProvider(transport=httpx.MockTransport(handler))

    async def fake_resolve(host, port):
        return ["93.184.216.34"] if host == "example.com" else ["10.0.0.7"]

    provider._resolve = fake_resolve
    with pytest.raises(WebFetchError) as exc:
        await provider.fetch("https://example.com/start")
    assert exc.value.code == "web_fetch_private_address"
    assert len(calls) == 1


# -- webfetch: extraction --------------------------------------------------


PAGE_HTML = """\
<html><head>
<title>  今日新闻  </title>
<meta property="article:published_time" content="2026-11-05T08:00:00Z">
<script>window.evil = "ignore me";</script>
<style>.x { color: red }</style>
</head><body>
<h1>标题</h1>
<p>第一段内容，包含 <b>重点</b>。</p>
<script>alert("never")</script>
<p>第二段内容。</p>
</body></html>
"""


def _public_provider(transport, *, resolve=None):
    provider = HttpWebFetchProvider(transport=transport)
    addresses = resolve or ["93.184.216.34"]

    async def fake_resolve(host, port):
        return list(addresses)

    provider._resolve = fake_resolve
    return provider


@pytest.mark.asyncio
async def test_page_text_is_extracted_without_scripts_or_styles():
    provider = _public_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=PAGE_HTML,
                headers={"content-type": "text/html; charset=utf-8"},
            )
        )
    )

    page = await provider.fetch("https://example.com/news")

    assert page["title"] == "今日新闻"
    assert page["published_at"] == "2026-11-05T08:00:00Z"
    assert "第一段内容" in page["text"]
    assert "第二段内容" in page["text"]
    assert "alert" not in page["text"]
    assert "color: red" not in page["text"]
    assert page["url"] == "https://example.com/news"
    assert page["source"] == "example.com"


@pytest.mark.asyncio
async def test_non_html_content_is_refused():
    provider = _public_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}
            )
        )
    )
    with pytest.raises(WebFetchError) as exc:
        await provider.fetch("https://example.com/file.pdf")
    assert exc.value.code == "web_fetch_unsupported_content_type"


@pytest.mark.asyncio
async def test_oversized_response_is_cut_off_with_an_error():
    body = b"<html><body>" + b"x" * 5000 + b"</body></html>"
    provider = HttpWebFetchProvider(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=body, headers={"content-type": "text/html"}
            )
        ),
        max_bytes=2048,
    )

    async def fake_resolve(host, port):
        return ["93.184.216.34"]

    provider._resolve = fake_resolve
    with pytest.raises(WebFetchError) as exc:
        await provider.fetch("https://example.com/big")
    assert exc.value.code == "web_fetch_too_large"


# -- webfetch handler: the search-result allowlist ------------------------


@pytest.mark.asyncio
async def test_web_fetch_refuses_a_url_the_search_did_not_return(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    provider = _public_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, text=PAGE_HTML, headers={"content-type": "text/html"}
            )
        )
    )
    handler = WebFetchHandler(provider)
    context = _context(
        rt,
        person.person_id,
        extras={"web_search_urls": {"https://example.com/allowed"}},
    )

    outcome = await handler.execute(
        {"url": "https://example.com/invented"}, context
    )

    assert outcome.status == ToolStatus.DENIED
    assert outcome.error_code == "web_fetch_url_not_from_search"


@pytest.mark.asyncio
async def test_web_fetch_reads_a_url_from_this_turns_search(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    provider = _public_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, text=PAGE_HTML, headers={"content-type": "text/html"}
            )
        )
    )
    context = _context(
        rt,
        person.person_id,
        extras={"web_search_urls": {"https://example.com/news"}},
    )

    outcome = await WebFetchHandler(provider).execute(
        {"url": "https://example.com/news#section"}, context
    )

    assert outcome.status == ToolStatus.SUCCESS
    assert outcome.untrusted is True
    assert "第一段内容" in outcome.data["text"]
    assert context.extras["web_fetch_count"] == 1


@pytest.mark.asyncio
async def test_web_fetch_respects_the_per_turn_page_limit(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    provider = _public_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, text=PAGE_HTML, headers={"content-type": "text/html"}
            )
        )
    )
    context = _context(
        rt,
        person.person_id,
        extras={
            "web_search_urls": {"https://example.com/news"},
            "web_fetch_count": MAX_PAGES_PER_TURN,
        },
    )

    outcome = await WebFetchHandler(provider).execute(
        {"url": "https://example.com/news"}, context
    )

    assert outcome.status == ToolStatus.DENIED
    assert outcome.error_code == "web_fetch_turn_limit"


# -- executor: trusted search URLs become the allowlist -------------------


class _StaticSearchHandler:
    def __init__(self, outcome: ToolHandlerResult) -> None:
        self.outcome = outcome

    async def execute(self, arguments, context):
        return self.outcome


def _search_registry(outcome: ToolHandlerResult) -> ToolRegistry:
    from companion.tools.handlers.web import WebSearchArguments

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="web.search",
            description="test",
            argument_model=WebSearchArguments,
            permission="web.read",
        ),
        _StaticSearchHandler(outcome),
    )
    return registry


@pytest.mark.asyncio
async def test_successful_search_records_its_urls_for_web_fetch(rt):
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    executor = ToolExecutor(
        _search_registry(
            ToolHandlerResult(
                status=ToolStatus.SUCCESS,
                data=[{"title": "t", "url": "https://example.com/a", "snippet": "s"}],
                source_refs=["https://example.com/a#frag"],
                untrusted=True,
            )
        )
    )
    context = _context(rt, person.person_id, provider=object())
    request = ToolRequest(tool_name="web.search", arguments={"query": "x"})

    result = await executor.execute_one(
        request, context, tool_call_id="tc_x"
    )

    assert result.status == ToolStatus.SUCCESS
    allowed = context.extras["web_search_urls"]
    assert "https://example.com/a#frag" in allowed
    assert "https://example.com/a" in allowed


@pytest.mark.asyncio
async def test_failed_search_records_no_fetchable_urls(rt):
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    executor = ToolExecutor(
        _search_registry(
            ToolHandlerResult(
                status=ToolStatus.ERROR,
                error_code="web_search_failed:duckduckgo_anomaly",
            )
        )
    )
    context = _context(rt, person.person_id, provider=object())
    request = ToolRequest(tool_name="web.search", arguments={"query": "x"})

    await executor.execute_one(request, context, tool_call_id="tc_y")

    assert "web_search_urls" not in context.extras


# -- web handler: bounded results, honest counts --------------------------


class _ManyResultsProvider:
    name = "fake-search"

    async def search(self, query, limit=5):
        return [
            SimpleNamespace(
                title=f"标题 {index} " + "t" * 400,
                url=f"https://example.com/result-{index}",
                snippet="片段 " + "s" * 900,
            )
            for index in range(limit)
        ]


@pytest.mark.asyncio
async def test_web_handler_budgets_results_and_reports_counts(rt):
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )
    context = _context(rt, person.person_id, provider=_ManyResultsProvider())

    outcome = await WebSearchHandler().execute(
        {"query": "long", "limit": 5}, context
    )

    assert outcome.status == ToolStatus.SUCCESS
    assert len(json.dumps(outcome.data, ensure_ascii=False)) <= RESULT_BUDGET_CHARS
    assert outcome.meta["returned_count"] == len(outcome.data)
    assert outcome.meta["returned_count"] + outcome.meta["truncated_count"] == 5
    assert all(
        len(item["title"]) <= 150 and len(item["snippet"]) <= 400
        for item in outcome.data
    )
    assert all(item["url"].startswith("https://example.com/result-") for item in outcome.data)


@pytest.mark.asyncio
async def test_web_handler_empty_stays_empty_and_untrusted(rt):
    class _EmptyProvider:
        name = "fake-search"

        async def search(self, query, limit=5):
            return []

    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_external_ai=True
    )

    outcome = await WebSearchHandler().execute(
        {"query": "nothing"}, _context(rt, person.person_id, provider=_EmptyProvider())
    )

    assert outcome.status == ToolStatus.EMPTY
    assert outcome.data == []
    assert outcome.untrusted is True


# -- Speaker-side projection ---------------------------------------------


def test_web_results_keep_whole_urls_and_drop_duplicates():
    data = [
        {"title": "a" * 300, "url": "https://example.com/1", "snippet": "s" * 900},
        {"title": "dup", "url": "https://example.com/1", "snippet": "dup"},
        {"title": "bad", "url": "ftp://example.com/x", "snippet": "bad"},
        {"title": "b", "url": "https://example.com/2", "snippet": "ok"},
    ]
    projected = _project_web_result(data)

    assert [item["url"] for item in projected] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert len(projected[0]["title"]) == 120
    assert len(projected[0]["snippet"]) == 250


def test_weather_projection_keeps_the_numbers_and_clips_text():
    projected = _project_weather_result(
        {
            "location": "Houston" * 100,
            "temperature": 25.5,
            "temperature_unit": "°C",
            "daily_max": 29.0,
            "daily_min": 18.0,
            "precipitation_probability": 20.0,
            "weather": "大部晴朗",
            "observed_at": "2026-11-05T10:00",
            "source": "Open-Meteo",
            "junk": "x" * 5000,
        }
    )
    assert len(projected["location"]) == 150
    assert projected["temperature"] == 25.5
    assert projected["daily_max"] == 29.0
    assert "junk" not in projected


def test_web_fetch_projection_keeps_title_url_and_excerpt():
    projected = _project_web_fetch_result(
        {
            "url": "https://example.com/a",
            "title": "t" * 300,
            "text": "正文" * 5000,
            "published_at": "2026-11-05T08:00:00Z",
            "source": "example.com",
        }
    )
    assert projected["url"] == "https://example.com/a"
    assert len(projected["title"]) == 120
    assert len(projected["text"]) == 1200
    assert projected["published_at"] == "2026-11-05T08:00:00Z"


def test_failed_tool_status_and_error_survive_the_budget():
    results = [
        {
            "tool_name": "web.search",
            "tool_call_id": f"tc_{index}",
            "status": "SUCCESS",
            "untrusted": True,
            "data": [{"title": "t", "url": f"https://e.com/{index}", "snippet": "s" * 400}]
            * 5,
        }
        for index in range(12)
    ]
    results.append(
        {
            "tool_name": "weather.forecast",
            "tool_call_id": "tc_weather",
            "status": "ERROR",
            "error_code": "weather_failed:weather_timeout",
            "untrusted": True,
            "data": None,
        }
    )

    items = _tool_payload(results)

    assert len(json.dumps(items, ensure_ascii=False)) <= MAX_TOOL_CONTEXT_CHARS + 400
    weather = [item for item in items if item["tool"] == "weather.forecast"]
    assert weather and weather[0]["status"] == "ERROR"
    assert weather[0]["error_code"] == "weather_failed:weather_timeout"


# -- full dialogue ---------------------------------------------------------


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
                "reason": "weather needs live data",
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


@pytest.mark.asyncio
async def test_weather_question_reaches_the_speaker_as_live_data(db):
    router = _FakeRouter(
        tools=[
            {
                "tool_name": "weather.forecast",
                "arguments": {"location": "Houston"},
                "purpose": "today's weather",
                "required": True,
            }
        ]
    )
    response = _RecordingResponse()
    registry = build_default_tool_registry(
        weather_provider=OpenMeteoProvider(transport=_weather_transport())
    )
    rt = build_runtime(
        db=db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=router,
        deepseek_client=object(),
        cognitive=_ExplodingCognitive(),
        response=response,
        allow_mixed_providers=True,
        tool_registry=registry,
    )
    await rt.repos.settings.set("web_search.enabled", "true")
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_external_ai=True,
    )
    from companion.domain.enums import EventSource
    from companion.domain.models import make_utterance_event

    real_event = make_utterance_event(
        conversation_id="conv",
        text="看看今天休斯顿天气",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.orchestrator.process_utterance(real_event)

    tool = response.calls[0].tool_results[0]
    assert tool["tool_name"] == "weather.forecast"
    assert tool["status"] == "SUCCESS"
    assert tool["untrusted"] is True
    assert tool["data"]["temperature"] == 25.5

    from companion.prompts.response import build_response_messages

    prompt = json.dumps(
        build_response_messages(response.calls[0], None), ensure_ascii=False
    )
    assert "25.5" in prompt
    assert "UNTRUSTED_EXTERNAL" in prompt
    assert "weather.forecast" in prompt
