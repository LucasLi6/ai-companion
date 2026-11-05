"""V0.9.2: real web search providers.

All requests go through ``httpx.MockTransport``; no real API is ever called.
The tests pin the properties the offline suite depends on:

- Brave JSON (Chinese/English) parses into real URLs and snippets;
- an absent credential is a configuration error, never a silent mock fallback;
- 401/403 are not retried, 429/5xx/timeouts are retried and reported truthfully;
- the configured provider is resolved at request time.
"""

from __future__ import annotations

import httpx
import pytest

from companion.providers.websearch import (
    BRAVE_KEY_ENV,
    BraveWebSearchProvider,
    MockWebSearchProvider,
    SettingsWebSearchProvider,
    WebSearchConfigError,
    WebSearchError,
)


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _brave_payload(results: list[dict]) -> dict:
    return {"web": {"results": results}}


@pytest.mark.asyncio
async def test_brave_parses_chinese_and_english_results_and_preserves_urls():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["token"] = request.headers.get("X-Subscription-Token")
        seen["accept"] = request.headers.get("Accept")
        seen["q"] = request.url.params.get("q")
        seen["count"] = request.url.params.get("count")
        return httpx.Response(
            200,
            json=_brave_payload(
                [
                    {
                        "title": "中文标题",
                        "url": "https://example.cn/article/1",
                        "description": "这是中文摘要。",
                    },
                    {
                        "title": "English title",
                        "url": "https://example.com/article/2",
                        "description": "An English snippet.",
                    },
                    # Invalid entries are skipped, never surfaced as results.
                    {"title": "bad", "url": "javascript:alert(1)"},
                    "not-a-result",
                ]
            ),
        )

    provider = BraveWebSearchProvider(
        api_key="test-key", transport=_transport(handler)
    )
    results = await provider.search("今天 北京 新闻", limit=2)

    assert seen["method"] == "GET"
    assert seen["token"] == "test-key"
    assert seen["accept"] == "application/json"
    assert seen["q"] == "今天 北京 新闻"
    assert seen["count"] == "2"
    assert [r.url for r in results] == [
        "https://example.cn/article/1",
        "https://example.com/article/2",
    ]
    assert results[0].title == "中文标题"
    assert results[0].snippet == "这是中文摘要。"
    assert results[1].title == "English title"
    assert all(r.provider == "brave" for r in results)


@pytest.mark.asyncio
async def test_brave_no_results_is_an_empty_successful_lookup():
    provider = BraveWebSearchProvider(
        api_key="test-key",
        transport=_transport(
            lambda request: httpx.Response(200, json=_brave_payload([]))
        ),
    )
    assert await provider.search("nothing matches", limit=5) == []

    # A well-formed response without a ``web`` block is also an empty lookup.
    provider = BraveWebSearchProvider(
        api_key="test-key",
        transport=_transport(
            lambda request: httpx.Response(200, json={"query": "x"})
        ),
    )
    assert await provider.search("x") == []


@pytest.mark.asyncio
async def test_brave_invalid_json_is_an_error_not_a_fake_result():
    provider = BraveWebSearchProvider(
        api_key="test-key",
        transport=_transport(
            lambda request: httpx.Response(200, text="<html>not json</html>")
        ),
    )
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "brave_invalid_response"


@pytest.mark.asyncio
async def test_missing_brave_key_is_a_config_error_and_never_a_mock(rt, monkeypatch):
    monkeypatch.delenv(BRAVE_KEY_ENV, raising=False)
    calls: list[httpx.Request] = []

    def exploding(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("no network call without a key")

    provider = BraveWebSearchProvider(transport=_transport(exploding))
    assert await provider.network_state() == "unconfigured"
    with pytest.raises(WebSearchConfigError) as exc:
        await provider.search("x")
    assert exc.value.code == "brave_api_key_missing"
    assert calls == []

    await rt.repos.settings.set("web_search.enabled", "true")
    await rt.repos.settings.set("web_search.provider", "brave")
    configured = SettingsWebSearchProvider(
        rt.repos.settings, allow_mock=False, transport=_transport(exploding)
    )
    assert await configured.network_state() == "unconfigured"
    with pytest.raises(WebSearchConfigError) as exc:
        await configured.search("x")
    assert exc.value.code == "brave_api_key_missing"
    assert calls == []


@pytest.mark.asyncio
async def test_mock_provider_is_never_presented_as_real(rt):
    mock = MockWebSearchProvider()
    assert mock.is_real_network is False
    assert await mock.network_state() == "mock"
    results = await mock.search("ota")
    assert results and all(r.provider == "mock" for r in results)

    # Even when the Router asks for "a real provider only", a configured mock
    # is refused instead of being silently upgraded or downgraded.
    await rt.repos.settings.set("web_search.enabled", "true")
    await rt.repos.settings.set("web_search.provider", "mock")
    provider = SettingsWebSearchProvider(rt.repos.settings, allow_mock=False)
    with pytest.raises(WebSearchConfigError) as exc:
        await provider.search("ota")
    assert exc.value.code == "mock_provider_not_allowed"


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.asyncio
async def test_brave_auth_errors_are_not_retried(status):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(status)
        return httpx.Response(status, json={"error": "denied"})

    provider = BraveWebSearchProvider(
        api_key="bad-key",
        retries=2,
        backoff_base=0,
        transport=_transport(handler),
    )
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "brave_auth_failed"
    assert exc.value.status_code == status
    assert exc.value.retryable is False
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("status", "code"),
    [(429, "brave_rate_limited"), (500, "brave_server_error")],
)
@pytest.mark.asyncio
async def test_brave_transient_errors_are_retried_then_reported(status, code):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(status)
        return httpx.Response(status, json={"error": "transient"})

    provider = BraveWebSearchProvider(
        api_key="key",
        retries=2,
        backoff_base=0,
        transport=_transport(handler),
    )
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == code
    assert exc.value.status_code == status
    assert exc.value.retryable is True
    assert len(calls) == 3  # initial request + two retries


@pytest.mark.asyncio
async def test_brave_other_4xx_fails_without_retrying():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(400)
        return httpx.Response(400, json={"error": "bad request"})

    provider = BraveWebSearchProvider(
        api_key="key", retries=2, backoff_base=0, transport=_transport(handler)
    )
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "brave_http_error"
    assert exc.value.retryable is False
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_timeout_is_retried_and_can_recover():
    calls: list[int] = []
    results = _brave_payload(
        [{"title": "t", "url": "https://example.com/a", "description": "s"}]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=results)

    provider = BraveWebSearchProvider(
        api_key="key", retries=2, backoff_base=0, transport=_transport(handler)
    )
    found = await provider.search("x", limit=1)
    assert [r.url for r in found] == ["https://example.com/a"]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_exhausted_timeout_is_reported_truthfully():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectTimeout("timed out", request=request)

    provider = BraveWebSearchProvider(
        api_key="key", retries=2, backoff_base=0, transport=_transport(handler)
    )
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "brave_timeout"
    assert exc.value.retryable is True
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_transport_error_is_reported_as_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = BraveWebSearchProvider(
        api_key="key", retries=1, backoff_base=0, transport=_transport(handler)
    )
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "brave_network_error"
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_settings_are_resolved_at_request_time(rt, monkeypatch):
    monkeypatch.setenv(BRAVE_KEY_ENV, "test-key")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json=_brave_payload(
                [
                    {
                        "title": "live",
                        "url": "https://example.com/live",
                        "description": "fresh",
                    }
                ]
            ),
        )

    provider = SettingsWebSearchProvider(
        rt.repos.settings, allow_mock=True, transport=_transport(handler)
    )
    # Off by default: mock, no network.
    assert await provider.network_state() == "mock"
    assert (await provider.search("x"))[0].provider == "mock"
    assert calls == []

    # Turning the setting on makes the very next call real.
    await rt.repos.settings.set("web_search.enabled", "true")
    await rt.repos.settings.set("web_search.provider", "brave")
    assert await provider.network_state() == "real"
    results = await provider.search("x")
    assert results[0].provider == "brave"
    assert results[0].url == "https://example.com/live"
    assert len(calls) == 1

    # Turning it off again returns to mock without a request.
    await rt.repos.settings.set("web_search.enabled", "false")
    assert await provider.network_state() == "mock"
    assert (await provider.search("x"))[0].provider == "mock"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_router_mode_reports_disabled_instead_of_mock(rt):
    provider = SettingsWebSearchProvider(rt.repos.settings, allow_mock=False)
    assert await provider.network_state() == "disabled"
    with pytest.raises(WebSearchError) as exc:
        await provider.search("x")
    assert exc.value.code == "web_search_disabled"
