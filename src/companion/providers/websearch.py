"""Web-search providers.

Real network search is **off by default** and must be explicitly enabled by
setting ``web_search.enabled=true``. Providers expose ``network_state`` so the
research engine and the Router tool can tell a real network provider from an
offline mock *before* calling it; a mock is never silently substituted for a
missing real configuration.

The Brave Search API key is read from ``BRAVE_SEARCH_API_KEY`` at request
time. It is never written to settings, the CMF database, logs or Git.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_KEY_ENV = "BRAVE_SEARCH_API_KEY"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: Optional[str] = None


class WebSearchError(RuntimeError):
    """A truthful search failure. Never converted into a fake result."""

    def __init__(
        self,
        code: str,
        message: Optional[str] = None,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message or code)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class WebSearchConfigError(WebSearchError):
    """A required provider setting or credential is missing."""


class MockWebSearchProvider:
    """Deterministic, offline search results derived from the query."""

    name = "mock-web-search"
    is_real_network = False

    async def network_state(self) -> str:
        return "mock"

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        q = query.strip() or "query"
        return [
            SearchResult(
                title=f"[mock] 来源 {i + 1}：{q[:40]}",
                url=f"https://example.com/result/{i + 1}",
                snippet=f"这是「{q}」的模拟摘要 {i + 1}。",
                provider="mock",
            )
            for i in range(min(limit, 3))
        ]


class HttpWebSearchProvider:
    """Experimental real search via DuckDuckGo's HTML endpoint.

    Constructing this provider is not enough to search: :meth:`search` refuses
    to make a network call unless ``enabled`` is ``True``. It is not the
    default real provider; Brave is. No API key is required by this endpoint.
    """

    name = "http-web-search"
    is_real_network = True

    def __init__(self, enabled: bool = False, *, timeout: float = 10.0) -> None:
        self.enabled = enabled
        self.timeout = timeout

    async def network_state(self) -> str:
        return "real" if self.enabled else "disabled"

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not self.enabled:
            raise WebSearchError(
                "web_search_disabled",
                "real web search is disabled; set web_search.enabled=true to use it",
            )
        import httpx
        import re

        url = "https://html.duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=self.timeout
            ) as client:
                resp = await client.post(url, data={"q": query})
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise WebSearchError(
                "duckduckgo_network_error", retryable=True
            ) from exc
        if resp.status_code in (401, 403):
            raise WebSearchError(
                "duckduckgo_auth_failed", status_code=resp.status_code
            )
        if resp.status_code == 429:
            raise WebSearchError(
                "duckduckgo_rate_limited",
                status_code=429,
                retryable=True,
            )
        if resp.status_code >= 500:
            raise WebSearchError(
                "duckduckgo_server_error",
                status_code=resp.status_code,
                retryable=True,
            )
        if resp.status_code != 200:
            raise WebSearchError(
                "duckduckgo_http_error", status_code=resp.status_code
            )
        html = resp.text

        results: list[SearchResult] = []
        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html
        ):
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            href = m.group(1)
            results.append(
                SearchResult(
                    title=title,
                    url=href,
                    snippet="",
                    provider="duckduckgo",
                )
            )
            if len(results) >= limit:
                break
        return results


class BraveWebSearchProvider:
    """Brave Search API adapter.

    ``api_key`` may be injected by a test; when ``None`` the key is read from
    the environment on every call. An absent key is a configuration error, not
    an empty successful search and not a silent mock fallback.
    """

    name = "brave-web-search"
    is_real_network = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: float = 10.0,
        max_limit: int = 10,
        retries: int = 2,
        backoff_base: float = 0.25,
        transport=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._api_key = api_key
        self.timeout = timeout
        self.max_limit = max(1, int(max_limit))
        self.retries = max(0, int(retries))
        self.backoff_base = max(0.0, float(backoff_base))
        self.transport = transport
        self.logger = logger or logging.getLogger("companion.websearch")

    def _key(self) -> str:
        return (self._api_key or os.environ.get(BRAVE_KEY_ENV, "")).strip()

    async def network_state(self) -> str:
        return "real" if self._key() else "unconfigured"

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        import httpx

        query = str(query or "").strip()
        if not query:
            raise WebSearchError("empty_query")
        key = self._key()
        if not key:
            raise WebSearchConfigError(
                "brave_api_key_missing",
                f"{BRAVE_KEY_ENV} is not set; real web search cannot run",
            )
        count = min(max(1, int(limit)), self.max_limit)
        attempts = self.retries + 1
        last_error: Optional[WebSearchError] = None

        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    transport=self.transport,
                    follow_redirects=False,
                ) as client:
                    response = await client.get(
                        BRAVE_SEARCH_ENDPOINT,
                        headers={
                            "Accept": "application/json",
                            "X-Subscription-Token": key,
                        },
                        params={"q": query, "count": count},
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = WebSearchError(
                    "brave_timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "brave_network_error",
                    retryable=True,
                )
                if attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                raise last_error from exc

            status = response.status_code
            if status == 200:
                return self._parse(response, count)
            if status in (401, 403):
                raise WebSearchError(
                    "brave_auth_failed",
                    "Brave Search rejected the API key",
                    status_code=status,
                )
            if status == 429:
                last_error = WebSearchError(
                    "brave_rate_limited",
                    status_code=status,
                    retryable=True,
                )
            elif status >= 500:
                last_error = WebSearchError(
                    "brave_server_error",
                    status_code=status,
                    retryable=True,
                )
            else:
                raise WebSearchError(
                    "brave_http_error", status_code=status
                )
            if attempt + 1 < attempts:
                await self._backoff(attempt)
                continue
            raise last_error
        raise last_error or WebSearchError("brave_search_failed")

    async def _backoff(self, attempt: int) -> None:
        if self.backoff_base > 0:
            await asyncio.sleep(min(self.backoff_base * (2**attempt), 2.0))

    @staticmethod
    def _parse(response, count: int) -> list[SearchResult]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError("brave_invalid_response") from exc
        if not isinstance(payload, dict):
            raise WebSearchError("brave_invalid_response")
        web = payload.get("web")
        items = web.get("results") if isinstance(web, dict) else None
        if not isinstance(items, list):
            return []
        results: list[SearchResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title") or "").strip()[:300],
                    url=url[:2000],
                    snippet=str(item.get("description") or "").strip()[:2000],
                    provider="brave",
                )
            )
            if len(results) >= count:
                break
        return results


class SettingsWebSearchProvider:
    """Resolve the configured provider at every call.

    ``allow_mock=True`` is the offline/research mode: with network search
    disabled it falls back to deterministic mock results, which are clearly
    labelled and can never be mistaken for real research. ``allow_mock=False``
    is the Router mode: a real result is required, so disabled/unconfigured
    states surface as truthful errors instead of mock hits.
    """

    name = "configured-web-search"

    def __init__(
        self,
        settings,
        *,
        allow_mock: bool = False,
        transport=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.settings = settings
        self.allow_mock = allow_mock
        self.transport = transport
        self.logger = logger or logging.getLogger("companion.websearch")

    async def _provider_name(self) -> str:
        raw = await self.settings.get("web_search.provider")
        return (raw or "brave").strip().lower()

    async def _resolve(self):
        enabled = await self.settings.get_bool("web_search.enabled", False)
        if not enabled:
            if self.allow_mock:
                return "mock", MockWebSearchProvider()
            return "disabled", None
        name = await self._provider_name()
        if name in ("mock", "offline"):
            if self.allow_mock:
                return "mock", MockWebSearchProvider()
            raise WebSearchConfigError(
                "mock_provider_not_allowed",
                "web_search.provider=mock cannot be used by the Router",
            )
        if name in ("brave", ""):
            provider = BraveWebSearchProvider(
                transport=self.transport, logger=self.logger
            )
            if await provider.network_state() != "real":
                return "unconfigured", None
            return "real", provider
        if name in ("duckduckgo", "http", "html"):
            return "real", HttpWebSearchProvider(enabled=True)
        raise WebSearchConfigError(
            "unknown_web_search_provider",
            f"unknown web_search.provider={name!r}",
        )

    async def network_state(self) -> str:
        try:
            state, _ = await self._resolve()
        except WebSearchConfigError:
            return "misconfigured"
        return state

    async def describe(self) -> tuple[str, str]:
        state, provider = await self._resolve()
        if provider is None:
            return state, self.name
        return state, getattr(provider, "name", self.name)

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        state, provider = await self._resolve()
        if state == "mock":
            return await provider.search(query, limit=limit)
        if state == "disabled":
            raise WebSearchError(
                "web_search_disabled",
                "real web search is disabled; set web_search.enabled=true",
            )
        if state == "unconfigured":
            raise WebSearchConfigError(
                "brave_api_key_missing",
                f"{BRAVE_KEY_ENV} is not set; real web search cannot run",
            )
        return await provider.search(query, limit=limit)


async def make_web_search(
    settings, *, enabled: Optional[bool] = None, allow_mock: bool = True
) -> object:
    """Build a settings-aware provider for idle/research use.

    With network search disabled this returns a mock provider (offline default,
    matching the V0.9 behaviour). When enabled, it returns the configured real
    provider wrapper; a missing key stays an explicit configuration error at
    call time instead of silently degrading to mock.
    """
    if enabled is None:
        enabled = await settings.get_bool("web_search.enabled", False)
    if not enabled:
        return MockWebSearchProvider()
    return SettingsWebSearchProvider(settings, allow_mock=allow_mock)


__all__ = [
    "SearchResult",
    "WebSearchError",
    "WebSearchConfigError",
    "MockWebSearchProvider",
    "HttpWebSearchProvider",
    "BraveWebSearchProvider",
    "SettingsWebSearchProvider",
    "make_web_search",
    "BRAVE_KEY_ENV",
]
