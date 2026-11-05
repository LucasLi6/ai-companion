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


class _DuckDuckGoHTMLParser:
    """Streaming, container-safe parser for DuckDuckGo's HTML result page.

    A ``result__a`` anchor opens one result; the next ``result__snippet``
    element in document order supplies its snippet. Titles and snippets are
    therefore attached per result instead of being paired by list index.
    Class names are matched as a token set, never as an exact string.
    """

    _VOID = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        self._parser = None
        self.results: list[dict] = []
        self.no_results = False
        self._current: Optional[dict] = None
        self._capture: Optional[str] = None
        self._capture_depth = 0
        self._depth = 0
        self._buffer: list[str] = []

    def feed(self, body: str) -> None:
        from html.parser import HTMLParser

        parser = self

        class _Handler(HTMLParser):
            def handle_starttag(self, tag, attrs):
                parser._start(tag, attrs)

            def handle_startendtag(self, tag, attrs):
                parser._start(tag, attrs)

            def handle_endtag(self, tag):
                parser._end(tag)

            def handle_data(self, data):
                if parser._capture is not None:
                    parser._buffer.append(data)

        self._parser = _Handler(convert_charrefs=True)
        self._parser.feed(body)

    def close(self) -> None:
        if self._parser is not None:
            self._parser.close()
        self._finish_current()

    @staticmethod
    def _attrs(attrs) -> dict[str, str]:
        return {str(k).lower(): str(v or "") for k, v in attrs}

    @staticmethod
    def _classes(attrs) -> set[str]:
        raw = " ".join(
            str(v or "") for k, v in attrs if str(k).lower() == "class"
        )
        return set(raw.split())

    def _start(self, tag: str, attrs) -> None:
        classes = self._classes(attrs)
        if self._capture is None:
            if tag == "a" and "result__a" in classes:
                values = self._attrs(attrs)
                self._finish_current()
                self._current = {
                    "title": "",
                    "url": _decode_duckduckgo_url(values.get("href", "")),
                    "snippet": "",
                }
                self._begin_capture("title")
            elif "result__snippet" in classes and self._current is not None:
                self._begin_capture("snippet")
            elif "no-results" in classes or "result--no-result" in classes:
                self.no_results = True
        if tag in self._VOID:
            return
        self._depth += 1

    def _end(self, tag: str) -> None:
        if tag in self._VOID:
            return
        if self._capture is not None and self._depth == self._capture_depth:
            self._end_capture()
        self._depth = max(0, self._depth - 1)

    def _begin_capture(self, field: str) -> None:
        self._capture = field
        self._capture_depth = self._depth + 1
        self._buffer = []

    def _end_capture(self) -> None:
        text = " ".join("".join(self._buffer).split())
        if self._current is not None and self._capture is not None:
            existing = str(self._current.get(self._capture) or "")
            self._current[self._capture] = (existing + " " + text).strip()
        self._capture = None
        self._buffer = []

    def _finish_current(self) -> None:
        if self._capture is not None:
            self._end_capture()
        if self._current is not None:
            self.results.append(self._current)
            self._current = None


def _decode_duckduckgo_url(href: str) -> str:
    """Decode a DDG redirect (``//duckduckgo.com/l/?uddg=…``) to its target."""
    from urllib.parse import parse_qs, urlparse

    href = (href or "").strip()
    if not href:
        return ""
    if "uddg=" in href:
        target = parse_qs(urlparse(href).query).get("uddg", [""])[0]
        if target:
            href = target
    if href.startswith("//"):
        href = "https:" + href
    return href


class HttpWebSearchProvider:
    """Real search via DuckDuckGo's HTML endpoint (no API key required).

    Constructing this provider is not enough to search: :meth:`search` refuses
    to make a network call unless ``enabled`` is ``True``. It is not the
    default real provider; Brave is. No API key is required by this endpoint.

    DuckDuckGo serves an "anomaly" challenge page (HTTP 202) to clients it
    identifies as bots. Sending a normal browser ``User-Agent`` avoids it; if a
    challenge page is still returned, it is reported as a retryable error
    rather than parsed into an empty (and therefore falsely "no results")
    success.
    """

    name = "http-web-search"
    is_real_network = True

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        enabled: bool = False,
        *,
        timeout: float = 8.0,
        retries: int = 2,
        backoff_base: float = 0.25,
        transport=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.enabled = enabled
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff_base = max(0.0, float(backoff_base))
        self.transport = transport
        self.logger = logger or logging.getLogger("companion.websearch")

    async def network_state(self) -> str:
        return "real" if self.enabled else "disabled"

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        import httpx

        if not self.enabled:
            raise WebSearchError(
                "web_search_disabled",
                "real web search is disabled; set web_search.enabled=true to use it",
            )
        query = str(query or "").strip()
        if not query:
            raise WebSearchError("empty_query")

        url = "https://html.duckduckgo.com/html/"
        attempts = self.retries + 1
        last_error: Optional[WebSearchError] = None

        for attempt in range(attempts):
            last_error = None
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=self.timeout,
                    transport=self.transport,
                    headers={"User-Agent": self.USER_AGENT},
                ) as client:
                    resp = await client.get(url, params={"q": query})
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = WebSearchError(
                    "duckduckgo_timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "duckduckgo_network_error",
                    retryable=True,
                )
                if attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                raise last_error from exc

            status = resp.status_code
            body = resp.text or ""
            lower = body.lower()
            # The anti-bot challenge page is an error, never an empty success.
            if status == 202 or ("anomaly" in lower and "challenge" in lower):
                last_error = WebSearchError(
                    "duckduckgo_anomaly",
                    status_code=status,
                    retryable=True,
                )
                if attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                raise last_error
            if status in (401, 403):
                raise WebSearchError(
                    "duckduckgo_auth_failed", status_code=status
                )
            if status == 429:
                last_error = WebSearchError(
                    "duckduckgo_rate_limited",
                    status_code=status,
                    retryable=True,
                )
            elif status >= 500:
                last_error = WebSearchError(
                    "duckduckgo_server_error",
                    status_code=status,
                    retryable=True,
                )
            elif status != 200:
                raise WebSearchError(
                    "duckduckgo_http_error", status_code=status
                )
            if last_error is not None:
                if attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                raise last_error

            if not body.strip():
                last_error = WebSearchError(
                    "duckduckgo_empty_response", retryable=True
                )
                if attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                raise last_error
            return self._parse(body, limit)
        raise last_error or WebSearchError("duckduckgo_search_failed")

    async def _backoff(self, attempt: int) -> None:
        if self.backoff_base > 0:
            await asyncio.sleep(min(self.backoff_base * (2**attempt), 2.0))

    @staticmethod
    def _parse(body: str, limit: int) -> list[SearchResult]:
        """Parse the DuckDuckGo HTML result page.

        One streaming parse walks the document in order: a ``result__a`` anchor
        starts a result and the ``result__snippet`` that follows belongs to that
        same result. This is more robust than extracting all titles and all
        snippets separately and pairing them by index, which breaks as soon as
        one result lacks a snippet or an extra anchor appears.

        Class matching is by token set, so ``class="result__a js-link"`` and a
        different attribute order both work. An unrecognised page is reported
        as a parse error instead of being mistaken for "no results".
        """
        parser = _DuckDuckGoHTMLParser()
        try:
            parser.feed(body)
            parser.close()
        except Exception as exc:  # malformed HTML must not become "no results"
            raise WebSearchError("duckduckgo_parse_error") from exc

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in parser.results:
            title = " ".join(str(item.get("title") or "").split())
            url = str(item.get("url") or "").strip()
            if not title or not url or url in seen:
                continue
            if not url.startswith(("http://", "https://")):
                continue
            seen.add(url)
            results.append(
                SearchResult(
                    title=title[:300],
                    url=url[:2000],
                    snippet=" ".join(str(item.get("snippet") or "").split())[:2000],
                    provider="duckduckgo",
                )
            )
            if len(results) >= limit:
                break

        if results:
            return results
        if parser.no_results:
            return []
        # A 200 page that neither parsed into results nor says "no results" has
        # an unknown structure (layout change, interstitial). Reporting that as
        # an empty success would fabricate "nothing found".
        raise WebSearchError("duckduckgo_parse_error", retryable=False)


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
