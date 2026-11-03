"""Web-search providers.

Real network search is **off by default** and must be explicitly enabled by
setting ``web_search.enabled=true``. Tests always use
:class:`MockWebSearchProvider` and never touch the network (see the V0.1
constraints: no test may call a real API and pretend it costs nothing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class MockWebSearchProvider:
    """Deterministic, offline search results derived from the query."""

    name = "mock-web-search"

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        q = query.strip() or "query"
        return [
            SearchResult(
                title=f"[mock] 来源 {i + 1}：{q[:40]}",
                url=f"https://example.com/result/{i + 1}",
                snippet=f"这是「{q}」的模拟摘要 {i + 1}。",
            )
            for i in range(min(limit, 3))
        ]


class HttpWebSearchProvider:
    """Real search via DuckDuckGo's HTML endpoint.

    Constructing this provider is not enough to search: :meth:`search` refuses
    to make a network call unless ``enabled`` is ``True``, so the provider is
    inert until the operator flips the ``web_search.enabled`` setting. No API
    key or access token is required or stored.
    """

    name = "http-web-search"

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not self.enabled:
            raise RuntimeError(
                "real web search is disabled; set web_search.enabled=true to use it"
            )
        import httpx
        import re

        url = "https://html.duckduckgo.com/html/"
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            resp = await client.post(url, data={"q": query})
            resp.raise_for_status()
            html = resp.text

        results: list[SearchResult] = []
        # DuckDuckGo HTML results: <a class="result__a" href="...">title</a>.
        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html
        ):
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            href = m.group(1)
            results.append(SearchResult(title=title, url=href, snippet=""))
            if len(results) >= limit:
                break
        return results


async def make_web_search(settings, *, enabled: Optional[bool] = None) -> object:
    """Build the active search provider from application settings.

    Returns a mock provider unless ``web_search.enabled`` is true (either
    explicitly passed or read from settings). The explicit ``enabled`` flag
    exists so tests can force a real-provider object without enabling it.
    """
    if enabled is None:
        enabled = await settings.get_bool("web_search.enabled", False)
    return HttpWebSearchProvider(enabled=enabled) if enabled else MockWebSearchProvider()
