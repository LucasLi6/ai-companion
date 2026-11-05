"""Read the main text of a public web page (no JS, no PDF, no browser).

The Router may only fetch a URL that this turn's ``web.search`` actually
returned; :class:`~companion.tools.handlers.webfetch.WebFetchHandler` enforces
that allowlist. This provider adds the network-level defence:

- only ``http``/``https`` URLs, no embedded credentials;
- every hostname is resolved and **all** addresses must be public, so an
  internal, loopback, link-local or metadata address is refused;
- redirects are followed manually (bounded), and every hop is re-validated;
- environment proxies are ignored, so a proxy cannot silently forward the
  request to a blocked or internal destination;
- the connected peer address is checked after the response when the transport
  exposes it;
- response size and extracted text are capped.

Anything that cannot be verified is refused rather than attempted.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urljoin, urlsplit

MAX_REDIRECTS = 3
MAX_BYTES = 1_500_000
MAX_TEXT_CHARS = 20_000
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)


class WebFetchError(RuntimeError):
    """A truthful fetch failure. Never replaced with page-like text."""

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


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_reserved
        and not address.is_unspecified
    )


def _blocked_hostname(host: str) -> bool:
    name = (host or "").strip().lower().rstrip(".")
    if not name:
        return True
    if name in {"localhost", "localhost.localdomain"}:
        return True
    if name.endswith(".localhost") or name.endswith(".local") or name.endswith(
        ".internal"
    ):
        return True
    return False


def normalize_web_url(url: str) -> str:
    """Canonical form used for the search-result allowlist and checks."""
    text = str(url or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    fragment = ""
    return parts._replace(fragment=fragment).geturl()


class HttpWebFetchProvider:
    name = "http-web-fetch"

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_bytes: int = MAX_BYTES,
        max_text_chars: int = MAX_TEXT_CHARS,
        transport=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max(1024, int(max_bytes))
        self.max_text_chars = max(500, int(max_text_chars))
        self.transport = transport
        self.logger = logger or logging.getLogger("companion.webfetch")

    async def fetch(self, url: str) -> dict:
        target = str(url or "").strip()
        if not target:
            raise WebFetchError("empty_url")
        current = target
        for _ in range(MAX_REDIRECTS + 1):
            await self._assert_public_url(current)
            response, body = await self._request(current)
            status = response.status_code
            if status in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    raise WebFetchError(
                        "web_fetch_redirect_without_location", status_code=status
                    )
                current = urljoin(current, location)
                continue
            if status in (401, 403):
                raise WebFetchError(
                    "web_fetch_forbidden", status_code=status
                )
            if status == 404:
                raise WebFetchError("web_fetch_not_found", status_code=status)
            if status == 429:
                raise WebFetchError(
                    "web_fetch_rate_limited", status_code=429, retryable=True
                )
            if status >= 500:
                raise WebFetchError(
                    "web_fetch_server_error",
                    status_code=status,
                    retryable=True,
                )
            if status != 200:
                raise WebFetchError("web_fetch_http_error", status_code=status)
            return self._extract(current, response, body)
        raise WebFetchError("web_fetch_too_many_redirects")

    # -- validation --------------------------------------------------------

    async def _assert_public_url(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise WebFetchError("web_fetch_scheme_not_allowed")
        if parts.username or parts.password:
            raise WebFetchError("web_fetch_credentials_not_allowed")
        host = parts.hostname or ""
        if _blocked_hostname(host):
            raise WebFetchError("web_fetch_private_address")
        addresses = await self._resolve(host, parts.port)
        if not addresses:
            raise WebFetchError("web_fetch_dns_failed", retryable=True)
        for address in addresses:
            if not _is_public_ip(address):
                raise WebFetchError("web_fetch_private_address")

    @staticmethod
    async def _resolve(host: str, port: Optional[int]) -> list[str]:
        # An IP literal is checked directly and never re-resolved.
        try:
            return [str(ipaddress.ip_address(host))]
        except ValueError:
            pass
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.run_in_executor(
                None,
                lambda: socket.getaddrinfo(
                    host, port or 443, type=socket.SOCK_STREAM
                ),
            )
        except (socket.gaierror, OSError):
            return []
        found: list[str] = []
        for info in infos:
            address = info[4][0]
            if address not in found:
                found.append(address)
        return found

    # -- request -----------------------------------------------------------

    async def _request(self, url: str):
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
                follow_redirects=False,
                trust_env=False,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                },
            ) as client:
                async with client.stream("GET", url) as response:
                    self._check_peer(response)
                    content_type = response.headers.get(
                        "content-type", ""
                    ).split(";")[0].strip().lower()
                    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                        raise WebFetchError(
                            "web_fetch_unsupported_content_type",
                            content_type,
                        )
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise WebFetchError("web_fetch_too_large")
                        chunks.append(chunk)
                    return response, b"".join(chunks)
        except WebFetchError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise WebFetchError(
                "web_fetch_timeout"
                if isinstance(exc, httpx.TimeoutException)
                else "web_fetch_network_error",
                retryable=True,
            ) from exc

    @staticmethod
    def _check_peer(response) -> None:
        """Reject a connection whose actual peer is not a public address.

        This closes the DNS-rebinding window between the resolution check and
        the request. Transports that do not expose a peer address (e.g. a test
        MockTransport) skip this check.
        """
        try:
            stream = response.extensions.get("network_stream")
            if stream is None:
                return
            peer = stream.get_extra_info("server_addr") or stream.get_extra_info(
                "peername"
            )
        except Exception:
            return
        if not peer:
            return
        address = peer[0] if isinstance(peer, (tuple, list)) and peer else str(peer)
        if not _is_public_ip(str(address)):
            raise WebFetchError("web_fetch_private_address")

    # -- extraction --------------------------------------------------------

    def _extract(self, url: str, response, body: bytes) -> dict:
        encoding = response.encoding or "utf-8"
        try:
            text = body.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = body.decode("utf-8", errors="replace")
        extractor = _HtmlTextExtractor()
        try:
            extractor.feed(text)
            extractor.close()
        except Exception as exc:  # malformed markup: report, never fake text
            raise WebFetchError("web_fetch_parse_error") from exc
        content = extractor.text()[: self.max_text_chars]
        if not content:
            raise WebFetchError("web_fetch_empty_content")
        return {
            "url": url,
            "title": (extractor.title or "").strip()[:300],
            "text": content,
            "published_at": extractor.published_at,
            "fetched_at": _now_iso(),
            "source": urlsplit(url).hostname or "",
        }


class _HtmlTextExtractor:
    """Small visible-text extractor with a title and publish date."""

    _SKIP = {"script", "style", "noscript", "svg", "template", "iframe"}
    _PUBLISH_KEYS = {
        "article:published_time",
        "og:updated_time",
        "date",
        "pubdate",
        "publish-date",
        "publishdate",
    }

    def __init__(self) -> None:
        self.title = ""
        self.published_at: Optional[str] = None
        self._parts: list[str] = []
        self._depth = 0
        self._skip_depth: Optional[int] = None
        self._in_title = False
        self._title_buffer: list[str] = []

    def feed(self, body: str) -> None:
        from html.parser import HTMLParser

        extractor = self

        class _Handler(HTMLParser):
            def handle_starttag(self, tag, attrs):
                extractor._start(tag, attrs)

            def handle_startendtag(self, tag, attrs):
                extractor._start(tag, attrs, self_closing=True)

            def handle_endtag(self, tag):
                extractor._end(tag)

            def handle_data(self, data):
                extractor._data(data)

        self._handler = _Handler(convert_charrefs=True)
        self._handler.feed(body)

    def close(self) -> None:
        handler = getattr(self, "_handler", None)
        if handler is not None:
            handler.close()

    def _start(self, tag: str, attrs, *, self_closing: bool = False) -> None:
        tag = tag.lower()
        values = {str(k).lower(): str(v or "") for k, v in attrs}
        if tag == "meta":
            key = (
                values.get("property") or values.get("name") or ""
            ).strip().lower()
            content = values.get("content") or ""
            if key in self._PUBLISH_KEYS and content and not self.published_at:
                self.published_at = content.strip()[:64]
        if tag == "title":
            self._in_title = True
            self._title_buffer = []
            return
        if tag in self._SKIP:
            if not self_closing:
                self._skip_depth = self._depth
            return
        if self_closing:
            if tag in {"br", "p", "div", "li", "tr"}:
                self._parts.append(" ")
            return
        self._depth += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section",
                   "article", "header", "footer"}:
            self._parts.append(" ")

    def _end(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            self.title = " ".join("".join(self._title_buffer).split())
            return
        if tag in self._SKIP:
            self._skip_depth = None
            return
        if self._skip_depth is not None:
            return
        self._depth = max(0, self._depth - 1)
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "section",
                   "article", "header", "footer"}:
            self._parts.append(" ")

    def _data(self, data: str) -> None:
        if self._in_title:
            self._title_buffer.append(data)
            return
        if self._skip_depth is not None:
            return
        if data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def _now_iso() -> str:
    from companion.domain.models import utcnow

    return utcnow().isoformat()


__all__ = [
    "HttpWebFetchProvider",
    "WebFetchError",
    "MAX_REDIRECTS",
    "MAX_BYTES",
    "normalize_web_url",
    "_is_public_ip",
]
