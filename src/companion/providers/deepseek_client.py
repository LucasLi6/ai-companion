"""Real DeepSeek Chat Completions client (Phase 1 of V0.1 DeepSeek integration).

This module is the *only* place that talks to the DeepSeek API. Everything
above it (cognitive provider, response provider, privacy gateway) depends on
the small, well-defined surface here, never on the ``openai`` SDK directly.

Design rules (mirroring the V0.1 constraints):

- **Key security.** The API key is read from the ``DEEPSEEK_API_KEY``
  environment variable (or an explicit config object). It is never written to
  SQLite, never logged, never passed as a CLI flag, and never embedded in a
  test snapshot. ``build_deepseek_client`` refuses an empty key.
- **No sensitive logging.** Request audit lines carry only
  ``request_id``/``event_id``/``provider``/``model``/``request_type``/
  ``latency_ms``/token counts/``success``/``error_type``. Message content,
  the full prompt, private conversation text and ``reasoning_content`` are
  never logged.
- **Fail fast vs retry.** Auth (401/402) and deterministic errors (400,
  empty config) raise immediately. Transient errors — 429, 5xx, timeout,
  connection loss — retry with exponential backoff up to ``max_retries``.
- **Lazy SDK import.** ``openai`` is imported only inside
  :func:`build_deepseek_client`, so the core runtime (and the mock path)
  never needs it installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from companion.domain.models import new_id

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

logger = logging.getLogger("companion.deepseek")


# -- configuration ---------------------------------------------------------


@dataclass(frozen=True)
class DeepSeekConfig:
    """Immutable DeepSeek client configuration.

    ``api_key`` must be set before any network call. Everything else has a
    safe default and can be overridden via the ``DEEPSEEK_*`` environment
    variables (:func:`load_deepseek_config`).
    """

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    analysis_max_tokens: int = 1024
    response_max_tokens: int = 2048
    # V0.4: Thinking Mode is OFF by default for both request types. The
    # explicit "none" is sent on the wire (not omitted) so the provider cannot
    # silently apply a reasoning default. An environment override to "high"
    # remains possible for both request types.
    analysis_reasoning_effort: Optional[str] = "none"
    response_reasoning_effort: Optional[str] = "none"


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_str(value: str | None, default: str | None) -> str | None:
    if value is None or value.strip() == "":
        return default
    return value.strip()


def load_deepseek_config(environ: dict | None = None) -> DeepSeekConfig:
    """Build :class:`DeepSeekConfig` from ``DEEPSEEK_*`` environment variables.

    ``environ`` defaults to ``os.environ``; tests pass an explicit mapping so a
    developer's real key is never read accidentally.
    """
    env = environ if environ is not None else os.environ
    return DeepSeekConfig(
        api_key=env.get("DEEPSEEK_API_KEY", "").strip(),
        base_url=_env_str(env.get("DEEPSEEK_BASE_URL"), DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
        model=_env_str(env.get("DEEPSEEK_MODEL"), DEFAULT_MODEL) or DEFAULT_MODEL,
        timeout_seconds=_env_float(env.get("DEEPSEEK_TIMEOUT"), 60.0),
        max_retries=_env_int(env.get("DEEPSEEK_MAX_RETRIES"), 2),
        analysis_max_tokens=_env_int(env.get("DEEPSEEK_ANALYSIS_MAX_TOKENS"), 1024),
        response_max_tokens=_env_int(env.get("DEEPSEEK_RESPONSE_MAX_TOKENS"), 2048),
        analysis_reasoning_effort=_env_str(
            env.get("DEEPSEEK_ANALYSIS_REASONING_EFFORT"), "none"
        ),
        response_reasoning_effort=_env_str(
            env.get("DEEPSEEK_RESPONSE_REASONING_EFFORT"), "none"
        ),
    )


# -- errors -----------------------------------------------------------------


class DeepSeekError(Exception):
    """Base class for every DeepSeek client failure.

    ``retryable`` controls whether :meth:`DeepSeekClient.complete` retries the
    request before giving up.
    """

    error_type = "deepseek_error"
    retryable = False


class DeepSeekConfigError(DeepSeekError):
    """Missing/invalid configuration (e.g. no API key in Real mode)."""

    error_type = "config"
    retryable = False


class DeepSeekAuthError(DeepSeekError):
    """Authentication or balance problem (HTTP 401/402). Fail fast."""

    error_type = "auth"
    retryable = False


class DeepSeekInvalidRequestError(DeepSeekError):
    """Deterministic request error (HTTP 400). Fail fast."""

    error_type = "invalid_request"
    retryable = False


class DeepSeekRateLimitError(DeepSeekError):
    """HTTP 429 — retry with backoff."""

    error_type = "rate_limit"
    retryable = True


class DeepSeekServerError(DeepSeekError):
    """HTTP 5xx — retry with backoff."""

    error_type = "server_error"
    retryable = True


class DeepSeekTimeoutError(DeepSeekError):
    """Request timed out — retry with backoff."""

    error_type = "timeout"
    retryable = True


class DeepSeekConnectionError(DeepSeekError):
    """Network failure — retry with backoff."""

    error_type = "connection"
    retryable = True


class DeepSeekResponseError(DeepSeekError):
    """Empty/malformed response body — retry with backoff."""

    error_type = "invalid_response"
    retryable = True


# -- response ---------------------------------------------------------------


@dataclass(frozen=True)
class ChatResponse:
    """The parts of a Chat Completions response the rest of the app needs."""

    content: str
    model: str
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _classify_exception(exc: Exception) -> tuple[str, bool]:
    """Map an SDK/transport exception to ``(error_type, retryable)``.

    Duck-typing is used rather than ``isinstance`` on ``openai`` exceptions so
    this module — and its tests — work with a fake SDK object that raises
    look-alike exceptions.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None
    if status == 401 or status == 402:
        return "auth", False
    if status == 429:
        return "rate_limit", True
    if status == 400:
        return "invalid_request", False
    if status is not None and 500 <= status < 600:
        return "server_error", True
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout", True
    if "connection" in name or "connect" in name:
        return "connection", True
    return "unknown", False


_ERROR_CLASSES = {
    "auth": DeepSeekAuthError,
    "invalid_request": DeepSeekInvalidRequestError,
    "rate_limit": DeepSeekRateLimitError,
    "server_error": DeepSeekServerError,
    "timeout": DeepSeekTimeoutError,
    "connection": DeepSeekConnectionError,
    "invalid_response": DeepSeekResponseError,
    "config": DeepSeekConfigError,
}


def _wrap_error(error_type: str, detail: str) -> DeepSeekError:
    cls = _ERROR_CLASSES.get(error_type, DeepSeekError)
    return cls(detail)


def _safe_detail(exc: Exception) -> str:
    """A log-safe error detail that never echoes a response/request body."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return f"HTTP {status}"
    return type(exc).__name__


# -- client -----------------------------------------------------------------


class DeepSeekClient:
    """Thin, retrying wrapper over an AsyncOpenAI-compatible client.

    Two ways to construct it:

    - Inject ``client`` directly (tests use a fake). No ``openai`` import.
    - Use :func:`build_deepseek_client`, which creates a real ``AsyncOpenAI``.
    """

    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        client=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self._client = client
        self.logger = logger or logging.getLogger("companion.deepseek")

    async def complete(
        self,
        messages: list[dict],
        *,
        request_type: str,
        event_id: Optional[str] = None,
        request_id: Optional[str] = None,
        json_mode: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> ChatResponse:
        """Run one Chat Completions call with retry + audit logging.

        ``request_type`` is ``"analysis"`` or ``"response"``; it selects the
        per-type token/effort defaults from the config when those arguments are
        left ``None``. ``json_mode`` defaults to ``True`` for ``"analysis"``
        and ``False`` for ``"response"``.
        """
        if not self.config.api_key and self._client is None:
            raise DeepSeekConfigError(
                "no DEEPSEEK_API_KEY configured (Real mode requires a key)"
            )
        if json_mode is None:
            json_mode = request_type == "analysis"
        if max_tokens is None:
            max_tokens = (
                self.config.analysis_max_tokens
                if request_type == "analysis"
                else self.config.response_max_tokens
            )
        if reasoning_effort is None:
            reasoning_effort = (
                self.config.analysis_reasoning_effort
                if request_type == "analysis"
                else self.config.response_reasoning_effort
            )
        request_id = request_id or new_id("dsrq")

        last_error: Optional[DeepSeekError] = None
        attempt = 0
        while True:
            t0 = time.monotonic()
            try:
                response = await self._request(
                    messages,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
            except DeepSeekError as exc:
                last_error = exc
            except Exception as exc:  # SDK/transport error -> classify it
                error_type, retryable = _classify_exception(exc)
                last_error = _wrap_error(error_type, _safe_detail(exc))
                if not retryable:
                    last_error.retryable = False
            else:
                latency_ms = (time.monotonic() - t0) * 1000
                self._audit(
                    request_id,
                    event_id,
                    request_type,
                    response.prompt_tokens,
                    response.completion_tokens,
                    response.total_tokens,
                    latency_ms,
                    success=True,
                    error_type=None,
                )
                return response

            latency_ms = (time.monotonic() - t0) * 1000
            assert last_error is not None
            self._audit(
                request_id,
                event_id,
                request_type,
                0,
                0,
                0,
                latency_ms,
                success=False,
                error_type=last_error.error_type,
            )
            if not last_error.retryable or attempt >= self.config.max_retries:
                raise last_error
            attempt += 1
            await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** (attempt - 1)))

    async def _request(
        self,
        messages: list[dict],
        *,
        json_mode: bool,
        max_tokens: int,
        reasoning_effort: Optional[str],
    ) -> ChatResponse:
        if self._client is None:
            raise DeepSeekConfigError("no DeepSeek client configured")

        params: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort

        try:
            raw = await self._client.chat.completions.create(**params)
        except DeepSeekError:
            raise
        except Exception as exc:
            error_type, retryable = _classify_exception(exc)
            raise _wrap_error(error_type, _safe_detail(exc)) from exc

        return self._parse_response(raw)

    def _parse_response(self, raw) -> ChatResponse:
        try:
            choice = raw.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise DeepSeekResponseError(f"malformed response: {exc!r}") from exc

        content = content or ""
        if not content.strip():
            raise DeepSeekResponseError("empty completion content")

        usage = getattr(raw, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        finish_reason = None
        try:
            finish_reason = getattr(choice, "finish_reason", None)
        except AttributeError:
            finish_reason = None

        model = getattr(raw, "model", self.config.model) or self.config.model
        return ChatResponse(
            content=content,
            model=model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _audit(
        self,
        request_id: str,
        event_id: Optional[str],
        request_type: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        latency_ms: float,
        *,
        success: bool,
        error_type: Optional[str],
    ) -> None:
        """Structured request log — never message content or the API key."""
        self.logger.info(
            "deepseek_request request_id=%s event_id=%s provider=deepseek model=%s"
            " request_type=%s latency_ms=%.2f input_tokens=%d output_tokens=%d"
            " total_tokens=%d success=%s error_type=%s",
            request_id,
            event_id,
            self.config.model,
            request_type,
            latency_ms,
            input_tokens,
            output_tokens,
            total_tokens,
            "true" if success else "false",
            error_type or "none",
        )


def build_deepseek_client(
    config: DeepSeekConfig,
    *,
    logger: Optional[logging.Logger] = None,
) -> DeepSeekClient:
    """Create a real ``DeepSeekClient`` backed by ``AsyncOpenAI``.

    Refuses to construct a real client without an API key (the "Real 模式没有
    Key 时，必须明确报错" constraint). The ``openai`` SDK is imported lazily so
    the core runtime never needs it.
    """
    if not config.api_key:
        raise DeepSeekConfigError(
            "no DEEPSEEK_API_KEY configured — set it in the environment"
        )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DeepSeekConfigError(
            "the 'openai' package is required for DeepSeek; install with"
            " `pip install -e '.[deepseek]'`"
        ) from exc

    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        # Our own retry loop owns retries; the SDK must not double-retry.
        max_retries=0,
    )
    return DeepSeekClient(config, client=client, logger=logger)
