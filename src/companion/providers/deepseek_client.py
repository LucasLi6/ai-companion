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
    # V0.4: Thinking Mode is OFF by default for both request types. V0.6.3
    # makes that switch explicit on the wire (``thinking.type``), because the
    # V4 API defaults to thinking ON and ``reasoning_effort`` alone cannot
    # reliably turn it off. Enable with DEEPSEEK_THINKING=enabled or per type
    # with DEEPSEEK_ANALYSIS_THINKING / DEEPSEEK_RESPONSE_THINKING.
    analysis_thinking: str = "disabled"
    response_thinking: str = "disabled"
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


_ENABLED_WORDS = ("1", "true", "yes", "on", "enabled", "enable")
_DISABLED_WORDS = ("0", "false", "no", "off", "disabled", "disable", "none")


def _env_thinking(value: str | None, default: str) -> str:
    """Normalise a thinking-mode env value to ``"enabled"``/``"disabled"``."""
    if value is None or value.strip() == "":
        return default
    word = value.strip().lower()
    if word in _ENABLED_WORDS:
        return "enabled"
    if word in _DISABLED_WORDS:
        return "disabled"
    return default


def _thinking_wire(request_type: str, config: DeepSeekConfig) -> dict:
    """The provider's explicit thinking switch, sent in ``extra_body``."""
    mode = (
        config.analysis_thinking
        if request_type == "analysis"
        else config.response_thinking
    )
    return {"thinking": {"type": "enabled" if mode == "enabled" else "disabled"}}


def _reasoning_effort_wire(effort: Optional[str], thinking: dict) -> Optional[str]:
    """Reasoning effort only makes sense while thinking is enabled.

    ``"none"`` is the project's internal "unset" marker; the V4 API expresses
    off via ``thinking.type``, not via an effort value.
    """
    if thinking["thinking"]["type"] != "enabled":
        return None
    if effort is None:
        return None
    cleaned = effort.strip()
    if cleaned.lower() in ("", "none", "disabled", "off"):
        return None
    return cleaned


def load_deepseek_config(environ: dict | None = None) -> DeepSeekConfig:
    """Build :class:`DeepSeekConfig` from ``DEEPSEEK_*`` environment variables.

    ``environ`` defaults to ``os.environ``; tests pass an explicit mapping so a
    developer's real key is never read accidentally.
    """
    env = environ if environ is not None else os.environ
    # One global switch, with optional per-request-type overrides.
    thinking = _env_thinking(env.get("DEEPSEEK_THINKING"), "disabled")
    return DeepSeekConfig(
        api_key=env.get("DEEPSEEK_API_KEY", "").strip(),
        base_url=_env_str(env.get("DEEPSEEK_BASE_URL"), DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
        model=_env_str(env.get("DEEPSEEK_MODEL"), DEFAULT_MODEL) or DEFAULT_MODEL,
        timeout_seconds=_env_float(env.get("DEEPSEEK_TIMEOUT"), 60.0),
        max_retries=_env_int(env.get("DEEPSEEK_MAX_RETRIES"), 2),
        analysis_max_tokens=_env_int(env.get("DEEPSEEK_ANALYSIS_MAX_TOKENS"), 1024),
        response_max_tokens=_env_int(env.get("DEEPSEEK_RESPONSE_MAX_TOKENS"), 2048),
        analysis_thinking=_env_thinking(
            env.get("DEEPSEEK_ANALYSIS_THINKING"), thinking
        ),
        response_thinking=_env_thinking(
            env.get("DEEPSEEK_RESPONSE_THINKING"), thinking
        ),
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


@dataclass(frozen=True)
class StreamDelta:
    """One incremental piece of a streaming completion.

    ``reasoning_content`` is deliberately not represented: the thinking stream
    is never displayed, logged or forwarded. Only the visible ``content`` and
    the terminal metadata leave this module.
    """

    content: str = ""
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_reported: bool = False


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


class _Usage:
    """Accumulate streaming usage without assuming every chunk carries it."""

    __slots__ = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reported",
    )

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.reported = False

    def absorb(self, usage) -> None:
        if usage is None:
            return
        try:
            self.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            self.total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            self.reported = True
        except (TypeError, ValueError):
            return


async def _close_stream(stream) -> None:
    if stream is None:
        return
    close = getattr(stream, "aclose", None)
    if close is None:
        close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception:  # closing a broken stream must never mask the real error
        return


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
        thinking = _thinking_wire(request_type, self.config)
        reasoning_effort = _reasoning_effort_wire(reasoning_effort, thinking)
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
                    thinking=thinking,
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

    async def stream_completion(
        self,
        messages: list[dict],
        *,
        request_type: str = "response",
        event_id: Optional[str] = None,
        request_id: Optional[str] = None,
        json_mode: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Yield visible deltas from a real streaming Chat Completions call.

        This is a true stream: each delta is yielded as it arrives, not after
        the response is complete. ``reasoning_content`` is ignored entirely.

        Retries are only safe *before* the first visible delta. Once any text
        has been shown to the caller, an error propagates so the orchestration
        layer can persist an explicit interrupted/partial status instead of
        silently restarting the reply. Cancellation closes the upstream stream
        and re-raises, so the caller can record the partial output.
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
        thinking = _thinking_wire(request_type, self.config)
        reasoning_effort = _reasoning_effort_wire(reasoning_effort, thinking)
        request_id = request_id or new_id("dsrq")

        attempt = 0
        while True:
            t0 = time.monotonic()
            first_token_ms: Optional[float] = None
            emitted = False
            finish_reason: Optional[str] = None
            usage = _Usage()
            stream = None
            try:
                stream = await self._open_stream(
                    messages,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    thinking=thinking,
                )
                iterator = stream.__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            anext(iterator), timeout=self.config.timeout_seconds
                        )
                    except StopAsyncIteration:
                        break
                    usage.absorb(getattr(chunk, "usage", None))
                    choices = getattr(chunk, "choices", None) or []
                    choice = choices[0] if choices else None
                    content = ""
                    if choice is not None:
                        delta = getattr(choice, "delta", None)
                        # ``reasoning_content`` is intentionally never read.
                        content = getattr(delta, "content", None) or ""
                        reason = getattr(choice, "finish_reason", None)
                        if reason:
                            finish_reason = reason
                    if content:
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - t0) * 1000
                        emitted = True
                        yield StreamDelta(
                            content=content, finish_reason=finish_reason
                        )
                    elif finish_reason is not None:
                        yield StreamDelta(finish_reason=finish_reason)

                self._audit_stream(
                    request_id,
                    event_id,
                    request_type,
                    usage,
                    (time.monotonic() - t0) * 1000,
                    first_token_ms,
                    success=True,
                    error_type=None,
                )
                # A final metadata delta lets the caller see usage and the
                # finish reason even when the provider sent them separately.
                yield StreamDelta(
                    finish_reason=finish_reason,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    usage_reported=usage.reported,
                )
                return
            except asyncio.CancelledError:
                await _close_stream(stream)
                self._audit_stream(
                    request_id,
                    event_id,
                    request_type,
                    usage,
                    (time.monotonic() - t0) * 1000,
                    first_token_ms,
                    success=False,
                    error_type="cancelled",
                )
                raise
            except Exception as exc:
                await _close_stream(stream)
                if isinstance(exc, DeepSeekError):
                    error = exc
                else:
                    error_type, retryable = _classify_exception(exc)
                    error = _wrap_error(error_type, _safe_detail(exc))
                    error.retryable = retryable
                self._audit_stream(
                    request_id,
                    event_id,
                    request_type,
                    usage,
                    (time.monotonic() - t0) * 1000,
                    first_token_ms,
                    success=False,
                    error_type=error.error_type,
                )
                # Text already shown to a person cannot be silently restarted.
                if emitted or not error.retryable or attempt >= self.config.max_retries:
                    raise error
                attempt += 1
                await asyncio.sleep(
                    self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                )

    async def _open_stream(
        self,
        messages: list[dict],
        *,
        json_mode: bool,
        max_tokens: int,
        reasoning_effort: Optional[str],
        thinking: dict,
    ):
        if self._client is None:
            raise DeepSeekConfigError("no DeepSeek client configured")
        params: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "extra_body": thinking,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        try:
            return await self._client.chat.completions.create(**params)
        except DeepSeekError:
            raise
        except Exception as exc:
            error_type, retryable = _classify_exception(exc)
            error = _wrap_error(error_type, _safe_detail(exc))
            error.retryable = retryable
            raise error from exc

    def _audit_stream(
        self,
        request_id: str,
        event_id: Optional[str],
        request_type: str,
        usage,
        latency_ms: float,
        first_token_ms: Optional[float],
        *,
        success: bool,
        error_type: Optional[str],
    ) -> None:
        """Structured stream log — never content, never a reasoning stream."""
        self.logger.info(
            "deepseek_stream request_id=%s event_id=%s provider=deepseek model=%s"
            " request_type=%s latency_ms=%.2f time_to_first_token_ms=%s"
            " input_tokens=%d output_tokens=%d total_tokens=%d success=%s"
            " error_type=%s",
            request_id,
            event_id,
            self.config.model,
            request_type,
            latency_ms,
            f"{first_token_ms:.2f}" if first_token_ms is not None else "none",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            "true" if success else "false",
            error_type or "none",
        )

    async def _request(
        self,
        messages: list[dict],
        *,
        json_mode: bool,
        max_tokens: int,
        reasoning_effort: Optional[str],
        thinking: dict,
    ) -> ChatResponse:
        if self._client is None:
            raise DeepSeekConfigError("no DeepSeek client configured")

        params: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "extra_body": thinking,
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
