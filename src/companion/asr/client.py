"""DashScope Qwen streaming ASR session for the independent demo.

Verified against the installed DashScope SDK 2.0.4: the streaming entry point
is ``client.audio.asr.recognize(...)`` returning a duplex ``Recognition``
session with ``start`` / ``send_audio_frame`` / ``stop`` / ``events`` /
``close``. ``recognize_file`` (whole-file recognition) is deliberately not
used. ``intermediate_result_enabled`` and ``max_sentence_silence`` are passed
through the supported ``parameters=`` escape hatch; the wire format and rate
are named arguments.

The session owns a small thread-safe lifecycle state machine so the sender and
receiver threads can share one connection without racing each other:

* one re-entrant lock guards the local state; ``send_audio``/``stop`` perform
  the check and the network call under the same lock, closing the
  check-then-send window that produced ``Bad file descriptor`` on a socket the
  receiver had already torn down;
* the receive iterator is *never* held under that lock (a blocked ``recv`` must
  not starve the sender's close path);
* the first unrecoverable error becomes ``primary_error`` and the state
  ``FAILED``; every later symptom (sending into a closed socket, a receiver
  unblocked by our own close) is recorded as ``secondary_error`` and surfaced
  as :class:`ASRSessionStateError` instead of a second failure;
* ``close`` is idempotent and records who requested it via ``close_origin``.

The SDK may close the socket internally inside its own ``recv``/``_send`` error
handlers. We cannot stop that; the state machine makes the *other* thread able
to recognise the closure safely instead of re-reporting it as a new root cause.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from companion.asr.config import ASRConfig

# ------------------------------------------------------------------ session state
STATE_CREATED = "CREATED"
STATE_CONNECTING = "CONNECTING"
STATE_STREAMING = "STREAMING"
STATE_INPUT_FINISHING = "INPUT_FINISHING"
STATE_WAITING_FOR_RESULT = "WAITING_FOR_RESULT"
STATE_TASK_FINISHED = "TASK_FINISHED"
STATE_FAILED = "FAILED"
STATE_CLOSED = "CLOSED"

# --------------------------------------------------------------- close origins
CLOSE_ORIGIN_LOCAL_NORMAL = "local_normal"
CLOSE_ORIGIN_LOCAL_CANCEL = "local_cancel"
CLOSE_ORIGIN_LOCAL_TIMEOUT = "local_timeout"
CLOSE_ORIGIN_SDK_INTERNAL = "sdk_internal"
CLOSE_ORIGIN_EXTERNAL = "external_or_unknown"

# ---------------------------------------------------------------- failure kinds
FAILURE_KIND_AUTH = "auth"
FAILURE_KIND_NETWORK = "network"
FAILURE_KIND_EOF = "eof"
FAILURE_KIND_LOCAL_SOCKET = "local_socket"
FAILURE_KIND_TIMEOUT = "timeout"
FAILURE_KIND_CANCELLED = "cancelled"
FAILURE_KIND_TASK_FAILED = "task_failed"
FAILURE_KIND_UNKNOWN = "unknown"


class ASRClientError(Exception):
    """ASR connection, send or receive failure (display-safe)."""


class ASRSessionStateError(ASRClientError):
    """A send/stop was attempted after the session left its sendable state.

    This is always a *secondary* symptom of an already-terminated session: the
    sender observed a receiver/manager close it first. It must never overwrite
    the primary error, start a new connection, or be reported as a recognition
    failure.
    """

    def __init__(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.origin = origin


class ASRFailure(ASRClientError):
    """A classified, display-safe ASR failure carrying its cause metadata."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = FAILURE_KIND_UNKNOWN,
        origin: str = CLOSE_ORIGIN_EXTERNAL,
        request_id: Optional[str] = None,
        exc_type: str = "",
        code: Any = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.origin = origin
        self.request_id = request_id
        self.exc_type = exc_type
        self.code = code


@dataclass(frozen=True)
class ASREvent:
    """A parsed ASR event: provisional sentence, confirmed sentence or error."""

    kind: str  # "sentence" | "error"
    text: str = ""
    sentence_id: str = ""
    is_final: bool = False
    error: str = ""


def _module_name(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__name__}"


def _safe_detail(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    lowered = text.lower()
    for marker in ("authorization", "api_key", "bearer", "secret", "token"):
        if marker in lowered:
            return ""
    return text[:160]


def _is_auth(exc: BaseException, http_code: Any, code: Any) -> bool:
    """Auth is decided by HTTP/business codes and exception type, not text."""
    if http_code in (401, 403) or str(http_code) in ("401", "403"):
        return True
    if code in (401, 403) or str(code) in ("401", "403"):
        return True
    name = _module_name(exc).lower()
    for marker in (
        "authentication",
        "invalidapikey",
        "unauthorized",
        "forbidden",
        "accessdenied",
        "auth",
    ):
        if marker in name:
            return True
    if isinstance(code, str):
        compact = code.lower().replace("_", "").replace("-", "").replace(" ", "")
        for marker in (
            "invalidapikey",
            "unauthorized",
            "forbidden",
            "authenticationfailed",
            "accessdenied",
        ):
            if marker in compact:
                return True
    # Last-resort hint for the *message*, never for the lifecycle state.
    text = " ".join(str(exc).split()).lower()
    return any(
        marker in text
        for marker in (
            "unauthorized",
            "invalid api key",
            "forbidden",
            "access denied",
        )
    )


def classify_failure(
    exc: BaseException,
    *,
    action: str,
    request_id: Optional[str] = None,
    origin: str = CLOSE_ORIGIN_EXTERNAL,
) -> ASRFailure:
    """Classify an SDK/OS exception into a display-safe :class:`ASRFailure`.

    The decision is driven by exception type, HTTP/business codes and
    ``errno``. Error text is only used as a hint for the message shown to the
    user, never to decide the lifecycle state.
    """
    if isinstance(exc, ASRFailure):
        return exc
    if isinstance(exc, ASRSessionStateError):
        return ASRFailure(
            str(exc),
            kind=FAILURE_KIND_CANCELLED,
            origin=exc.origin or origin,
            request_id=exc.request_id or request_id,
            exc_type=_module_name(exc),
        )
    if isinstance(exc, ASRClientError):
        return ASRFailure(
            str(exc),
            kind=FAILURE_KIND_UNKNOWN,
            origin=origin,
            request_id=request_id,
            exc_type=_module_name(exc),
        )

    exc_type = _module_name(exc)
    rid = (
        request_id
        or getattr(exc, "request_id", None)
        or getattr(exc, "task_id", None)
    )
    http_code = getattr(exc, "http_code", None)
    if http_code is None:
        http_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)

    if _is_auth(exc, http_code, code):
        return ASRFailure(
            "鉴权失败，请检查 API Key、工作空间与模型访问权限。",
            kind=FAILURE_KIND_AUTH,
            origin=origin,
            request_id=rid,
            exc_type=exc_type,
            code=code,
        )
    if isinstance(exc, TimeoutError) or "timeout" in exc_type.lower():
        return ASRFailure(
            "连接或识别超时。",
            kind=FAILURE_KIND_TIMEOUT,
            origin=origin,
            request_id=rid,
            exc_type=exc_type,
            code=code,
        )
    if isinstance(exc, OSError):
        errno = getattr(exc, "errno", None)
        if errno == 9:
            return ASRFailure(
                "本地 socket 已失效（Bad file descriptor）。",
                kind=FAILURE_KIND_LOCAL_SOCKET,
                origin=origin,
                request_id=rid,
                exc_type=exc_type,
                code=errno,
            )
        if errno in (32, 54, 104):
            return ASRFailure(
                "网络连接已中断。",
                kind=FAILURE_KIND_NETWORK,
                origin=origin,
                request_id=rid,
                exc_type=exc_type,
                code=errno,
            )
        return ASRFailure(
            "网络不可用或连接被拒绝。",
            kind=FAILURE_KIND_NETWORK,
            origin=origin,
            request_id=rid,
            exc_type=exc_type,
            code=errno,
        )
    if "dashscope" in exc_type.lower() and (
        code or getattr(exc, "message", None)
    ):
        detail = _safe_detail(exc) or type(exc).__name__
        return ASRFailure(
            f"服务端返回任务失败：{detail}",
            kind=FAILURE_KIND_TASK_FAILED,
            origin=origin,
            request_id=rid,
            exc_type=exc_type,
            code=code,
        )
    name = exc_type.lower()
    if (
        "connectionclosed" in name
        or "apiconnection" in name
        or "connectionerror" in name
        or "websocketconnection" in name
    ):
        return ASRFailure(
            "WebSocket 提前关闭，任务未正常完成（根因需结合生命周期日志确认）。",
            kind=FAILURE_KIND_EOF,
            origin=origin,
            request_id=rid,
            exc_type=exc_type,
            code=code,
        )

    detail = _safe_detail(exc)
    message = f"{action}失败" + (
        f"：{detail}" if detail else f"（{type(exc).__name__}）"
    )
    return ASRFailure(
        message,
        kind=FAILURE_KIND_UNKNOWN,
        origin=origin,
        request_id=rid,
        exc_type=exc_type,
        code=code,
    )


def _event_name(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    header = raw.get("header")
    if not isinstance(header, dict):
        return None
    return str(header.get("event") or "").lower() or None


def _default_client_factory(config: ASRConfig) -> Any:
    try:
        import dashscope  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ASRClientError(
            "dashscope 未安装：请执行 pip install 'ai-companion[asr]'"
        ) from exc
    return dashscope.Client(
        api_key=config.api_key,
        workspace=config.workspace_id,
        region=config.region,
        timeout=config.timeout_seconds,
        max_retries=0,
    )


def parse_event(raw: Any) -> Optional[ASREvent]:
    """Parse ``event["payload"]["output"]["sentence"]`` safely.

    Heartbeats, textless status payloads and non-result events are ignored.
    Both the explicit ``sentence_end`` flag and the older ``end_time`` marker
    are accepted; ``heartbeat`` is always dropped.
    """
    if not isinstance(raw, dict):
        return None
    header = raw.get("header")
    if not isinstance(header, dict):
        header = {}
    name = str(header.get("event") or "").lower()
    if name in ("task-failed", "task-canceled", "task-cancelled"):
        return ASREvent(
            kind="error",
            error="ASR 服务返回任务失败，请检查模型、网络与账户权限。",
        )
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return None
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    sentence = output.get("sentence")
    if not isinstance(sentence, dict):
        return None
    if sentence.get("heartbeat"):
        return None
    text = str(sentence.get("text") or "").strip()
    if not text:
        return None

    end = sentence.get("sentence_end")
    if isinstance(end, str):
        end = end.strip().lower() in ("1", "true", "yes", "end")
    if end is None:
        end = sentence.get("end_time") is not None
    sentence_id = sentence.get("sentence_id", sentence.get("sentenceId", ""))
    return ASREvent(
        kind="sentence",
        text=text,
        sentence_id=str(sentence_id or ""),
        is_final=bool(end),
    )


class ASRSession:
    """One duplex recognition session with a thread-safe lifecycle."""

    def __init__(
        self,
        config: ASRConfig,
        *,
        client_factory: Optional[Callable[[ASRConfig], Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or _default_client_factory
        self._client: Any = None
        self._recognition: Any = None
        self._lock = threading.RLock()
        self._state = STATE_CREATED
        self.request_id: Optional[str] = None
        self.utterance_id: Optional[int] = None
        self.close_origin: Optional[str] = None
        self.primary_error: Optional[str] = None
        self.secondary_error: Optional[str] = None
        self.primary_failure: Optional["ASRFailure"] = None
        self.last_server_event: Optional[str] = None
        self.logger = logger or logging.getLogger("companion.asr.session")
        self._first_audio_sent = False
        self._stop_sent = False

    # ------------------------------------------------------ observable state
    @property
    def closed(self) -> bool:
        with self._lock:
            return self._state == STATE_CLOSED

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._state == STATE_FAILED

    def bind_utterance(self, utterance_id: int) -> None:
        self.utterance_id = int(utterance_id)

    # ----------------------------------------------------------- lifecycle
    def open(self) -> None:
        """Connect and start the streaming task. Called from the sender thread."""
        with self._lock:
            if self._state != STATE_CREATED:
                return
            self._state = STATE_CONNECTING
        self._log("session_connect_start")
        try:
            client = self._client_factory(self._config)
            recognition = client.audio.asr.recognize(
                model=self._config.model,
                format="pcm",
                sample_rate=self._config.sample_rate,
                disfluency_removal_enabled=False,
                parameters={
                    "intermediate_result_enabled": True,
                    "max_sentence_silence": 1500,
                },
                timeout=self._config.timeout_seconds,
            )
            self._client = client
            self._recognition = recognition
            recognition.start()
            self.request_id = (
                getattr(recognition, "request_id", None)
                or getattr(
                    getattr(recognition, "session", None), "request_id", None
                )
            )
        except ASRClientError as exc:
            self._record_failure(ASRFailure(str(exc), exc_type=_module_name(exc)))
            self._log("session_connect_failed")
            raise
        except Exception as exc:  # noqa: BLE001 - sanitized
            failure = classify_failure(exc, action="连接", request_id=self.request_id)
            self._record_failure(failure)
            self._log("session_connect_failed")
            raise failure from exc
        with self._lock:
            if self._state == STATE_CONNECTING:
                self._state = STATE_STREAMING
        self._log("session_connect_success")
        self._log("task_started")

    def send_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            if self._state != STATE_STREAMING:
                raise self._state_error(
                    f"ASR 会话未处于可发送状态（{self._state}）。"
                )
            recognition = self._recognition
            try:
                recognition.send_audio_frame(pcm)
            except ASRClientError as exc:
                self._fail_send(
                    ASRFailure(str(exc), exc_type=_module_name(exc)), exc
                )
            except Exception as exc:  # noqa: BLE001 - sanitized
                self._fail_send(
                    classify_failure(
                        exc,
                        action="音频发送",
                        request_id=self.request_id,
                        origin=CLOSE_ORIGIN_SDK_INTERNAL,
                    ),
                    exc,
                )
        if not self._first_audio_sent:
            self._first_audio_sent = True
            self._log("first_audio_sent")

    def stop(self) -> None:
        """End input; the receiver keeps consuming through task-finished.

        Sending ``finish-task`` is *not* closing: the receiver keeps reading
        until the server confirms the task finished. Calling it twice is a
        no-op, and a call on a failed/closed/unopened session raises
        :class:`ASRSessionStateError` (secondary, never a new root cause).
        """
        with self._lock:
            if self._state in (STATE_FAILED, STATE_CLOSED):
                raise self._state_error(
                    f"ASR 会话未处于可发送状态（{self._state}）。"
                )
            if self._stop_sent:
                return
            if self._state not in (
                STATE_STREAMING,
                STATE_INPUT_FINISHING,
                STATE_WAITING_FOR_RESULT,
                STATE_TASK_FINISHED,
            ):
                raise self._state_error(
                    f"ASR 会话尚未连接（{self._state}）。"
                )
            recognition = self._recognition
            try:
                recognition.stop()
            except ASRClientError as exc:
                self._fail_send(
                    ASRFailure(str(exc), exc_type=_module_name(exc)), exc
                )
            except Exception as exc:  # noqa: BLE001 - sanitized
                self._fail_send(
                    classify_failure(
                        exc,
                        action="结束输入",
                        request_id=self.request_id,
                        origin=CLOSE_ORIGIN_SDK_INTERNAL,
                    ),
                    exc,
                )
            self._stop_sent = True
            if self._state == STATE_STREAMING:
                self._state = STATE_INPUT_FINISHING
        self._log("finish_task_sent")

    def mark_last_audio_sent(self) -> None:
        self._log("last_audio_sent")

    def events(self) -> Iterator[ASREvent]:
        """Iterate parsed events; raise only a *primary* failure.

        When the session was already failed/closed before iteration begins, the
        generator is empty so the receiver never fabricates a clean
        ``task-finished``. A receive error that is merely a symptom of an
        earlier close is surfaced as :class:`ASRSessionStateError`, which the
        caller treats as secondary.
        """
        with self._lock:
            recognition = self._recognition
            state = self._state
        if recognition is None or state in (STATE_CLOSED, STATE_FAILED):
            return
        if state == STATE_INPUT_FINISHING:
            self._set_state(STATE_WAITING_FOR_RESULT)
        try:
            for raw in recognition.events():
                self.last_server_event = _event_name(raw)
                parsed = parse_event(raw)
                if parsed is not None and parsed.kind == "error":
                    self._record_failure(
                        ASRFailure(
                            parsed.error,
                            kind=FAILURE_KIND_TASK_FAILED,
                            origin=CLOSE_ORIGIN_EXTERNAL,
                            request_id=self.request_id,
                        )
                    )
                    self._log("task_failed_received")
                if parsed is not None:
                    yield parsed
            with self._lock:
                if self._state not in (STATE_FAILED, STATE_CLOSED):
                    self._state = STATE_TASK_FINISHED
            self._log("task_finished_received")
        except BaseException as exc:
            if isinstance(exc, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                raise
            failure = classify_failure(exc, action="接收", request_id=self.request_id)
            primary = self._record_failure(failure)
            self._log("socket_read_eof")
            if primary:
                raise failure from exc
            raise ASRSessionStateError(
                str(failure),
                request_id=self.request_id,
                origin=self.close_origin,
            ) from exc

    def close(self, origin: str = CLOSE_ORIGIN_LOCAL_NORMAL) -> None:
        """Idempotently release the connection, recording who requested it."""
        with self._lock:
            if self._state == STATE_CLOSED:
                return
            if self.close_origin is None:
                self.close_origin = origin
            self._state = STATE_CLOSED
            recognition, self._recognition = self._recognition, None
            client, self._client = self._client, None
        self._log("close_requested")
        if recognition is not None:
            try:
                recognition.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        self._log("session_closed")

    # ---------------------------------------------------------- internals
    def _set_state(self, state: str) -> None:
        with self._lock:
            if self._state in (STATE_FAILED, STATE_CLOSED):
                return
            self._state = state

    def _fail_send(self, failure: ASRFailure, exc: BaseException) -> None:
        """Record a send/stop failure; raise secondary when already terminal.

        The first failure on a live session is primary. A send error that is
        only a symptom of an earlier receiver/manager close (the SDK closed the
        socket itself) must never overwrite the primary error or start a new
        connection: it is surfaced as :class:`ASRSessionStateError`.
        """
        primary = self._record_failure(failure)
        self._log("socket_send_failed")
        if primary:
            raise failure from exc
        raise ASRSessionStateError(
            str(failure),
            request_id=self.request_id,
            origin=self.close_origin,
        ) from exc

    def _record_failure(
        self, failure: ASRFailure, *, secondary: bool = False
    ) -> bool:
        """Store *failure*; return True only if it became the primary error."""
        with self._lock:
            if secondary or self._state in (STATE_FAILED, STATE_CLOSED):
                self.secondary_error = str(failure)
                return False
            self._state = STATE_FAILED
            self.primary_error = str(failure)
            self.primary_failure = failure
            if self.close_origin is None:
                self.close_origin = failure.origin
            return True

    def _record_secondary(self, message: str) -> None:
        with self._lock:
            self.secondary_error = message

    def _state_error(self, message: str) -> ASRSessionStateError:
        error = ASRSessionStateError(
            message,
            request_id=self.request_id,
            origin=self.close_origin,
        )
        self._record_secondary(message)
        self._log("secondary_state_error")
        return error

    def _log(self, event_type: str) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        with self._lock:
            state = self._state
            request_id = self.request_id
            origin = self.close_origin
            primary = self.primary_error
            secondary = self.secondary_error
            last_event = self.last_server_event
            failure = self.primary_failure
        self.logger.debug(
            "asr_lifecycle utterance_id=%s request_id=%s thread_name=%s"
            " timestamp_monotonic=%.3f session_state=%s event_type=%s"
            " last_server_event=%s close_origin=%s primary_error=%s"
            " secondary_error=%s error_type=%s error_code=%s",
            self.utterance_id,
            request_id or "-",
            threading.current_thread().name,
            time.monotonic(),
            state,
            event_type,
            last_event or "-",
            origin or "-",
            _one_line(primary),
            _one_line(secondary),
            getattr(failure, "exc_type", "") or "-",
            getattr(failure, "code", "") if getattr(failure, "code", None) is not None else "-",
        )


def _one_line(text: Optional[str]) -> str:
    return " ".join(str(text or "").split()) or "-"


__all__ = [
    "ASRClientError",
    "ASREvent",
    "ASRFailure",
    "ASRSession",
    "ASRSessionStateError",
    "CLOSE_ORIGIN_EXTERNAL",
    "CLOSE_ORIGIN_LOCAL_CANCEL",
    "CLOSE_ORIGIN_LOCAL_NORMAL",
    "CLOSE_ORIGIN_LOCAL_TIMEOUT",
    "CLOSE_ORIGIN_SDK_INTERNAL",
    "FAILURE_KIND_AUTH",
    "FAILURE_KIND_CANCELLED",
    "FAILURE_KIND_EOF",
    "FAILURE_KIND_LOCAL_SOCKET",
    "FAILURE_KIND_NETWORK",
    "FAILURE_KIND_TASK_FAILED",
    "FAILURE_KIND_TIMEOUT",
    "FAILURE_KIND_UNKNOWN",
    "STATE_CLOSED",
    "STATE_CONNECTING",
    "STATE_CREATED",
    "STATE_FAILED",
    "STATE_INPUT_FINISHING",
    "STATE_STREAMING",
    "STATE_TASK_FINISHED",
    "STATE_WAITING_FOR_RESULT",
    "classify_failure",
    "parse_event",
]
