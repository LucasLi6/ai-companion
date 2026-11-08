"""DashScope Qwen streaming ASR session for the independent demo.

Verified against the installed DashScope SDK 2.0.4: the streaming entry point
is ``client.audio.asr.recognize(...)`` returning a duplex ``Recognition``
session with ``start`` / ``send_audio_frame`` / ``stop`` / ``events`` /
``close``. ``recognize_file`` (whole-file recognition) is deliberately not
used. ``intermediate_result_enabled`` and ``max_sentence_silence`` are passed
through the supported ``parameters=`` escape hatch; the wire format and rate
are named arguments.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from companion.asr.config import ASRConfig


class ASRClientError(Exception):
    """ASR connection, send or receive failure (display-safe)."""


@dataclass(frozen=True)
class ASREvent:
    """A parsed ASR event: provisional sentence, confirmed sentence or error."""

    kind: str  # "sentence" | "error"
    text: str = ""
    sentence_id: str = ""
    is_final: bool = False
    error: str = ""


def _safe_detail(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    lowered = text.lower()
    for marker in ("authorization", "api_key", "bearer", "secret", "token"):
        if marker in lowered:
            return ""
    return text[:160]


def _safe_failure(action: str, exc: BaseException) -> str:
    detail = _safe_detail(exc)
    suffix = f"：{detail}" if detail else f"（{type(exc).__name__}）"
    return (
        f"ASR {action}失败{suffix}。请检查网络、DASHSCOPE_API_KEY、"
        "工作空间与模型访问权限。"
    )


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
    """One duplex recognition session; open on the sender thread."""

    def __init__(
        self,
        config: ASRConfig,
        *,
        client_factory: Optional[Callable[[ASRConfig], Any]] = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or _default_client_factory
        self._client: Any = None
        self._recognition: Any = None
        self._lock = threading.Lock()
        self._closed = False
        self.request_id: Optional[str] = None

    @property
    def closed(self) -> bool:
        return self._closed

    def open(self) -> None:
        """Connect and start the streaming task. Called from the sender thread."""
        with self._lock:
            if self._recognition is not None or self._closed:
                return
            try:
                client = self._client_factory(self._config)
                self._client = client
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
                self._recognition = recognition
                recognition.start()
                self.request_id = getattr(recognition, "request_id", None)
            except ASRClientError:
                self._close_locked()
                raise
            except Exception as exc:  # noqa: BLE001 - sanitized
                self._close_locked()
                raise ASRClientError(_safe_failure("连接", exc)) from exc

    def send_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        recognition = self._recognition
        if recognition is None or self._closed:
            raise ASRClientError("ASR 会话尚未建立或已经关闭。")
        try:
            recognition.send_audio_frame(pcm)
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise ASRClientError(_safe_failure("音频发送", exc)) from exc

    def stop(self) -> None:
        """End input; the receiver keeps consuming through task-finished."""
        recognition = self._recognition
        if recognition is None or self._closed:
            return
        try:
            recognition.stop()
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise ASRClientError(_safe_failure("结束输入", exc)) from exc

    def events(self) -> Iterator[ASREvent]:
        recognition = self._recognition
        if recognition is None:
            return
        try:
            for raw in recognition.events():
                parsed = parse_event(raw)
                if parsed is not None:
                    yield parsed
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise ASRClientError(_safe_failure("接收", exc)) from exc

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        self._closed = True
        recognition, self._recognition = self._recognition, None
        client, self._client = self._client, None
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


__all__ = [
    "ASRClientError",
    "ASREvent",
    "ASRSession",
    "parse_event",
]
