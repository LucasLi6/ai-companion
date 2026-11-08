"""DashScope Qwen TTS client.

The SDK is imported lazily so the rest of the companion (and the test suite)
works without the optional ``tts`` extra installed. The client returns raw PCM
chunks only: it never touches Ava's Router, Speaker, Memory or Emotion code.

Secrets are held in memory and never logged. Error messages contain only the
error class / provider status, never user text.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, Callable, Iterator, Optional

from companion.tts.config import TTSConfig
from companion.tts.errors import (
    TTSClientError,
    TTSEmptyAudioError,
    TTSError,
)

# Provider failures are isolated here: the manager converts them into a status
# entry and the text reply is already on screen by then.


def _decode_audio(value: Any) -> bytes:
    """Extract PCM bytes from one response field.

    DashScope may return ``bytes`` or a base64 string depending on SDK version.
    Anything else (URLs, dict wrappers) is ignored rather than guessed at.
    """
    if value is None:
        return b""
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return b""
        try:
            return base64.b64decode(stripped, validate=True)
        except (binascii.Error, ValueError):
            return b""
    return b""


def extract_audio_bytes(response: Any) -> bytes:
    """Pull the audio payload out of one streamed DashScope response.

    Supports the shapes used by ``dashscope`` SDK releases:
    ``response.output.audio`` (bytes/base64), ``response.output.audio.data``,
    ``response.output.audio_data`` and dict equivalents.
    """
    candidates: list[Any] = []
    output = getattr(response, "output", None)
    if output is not None:
        candidates.append(output)
        if isinstance(output, dict):
            candidates.extend(
                output.get(key)
                for key in ("audio", "audio_data", "data")
                if output.get(key) is not None
            )
        else:
            for key in ("audio", "audio_data"):
                value = getattr(output, key, None)
                if value is not None:
                    candidates.append(value)
    if not isinstance(output, dict):
        direct = getattr(response, "audio", None)
        if direct is not None:
            candidates.append(direct)

    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("data", "audio", "bytes"):
                audio = _decode_audio(candidate.get(key))
                if audio:
                    return audio
            continue
        nested = getattr(candidate, "data", None)
        if nested is not None:
            audio = _decode_audio(nested)
            if audio:
                return audio
        audio = _decode_audio(candidate)
        if audio:
            return audio
    return b""


def _raise_for_provider_error(response: Any) -> None:
    status_code = getattr(response, "status_code", None)
    if status_code is None or status_code == 200:
        return
    code = getattr(response, "code", "") or ""
    message = getattr(response, "message", "") or ""
    # Trim provider text; it is not user content, but keep the line short.
    detail = f"{code}: {message}".strip(": ")[:160]
    raise TTSClientError(
        f"DashScope TTS 请求失败 status={status_code}"
        + (f" ({detail})" if detail else "")
    )


class TTSClient:
    """Thin wrapper around ``dashscope.Client(...).audio.tts.stream``.

    ``client_factory`` lets tests inject a fake SDK object without network or
    credentials. No request is sent until the worker actually synthesizes.
    """

    def __init__(
        self,
        config: TTSConfig,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory
        self._client: Any = None

    @property
    def config(self) -> TTSConfig:
        return self._config

    def _provider_parameters(self) -> dict:
        """Provider parameters set once per session.

        Returns an empty dict when ``instruction`` is disabled, so the A/B
        listening test can compare "with vs without instruction".
        """
        if self._config.instruction:
            return {"instruction": self._config.instruction}
        return {}

    def _sdk(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        try:
            import dashscope  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TTSClientError(
                "dashscope 未安装：请执行 pip install 'ai-companion[tts]'"
            ) from exc
        try:
            self._client = dashscope.Client(
                api_key=self._config.api_key,
                workspace=self._config.workspace_id,
                region=self._config.region,
            )
        except Exception as exc:  # noqa: BLE001 - message is sanitized
            raise TTSClientError(
                f"无法初始化 DashScope 客户端: {type(exc).__name__}"
            ) from exc
        return self._client

    def stream_pcm(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks for one complete reply.

        Raises :class:`TTSClientError` on transport/provider failure and
        :class:`TTSEmptyAudioError` when the stream yields no audio at all.
        """
        client = self._sdk()
        method = getattr(getattr(getattr(client, "audio", None), "tts", None), "stream", None)
        if not callable(method):
            raise TTSClientError("DashScope SDK 不支持 audio.tts.stream")

        try:
            responses = method(
                text,
                model=self._config.model,
                voice=self._config.voice,
                format=self._config.audio_format,
                sample_rate=self._config.sample_rate,
                rate=self._config.speech_rate,
                parameters=self._provider_parameters(),
            )
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise TTSClientError(
                f"TTS 请求失败: {type(exc).__name__}"
            ) from exc

        produced = False
        try:
            for response in responses:
                # The official DashScope SDK yields raw PCM bytes directly.
                # Handle that shape before looking for response fields.
                if isinstance(response, (bytes, bytearray, memoryview)):
                    chunk = bytes(response)
                else:
                    _raise_for_provider_error(response)
                    chunk = extract_audio_bytes(response)
                if chunk:
                    produced = True
                    yield chunk
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001 - streaming transport failure
            raise TTSClientError(
                f"TTS 音频流中断: {type(exc).__name__}"
            ) from exc

        if not produced:
            raise TTSEmptyAudioError("TTS 返回空音频")

    def open_stream(self) -> Any:
        """Open one bidirectional TTS session for a single reply (Phase B).

        The caller must run ``send_text`` and ``audio_chunks`` concurrently;
        this method only establishes the session. It never touches the chat
        event loop by itself.

        ``audio.tts.connect`` returns an unstarted ``SynthesisSession`` in the
        DashScope SDK; ``start()`` opens the websocket and waits for the
        task-started handshake, which ``send_text`` requires. This blocks the
        calling (TTS) thread only, never the chat loop.
        """
        client = self._sdk()
        method = getattr(
            getattr(getattr(client, "audio", None), "tts", None),
            "connect",
            None,
        )
        if not callable(method):
            raise TTSClientError("DashScope SDK 不支持 audio.tts.connect")
        try:
            session = method(
                model=self._config.model,
                voice=self._config.voice,
                format=self._config.audio_format,
                sample_rate=self._config.sample_rate,
                rate=self._config.speech_rate,
                parameters=self._provider_parameters(),
            )
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise TTSClientError(
                f"TTS 会话建立失败: {type(exc).__name__}"
            ) from exc

        starter = getattr(session, "start", None)
        if callable(starter):
            try:
                starter()
            except Exception as exc:  # noqa: BLE001 - sanitized
                closer = getattr(session, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # noqa: BLE001 - cleanup must not raise
                        pass
                raise TTSClientError(
                    f"TTS 会话建立失败: {type(exc).__name__}"
                ) from exc
        return session

    def close(self) -> None:
        client, self._client = self._client, None
        closer = getattr(client, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001 - cleanup must not raise
                pass


__all__ = [
    "TTSClient",
    "TTSClientError",
    "TTSEmptyAudioError",
    "TTSError",
    "extract_audio_bytes",
]
