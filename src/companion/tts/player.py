"""PyAudio playback for the TTS manager's dedicated player thread.

PyAudio is imported lazily so the package and its tests do not require the
optional ``tts`` extra. One stream (and one PyAudio instance) is reused for all
replies and closed on shutdown. Raw PCM is written directly to the output
device; no temporary WAV file is created.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from companion.tts.config import TTSConfig
from companion.tts.errors import TTSPlayerError


class PyAudioPlayer:
    """Mono 16-bit PCM playback on the Mac's default output device."""

    def __init__(
        self,
        config: TTSConfig,
        pyaudio_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._config = config
        self._pyaudio_factory = pyaudio_factory
        self._pyaudio: Any = None
        self._stream: Any = None
        # Serializes write/stop/close: interrupt() waits for the chunk that is
        # already in flight and then flushes the device buffer.
        self._io_lock = threading.Lock()

    def open(self) -> None:
        """Open the output stream; idempotent."""
        if self._stream is not None:
            return
        if self._pyaudio_factory is not None:
            self._pyaudio = self._pyaudio_factory()
        else:
            try:
                import pyaudio  # type: ignore[import-not-found]
            except ImportError as exc:
                raise TTSPlayerError(
                    "pyaudio 未安装：请执行 pip install 'ai-companion[tts]'"
                ) from exc
            self._pyaudio = pyaudio.PyAudio()
        try:
            self._stream = self._pyaudio.open(
                format=self._pyaudio.paInt16,
                channels=self._config.channels,
                rate=self._config.sample_rate,
                output=True,
            )
        except Exception as exc:  # noqa: BLE001 - sanitized
            self._safe_terminate()
            raise TTSPlayerError(
                f"无法打开音频输出设备: {type(exc).__name__}"
            ) from exc

    def write(self, pcm: bytes) -> None:
        if self._stream is None:
            self.open()
        try:
            with self._io_lock:
                stream = self._stream
                if stream is None:
                    raise TTSPlayerError("音频输出设备已关闭")
                stream.write(pcm)
        except TTSPlayerError:
            raise
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise TTSPlayerError(
                f"音频播放失败: {type(exc).__name__}"
            ) from exc

    def interrupt(self) -> None:
        """Flush buffered audio for the current reply, keeping the stream open."""
        with self._io_lock:
            stream = self._stream
            if stream is None:
                return
            try:
                stream.stop_stream()
            except Exception:  # noqa: BLE001 - best effort
                return
            try:
                stream.start_stream()
            except Exception:  # noqa: BLE001 - reopen lazily on next write
                self._safe_close_stream()

    def close(self) -> None:
        with self._io_lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
        self._safe_terminate()

    def _safe_close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _safe_terminate(self) -> None:
        pyaudio, self._pyaudio = self._pyaudio, None
        if pyaudio is not None:
            try:
                pyaudio.terminate()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["PyAudioPlayer", "TTSPlayerError"]
