"""PyAudio playback for the TTS manager's dedicated player thread.

PyAudio is imported lazily so the package and its tests do not require the
optional ``tts`` extra. One stream (and one PyAudio instance) is reused for all
replies and closed on shutdown. Raw PCM is written directly to the output
device; no temporary WAV file is created.

Output device
-------------
By default the stream follows the system default output device, which includes
Bluetooth headphones/speakers once macOS selects them. ``AVA_TTS_OUTPUT_DEVICE``
can pin a specific PortAudio device by index (``1``) or by name substring
(``WH-1000XM6``). :meth:`refresh` re-opens the stream when the resolved device
changes (for example after connecting Bluetooth), so a long-running REPL does
not stay stuck on the previous speaker.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from companion.tts.config import TTSConfig
from companion.tts.errors import TTSPlayerError


class PyAudioPlayer:
    """Mono 16-bit PCM playback on the default or selected output device."""

    def __init__(
        self,
        config: TTSConfig,
        pyaudio_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._config = config
        self._pyaudio_factory = pyaudio_factory
        self._pyaudio: Any = None
        self._stream: Any = None
        self._device_index: Optional[int] = None
        self._device_name: Optional[str] = None
        # Serializes write/stop/close: interrupt() waits for the chunk that is
        # already in flight and then flushes the device buffer.
        self._io_lock = threading.Lock()

    @property
    def device_name(self) -> str:
        return self._device_name or "system default"

    def open(self) -> None:
        """Open the output stream; idempotent."""
        if self._stream is not None:
            return
        if self._pyaudio is None:
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
            self._device_index, self._device_name = self._resolve_device()
            self._open_stream()
        except TTSPlayerError:
            self._safe_terminate()
            raise
        except Exception as exc:  # noqa: BLE001 - sanitized
            self._safe_terminate()
            raise TTSPlayerError(
                f"无法打开音频输出设备: {type(exc).__name__}"
            ) from exc

    def refresh(self) -> None:
        """Re-open on the newly resolved device (e.g. Bluetooth connected).

        Called before a new reply starts speaking. It never raises: a failed
        refresh leaves the player closed, and the next ``write`` reopens it.
        """
        with self._io_lock:
            if self._pyaudio is None or self._stream is None:
                return
            try:
                index, name = self._resolve_device()
            except Exception:  # noqa: BLE001 - keep the current device
                return
            if index == self._device_index and name == self._device_name:
                return
            self._safe_close_stream()
            self._device_index, self._device_name = index, name
            try:
                self._open_stream()
            except Exception:  # noqa: BLE001 - next write retries
                self._safe_close_stream()

    def _resolve_device(self) -> tuple[Optional[int], Optional[str]]:
        """Resolve the configured or system-default output device."""
        pyaudio = self._pyaudio
        spec = (self._config.output_device or "").strip()
        if spec:
            if spec.isdigit():
                index = int(spec)
                info = pyaudio.get_device_info_by_index(index)
            else:
                info = self._find_device_by_name(spec)
                if info is None:
                    raise TTSPlayerError(
                        f"找不到音频输出设备: {spec}"
                    )
                index = int(info.get("index", -1))
            if int(info.get("maxOutputChannels", 0)) <= 0:
                raise TTSPlayerError(
                    f"设备不是输出设备: {info.get('name', spec)}"
                )
            return index, str(info.get("name", spec))

        getter = getattr(pyaudio, "get_default_output_device_info", None)
        if not callable(getter):
            # Third-party/test doubles without device enumeration: let
            # PortAudio use its own default.
            return None, None
        info = getter()
        return (
            int(info.get("index", -1)),
            str(info.get("name", "")) or None,
        )

    def _find_device_by_name(self, needle: str) -> Optional[dict]:
        pyaudio = self._pyaudio
        count = getattr(pyaudio, "get_device_count", None)
        get_info = getattr(pyaudio, "get_device_info_by_index", None)
        if not callable(count) or not callable(get_info):
            return None
        lowered = needle.lower()
        for index in range(count()):
            info = get_info(index)
            if int(info.get("maxOutputChannels", 0)) <= 0:
                continue
            if lowered in str(info.get("name", "")).lower():
                return info
        return None

    def _open_stream(self) -> None:
        kwargs = {
            # ``paInt16`` is a module-level constant, not an attribute of a
            # PyAudio() instance; the instance method also works with test
            # doubles. Width 2 == 16-bit PCM.
            "format": self._pyaudio.get_format_from_width(
                self._config.sample_width
            ),
            "channels": self._config.channels,
            "rate": self._config.sample_rate,
            "output": True,
        }
        if self._device_index is not None:
            kwargs["output_device_index"] = self._device_index
        self._stream = self._pyaudio.open(**kwargs)

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
