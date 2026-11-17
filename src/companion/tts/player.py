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

Stream health
-------------
``_stream is not None`` does **not** mean the PortAudio stream can play: after
``interrupt()`` (or a device switch) the object can exist while the stream is
stopped, or ``start_stream()`` can fail and leave a dead handle behind. Every
write therefore passes through :meth:`_ensure_playable_stream_locked`, which
verifies the real PortAudio state, restarts a stopped-but-recoverable stream and
otherwise rebuilds it. A stream that cannot be rebuilt raises
:class:`TTSPlayerError` instead of silently dropping audio.

``stream.write()`` proves only that PortAudio accepted the PCM. It says nothing
about whether the speaker physically produced sound; diagnostics must not treat
a successful write as proof of audible output.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from companion.tts.config import TTSConfig
from companion.tts.errors import TTSPlayerError

# Display-safe failure codes stored in ``last_error`` and reported by
# ``/tts status``. They contain no user text and no credentials.
OUTPUT_DEVICE_OPEN_FAILED = "OUTPUT_DEVICE_OPEN_FAILED"
OUTPUT_STREAM_INACTIVE = "OUTPUT_STREAM_INACTIVE"
OUTPUT_STREAM_RESTART_FAILED = "OUTPUT_STREAM_RESTART_FAILED"
OUTPUT_STREAM_STOP_FAILED = "OUTPUT_STREAM_STOP_FAILED"
OUTPUT_WRITE_FAILED = "OUTPUT_WRITE_FAILED"


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
        # Serializes open/write/interrupt/refresh/close: interrupt() waits for
        # the chunk already in flight, and no close can race a device write.
        # RLock: the helper methods below are called both directly and from
        # each other while the lock is already held.
        self._io_lock = threading.RLock()
        # Set when the last open/start attempt failed, so the next write knows
        # the device must be rebuilt instead of trusting the old handle.
        self._recovery_required = False
        self._last_error: Optional[str] = None

    @property
    def device_name(self) -> str:
        return self._device_name or "system default"

    @property
    def last_error(self) -> Optional[str]:
        """Last device/write failure code (display-safe), if any."""
        return self._last_error

    def open(self) -> None:
        """Open the output stream; idempotent."""
        with self._io_lock:
            try:
                # Also repairs an existing-but-stopped stream: open() must
                # never report success while the handle cannot play.
                self._ensure_pyaudio_locked()
                self._ensure_playable_stream_locked()
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
        refresh leaves the player recoverable, and the next ``write`` reopens
        it. If the device did not change the stream is never rebuilt — only a
        stream that is genuinely not playable is repaired.
        """
        with self._io_lock:
            if self._pyaudio is None or self._stream is None:
                return
            try:
                index, name = self._resolve_device()
            except Exception:  # noqa: BLE001 - keep the current device
                self._last_error = OUTPUT_DEVICE_OPEN_FAILED
                return
            if index == self._device_index and name == self._device_name:
                try:
                    self._ensure_playable_stream_locked()
                except TTSPlayerError:
                    self._last_error = (
                        self._last_error or OUTPUT_DEVICE_OPEN_FAILED
                    )
                return
            self._safe_close_stream()
            self._device_index, self._device_name = index, name
            try:
                self._open_stream_locked()
                self._recovery_required = False
            except Exception:  # noqa: BLE001 - next write retries
                self._safe_close_stream()
                self._recovery_required = True
                self._last_error = OUTPUT_DEVICE_OPEN_FAILED

    # ----------------------------------------------------------- device setup
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

    def _ensure_pyaudio_locked(self) -> None:
        if self._pyaudio is not None:
            return
        if self._pyaudio_factory is not None:
            self._pyaudio = self._pyaudio_factory()
            return
        try:
            import pyaudio  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TTSPlayerError(
                "pyaudio 未安装：请执行 pip install 'ai-companion[tts]'"
            ) from exc
        self._pyaudio = pyaudio.PyAudio()

    def _open_stream_locked(self) -> None:
        """Resolve the device and open a new PortAudio stream."""
        self._device_index, self._device_name = self._resolve_device()
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

    # ------------------------------------------------------ stream life cycle
    @staticmethod
    def _stream_is_active(stream: Any) -> bool:
        """Real PortAudio state when the stream exposes it.

        Test doubles (and older SDK wrappers) may not implement
        ``is_active()``; those are assumed playable rather than forced through
        a rebuild, which keeps fault injection explicit.
        """
        checker = getattr(stream, "is_active", None)
        if not callable(checker):
            return True
        try:
            return bool(checker())
        except Exception:  # noqa: BLE001 - unqueryable means unusable
            return False

    def _ensure_playable_stream_locked(self) -> None:
        """Guarantee ``_stream`` can accept PCM, rebuilding it if necessary.

        Raises :class:`TTSPlayerError` when no playable stream can be
        established. The caller holds ``_io_lock``.
        """
        self._ensure_pyaudio_locked()
        stream = self._stream
        if stream is not None and self._stream_is_active(stream):
            self._recovery_required = False
            return
        if stream is not None:
            # A handle exists but PortAudio says it is not running: try to
            # restart it before paying for a full rebuild.
            starter = getattr(stream, "start_stream", None)
            if callable(starter):
                try:
                    starter()
                    if self._stream_is_active(stream):
                        self._recovery_required = False
                        return
                except Exception as exc:  # noqa: BLE001 - rebuild below
                    self._last_error = (
                        f"{OUTPUT_STREAM_RESTART_FAILED}: {type(exc).__name__}"
                    )
            self._last_error = self._last_error or OUTPUT_STREAM_INACTIVE
            self._safe_close_stream()
        try:
            self._open_stream_locked()
        except TTSPlayerError:
            self._recovery_required = True
            self._last_error = OUTPUT_DEVICE_OPEN_FAILED
            raise
        except Exception as exc:  # noqa: BLE001 - sanitized
            self._recovery_required = True
            self._last_error = f"{OUTPUT_DEVICE_OPEN_FAILED}: {type(exc).__name__}"
            raise TTSPlayerError(
                f"无法打开音频输出设备: {type(exc).__name__}"
            ) from exc
        self._recovery_required = False

    def write(self, pcm: bytes) -> None:
        with self._io_lock:
            self._ensure_playable_stream_locked()
            stream = self._stream
            if stream is None:
                raise TTSPlayerError("音频输出设备已关闭")
            try:
                stream.write(pcm)
            except Exception as exc:  # noqa: BLE001 - sanitized
                # The handle accepted no data; do not pretend it still works.
                self._safe_close_stream()
                self._recovery_required = True
                self._last_error = f"{OUTPUT_WRITE_FAILED}: {type(exc).__name__}"
                raise TTSPlayerError(
                    f"音频播放失败: {type(exc).__name__}"
                ) from exc

    def interrupt(self) -> None:
        """Flush buffered audio for the current reply, keeping the stream open."""
        with self._io_lock:
            stream = self._stream
            if stream is None:
                return
            if not self._stream_is_active(stream):
                # Nothing is buffered in a stopped stream; make sure the next
                # write rebuilds instead of trusting this dead handle.
                self._safe_close_stream()
                self._recovery_required = True
                return
            try:
                stream.stop_stream()
            except Exception as exc:  # noqa: BLE001 - stream unusable
                self._last_error = (
                    f"{OUTPUT_STREAM_STOP_FAILED}: {type(exc).__name__}"
                )
                self._safe_close_stream()
                self._recovery_required = True
                return
            try:
                stream.start_stream()
            except Exception as exc:  # noqa: BLE001 - reopen lazily on next write
                self._last_error = (
                    f"{OUTPUT_STREAM_RESTART_FAILED}: {type(exc).__name__}"
                )
                # Never keep a stream that failed to restart: it looks open
                # but cannot play.
                self._safe_close_stream()
                self._recovery_required = True

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
            self._recovery_required = False
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
