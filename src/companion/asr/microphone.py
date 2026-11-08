"""Microphone capture for the independent real-time ASR demo.

PyAudio/PortAudio is imported lazily so importing the package (and its offline
tests) never requires the optional ``asr`` dependencies.

The capture thread only copies audio into a bounded in-memory queue. It never
waits for the network and never writes audio to disk. If the device cannot run
at 16 kHz, capture uses a rate it natively supports and converts to 16 kHz with
a stateful streaming resampler; the reported rate is always the real one.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import warnings
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Optional

from companion.asr.config import ASRConfig, SAMPLE_RATE, SAMPLE_WIDTH


class MicrophoneError(Exception):
    """Microphone permission, device or stream failure (display-safe)."""


@dataclass(frozen=True)
class AudioDevice:
    """One PortAudio input device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int
    is_default_input: bool = False


@dataclass(frozen=True)
class AudioChunk:
    """Captured PCM already converted to the ASR rate (mono int16)."""

    pcm: bytes
    sample_rate: int
    native_rate: int
    timestamp: float


@dataclass(frozen=True)
class AudioInterrupted:
    """Capture was interrupted; any in-progress utterance must be aborted."""

    reason: str
    timestamp: float


@dataclass(frozen=True)
class AudioNotice:
    """Non-fatal capture status for the separate status line."""

    text: str
    is_error: bool
    timestamp: float


def _load_pyaudio() -> Any:
    try:
        import pyaudio  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MicrophoneError(
            "pyaudio 未安装：请执行 pip install 'ai-companion[asr]'"
        ) from exc
    return pyaudio


def _safe_detail(exc: BaseException) -> str:
    """A short, credential-free detail string for device errors."""
    text = " ".join(str(exc).split())
    lowered = text.lower()
    for marker in ("authorization", "api_key", "bearer", "secret", "token"):
        if marker in lowered:
            return ""
    return text[:160]


def _format_error(exc: BaseException, *, opening: bool = True) -> str:
    """Turn a PortAudio/PyAudio error into a friendly macOS hint."""
    detail = _safe_detail(exc)
    lowered = detail.lower()
    if (
        "-9996" in lowered
        or "invalid input device" in lowered
        or "no default input" in lowered
        or "permission" in lowered
        or "not permitted" in lowered
    ):
        return (
            "无法访问麦克风。请确认设备已连接，并在“系统设置 → 隐私与安全性 → "
            "麦克风”中允许终端或 IDE 访问后重试。"
        )
    if "-9997" in lowered or "invalid sample rate" in lowered:
        return "麦克风不支持所选采样率，请检查输入设备设置。"
    action = "打开" if opening else "读取"
    suffix = f"（{detail}）" if detail else ""
    return f"麦克风{action}失败{suffix}。请检查输入设备和系统权限。"


def _device_from_info(info: dict, *, is_default: bool = False) -> AudioDevice:
    return AudioDevice(
        index=int(info.get("index", -1)),
        name=str(info.get("name", "") or ""),
        max_input_channels=int(info.get("maxInputChannels", 0) or 0),
        max_output_channels=int(info.get("maxOutputChannels", 0) or 0),
        default_sample_rate=int(float(info.get("defaultSampleRate", 0) or 0)),
        is_default_input=is_default,
    )


def _default_input_index(pyaudio: Any) -> Optional[int]:
    getter = getattr(pyaudio, "get_default_input_device_info", None)
    if not callable(getter):
        return None
    try:
        return int(getter().get("index", -1))
    except Exception:  # noqa: BLE001 - no default is a normal state
        return None


def list_input_devices(
    pyaudio_factory: Optional[Callable[[], Any]] = None,
) -> list[AudioDevice]:
    """Enumerate devices that can capture audio; output-only devices are skipped."""
    pyaudio = pyaudio_factory() if pyaudio_factory is not None else _load_pyaudio()
    instance = pyaudio.PyAudio() if pyaudio_factory is None else pyaudio
    try:
        count = int(instance.get_device_count())
        default_index = _default_input_index(instance)
        devices: list[AudioDevice] = []
        for index in range(count):
            try:
                info = instance.get_device_info_by_index(index)
            except Exception:  # noqa: BLE001 - stale device index
                continue
            device = _device_from_info(
                info, is_default=index == default_index
            )
            if device.max_input_channels > 0:
                devices.append(device)
        return devices
    finally:
        terminate = getattr(instance, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:  # noqa: BLE001 - best effort
                pass


def resolve_input_device(
    pyaudio: Any,
    spec: str = "",
    devices: Optional[list[AudioDevice]] = None,
) -> AudioDevice:
    """Resolve an index, a name substring, or the system default input."""
    spec = (spec or "").strip()
    if devices is None:
        devices = _enumerate(pyaudio)
    if spec.isdigit():
        index = int(spec)
        for device in devices:
            if device.index == index:
                return device
        raise MicrophoneError(
            f"设备 {index} 不是可用的音频输入设备。使用 --list-devices 查看。"
        )
    if spec:
        lowered = spec.lower()
        matches = [d for d in devices if lowered in d.name.lower()]
        if not matches:
            raise MicrophoneError(
                f"找不到名称包含“{spec}”的音频输入设备。使用 --list-devices 查看。"
            )
        if len(matches) > 1:
            # Prefer the one macOS marked as default; otherwise the first.
            matches.sort(key=lambda d: (not d.is_default_input, d.index))
        return matches[0]

    for device in devices:
        if device.is_default_input:
            return device
    if not devices:
        raise MicrophoneError("没有找到可用的音频输入设备。")
    # PortAudio did not report a default; use its first input device rather
    # than pretending a specific one was chosen by the system.
    return devices[0]


def _enumerate(pyaudio: Any) -> list[AudioDevice]:
    count = int(pyaudio.get_device_count())
    default_index = _default_input_index(pyaudio)
    devices: list[AudioDevice] = []
    for index in range(count):
        try:
            info = pyaudio.get_device_info_by_index(index)
        except Exception:  # noqa: BLE001
            continue
        device = _device_from_info(info, is_default=index == default_index)
        if device.max_input_channels > 0:
            devices.append(device)
    return devices


# Rates PortAudio devices commonly expose; 16 kHz is always tried first.
_FALLBACK_RATES = (48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000)


def choose_capture_rate(
    pyaudio: Any,
    device: AudioDevice,
    preferred: int = SAMPLE_RATE,
) -> int:
    """Pick a native mono int16 rate; prefer 16 kHz when supported."""
    get_format = getattr(pyaudio, "get_format_from_width", None)
    try:
        sample_format = (
            get_format(SAMPLE_WIDTH)
            if callable(get_format)
            else getattr(pyaudio, "paInt16", 8)
        )
    except Exception:  # noqa: BLE001 - fake/older bindings
        sample_format = getattr(pyaudio, "paInt16", 8)
    is_supported = getattr(pyaudio, "is_format_supported", None)
    candidates = [preferred] + [
        rate for rate in _FALLBACK_RATES if rate != preferred
    ]
    if not callable(is_supported):
        if device.default_sample_rate > 0:
            return device.default_sample_rate
        raise MicrophoneError("无法检测麦克风支持的采样率。")
    for rate in candidates:
        try:
            if is_supported(
                rate,
                input_device=device.index,
                input_channels=1,
                input_format=sample_format,
            ):
                return rate
        except Exception:  # noqa: BLE001 - unsupported rate/device
            continue
    raise MicrophoneError(
        "麦克风不支持 16-bit 单声道采集（已尝试 16/48/44.1 kHz 等常见采样率）。"
    )


class StreamingResampler:
    """Stateful mono int16 resampler; chunks may end anywhere.

    ``audioop.ratecv`` is used when the standard library still provides it
    (Python 3.12 and earlier). On newer interpreters a small linear
    interpolator carries its fractional position across chunks, so no sample
    is duplicated or dropped at a chunk boundary.
    """

    def __init__(self, source_rate: int, target_rate: int = SAMPLE_RATE) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._passthrough = self.source_rate == self.target_rate
        self._pending = b""
        self._audioop: Any = None
        self._state: Any = None
        self._carry: list[int] = []
        self._position = 0.0
        if not self._passthrough:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    import audioop  # type: ignore[import-not-found]

                self._audioop = audioop
            except ImportError:  # pragma: no cover - Python 3.13+
                self._audioop = None

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if self._passthrough:
            return pcm
        if self._pending:
            pcm = self._pending + pcm
        if len(pcm) % SAMPLE_WIDTH:
            self._pending = pcm[-1:]
            pcm = pcm[:-SAMPLE_WIDTH + 1]
        else:
            self._pending = b""
        if not pcm:
            return b""
        if self._audioop is not None:
            out, self._state = self._audioop.ratecv(
                pcm,
                SAMPLE_WIDTH,
                1,
                self.source_rate,
                self.target_rate,
                self._state,
            )
            return out
        return self._linear(pcm)

    def _linear(self, pcm: bytes) -> bytes:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        data = self._carry + list(samples)
        if not data:
            self._carry = []
            return b""
        step = self.source_rate / self.target_rate
        out = array("h")
        position = self._position
        last = len(data) - 1
        while position < last:
            index = int(position)
            fraction = position - index
            value = data[index] + (data[index + 1] - data[index]) * fraction
            value_int = int(round(value))
            if value_int > 32767:
                value_int = 32767
            elif value_int < -32768:
                value_int = -32768
            out.append(value_int)
            position += step
        keep = int(position)
        if keep > len(data):
            keep = len(data)
        self._carry = data[keep:]
        self._position = position - keep
        if sys.byteorder != "little":
            out.byteswap()
        return out.tobytes()


class MicrophoneCapture:
    """One capture thread feeding a bounded queue of 16 kHz PCM chunks."""

    def __init__(
        self,
        config: ASRConfig,
        audio_queue: "queue.Queue[Any]",
        pyaudio_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._config = config
        self._audio_queue = audio_queue
        self._pyaudio_factory = pyaudio_factory
        self._pyaudio: Any = None
        self._own_pyaudio = False
        self._stream: Any = None
        self._resampler: Optional[StreamingResampler] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.device: Optional[AudioDevice] = None
        self.native_rate = 0
        self.resampling = False
        self.dropped_chunks = 0
        self._last_default_check = 0.0
        self._empty_reads = 0

    # ------------------------------------------------------------------ public
    @property
    def description(self) -> str:
        if self.device is None:
            return "未选择设备"
        rate = (
            f"{self.native_rate} Hz → {SAMPLE_RATE} Hz"
            if self.resampling
            else f"{self.native_rate} Hz"
        )
        return f"{self.device.name}（{rate}，单声道 16-bit）"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._pyaudio = self._ensure_pyaudio()
        self._open_stream()
        self._thread = threading.Thread(
            target=self._run, name="asr-microphone", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._close_stream()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._release_pyaudio()

    # ------------------------------------------------------------------ setup
    def _ensure_pyaudio(self) -> Any:
        if self._pyaudio is not None:
            return self._pyaudio
        if self._pyaudio_factory is not None:
            instance = self._pyaudio_factory()
            self._own_pyaudio = False
        else:
            module = _load_pyaudio()
            instance = module.PyAudio()
            self._own_pyaudio = True
        self._pyaudio = instance
        return instance

    def _open_stream(self) -> None:
        pyaudio = self._pyaudio
        if pyaudio is None:
            raise MicrophoneError("麦克风尚未初始化。")
        try:
            devices = _enumerate(pyaudio)
            device = resolve_input_device(pyaudio, self._config.device, devices)
            native_rate = choose_capture_rate(pyaudio, device)
        except MicrophoneError:
            raise
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise MicrophoneError(_format_error(exc)) from exc

        resampler = (
            None
            if native_rate == SAMPLE_RATE
            else StreamingResampler(native_rate, SAMPLE_RATE)
        )
        frames_per_buffer = max(1, native_rate // 50)  # 20 ms
        get_format = getattr(pyaudio, "get_format_from_width", None)
        sample_format = (
            get_format(SAMPLE_WIDTH)
            if callable(get_format)
            else getattr(pyaudio, "paInt16", 8)
        )
        kwargs = {
            "format": sample_format,
            "channels": 1,
            "rate": native_rate,
            "input": True,
            "input_device_index": device.index,
            "frames_per_buffer": frames_per_buffer,
        }
        try:
            stream = pyaudio.open(**kwargs)
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise MicrophoneError(_format_error(exc)) from exc

        self._stream = stream
        self._resampler = resampler
        self.device = device
        self.native_rate = native_rate
        self.resampling = resampler is not None
        self._last_default_check = time.monotonic()
        self._empty_reads = 0

    # ------------------------------------------------------------------- loop
    def _run(self) -> None:
        while not self._stop.is_set():
            stream = self._stream
            if stream is None:
                if not self._recover():
                    break
                continue
            try:
                data = stream.read(
                    max(1, self.native_rate // 50),
                    exception_on_overflow=False,
                )
            except Exception as exc:  # noqa: BLE001 - device lost
                self._notify(
                    f"{_format_error(exc, opening=False)} 正在尝试恢复…",
                    is_error=True,
                )
                self._queue_interrupt("麦克风读取中断，正在尝试恢复。")
                self._close_stream()
                self._backoff()
                continue
            if not data:
                self._empty_reads += 1
                if self._empty_reads > 50:
                    self._queue_interrupt("麦克风长时间没有返回音频，正在重新打开。")
                    self._close_stream()
                    self._backoff()
                continue
            self._empty_reads = 0
            resampler = self._resampler
            try:
                pcm = resampler.process(data) if resampler is not None else data
            except Exception as exc:  # noqa: BLE001 - resampler bug/state
                self._notify(
                    f"音频重采样失败（{type(exc).__name__}），已重新打开设备。",
                    is_error=True,
                )
                self._queue_interrupt("音频重采样失败，正在重新打开设备。")
                self._close_stream()
                self._backoff()
                continue
            if pcm:
                self._put_audio(pcm)
            if self._should_check_default():
                self._check_default_device()

    def _recover(self) -> bool:
        """Re-enumerate and reopen after a device failure; False on stop."""
        while not self._stop.is_set():
            try:
                self._open_stream()
            except MicrophoneError as exc:
                self._notify(f"{exc} 正在重试…", is_error=True)
                self._backoff()
                continue
            self._notify(
                f"麦克风已恢复：{self.description}", is_error=False
            )
            return True
        return False

    def _backoff(self) -> None:
        self._stop.wait(0.5)

    def _should_check_default(self) -> bool:
        if (self._config.device or "").strip():
            return False
        now = time.monotonic()
        if now - self._last_default_check < 1.0:
            return False
        self._last_default_check = now
        return True

    def _check_default_device(self) -> None:
        pyaudio = self._pyaudio
        if pyaudio is None or self.device is None:
            return
        current = _default_input_index(pyaudio)
        if current is None or current == self.device.index:
            return
        self._notify(
            "系统默认输入设备已变化，正在切换…", is_error=False
        )
        self._queue_interrupt("系统默认输入设备已变化。")
        self._close_stream()
        self._backoff()

    # ------------------------------------------------------------------ queue
    def _put_audio(self, pcm: bytes) -> None:
        chunk = AudioChunk(
            pcm=pcm,
            sample_rate=SAMPLE_RATE,
            native_rate=self.native_rate,
            timestamp=time.monotonic(),
        )
        try:
            self._audio_queue.put_nowait(chunk)
            return
        except queue.Full:
            pass
        # Keep listening: drop the oldest queued chunk, never the newest audio.
        try:
            self._audio_queue.get_nowait()
            self.dropped_chunks += 1
        except queue.Empty:
            pass
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            self.dropped_chunks += 1

    def _queue_interrupt(self, reason: str) -> None:
        self._put_control(
            AudioInterrupted(reason=reason, timestamp=time.monotonic())
        )

    def _notify(self, text: str, *, is_error: bool) -> None:
        if self._config.debug or is_error:
            self._put_control(
                AudioNotice(
                    text=text,
                    is_error=is_error,
                    timestamp=time.monotonic(),
                )
            )

    def _put_control(self, item: Any) -> None:
        # Control items are rare and must not be dropped: retry until the VAD
        # worker drains them or shutdown begins.
        while not self._stop.is_set():
            try:
                self._audio_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    # ---------------------------------------------------------------- teardown
    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop_stream()
        except Exception:  # noqa: BLE001 - best effort
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def _release_pyaudio(self) -> None:
        pyaudio, self._pyaudio = self._pyaudio, None
        if pyaudio is None or not self._own_pyaudio:
            return
        terminate = getattr(pyaudio, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:  # noqa: BLE001 - best effort
                pass


__all__ = [
    "AudioChunk",
    "AudioDevice",
    "AudioInterrupted",
    "AudioNotice",
    "MicrophoneCapture",
    "MicrophoneError",
    "StreamingResampler",
    "choose_capture_rate",
    "list_input_devices",
    "resolve_input_device",
]
