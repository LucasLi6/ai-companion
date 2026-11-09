"""Optional WebRTC AEC3 channel for real-time barge-in.

This module is deliberately additive: it never replaces the microphone,
Silero VAD, DashScope ASR or Qwen TTS. The TTS player hands it the PCM it is
about to write (the far-end reference); the ASR VAD thread hands it captured
microphone PCM (the near end). The pipeline resamples the 24 kHz TTS stream to
16 kHz, keeps a short time-aligned reference buffer, and returns echo-cancelled
16 kHz PCM. Only when that returned block is valid may the ASR side run VAD on
it; raw, un-cancelled microphone audio is never used for a barge-in decision.

Threading
---------
* :meth:`push_playback` runs on the TTS player thread and only appends to a
  tiny bounded deque under a short lock. It never resamples, never calls AEC
  and never blocks on the audio device.
* :meth:`process_microphone` runs on the ASR VAD thread and owns the canceller
  and the streaming resampler, so those objects are never touched by two
  threads at once.
* :meth:`start`/:meth:`stop`/:meth:`reset` are called from the asyncio loop;
  they take the processing lock for at most one 20 ms block.

If ``pywebrtc-audio`` is missing, cannot be constructed, or the canceller
raises, the pipeline reports itself unavailable/degraded and every caller falls
back to the existing half-duplex path. Nothing here raises into playback or
recognition.
"""

from __future__ import annotations

import sys
import threading
import time
from array import array
from collections import Counter, deque
from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable, Deque, Optional, Tuple

from companion.asr.microphone import StreamingResampler

AEC_SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(AEC_SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2

# Reference frames are only useful for about one second of playback.
DEFAULT_REFERENCE_SECONDS = 1.0
DEFAULT_PLAYBACK_QUEUE_FRAMES = 64
DEFAULT_SYNC_TOLERANCE_SECONDS = 0.10
DEFAULT_FALLBACK_AFTER_INVALID_SECONDS = 2.0
_MAX_METRIC_SAMPLES = 256


class EchoCancellerUnavailable(RuntimeError):
    """The optional WebRTC AEC3 dependency is missing or unusable."""


@dataclass(frozen=True)
class MicProcessingResult:
    """One processed microphone chunk.

    ``valid`` is true only when a matching far-end reference existed for every
    block in the chunk and the canceller returned a well-formed frame. Callers
    must not run VAD (or make a barge-in decision) on an invalid result.
    """

    pcm: bytes = b""
    valid: bool = False
    reason: str = ""
    generation: Optional[int] = None
    near_rms: float = 0.0
    clean_rms: float = 0.0


def _default_canceller_factory(
    sample_rate: int, num_channels: int, stream_delay_ms: int
) -> Any:
    try:
        from pywebrtc_audio import EchoCanceller  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise EchoCancellerUnavailable(
            "pywebrtc-audio 未安装：请执行 pip install 'ai-companion[bargein]'"
        ) from exc
    try:
        return EchoCanceller(
            sample_rate=sample_rate,
            num_channels=num_channels,
            stream_delay_ms=stream_delay_ms,
        )
    except TypeError:
        # Older/newer builds may only accept positional arguments.
        return EchoCanceller(sample_rate, num_channels, stream_delay_ms)


def _pcm16(value: Any) -> bytes:
    """Normalise a canceller result (bytes or int16 array) to little-endian PCM."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if hasattr(value, "tobytes"):
        dtype = str(getattr(value, "dtype", "") or "")
        if dtype in ("int16", "<i2", ">i2", "=i2"):
            data = value.tobytes()
            if sys.byteorder != "little" and dtype in (">i2", "=i2"):
                samples = array("h")
                samples.frombytes(data)
                samples.byteswap()
                return samples.tobytes()
            return data
        try:
            import numpy as np  # type: ignore[import-not-found]

            array_value = np.asarray(value)
            if array_value.dtype.kind == "f":
                array_value = (
                    np.clip(array_value, -1.0, 1.0) * 32767.0
                ).astype("<i2")
            else:
                array_value = array_value.astype("<i2")
            return array_value.tobytes()
        except Exception:  # noqa: BLE001 - unknown result type
            return b""
    return b""


def _rms(pcm: bytes) -> float:
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[:usable])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    return sqrt(total / len(samples))


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 2)


class EchoCancellationPipeline:
    """Stateful far/near alignment plus one WebRTC AEC3 instance.

    Parameters are intentionally few; alignment relies on AEC3's own internal
    delay estimation, with ``stream_delay_ms`` as a coarse manual offset.
    """

    def __init__(
        self,
        *,
        sample_rate: int = AEC_SAMPLE_RATE,
        playback_sample_rate: int = 24000,
        stream_delay_ms: int = 0,
        frame_ms: int = FRAME_MS,
        reference_seconds: float = DEFAULT_REFERENCE_SECONDS,
        playback_queue_frames: int = DEFAULT_PLAYBACK_QUEUE_FRAMES,
        sync_tolerance_seconds: float = DEFAULT_SYNC_TOLERANCE_SECONDS,
        fallback_after_invalid_seconds: float = (
            DEFAULT_FALLBACK_AFTER_INVALID_SECONDS
        ),
        canceller_factory: Optional[Callable[[int, int, int], Any]] = None,
    ) -> None:
        if sample_rate <= 0 or playback_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.sample_rate = int(sample_rate)
        self.playback_sample_rate = int(playback_sample_rate)
        self.stream_delay_ms = max(0, int(stream_delay_ms))
        self.frame_ms = max(1, int(frame_ms))
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.frame_seconds = self.frame_samples / self.sample_rate
        self.sync_tolerance_seconds = max(
            0.0, float(sync_tolerance_seconds)
        )
        self._fallback_after_invalid_seconds = max(
            0.1, float(fallback_after_invalid_seconds)
        )
        self._factory = canceller_factory or _default_canceller_factory

        # Only process_microphone/start/stop/reset touch these.
        self._audio_lock = threading.Lock()
        # Only push_playback touches the queue; stats use their own lock.
        self._queue_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._playback: Deque[Tuple[int, float, bytes]] = deque(
            maxlen=max(2, int(playback_queue_frames))
        )
        self._reference: Deque[Tuple[int, float, bytes]] = deque(
            maxlen=max(2, int(reference_seconds / self.frame_seconds))
        )
        self._generation: Optional[int] = None
        self._active = False
        self._canceller: Any = None
        self._factory_ok = False
        self._failed = False
        self._degraded = False
        self._unavailable_reason = ""
        self._last_error = ""
        self._resampler = StreamingResampler(
            self.playback_sample_rate, self.sample_rate
        )
        self._ref_carry = bytearray()
        self._ref_carry_ts = 0.0
        self._ref_carry_generation: Optional[int] = None
        self._mic_buffer = bytearray()
        self._mic_buffer_ts = 0.0
        self._last_valid_at: Optional[float] = None
        self._invalid_run_started: Optional[float] = None
        self._consecutive_invalid_seconds = 0.0

        self._playback_frames = 0
        self._playback_dropped = 0
        self._mic_blocks = 0
        self._valid_blocks = 0
        self._invalid_blocks = 0
        self._invalid_reasons: Counter[str] = Counter()
        self._resample_errors = 0
        self._aec_errors = 0
        self._sync_discontinuities = 0
        self._aec_ms: Deque[float] = deque(maxlen=_MAX_METRIC_SAMPLES)
        self._near_rms: Deque[float] = deque(maxlen=_MAX_METRIC_SAMPLES)
        self._clean_rms: Deque[float] = deque(maxlen=_MAX_METRIC_SAMPLES)

        self._build_canceller()

    # ------------------------------------------------------------- capability
    def _build_canceller(self) -> bool:
        try:
            self._canceller = self._factory(
                self.sample_rate, 1, self.stream_delay_ms
            )
            self._factory_ok = True
            self._unavailable_reason = ""
            return True
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._canceller = None
            self._factory_ok = False
            self._unavailable_reason = str(exc) or type(exc).__name__
            return False

    @property
    def available(self) -> bool:
        """The optional canceller was constructed successfully."""
        return self._factory_ok

    @property
    def healthy(self) -> bool:
        """AEC is usable right now (not failed and not persistently invalid)."""
        return self._factory_ok and not self._failed and not self._degraded

    @property
    def degraded(self) -> bool:
        return self._degraded or self._failed

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason or self._last_error

    # --------------------------------------------------------------- lifecycle
    def start(self, generation: int) -> bool:
        """Arm the pipeline for one TTS generation and reset alignment state."""
        with self._audio_lock:
            if not self._build_canceller():
                return False
            self._generation = int(generation)
            self._active = True
            self._failed = False
            self._degraded = False
            self._last_error = ""
            with self._queue_lock:
                self._playback.clear()
            self._reference.clear()
            self._ref_carry = bytearray()
            self._ref_carry_ts = 0.0
            self._ref_carry_generation = None
            self._mic_buffer = bytearray()
            self._mic_buffer_ts = 0.0
            self._last_valid_at = None
            self._invalid_run_started = None
            self._consecutive_invalid_seconds = 0.0
            self._resampler = StreamingResampler(
                self.playback_sample_rate, self.sample_rate
            )
        return True

    def stop(self, generation: Optional[int] = None) -> None:
        """Stop accepting new playback for a generation.

        Reference frames already received are kept so the echo tail of the
        interrupted reply can still be cancelled while the user's first words
        are being recognized.
        """
        with self._audio_lock:
            if generation is not None and self._generation != int(generation):
                return
            self._active = False

    def reset(self) -> None:
        """Drop all state; the next :meth:`start` rebuilds the canceller."""
        with self._audio_lock:
            self._generation = None
            self._active = False
            with self._queue_lock:
                self._playback.clear()
            self._reference.clear()
            self._ref_carry = bytearray()
            self._ref_carry_generation = None
            self._mic_buffer = bytearray()
            self._last_valid_at = None
            self._invalid_run_started = None
            self._consecutive_invalid_seconds = 0.0
            self._degraded = False
            self._failed = False
            self._resampler = StreamingResampler(
                self.playback_sample_rate, self.sample_rate
            )
            self._build_canceller()

    # ---------------------------------------------------------------- far end
    def push_playback(
        self, pcm: bytes, timestamp: float, generation: int
    ) -> bool:
        """Enqueue one playback chunk from the TTS player thread.

        Fast path only: validate, append, return. Never resamples or calls
        AEC. A full queue drops the oldest frame instead of blocking playback.
        """
        if not pcm:
            return False
        try:
            data = bytes(pcm)
        except Exception:  # noqa: BLE001 - defensive
            return False
        with self._audio_lock:
            current = self._generation
            active = self._active
        if current is None or not active or int(generation) != current:
            return False
        with self._queue_lock:
            if len(self._playback) >= (self._playback.maxlen or 0):
                self._playback.popleft()
                with self._stats_lock:
                    self._playback_dropped += 1
            self._playback.append((int(generation), float(timestamp), data))
        with self._stats_lock:
            self._playback_frames += 1
        return True

    # ---------------------------------------------------------------- near end
    def process_microphone(
        self, pcm: bytes, timestamp: float
    ) -> MicProcessingResult:
        """Return echo-cancelled 16 kHz PCM for one captured chunk.

        The chunk is split into 20 ms blocks. If any block lacks a well-formed,
        time-aligned reference for the current generation, the whole chunk is
        reported invalid and its PCM is discarded. The caller must then skip
        VAD for this audio rather than fall back to the raw microphone.
        """
        if not pcm:
            return MicProcessingResult()
        with self._audio_lock:
            generation = self._generation
            if not self._factory_ok or self._canceller is None:
                self._count_invalid_block("unavailable", 0.0, 0.0)
                return MicProcessingResult(
                    valid=False,
                    reason="unavailable",
                    generation=generation,
                )

            if not self._mic_buffer:
                self._mic_buffer_ts = float(timestamp)
            self._mic_buffer.extend(pcm)
            blocks: list[Tuple[bytes, float]] = []
            while len(self._mic_buffer) >= self.frame_bytes:
                block = bytes(self._mic_buffer[: self.frame_bytes])
                del self._mic_buffer[: self.frame_bytes]
                blocks.append((block, self._mic_buffer_ts))
                self._mic_buffer_ts += self.frame_seconds
            if not blocks:
                return MicProcessingResult(
                    valid=True,
                    reason="buffering",
                    generation=generation,
                )

            # Drain playback once; the reference is shared by these blocks.
            self._drain_playback_locked()

            cleaned: list[bytes] = []
            near_values: list[float] = []
            clean_values: list[float] = []
            failure: Optional[str] = None
            for block, block_ts in blocks:
                reference, reason = self._reference_for_locked(block_ts)
                if reference is None:
                    failure = failure or reason
                    self._count_invalid_block(reason, 0.0, 0.0)
                    continue
                near_level = _rms(block)
                started = time.monotonic()
                try:
                    result = self._canceller.process(block, reference)
                except Exception as exc:  # noqa: BLE001 - optional engine
                    self._aec_error_locked(exc)
                    failure = failure or "aec_error"
                    self._count_invalid_block("aec_error", near_level, 0.0)
                    continue
                clean = _pcm16(result)
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if len(clean) != self.frame_bytes:
                    self._last_error = "aec_length_mismatch"
                    failure = failure or "aec_length"
                    self._count_invalid_block(
                        "aec_length", near_level, 0.0
                    )
                    continue
                clean_level = _rms(clean)
                cleaned.append(clean)
                near_values.append(near_level)
                clean_values.append(clean_level)
                self._valid_blocks += 1
                with self._stats_lock:
                    self._aec_ms.append(elapsed_ms)
                    self._near_rms.append(near_level)
                    self._clean_rms.append(clean_level)
                self._last_valid_at = time.monotonic()
                self._invalid_run_started = None
                self._consecutive_invalid_seconds = 0.0

            if failure is not None:
                # Conservative: never let a partially un-cancelled chunk reach
                # VAD as if it were clean.
                return MicProcessingResult(
                    valid=False,
                    reason=failure,
                    generation=generation,
                    near_rms=_mean(near_values),
                    clean_rms=_mean(clean_values),
                )
            self._mic_blocks += len(blocks)
            return MicProcessingResult(
                pcm=b"".join(cleaned),
                valid=True,
                reason="ok",
                generation=generation,
                near_rms=_mean(near_values),
                clean_rms=_mean(clean_values),
            )

    # -------------------------------------------------------------- internals
    def _drain_playback_locked(self) -> None:
        with self._queue_lock:
            if not self._playback:
                return
            pending = list(self._playback)
            self._playback.clear()
        for generation, timestamp, raw in pending:
            if generation != self._generation:
                continue
            try:
                resampled = self._resampler.process(raw)
            except Exception as exc:  # noqa: BLE001 - resampler state/bug
                with self._stats_lock:
                    self._resample_errors += 1
                self._last_error = f"resample:{type(exc).__name__}"
                continue
            if not resampled:
                continue
            if (
                self._ref_carry_generation is not None
                and self._ref_carry_generation != generation
            ):
                self._ref_carry = bytearray()
                self._ref_carry_generation = None
                self._sync_discontinuities += 1
            if (
                self._ref_carry
                and abs(timestamp - self._ref_carry_ts) > 0.10
            ):
                # Device switch or a gap in playback: restart the reference
                # timeline instead of pretending the old samples are adjacent.
                self._ref_carry = bytearray()
                self._sync_discontinuities += 1
            if not self._ref_carry:
                self._ref_carry_ts = float(timestamp)
                self._ref_carry_generation = generation
            self._ref_carry.extend(resampled)
            while len(self._ref_carry) >= self.frame_bytes:
                frame = bytes(self._ref_carry[: self.frame_bytes])
                del self._ref_carry[: self.frame_bytes]
                self._reference.append(
                    (generation, self._ref_carry_ts, frame)
                )
                self._ref_carry_ts += self.frame_seconds

    def _reference_for_locked(
        self, timestamp: float
    ) -> Tuple[Optional[bytes], str]:
        if self._generation is None:
            return None, "inactive"
        if not self._reference:
            self._note_invalid_time()
            return None, "no_reference"
        target = timestamp - (self.stream_delay_ms / 1000.0)
        best = min(self._reference, key=lambda item: abs(item[1] - target))
        reference_generation, reference_ts, reference_pcm = best
        if reference_generation != self._generation:
            self._note_invalid_time()
            return None, "generation_mismatch"
        if abs(reference_ts - target) > self.sync_tolerance_seconds:
            self._note_invalid_time()
            return None, (
                "stale_reference" if reference_ts < target else "ahead_reference"
            )
        if len(reference_pcm) != self.frame_bytes:
            self._note_invalid_time()
            return None, "bad_reference"
        return reference_pcm, ""

    def _note_invalid_time(self) -> None:
        now = time.monotonic()
        if self._invalid_run_started is None:
            self._invalid_run_started = now
        self._consecutive_invalid_seconds = max(
            0.0, now - self._invalid_run_started
        )
        # Only declare persistent failure after playback actually produced a
        # reference at least once; a slow TTS handshake is not a failure.
        if (
            self._playback_frames > 0
            and self._consecutive_invalid_seconds
            >= self._fallback_after_invalid_seconds
            and not self._degraded
        ):
            self._degraded = True
            self._last_error = "reference_missing"

    def _count_invalid_block(
        self, reason: str, near_rms: float, clean_rms: float
    ) -> None:
        self._invalid_blocks += 1
        with self._stats_lock:
            self._invalid_reasons[reason] += 1

    def _aec_error_locked(self, exc: Exception) -> None:
        with self._stats_lock:
            self._aec_errors += 1
        self._last_error = f"aec:{type(exc).__name__}"
        if self._aec_errors >= 3:
            self._failed = True

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        with self._audio_lock:
            reference_frames = len(self._reference)
            pending_playback = len(self._playback)
            generation = self._generation
            active = self._active
            with self._stats_lock:
                reasons = dict(self._invalid_reasons)
                aec_ms = list(self._aec_ms)
                near = list(self._near_rms)
                clean = list(self._clean_rms)
                playback_frames = self._playback_frames
                playback_dropped = self._playback_dropped
                mic_blocks = self._mic_blocks
                valid_blocks = self._valid_blocks
                invalid_blocks = self._invalid_blocks
                resample_errors = self._resample_errors
                aec_errors = self._aec_errors
                discontinuities = self._sync_discontinuities
        total = valid_blocks + invalid_blocks
        return {
            "available": self._factory_ok,
            "healthy": self.healthy,
            "degraded": self.degraded,
            "reason": self.unavailable_reason or None,
            "active": active,
            "generation": generation,
            "playback_frames": playback_frames,
            "playback_dropped": playback_dropped,
            "pending_playback_frames": pending_playback,
            "reference_frames": reference_frames,
            "mic_blocks": mic_blocks + invalid_blocks,
            "valid_blocks": valid_blocks,
            "invalid_blocks": invalid_blocks,
            "valid_ratio": (
                round(valid_blocks / total, 4) if total else 0.0
            ),
            "invalid_reasons": reasons,
            "aec_p50_ms": _percentile(aec_ms, 0.50),
            "aec_p95_ms": _percentile(aec_ms, 0.95),
            "near_rms_p50": _percentile(near, 0.50),
            "clean_rms_p50": _percentile(clean, 0.50),
            "resample_errors": resample_errors,
            "aec_errors": aec_errors,
            "sync_discontinuities": discontinuities,
            "last_error": self._last_error or None,
        }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


__all__ = [
    "AEC_SAMPLE_RATE",
    "EchoCancellationPipeline",
    "EchoCancellerUnavailable",
    "FRAME_BYTES",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "MicProcessingResult",
]
