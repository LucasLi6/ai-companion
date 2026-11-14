"""Optional WebRTC AEC3 channel for real-time barge-in.

This module is deliberately additive: it never replaces the microphone,
Silero VAD, DashScope ASR or Qwen TTS. The TTS player hands it the PCM it is
about to write (the far-end reference); the ASR VAD thread hands it captured
microphone PCM (the near end). The pipeline resamples the 24 kHz TTS stream to
16 kHz, keeps a short monotonic reference timeline, and returns echo-cancelled
16 kHz PCM. Only when that returned block is valid may the ASR side run VAD on
it; raw, un-cancelled microphone audio is never used for a barge-in decision.

Threading
---------
* :meth:`push_playback` runs on the TTS player thread and only validates and
  appends to a bounded deque under ``_queue_lock``. It never touches the
  canceller, never resamples and never waits for AEC work, so a slow AEC
  frame can never stall playback.
* :meth:`process_microphone` runs on the ASR VAD thread. It snapshots pending
  playback under ``_queue_lock`` (a short copy), releases it, and then owns the
  canceller and the streaming resampler under ``_audio_lock``.
* :meth:`start`/:meth:`stop`/:meth:`reset` update generation/enable state
  under ``_queue_lock`` first and the audio state under ``_audio_lock``
  second; no path takes ``_queue_lock`` while holding ``_audio_lock``, so the
  two locks cannot deadlock.

Timing
------
Far-end frames carry an estimated play time derived from sample counts
(frame 0 at ``t0``, frame 1 at ``t0 + 20 ms`` ...), not from a fresh
``time.monotonic()`` per frame. The TTS callback timestamp anchors the start
of that timeline and detects device/timeline jumps. Microphone chunk
timestamps are taken *after* a read completes, so a block's start is derived
as ``timestamp - duration``. ``stream_delay_ms`` is passed only to WebRTC AEC3
as its device-buffer hint; an independent ``timeline_offset_ms`` (default 0)
may calibrate the reference/mic timeline without double compensation.

If ``pywebrtc-audio`` or NumPy is missing, cannot be constructed, or the
canceller raises, the pipeline reports itself unavailable/degraded and every
caller falls back to the existing half-duplex path. Nothing here raises into
playback or recognition.
"""

from __future__ import annotations

import sys
import threading
import time
from array import array
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
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
DEFAULT_TAIL_MAX_SECONDS = 2.0
# How long the absence of a new far-end frame is still treated as "the audio
# chunk is on its way" (synthesis burst gaps) rather than confirmed silence.
DEFAULT_SYNTHESIS_GAP_SECONDS = 0.35
# A reply needs some time before its first audio arrives. Missing references
# during this window are UNKNOWN, never a fault.
DEFAULT_START_GRACE_SECONDS = 0.5
# Right after playback reports "finished", the device may still hold buffered
# audio: the far end is not silent yet.
DEFAULT_PLAYBACK_TAIL_GRACE_SECONDS = 0.25
# Missing references while the playback timeline says audio *is* expected are
# a synchronization fault, not a sentence pause. Bound both by time and by
# frames so a stalled/looping caller cannot hide a broken reference channel.
DEFAULT_SYNC_FAILURE_SECONDS = 0.6
# A captured chunk whose start jumped further than this from the expected
# continuation is a device/timebase discontinuity, not normal jitter.
DEFAULT_MIC_JUMP_SECONDS = 0.15
_MAX_METRIC_SAMPLES = 256


class EchoCancellerUnavailable(RuntimeError):
    """The optional WebRTC AEC3 dependency is missing or unusable."""


class AECFormatError(ValueError):
    """The canceller accepted a frame but returned an unusable result."""


class ReferenceStatus(str, Enum):
    """Whether the playback timeline expects a far-end reference right now.

    ``AUDIO_EXPECTED`` means the playback timeline covers this microphone
    block (or TTS is actively writing audio), so a missing reference is a real
    synchronization fault. ``KNOWN_SILENCE`` means playback is genuinely
    paused/over: the far end is silent and AEC may run with zero references.
    ``UNKNOWN`` means the channel cannot tell yet (reply start, a short gap
    between synthesis bursts, the device tail after "finished"): be
    conservative, but do not count it as a fault.
    """

    AUDIO_EXPECTED = "AUDIO_EXPECTED"
    KNOWN_SILENCE = "KNOWN_SILENCE"
    UNKNOWN = "UNKNOWN"


class AECState(str, Enum):
    """Coarse lifecycle of the echo-cancellation channel.

    ``GAP`` is deliberately distinct from ``FAILED``: while a generation is
    still playing, a missing far-end reference is just TTS silence between
    sentences, not evidence that AEC broke. Only a real engine error, a
    permanent format/unavailable problem, or a missing reference *after*
    playback ended is a ``FAILED``. Keeping the two apart is what lets barge-in
    survive a normal sentence pause instead of falling back permanently.
    """

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    GAP = "GAP"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ReferenceFrame:
    """One far-end frame on the playback timeline.

    ``sequence`` increases monotonically within a generation and is used to
    guarantee that a reference frame is consumed at most once and never
    backwards. ``timestamp`` is the estimated start of the frame's playback,
    derived from the anchor callback plus the sample count.
    """

    generation: int
    sequence: int
    timestamp: float
    pcm: bytes


@dataclass(frozen=True)
class MicProcessingResult:
    """One processed microphone chunk.

    ``valid`` is true only when a matching far-end reference existed for every
    block in the chunk and the canceller returned a well-formed frame. Callers
    must not make a barge-in decision on an invalid result.
    """

    pcm: bytes = b""
    valid: bool = False
    reason: str = ""
    generation: Optional[int] = None
    near_rms: float = 0.0
    clean_rms: float = 0.0
    # True only when near_rms/clean_rms were actually measured for a valid,
    # reference-aligned block. A fake/partial pipeline that cannot meter the
    # signal must not have that absence count as a barge-in confirmation.
    metrics_valid: bool = True


_NUMPY: Any = None
_NUMPY_FAILED = False


def _numpy() -> Any:
    """Import NumPy lazily; it is part of the optional ASR/barge-in extras."""
    global _NUMPY, _NUMPY_FAILED
    if _NUMPY is not None:
        return _NUMPY
    if _NUMPY_FAILED:
        raise EchoCancellerUnavailable(
            "numpy 未安装：WebRTC AEC3 需要 NumPy 数组输入"
        )
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        _NUMPY_FAILED = True
        raise EchoCancellerUnavailable(
            "numpy 未安装：请执行 pip install 'ai-companion[bargein]'"
        ) from exc
    _NUMPY = np
    return np


def _aec_process_frame(
    canceller: Any,
    near_pcm: bytes,
    far_pcm: bytes,
    *,
    frame_bytes: int = FRAME_BYTES,
) -> bytes:
    """Run one AEC frame with the real library's NumPy int16 interface.

    Both inputs must be the same 16 kHz mono int16 block. The result must be a
    one-dimensional int16 array of the same sample count; anything else is a
    format error, never silently converted into a barge-in decision.
    """
    if len(near_pcm) != len(far_pcm):
        raise AECFormatError("AEC near/far frame length mismatch")
    if len(near_pcm) != frame_bytes:
        raise AECFormatError(
            f"AEC expected {frame_bytes} bytes ({FRAME_MS} ms) per frame, "
            f"got {len(near_pcm)}"
        )
    np = _numpy()
    near = np.frombuffer(near_pcm, dtype="<i2").copy()
    far = np.frombuffer(far_pcm, dtype="<i2").copy()
    result = canceller.process(near, far)
    output = np.asarray(result)
    if output.dtype != np.int16:
        raise AECFormatError(f"invalid AEC output dtype {output.dtype}")
    if output.ndim != 1 or output.size != near.size:
        raise AECFormatError("invalid AEC output shape")
    return output.astype("<i2", copy=False).tobytes()


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
    """Stateful far/near alignment plus one continuous WebRTC AEC3 instance.

    ``stream_delay_ms`` is only the AEC3 device hint. ``timeline_offset_ms`` is
    an independent reference/mic alignment calibration (set it from measured
    hardware latency); it is never fed to AEC3.
    """

    def __init__(
        self,
        *,
        sample_rate: int = AEC_SAMPLE_RATE,
        playback_sample_rate: int = 24000,
        stream_delay_ms: int = 0,
        timeline_offset_ms: float = 0.0,
        frame_ms: int = FRAME_MS,
        reference_seconds: float = DEFAULT_REFERENCE_SECONDS,
        playback_queue_frames: int = DEFAULT_PLAYBACK_QUEUE_FRAMES,
        sync_tolerance_seconds: float = DEFAULT_SYNC_TOLERANCE_SECONDS,
        fallback_after_invalid_seconds: float = (
            DEFAULT_FALLBACK_AFTER_INVALID_SECONDS
        ),
        tail_max_seconds: float = DEFAULT_TAIL_MAX_SECONDS,
        synthesis_gap_seconds: float = DEFAULT_SYNTHESIS_GAP_SECONDS,
        start_grace_seconds: float = DEFAULT_START_GRACE_SECONDS,
        playback_tail_grace_seconds: float = DEFAULT_PLAYBACK_TAIL_GRACE_SECONDS,
        sync_failure_seconds: float = DEFAULT_SYNC_FAILURE_SECONDS,
        canceller_factory: Optional[Callable[[int, int, int], Any]] = None,
    ) -> None:
        if sample_rate <= 0 or playback_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.sample_rate = int(sample_rate)
        self.playback_sample_rate = int(playback_sample_rate)
        # AEC3's own device-buffer hint; not applied to the timeline.
        self.stream_delay_ms = max(0, int(stream_delay_ms))
        # Independent reference/mic timeline shift (positive = playback reaches
        # the microphone later than the callback timestamp suggests).
        self.timeline_offset_seconds = float(timeline_offset_ms) / 1000.0
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
        self._tail_max_seconds = max(0.01, float(tail_max_seconds))
        self._synthesis_gap_seconds = max(0.0, float(synthesis_gap_seconds))
        self._start_grace_seconds = max(0.0, float(start_grace_seconds))
        self._playback_tail_grace_seconds = max(
            0.0, float(playback_tail_grace_seconds)
        )
        self._sync_failure_seconds = max(0.0, float(sync_failure_seconds))
        self._sync_failure_frames = max(
            3, int(round(self._sync_failure_seconds / self.frame_seconds))
        )
        self._factory = canceller_factory or _default_canceller_factory

        # Lock order: _queue_lock (short, never held while waiting for audio)
        # then _audio_lock. _stats_lock is a leaf taken last.
        self._audio_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._playback: Deque[Tuple[int, float, bytes]] = deque(
            maxlen=max(2, int(playback_queue_frames))
        )
        self._reference_frames_capacity = max(
            2, int(reference_seconds / self.frame_seconds)
        )
        self._reference: Deque[ReferenceFrame] = deque(
            maxlen=self._reference_frames_capacity
        )
        self._generation: Optional[int] = None
        self._active = False
        self._canceller: Any = None
        self._factory_ok = False
        self._failed = False
        self._degraded = False
        self._gap = False
        self._unavailable_reason = ""
        self._last_error = ""
        self._resampler = StreamingResampler(
            self.playback_sample_rate, self.sample_rate
        )
        # Far-end timeline: next frame start and leftover samples.
        self._ref_carry = bytearray()
        self._ref_frame_ts: Optional[float] = None
        self._next_sequence = 0
        self._last_consumed_sequence: Optional[int] = None
        # Near-end buffering and timebase tracking.
        self._mic_buffer = bytearray()
        self._mic_buffer_ts = 0.0
        self._mic_expected_end: Optional[float] = None
        self._last_valid_at: Optional[float] = None
        self._invalid_run_started: Optional[float] = None
        self._consecutive_invalid_seconds = 0.0
        # Playback timeline evidence. ``_started_at``/``_playback_activity_at``
        # are wall-clock monotonic values; ``_ref_frame_ts`` is the far-end
        # playback clock. The three reference-status counters are per
        # generation and decide failure; they are never lifetime diagnostics.
        self._started_at = time.monotonic()
        self._playback_activity_at: Optional[float] = None
        self._playback_finished_at: Optional[float] = None
        # Set only by frame_submitted/frame_write_completed. Unlike
        # ``_playback_activity_at`` (which also notes device open/close), this
        # proves the player is *writing audio right now*, so a reference that
        # is missing then is a sync fault rather than a synthesis pause.
        self._timing_activity_at: Optional[float] = None
        self._reference_status = ReferenceStatus.UNKNOWN
        self._known_silence_frames = 0
        self._known_silence_seconds = 0.0
        self._known_silence_run_started: Optional[float] = None
        self._known_silence_last_at: Optional[float] = None
        self._consecutive_expected_reference_errors = 0
        self._expected_error_run_started: Optional[float] = None
        self._consecutive_unknown_reference_frames = 0
        self._unknown_run_started: Optional[float] = None
        self._reference_sync_errors = 0
        self._mic_jump_pending = False
        self._timing_events = 0
        # Last pairing diagnostics (``/mic status`` only; never logged per
        # frame, so terminal I/O cannot add latency).
        self._last_alignment_offset_ms: Optional[float] = None
        self._last_reference_age_ms: Optional[float] = None
        # Confirmed-barge-in echo tail: real references first, then silent far
        # frames so the already-confirmed user turn is never dropped.
        self._tail_active = False
        self._tail_generation: Optional[int] = None
        self._tail_started_at: Optional[float] = None
        self._zero_reference = bytes(self.frame_bytes)

        self._playback_frames = 0
        self._playback_dropped = 0
        # Per-generation counters. Only these decide whether *this* generation
        # failed; the lifetime totals above stay diagnostics for ``/mic
        # status`` and must never make a fresh generation look broken.
        self._generation_playback_frames = 0
        self._generation_aec_errors = 0
        self._mic_blocks = 0
        self._valid_blocks = 0
        self._invalid_blocks = 0
        self._reference_consumed = 0
        self._reference_expired = 0
        self._reference_dropped = 0
        self._tail_frames = 0
        self._invalid_reasons: Counter[str] = Counter()
        self._resample_errors = 0
        self._aec_errors = 0
        self._sync_discontinuities = 0
        self._mic_discontinuities = 0
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
    def state(self) -> AECState:
        """Lifecycle state; ``GAP`` (TTS silence) is not a failure."""
        if not self._factory_ok or self._failed or self._degraded:
            return AECState.FAILED
        if not self._active and not self._tail_active:
            return AECState.IDLE
        if self._gap:
            return AECState.GAP
        return AECState.ACTIVE

    @property
    def degraded(self) -> bool:
        return self._degraded or self._failed

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason or self._last_error

    @property
    def tail_active(self) -> bool:
        """True from a confirmed barge-in until the echo tail is ended."""
        return self._tail_active

    # --------------------------------------------------------------- lifecycle
    def start(self, generation: int) -> bool:
        """Arm the pipeline for one TTS generation and reset alignment state."""
        generation = int(generation)
        with self._queue_lock:
            self._playback.clear()
            self._generation = generation
            self._active = False
        with self._audio_lock:
            if not self._build_canceller():
                return False
            self._reset_audio_state_locked()
        with self._queue_lock:
            self._active = True
            self._gap = False
        return True

    def stop(self, generation: Optional[int] = None) -> None:
        """Stop accepting new playback for a generation.

        Reference frames already received are kept so the echo tail of the
        interrupted reply can still be cancelled while the user's first words
        are being recognized.
        """
        with self._queue_lock:
            if generation is not None and self._generation != int(generation):
                return
            self._active = False

    def reset(self) -> None:
        """Drop all state; the next :meth:`start` rebuilds the canceller."""
        with self._queue_lock:
            self._generation = None
            self._active = False
            self._playback.clear()
        with self._audio_lock:
            self._reset_audio_state_locked()
            self._build_canceller()

    def _reset_audio_state_locked(self) -> None:
        self._reference.clear()
        self._ref_carry = bytearray()
        self._ref_frame_ts = None
        self._next_sequence = 0
        self._last_consumed_sequence = None
        self._mic_buffer = bytearray()
        self._mic_buffer_ts = 0.0
        self._mic_expected_end = None
        self._last_valid_at = None
        self._invalid_run_started = None
        self._consecutive_invalid_seconds = 0.0
        self._failed = False
        self._degraded = False
        self._gap = False
        self._last_error = ""
        self._generation_playback_frames = 0
        self._generation_aec_errors = 0
        # Playback-timeline evidence is per generation: a previous reply's
        # finish time or activity must never classify a fresh generation.
        self._started_at = time.monotonic()
        self._playback_activity_at = None
        self._playback_finished_at = None
        self._timing_activity_at = None
        self._reference_status = ReferenceStatus.UNKNOWN
        self._known_silence_frames = 0
        self._known_silence_seconds = 0.0
        self._known_silence_run_started = None
        self._known_silence_last_at = None
        self._consecutive_expected_reference_errors = 0
        self._expected_error_run_started = None
        self._consecutive_unknown_reference_frames = 0
        self._unknown_run_started = None
        self._reference_sync_errors = 0
        self._mic_jump_pending = False
        self._tail_active = False
        self._tail_generation = None
        self._tail_started_at = None
        self._resampler = StreamingResampler(
            self.playback_sample_rate, self.sample_rate
        )

    # ------------------------------------------------------------- echo tail
    def begin_echo_tail(self, generation: int) -> bool:
        """Switch to continuity mode after a *confirmed* barge-in.

        Real reference frames are still used while they exist. Once they are
        exhausted, bounded silent far-end frames keep the same AEC instance
        running so the confirmed user turn is not dropped. This must only be
        armed after the interruption was confirmed and TTS was stopped; using
        it during detection would risk under-cancelling Ava's own voice.
        """
        generation = int(generation)
        with self._queue_lock:
            if self._generation is None or generation != self._generation:
                return False
        with self._audio_lock:
            if self._generation is None or generation != self._generation:
                return False
            self._tail_active = True
            self._tail_generation = generation
            self._tail_started_at = time.monotonic()
            self._tail_frames = 0
            self._invalid_run_started = None
            self._consecutive_invalid_seconds = 0.0
            self._gap = False
        return True

    def end_echo_tail(self) -> None:
        with self._audio_lock:
            self._tail_active = False
            self._tail_generation = None
            self._tail_started_at = None

    # ---------------------------------------------------------------- far end
    def push_playback(
        self, pcm: bytes, timestamp: float, generation: int
    ) -> bool:
        """Enqueue one playback chunk from the TTS player thread.

        Fast path only: validate, append, return. This never resamples, never
        calls AEC and never waits on ``_audio_lock``, so playback cannot stall
        behind a slow AEC frame. A full queue drops the oldest frame instead of
        blocking playback.
        """
        if not pcm:
            return False
        try:
            data = bytes(pcm)
        except Exception:  # noqa: BLE001 - defensive
            return False
        dropped = False
        seen_at = time.monotonic()
        with self._queue_lock:
            current = self._generation
            if current is None or not self._active or int(generation) != current:
                return False
            if len(self._playback) >= (self._playback.maxlen or 0):
                self._playback.popleft()
                dropped = True
            self._playback.append((int(generation), float(timestamp), data))
            self._playback_activity_at = seen_at
        with self._stats_lock:
            self._playback_frames += 1
            self._generation_playback_frames += 1
            if dropped:
                self._playback_dropped += 1
        return True

    # ------------------------------------------------------ playback timeline
    def note_playback_timing(self, timing: Any) -> None:
        """Receive one ``PlaybackTiming`` notice from the TTS player thread.

        Independent of the PCM listener: even if the far-end reference frames
        are lost, the player still tells the pipeline that audio was submitted
        and written. That is what lets a dropped reference be classified as a
        sync fault instead of being mistaken for TTS silence. Fast and
        exception-free; never blocks playback.
        """
        try:
            generation = getattr(timing, "generation", None)
            kind = str(getattr(timing, "kind", "") or "").strip().lower()
            now = time.monotonic()
            with self._queue_lock:
                current = self._generation
                if current is None:
                    return
                if generation is not None and int(generation) != current:
                    return
                self._playback_activity_at = now
                if kind in ("frame_submitted", "frame_write_completed"):
                    # Continuous writes with no matching reference frames is
                    # exactly the dropped-reference fault this channel exists
                    # to expose.
                    self._timing_activity_at = now
            with self._stats_lock:
                self._timing_events += 1
        except Exception:  # noqa: BLE001 - optional side channel
            return

    def note_playback_event(
        self, kind: str, generation: Optional[int] = None
    ) -> None:
        """Record a playback end notice (``finished``/``failed``/``cancelled``).

        Only the current generation may change the timeline. A finished reply
        keeps a short device-tail grace of UNKNOWN; after that the far end is
        definitely silent and AEC runs with zero references instead of being
        declared broken.
        """
        try:
            kind = str(kind or "").strip().lower()
            with self._queue_lock:
                current = self._generation
                if current is None:
                    return
                if generation is not None and int(generation) != current:
                    return
                if kind in ("finished", "failed", "cancelled", "stopped"):
                    self._playback_finished_at = time.monotonic()
                if kind in ("started", "playing", "submitted"):
                    self._playback_activity_at = time.monotonic()
        except Exception:  # noqa: BLE001 - optional side channel
            return

    def _take_pending_playback(self) -> list[Tuple[int, float, bytes]]:
        with self._queue_lock:
            if not self._playback:
                return []
            pending = list(self._playback)
            self._playback.clear()
            return pending

    # ---------------------------------------------------------------- near end
    def process_microphone(
        self, pcm: bytes, timestamp: float
    ) -> MicProcessingResult:
        """Return echo-cancelled 16 kHz PCM for one captured chunk.

        The chunk is split into 20 ms blocks. Its timestamp is the end of the
        capture read, so the first block start is ``timestamp - duration`` and
        later blocks advance by sample counts. If any block lacks a usable
        reference the whole chunk is invalid (unless the confirmed echo tail is
        active) and its PCM must not reach VAD.
        """
        if not pcm:
            return MicProcessingResult()
        try:
            data = bytes(pcm)
        except Exception:  # noqa: BLE001 - defensive
            return MicProcessingResult()

        # Snapshot playback under the queue lock only; never hold it while AEC
        # runs, or the TTS player thread would wait on audio processing.
        pending = self._take_pending_playback()
        if not pending:
            with self._queue_lock:
                generation = self._generation
            if generation is None:
                return MicProcessingResult(reason="inactive")
        else:
            generation = pending[0][0]

        with self._audio_lock:
            if not self._factory_ok or self._canceller is None:
                self._count_invalid_block("unavailable", 0.0, 0.0)
                return MicProcessingResult(
                    valid=False,
                    reason="unavailable",
                    generation=self._generation,
                )
            # A start()/reset() between the snapshot and this lock changed the
            # generation; drop the stale audio instead of mixing timelines.
            if generation != self._generation:
                return MicProcessingResult(
                    valid=False,
                    reason="inactive",
                    generation=self._generation,
                )

            duration = len(data) / (self.sample_rate * 2.0)
            if not self._mic_buffer:
                block_start = float(timestamp) - duration
                expected = self._mic_expected_end
                if expected is not None and (
                    abs(block_start - expected) > DEFAULT_MIC_JUMP_SECONDS
                ):
                    self._mic_discontinuities += 1
                    # A device/timebase jump invalidates far/near alignment.
                    # Never treat the resulting mismatch as a sync fault or as
                    # TTS silence: it is UNKNOWN until the next clean pairing.
                    self._mic_jump_pending = True
                self._mic_buffer_ts = block_start
            self._mic_expected_end = float(timestamp)
            self._mic_buffer.extend(data)

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

            self._ingest_playback_locked(pending)

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
                    clean = _aec_process_frame(
                        self._canceller,
                        block,
                        reference,
                        frame_bytes=self.frame_bytes,
                    )
                except Exception as exc:  # noqa: BLE001 - optional engine
                    self._aec_error_locked(exc)
                    failure = failure or "aec_error"
                    self._count_invalid_block("aec_error", near_level, 0.0)
                    continue
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
                # VAD as if it were clean. This also applies during the echo
                # tail, so the manager can transition that confirmed turn to
                # NORMAL instead of silently losing the user's audio.
                return MicProcessingResult(
                    valid=False,
                    reason=(
                        "tail_error"
                        if self._tail_active and failure == "aec_error"
                        else failure
                    ),
                    generation=generation,
                    near_rms=_mean(near_values),
                    clean_rms=_mean(clean_values),
                    metrics_valid=False,
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
    def _ingest_playback_locked(
        self, pending: list[Tuple[int, float, bytes]]
    ) -> None:
        """Resample pending playback and advance the monotonic far timeline.

        Runs only under ``_audio_lock`` and never waits on ``_queue_lock``.
        """
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
            timestamp = float(timestamp)
            if self._ref_frame_ts is None:
                self._ref_frame_ts = timestamp
                self._ref_carry = bytearray()
            else:
                # Advance the reference clock by sample counts, not by a fresh
                # wall-clock read per frame. The callback timestamp is only an
                # anchor and a discontinuity detector.
                expected = self._ref_frame_ts + (
                    len(self._ref_carry) / 2.0
                ) / self.sample_rate
                if abs(timestamp - expected) > self.sync_tolerance_seconds:
                    self._sync_discontinuities += 1
                    self._ref_carry = bytearray()
                    self._ref_frame_ts = timestamp
            self._ref_carry.extend(resampled)
            while len(self._ref_carry) >= self.frame_bytes:
                frame_pcm = bytes(self._ref_carry[: self.frame_bytes])
                del self._ref_carry[: self.frame_bytes]
                self._append_reference_locked(
                    ReferenceFrame(
                        generation=self._generation or 0,
                        sequence=self._next_sequence,
                        timestamp=self._ref_frame_ts,
                        pcm=frame_pcm,
                    )
                )
                self._next_sequence += 1
                self._ref_frame_ts += self.frame_seconds

    def _append_reference_locked(self, frame: ReferenceFrame) -> None:
        if len(self._reference) >= self._reference_frames_capacity:
            self._reference.popleft()
            self._reference_dropped += 1
        self._reference.append(frame)

    def _reference_for_locked(
        self, block_start_ts: float
    ) -> Tuple[Optional[bytes], str]:
        """Consume the next reference frame for a microphone block.

        Frames are paired strictly forward: a frame is consumed at most once,
        expired frames are dropped, and a future frame beyond tolerance is
        never substituted for the current block (except in the confirmed echo
        tail, where the real future frame is still better than silence).
        """
        generation = self._generation
        if generation is None:
            return None, "inactive"
        target = float(block_start_ts) - self.timeline_offset_seconds
        tolerance = self.sync_tolerance_seconds

        while self._reference:
            head = self._reference[0]
            if head.generation != generation:
                self._reference.popleft()
                self._reference_expired += 1
                continue
            if (
                self._last_consumed_sequence is not None
                and head.sequence <= self._last_consumed_sequence
            ):
                self._reference.popleft()
                continue
            if head.timestamp < target - tolerance:
                self._reference.popleft()
                self._reference_expired += 1
                continue
            break

        candidate: Optional[ReferenceFrame] = None
        for frame in self._reference:
            if frame.generation != generation:
                continue
            if (
                self._last_consumed_sequence is not None
                and frame.sequence <= self._last_consumed_sequence
            ):
                continue
            candidate = frame
            break

        if candidate is None:
            return self._missing_reference_locked(target, "no_reference")
        if len(candidate.pcm) != self.frame_bytes:
            return self._missing_reference_locked(target, "bad_reference")

        behind = target - candidate.timestamp
        if behind > tolerance:
            return self._missing_reference_locked(target, "stale_reference")
        if -behind > tolerance and not self._tail_active:
            # The next frame has not arrived yet; never pass a future frame off
            # as the current one during detection.
            return self._missing_reference_locked(target, "ahead_reference")

        self._last_consumed_sequence = candidate.sequence
        self._reference_consumed += 1
        self._gap = False
        self._mic_jump_pending = False
        # A real pairing proves the channel is synchronized: clear every
        # per-kind error run so an isolated hiccup never accumulates.
        self._reference_status = ReferenceStatus.AUDIO_EXPECTED
        self._consecutive_expected_reference_errors = 0
        self._expected_error_run_started = None
        self._consecutive_unknown_reference_frames = 0
        self._unknown_run_started = None
        self._invalid_run_started = None
        self._consecutive_invalid_seconds = 0.0
        self._last_alignment_offset_ms = round(
            (candidate.timestamp - target) * 1000.0, 2
        )
        self._last_reference_age_ms = round(
            max(0.0, block_start_ts - candidate.timestamp) * 1000.0, 1
        )
        return candidate.pcm, ""

    def _reference_status_locked(self, target: float) -> ReferenceStatus:
        """Classify why a microphone block has no matching reference frame.

        The old code treated *every* missing reference during an active
        generation as a harmless gap. That hid a real sync fault (audio is
        playing, but its reference frames were dropped) and left the user
        unable to interrupt while AEC still reported healthy. This splits the
        three cases explicitly using the playback timeline.
        """
        if self._mic_jump_pending:
            return ReferenceStatus.UNKNOWN
        now = time.monotonic()
        last_end = self._ref_frame_ts
        tolerance = self.sync_tolerance_seconds
        activity_age = (
            None
            if self._playback_activity_at is None
            else max(0.0, now - self._playback_activity_at)
        )
        finished_age = (
            None
            if self._playback_finished_at is None
            else max(0.0, now - self._playback_finished_at)
        )
        timing_age = (
            None
            if self._timing_activity_at is None
            else max(0.0, now - self._timing_activity_at)
        )
        # The player reported writing frames moments ago. Frames are written in
        # short bursts during continuous playback, so this proves audio is
        # playing *now*; a missing reference here cannot be a sentence pause.
        writing_now = (
            timing_age is not None
            and timing_age <= self._synthesis_gap_seconds
        )
        in_device_tail = (
            finished_age is not None
            and finished_age <= self._playback_tail_grace_seconds
        )
        if last_end is not None:
            if target <= last_end + tolerance:
                # The known playback timeline covers this block, yet no
                # reference survived to pair with it.
                return ReferenceStatus.AUDIO_EXPECTED
            if writing_now:
                # The player is actively writing audio past the known
                # reference timeline: the reference channel lost frames.
                return ReferenceStatus.AUDIO_EXPECTED
            if in_device_tail:
                return ReferenceStatus.UNKNOWN
            if (
                activity_age is not None
                and activity_age <= self._synthesis_gap_seconds
            ):
                # A synthesis burst gap: the next chunk is on its way.
                return ReferenceStatus.UNKNOWN
            return ReferenceStatus.KNOWN_SILENCE
        # Nothing has reached the reference timeline in this generation yet.
        if in_device_tail:
            return ReferenceStatus.UNKNOWN
        if writing_now and now - self._started_at > self._start_grace_seconds:
            # The player is writing, yet not one reference frame arrived after
            # the start grace: the reference channel is broken.
            return ReferenceStatus.AUDIO_EXPECTED
        if activity_age is not None:
            if activity_age <= self._synthesis_gap_seconds:
                return ReferenceStatus.UNKNOWN
            # TTS is writing audio but no reference frame ever arrived.
            return ReferenceStatus.AUDIO_EXPECTED
        if now - self._started_at <= self._start_grace_seconds:
            return ReferenceStatus.UNKNOWN
        # No audio was ever submitted: TTS is legitimately silent (e.g. the
        # reply contained no speakable text).
        return ReferenceStatus.KNOWN_SILENCE

    def _missing_reference_locked(
        self, target: float, reason: str
    ) -> Tuple[Optional[bytes], str]:
        if (
            self._tail_active
            and self._generation is not None
            and self._tail_generation == self._generation
        ):
            started = self._tail_started_at
            if (
                started is not None
                and (time.monotonic() - started) > self._tail_max_seconds
            ):
                return None, "tail_expired"
            # Playback already stopped (TTS stop was requested before this
            # mode): the far end is silent, so keep AEC running with zeros so
            # the confirmed user turn stays continuous.
            self._tail_frames += 1
            return self._zero_reference, ""

        status = self._reference_status_locked(target)
        self._reference_status = status
        now = time.monotonic()

        if status is ReferenceStatus.KNOWN_SILENCE:
            # Confirmed TTS silence: AEC stays alive with a zero far end, so
            # the user's speech during a sentence pause is still recognized.
            if self._known_silence_run_started is None:
                self._known_silence_run_started = now
            self._known_silence_frames += 1
            self._known_silence_seconds = max(
                0.0, now - self._known_silence_run_started
            )
            self._consecutive_expected_reference_errors = 0
            self._expected_error_run_started = None
            self._consecutive_unknown_reference_frames = 0
            self._unknown_run_started = None
            self._invalid_run_started = None
            self._consecutive_invalid_seconds = 0.0
            self._gap = True
            self._known_silence_last_at = now
            return self._zero_reference, ""

        if status is ReferenceStatus.AUDIO_EXPECTED:
            self._known_silence_run_started = None
            if self._expected_error_run_started is None:
                self._expected_error_run_started = now
            self._consecutive_expected_reference_errors += 1
            self._reference_sync_errors += 1
            self._unknown_run_started = None
            self._consecutive_unknown_reference_frames = 0
            elapsed = now - self._expected_error_run_started
            if (
                (
                    self._consecutive_expected_reference_errors
                    >= self._sync_failure_frames
                    or elapsed >= self._sync_failure_seconds
                )
                and not self._failed
            ):
                # Audio is playing but its reference is gone: barge-in no
                # longer has a valid echo model. Fail loudly and fall back to
                # half-duplex for the rest of this generation only.
                self._failed = True
                self._last_error = "reference_sync_failure"
            self._gap = False
            return None, reason

        # UNKNOWN: be conservative for barge-in, but never count this as a
        # fault. It is bounded by the start grace / synthesis gap / device
        # tail and always resolves to AUDIO_EXPECTED or KNOWN_SILENCE, so it
        # cannot persist indefinitely. Audio that *is* expected but has no
        # reference is handled (and failed) in the AUDIO_EXPECTED branch.
        self._known_silence_run_started = None
        # Break the fault run: only consecutive AUDIO_EXPECTED observations
        # may accumulate toward a declared sync failure. A declared failure
        # (``_failed``) is never cleared here.
        self._consecutive_expected_reference_errors = 0
        self._expected_error_run_started = None
        if self._unknown_run_started is None:
            self._unknown_run_started = now
        self._consecutive_unknown_reference_frames += 1
        self._gap = True
        return None, reason

    def _note_invalid_time(self) -> None:
        """Legacy diagnostic hook kept for callers/tests.

        Missing-reference policy now lives in :meth:`_missing_reference_locked`;
        this only tracks the generic invalid duration for stats.
        """
        now = time.monotonic()
        if self._invalid_run_started is None:
            self._invalid_run_started = now
        self._consecutive_invalid_seconds = max(
            0.0, now - self._invalid_run_started
        )

    def _count_invalid_block(
        self, reason: str, near_rms: float, clean_rms: float
    ) -> None:
        self._invalid_blocks += 1
        with self._stats_lock:
            self._invalid_reasons[reason] += 1

    def _aec_error_locked(self, exc: Exception) -> None:
        with self._stats_lock:
            self._aec_errors += 1
            self._generation_aec_errors += 1
            generation_errors = self._generation_aec_errors
        detail = str(exc).strip()
        self._last_error = f"aec:{type(exc).__name__}"
        if detail:
            self._last_error = f"{self._last_error}:{detail[:80]}"
        # ``_aec_errors`` is a lifetime diagnostic; only this generation's
        # errors may mark the current generation as failed.
        if generation_errors >= 3:
            self._failed = True

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        with self._queue_lock:
            pending_playback = len(self._playback)
            active = self._active
            generation = self._generation
        with self._audio_lock:
            reference_frames = len(self._reference)
            with self._stats_lock:
                reasons = dict(self._invalid_reasons)
                aec_ms = list(self._aec_ms)
                near = list(self._near_rms)
                clean = list(self._clean_rms)
                playback_frames = self._playback_frames
                generation_playback_frames = self._generation_playback_frames
                generation_aec_errors = self._generation_aec_errors
                playback_dropped = self._playback_dropped
                mic_blocks = self._mic_blocks
                valid_blocks = self._valid_blocks
                invalid_blocks = self._invalid_blocks
                consumed = self._reference_consumed
                expired = self._reference_expired
                ref_dropped = self._reference_dropped
                tail_frames = self._tail_frames
                resample_errors = self._resample_errors
                aec_errors = self._aec_errors
                discontinuities = self._sync_discontinuities
                mic_discontinuities = self._mic_discontinuities
            last_consumed = self._last_consumed_sequence
            next_sequence = self._next_sequence
            tail_active = self._tail_active
            alignment_ms = self._last_alignment_offset_ms
            reference_age_ms = self._last_reference_age_ms
            reference_status = self._reference_status.value
            expected_errors = self._consecutive_expected_reference_errors
            unknown_frames = self._consecutive_unknown_reference_frames
            known_silence_frames = self._known_silence_frames
            known_silence_seconds = self._known_silence_seconds
            sync_errors = self._reference_sync_errors
            playback_finished = self._playback_finished_at is not None
            timing_events = self._timing_events
        total = valid_blocks + invalid_blocks
        return {
            "available": self._factory_ok,
            "healthy": self.healthy,
            "aec_valid": self.healthy,
            "degraded": self.degraded,
            "state": self.state.value,
            "reason": self.unavailable_reason or None,
            "active": active,
            "generation": generation,
            "stream_delay_ms": self.stream_delay_ms,
            "timeline_offset_ms": round(
                self.timeline_offset_seconds * 1000.0, 2
            ),
            "playback_frames": playback_frames,
            "generation_playback_frames": generation_playback_frames,
            "generation_aec_errors": generation_aec_errors,
            "playback_dropped": playback_dropped,
            "pending_playback_frames": pending_playback,
            "reference_frames": reference_frames,
            "reference_consumed": consumed,
            "reference_expired": expired,
            "reference_dropped": ref_dropped,
            "reference_sequence": next_sequence,
            "last_consumed_sequence": last_consumed,
            "far_sequence": next_sequence,
            "near_sequence": mic_blocks + invalid_blocks,
            "alignment_offset_ms": alignment_ms,
            "reference_age_ms": reference_age_ms,
            "reference_status": reference_status,
            "consecutive_expected_reference_errors": expected_errors,
            "consecutive_unknown_reference_frames": unknown_frames,
            "known_silence_frames": known_silence_frames,
            "known_silence_seconds": round(known_silence_seconds, 4),
            "reference_sync_errors": sync_errors,
            "playback_finished": playback_finished,
            "playback_timing_events": timing_events,
            "tail_active": tail_active,
            "tail_frames": tail_frames,
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
            "mic_discontinuities": mic_discontinuities,
            "last_error": self._last_error or None,
        }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


__all__ = [
    "AEC_SAMPLE_RATE",
    "AECFormatError",
    "AECState",
    "EchoCancellationPipeline",
    "EchoCancellerUnavailable",
    "FRAME_BYTES",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "MicProcessingResult",
    "ReferenceFrame",
    "ReferenceStatus",
    "_aec_process_frame",
]
