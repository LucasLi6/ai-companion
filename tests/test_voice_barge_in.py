"""Offline tests for optional WebRTC-AEC barge-in (V0.10).

Everything here is deterministic and network-free: a fake echo canceller, fake
microphone, fake playback and a fake streaming reply. The real Silero model,
pywebrtc-audio, PyAudio, DashScope and the LLM provider are never touched.

Coverage map (spec scenarios A-I):
A. normal playback with no user speech never confirms a barge-in
B. pure loud echo (cancelled to silence) never confirms a barge-in
C. a real interjection stops TTS immediately and enters the ASR turn
D. the pre-roll is preserved exactly once
E. the old reply stream is cancelled and the REPL keeps running
F. stale finished/cancelled/on_turn_finished callbacks restore nothing
G. AEC load/processing failure degrades without a wrong barge-in
H. duplicate barge-in signals stop playback once and create one turn
I. with the feature disabled, behavior is the old half-duplex path
"""

from __future__ import annotations

import asyncio
import threading
import time
from array import array
from types import SimpleNamespace

import numpy as np
import pytest

from companion.asr.config import ASRConfig
from companion.asr.manager import (
    ASRManager,
    FinalUtterance,
    InputMode,
    SpeechActivity,
    _SpeechStarted,
)
from companion.asr.microphone import AudioChunk
from companion.asr.terminal import TerminalRenderer
from companion.asr.vad import (
    SPEECH_STARTED,
    VADSegmenter,
)
from companion.cli import _run_streaming_turn
from companion.voice import VoiceCoordinator, VoiceState
from companion.voice.barge_in_config import (
    BargeInConfig,
    BargeInConfigError,
    load_barge_in_config,
)
from companion.voice.echo_canceller import (
    AECFormatError,
    EchoCancellationPipeline,
    MicProcessingResult,
    _aec_process_frame,
)

WINDOW = 1024  # 512 int16 samples = one 32 ms Silero window


# --------------------------------------------------------------------- helpers
async def settle(rounds: int = 4) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0.01)


def wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def tone(samples: int, amplitude: int = 8000) -> bytes:
    values = array("h", [amplitude if i % 2 else -amplitude for i in range(samples)])
    return values.tobytes()


def silence(samples: int) -> bytes:
    return b"\x00\x00" * samples


def energy_probability(window: bytes) -> float:
    values = array("h")
    values.frombytes(window)
    if not values:
        return 0.0
    level = sum(abs(v) for v in values) / len(values)
    return 0.95 if level > 1500 else 0.0


def _assert_aec_arrays(near, far) -> None:
    """The real pywebrtc-audio interface: ndarray int16, equal lengths."""
    assert isinstance(near, np.ndarray)
    assert isinstance(far, np.ndarray)
    assert near.dtype == np.int16
    assert far.dtype == np.int16
    assert len(near) == len(far)


class PassThroughCanceller:
    """Models perfect AEC: the near signal survives unchanged."""

    def process(self, near, far):
        _assert_aec_arrays(near, far)
        return near.copy()


class SilenceCanceller:
    """Models perfect echo removal: only residual silence remains."""

    def process(self, near, far):
        _assert_aec_arrays(near, far)
        return np.zeros_like(near)


class ExplodingCanceller:
    def process(self, near, far):
        _assert_aec_arrays(near, far)
        raise RuntimeError("aec exploded")


class EchoSubtractingCanceller:
    """Models a loudspeaker path: remove exactly the far-end echo."""

    def process(self, near, far):
        _assert_aec_arrays(near, far)
        return np.clip(
            near.astype(np.int32) - far.astype(np.int32), -32768, 32767
        ).astype(np.int16)


class RecordingCanceller:
    """Perfect pass-through that records every far-end frame it was given."""

    def __init__(self) -> None:
        self.fars: list[np.ndarray] = []

    def process(self, near, far):
        _assert_aec_arrays(near, far)
        self.fars.append(np.array(far, copy=True))
        return near.copy()


def constant(samples: int, value: int) -> bytes:
    return np.full(samples, value, dtype="<i2").tobytes()


def _config(**overrides) -> ASRConfig:
    base = dict(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        timeout_seconds=2.0,
        final_timeout_seconds=1.0,
    )
    base.update(overrides)
    return ASRConfig(**base)


def _renderer() -> TerminalRenderer:
    return TerminalRenderer(stream=_Sink(), status_stream=_Sink())


class _Sink:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


# ------------------------------------------------------ AEC pipeline (A/B/C/G)
def _feed_pipeline(
    pipeline: EchoCancellationPipeline,
    *,
    windows: int,
    near_pcm: bytes,
    generation: int = 1,
    start: float = 1000.0,
) -> list:
    results = []
    for index in range(windows):
        timestamp = start + index * 0.02
        pipeline.push_playback(tone(480), timestamp, generation)
        results.append(
            pipeline.process_microphone(near_pcm, timestamp)
        )
    return results


def test_a_normal_playback_without_user_speech_never_confirms() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: SilenceCanceller()
    )
    assert pipeline.start(1)
    segmenter = VADSegmenter(energy_probability, _config())
    segmenter.configure(threshold=0.65, min_speech_ms=192, pre_roll_ms=400)

    events = []
    for result in _feed_pipeline(
        pipeline, windows=30, near_pcm=tone(320)
    ):
        assert result.valid
        events.extend(segmenter.process(result.pcm))
    assert all(event.kind != SPEECH_STARTED for event in events)
    assert pipeline.stats()["valid_ratio"] == 1.0


def test_b_pure_echo_is_cancelled_and_never_confirms() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: SilenceCanceller()
    )
    assert pipeline.start(1)
    segmenter = VADSegmenter(energy_probability, _config())
    segmenter.configure(threshold=0.65, min_speech_ms=192, pre_roll_ms=400)

    events = []
    for result in _feed_pipeline(
        pipeline, windows=30, near_pcm=tone(320, amplitude=20000)
    ):
        # The near end is loud, but the reference is valid and the model
        # removes it; VAD only ever sees the cancelled output.
        assert result.valid
        events.extend(segmenter.process(result.pcm))
    assert all(event.kind != SPEECH_STARTED for event in events)


def test_c_user_interjection_survives_cancellation() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller()
    )
    assert pipeline.start(7)
    segmenter = VADSegmenter(energy_probability, _config())
    segmenter.configure(threshold=0.65, min_speech_ms=192, pre_roll_ms=400)

    events = []
    confirmed_windows = None
    for index, result in enumerate(
        _feed_pipeline(
            pipeline, windows=20, near_pcm=tone(320), generation=7
        )
    ):
        assert result.valid
        events.extend(segmenter.process(result.pcm))
        if confirmed_windows is None and any(
            event.kind == SPEECH_STARTED for event in events
        ):
            confirmed_windows = index
    assert confirmed_windows is not None
    # 192 ms of speech = six 32 ms VAD windows. The 20 ms AEC/mic blocks add
    # buffering slack, so confirmation lands around the tenth block.
    assert confirmed_windows <= 10


def test_d_pre_roll_is_preserved_exactly_once() -> None:
    segmenter = VADSegmenter(energy_probability, _config())
    segmenter.configure(threshold=0.65, min_speech_ms=64, pre_roll_ms=128)

    quiet_a = silence(512)
    quiet_b = silence(512)
    speech_1 = tone(512)
    speech_2 = tone(512)
    events = []
    for window in (quiet_a, quiet_b, speech_1, speech_2):
        events.extend(segmenter.process(window))
    started = [e for e in events if e.kind == SPEECH_STARTED]
    assert len(started) == 1
    # Two pre-roll windows plus the two confirming speech windows; each byte
    # appears once, in order.
    assert started[0].audio == quiet_a + quiet_b + speech_1 + speech_2

    # Subsequent audio events carry only new audio, never a duplicate pre-roll.
    more = tone(512)
    later = segmenter.process(more)
    assert [e.audio for e in later if e.kind == "audio"] == [more]


def test_g_canceller_failure_marks_the_channel_degraded() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: ExplodingCanceller()
    )
    assert pipeline.start(1)
    for index in range(5):
        timestamp = 50.0 + index * 0.02
        pipeline.push_playback(tone(480), timestamp, 1)
        result = pipeline.process_microphone(tone(320), timestamp)
        assert not result.valid
    stats = pipeline.stats()
    assert stats["aec_errors"] >= 3
    assert stats["healthy"] is False


def test_g_missing_reference_blocks_detection_without_raw_fallback() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller(),
        fallback_after_invalid_seconds=0.05,
    )
    assert pipeline.start(1)
    first = pipeline.process_microphone(tone(320), 10.0)
    assert not first.valid
    assert first.reason == "no_reference"
    assert first.pcm == b""

    # A far-end dropout after real playback: stale reference, still invalid.
    pipeline.push_playback(tone(480), 20.0, 1)
    assert pipeline.process_microphone(tone(320), 20.0).valid
    stale = pipeline.process_microphone(tone(320), 20.6)
    assert not stale.valid
    # The single reference frame was consumed once; with audio stopped there
    # is no reference left to pair with later microphone audio.
    assert stale.reason in (
        "stale_reference",
        "ahead_reference",
        "no_reference",
    )


def test_g_generation_mismatch_rejects_playback() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller()
    )
    assert pipeline.start(2)
    assert pipeline.push_playback(tone(480), 1.0, 3) is False
    assert pipeline.push_playback(tone(480), 1.0, 2) is True


# ------------------------------------------------------ real AEC interface (P0)
def test_aec_interface_passes_numpy_int16_arrays() -> None:
    seen: dict[str, np.ndarray] = {}

    class StrictCanceller:
        def process(self, near, far):
            seen["near"] = near
            seen["far"] = far
            return near.copy()

    near = tone(320)
    far = tone(320)
    output = _aec_process_frame(StrictCanceller(), near, far)
    assert seen["near"].dtype == np.int16
    assert seen["far"].dtype == np.int16
    assert seen["near"].shape == (320,)
    assert seen["far"].shape == (320,)
    assert np.array_equal(seen["near"], np.frombuffer(near, dtype="<i2"))
    assert np.array_equal(seen["far"], np.frombuffer(far, dtype="<i2"))
    assert isinstance(output, bytes) and len(output) == 640


def test_aec_interface_rejects_bad_frames_and_results() -> None:
    good = tone(320)

    class Good:
        def process(self, near, far):
            return near

    # Length mismatch / wrong frame size are rejected before the model runs.
    with pytest.raises(AECFormatError):
        _aec_process_frame(Good(), good, tone(480))
    with pytest.raises(AECFormatError):
        _aec_process_frame(Good(), good, b"")

    class BadDtype:
        def process(self, near, far):
            return np.zeros(near.size, dtype=np.float32)

    class BadShape:
        def process(self, near, far):
            return np.zeros(near.size * 2, dtype=np.int16)

    class BadBytes:
        def process(self, near, far):
            return good

    for canceller in (BadDtype(), BadShape(), BadBytes()):
        with pytest.raises((AECFormatError, TypeError)):
            _aec_process_frame(canceller, good, good)


def test_real_pywebrtc_audio_accepts_int16_frames() -> None:
    """Real library check. SKIPPED, not passed, when the extra is absent."""
    pywebrtc = pytest.importorskip(
        "pywebrtc_audio", reason="optional bargein extra not installed"
    )
    try:
        canceller = pywebrtc.EchoCanceller(
            sample_rate=16000, num_channels=1, stream_delay_ms=0
        )
    except TypeError:  # pragma: no cover - binding signature variant
        canceller = pywebrtc.EchoCanceller(16000, 1, 0)

    near = tone(320, amplitude=4000)
    far = tone(320, amplitude=4000)
    first = _aec_process_frame(canceller, near, far)
    assert len(first) == 640
    # The same continuous instance must handle consecutive frames.
    for _ in range(4):
        assert len(_aec_process_frame(canceller, near, far)) == 640
    with pytest.raises((AECFormatError, TypeError, ValueError)):
        _aec_process_frame(canceller, b"", b"")
    with pytest.raises((AECFormatError, TypeError, ValueError)):
        _aec_process_frame(canceller, near, tone(480))


# -------------------------------------------------- reference timeline (P1)
def test_reference_timeline_is_consumed_once_in_order() -> None:
    recorder = RecordingCanceller()
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: recorder
    )
    assert pipeline.start(3)
    jitter = (0.0, 0.003, -0.002, 0.001, -0.003)
    values: list[int] = []
    for index in range(20):
        value = 100 + index
        values.append(value)
        timestamp = 100.0 + index * 0.02 + jitter[index % len(jitter)]
        assert pipeline.push_playback(constant(480, value), timestamp, 3)
        result = pipeline.process_microphone(
            silence(320), 100.02 + index * 0.02
        )
        assert result.valid, result.reason
    consumed = [int(frame[0]) for frame in recorder.fars]
    assert consumed == values  # ordered, no duplicates, no skipped frames
    stats = pipeline.stats()
    assert stats["reference_consumed"] == 20
    assert stats["reference_expired"] == 0


def test_timeline_does_not_substitute_a_future_reference() -> None:
    recorder = RecordingCanceller()
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: recorder
    )
    assert pipeline.start(1)
    assert pipeline.push_playback(constant(480, 500), 50.5, 1)
    ahead = pipeline.process_microphone(silence(320), 50.0)
    assert not ahead.valid
    assert ahead.reason == "ahead_reference"
    assert recorder.fars == []  # a future frame was not consumed early

    reached = pipeline.process_microphone(silence(320), 50.52)
    assert reached.valid
    assert len(recorder.fars) == 1
    assert int(recorder.fars[0][0]) == 500


def test_playback_and_microphone_jumps_are_counted() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller()
    )
    assert pipeline.start(1)
    assert pipeline.push_playback(tone(480), 10.0, 1)
    assert pipeline.process_microphone(tone(320), 10.02).valid
    # A large playback timestamp jump restarts the reference timeline.
    assert pipeline.push_playback(tone(480), 11.0, 1)
    assert pipeline.process_microphone(tone(320), 11.02).valid
    # A microphone timebase jump is counted, not silently treated as continuous.
    jump = pipeline.process_microphone(tone(320), 12.5)
    assert not jump.valid
    stats = pipeline.stats()
    assert stats["sync_discontinuities"] >= 1
    assert stats["mic_discontinuities"] >= 1


# --------------------------------------------- non-blocking playback (P1)
def test_push_playback_never_waits_for_a_slow_aec_frame() -> None:
    entered = threading.Event()
    release = threading.Event()

    class SlowCanceller:
        def process(self, near, far):
            entered.set()
            assert release.wait(2.0)
            return near.copy()

    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: SlowCanceller()
    )
    assert pipeline.start(1)
    assert pipeline.push_playback(tone(480), 10.0, 1)
    box: dict[str, MicProcessingResult] = {}

    def run_aec() -> None:
        box["result"] = pipeline.process_microphone(tone(320), 10.02)

    worker = threading.Thread(target=run_aec, daemon=True)
    worker.start()
    assert entered.wait(1.0)
    started = time.monotonic()
    assert pipeline.push_playback(tone(480), 10.02, 1) is True
    elapsed = time.monotonic() - started
    assert elapsed < 0.2, f"push_playback waited {elapsed:.3f}s for AEC"
    release.set()
    worker.join(timeout=2.0)
    assert box["result"].valid


# ------------------------------------- confirmed-turn continuity (P0/P1)
def test_confirmed_tail_keeps_audio_continuous_after_references_run_out() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller()
    )
    assert pipeline.start(9)
    assert pipeline.push_playback(tone(480), 100.0, 9)
    assert pipeline.process_microphone(tone(320), 100.02).valid
    pipeline.stop(9)
    assert pipeline.begin_echo_tail(9) is True

    chunks = [tone(320, amplitude=6000) for _ in range(25)]
    cleaned: list[bytes] = []
    for index, chunk in enumerate(chunks):
        result = pipeline.process_microphone(
            chunk, 100.04 + index * 0.02
        )
        assert result.valid, result.reason
        assert len(result.pcm) == 640
        cleaned.append(result.pcm)
    # Every sample of the confirmed turn is kept, unchanged and in order.
    assert b"".join(cleaned) == b"".join(chunks)
    stats = pipeline.stats()
    assert stats["tail_active"] is True
    assert stats["tail_frames"] >= 20
    assert stats["reference_consumed"] == 1


def test_tail_mode_is_bounded_and_reports_expiry() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller(),
        tail_max_seconds=0.05,
    )
    assert pipeline.start(1)
    assert pipeline.begin_echo_tail(1)
    time.sleep(0.08)
    result = pipeline.process_microphone(tone(320), 10.0)
    assert not result.valid
    assert result.reason == "tail_expired"


def test_tail_mode_refuses_a_non_matching_generation() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller()
    )
    assert pipeline.start(4)
    assert pipeline.begin_echo_tail(5) is False
    assert pipeline.tail_active is False


def test_dual_speaker_voice_survives_echo_cancellation() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: EchoSubtractingCanceller()
    )
    assert pipeline.start(1)
    segmenter = VADSegmenter(energy_probability, _config())
    segmenter.configure(threshold=0.65, min_speech_ms=192, pre_roll_ms=400)

    echo = np.full(320, 3000, dtype=np.int32)
    voice = np.array(
        [8000 if index % 2 else -8000 for index in range(320)],
        dtype=np.int32,
    )
    near = np.clip(echo + voice, -32768, 32767).astype(np.int16).tobytes()

    events = []
    for index in range(20):
        timestamp = 500.0 + index * 0.02
        assert pipeline.push_playback(constant(480, 3000), timestamp, 1)
        result = pipeline.process_microphone(near, timestamp)
        assert result.valid, result.reason
        events.extend(segmenter.process(result.pcm))
    # The echo was removed but the independent speaker still reaches VAD.
    assert any(event.kind == SPEECH_STARTED for event in events)


# --------------------------------------------------------------- manager path
class _FakeMic:
    description = "Fake Mic"
    dropped_chunks = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _RecordingSegmenter:
    """Accepts any PCM (no real Silero) and records what it was fed."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def process(self, pcm: bytes):
        self.calls.append(bytes(pcm))
        return []

    def reset(self) -> None:
        pass


class _RecordingVAD(VADSegmenter):
    """A real segmenter (fake probability function) that records its input."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[bytes] = []

    def process(self, pcm: bytes):
        self.calls.append(bytes(pcm))
        return super().process(pcm)


class _FakePipeline:
    available = True
    healthy = True
    unavailable_reason = ""
    degraded = False

    def __init__(self) -> None:
        self.cleaned = b"clean-audio"
        self.valid = True
        self.raise_on_process = False
        self.started: list[int] = []
        self.stopped: list[int | None] = []
        self.processed = 0

    def start(self, generation: int) -> bool:
        self.started.append(generation)
        return True

    def stop(self, generation: int | None = None) -> None:
        self.stopped.append(generation)

    def reset(self) -> None:
        pass

    def stats(self) -> dict:
        return {"valid_ratio": 1.0}

    def process_microphone(self, pcm: bytes, timestamp: float):
        self.processed += 1
        if self.raise_on_process:
            raise RuntimeError("boom")
        return MicProcessingResult(
            pcm=self.cleaned if self.valid else b"",
            valid=self.valid,
            reason="ok" if self.valid else "no_reference",
        )


def test_manager_feeds_only_aec_valid_audio_to_vad() -> None:
    segmenter = _RecordingSegmenter()
    pipeline = _FakePipeline()
    manager = ASRManager(
        _config(),
        microphone=_FakeMic(),
        vad_segmenter=segmenter,
        renderer=_renderer(),
    )
    manager.attach_echo_canceller(pipeline)
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: manager._started)
        assert manager.begin_barge_in(4) is True
        assert pipeline.started == [4]
        manager.audio_queue.put(
            AudioChunk(tone(640), 16000, 16000, time.monotonic())
        )
        assert wait_for(lambda: segmenter.calls)
        assert segmenter.calls[0] == b"clean-audio"

        pipeline.valid = False
        before = len(segmenter.calls)
        manager.audio_queue.put(
            AudioChunk(tone(640), 16000, 16000, time.monotonic())
        )
        assert wait_for(lambda: pipeline.processed >= 2)
        time.sleep(0.1)
        assert len(segmenter.calls) == before
    finally:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()


def test_manager_emits_barge_in_before_started_with_one_uid() -> None:
    manager = ASRManager(
        _config(),
        microphone=_FakeMic(),
        vad_segmenter=_RecordingSegmenter(),
        renderer=_renderer(),
    )
    activities: list[SpeechActivity] = []
    manager.set_listeners(on_speech=activities.append)
    # Pretend a session is already running so no sender threads start.
    manager._active_sessions = manager.config.max_concurrent_sessions
    manager._barge_in_generation = 9

    manager._handle(_SpeechStarted(b"pcm", time.monotonic(), barge_in=True))
    kinds = [activity.kind for activity in activities]
    assert kinds[:2] == ["barge_in", "started"]
    assert activities[0].utterance_id == activities[1].utterance_id
    assert activities[0].generation == 9
    assert manager.barge_in_stats()["confirmed"] == 1


def test_manager_falls_back_when_pipeline_raises() -> None:
    segmenter = _RecordingSegmenter()
    pipeline = _FakePipeline()
    pipeline.raise_on_process = True
    manager = ASRManager(
        _config(),
        microphone=_FakeMic(),
        vad_segmenter=segmenter,
        renderer=_renderer(),
    )
    manager.attach_echo_canceller(pipeline)
    reasons: list[str] = []
    manager.set_barge_in_unavailable_listener(reasons.append)
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: manager._started)
        assert manager.begin_barge_in(5) is True
        manager.audio_queue.put(
            AudioChunk(tone(640), 16000, 16000, time.monotonic())
        )
        assert wait_for(lambda: reasons)
        assert wait_for(
            lambda: manager.input_mode in (InputMode.MUTED, InputMode.NORMAL)
        )
        # Raw microphone audio was never handed to VAD.
        assert segmenter.calls == []
    finally:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()


class _TailPipeline:
    """Pipeline that runs out of references once playback is stopped."""

    available = True
    healthy = True
    unavailable_reason = ""
    degraded = False

    def __init__(self) -> None:
        self.started: list[int] = []
        self.stopped: list[int | None] = []
        self.exhausted = False
        self._tail = False

    @property
    def tail_active(self) -> bool:
        return self._tail

    def start(self, generation: int) -> bool:
        self.started.append(generation)
        self.exhausted = False
        self._tail = False
        return True

    def stop(self, generation: int | None = None) -> None:
        self.stopped.append(generation)
        self.exhausted = True

    def reset(self) -> None:
        self.exhausted = False
        self._tail = False

    def begin_echo_tail(self, generation: int) -> bool:
        self._tail = True
        return True

    def end_echo_tail(self) -> None:
        self._tail = False

    def stats(self) -> dict:
        return {}

    def process_microphone(self, pcm: bytes, timestamp: float):
        if self._tail or not self.exhausted:
            return MicProcessingResult(
                pcm=bytes(pcm), valid=True, reason="ok"
            )
        return MicProcessingResult(valid=False, reason="no_reference")


def test_confirmed_barge_in_turn_survives_reference_exhaustion() -> None:
    config = _config()
    segmenter = _RecordingVAD(energy_probability, config)
    pipeline = _TailPipeline()
    manager = ASRManager(
        config,
        microphone=_FakeMic(),
        vad_segmenter=segmenter,
        renderer=_renderer(),
    )
    activities: list[SpeechActivity] = []
    manager.set_listeners(on_speech=activities.append)
    # No DashScope session in this offline test.
    manager._active_sessions = config.max_concurrent_sessions
    manager.attach_echo_canceller(pipeline)
    assert manager.begin_barge_in(3) is True
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: manager._started)
        now = time.monotonic()
        for index in range(12):
            manager.audio_queue.put(
                AudioChunk(tone(320), 16000, 16000, now + index * 0.02)
            )
        assert wait_for(
            lambda: any(item.kind == "barge_in" for item in activities)
        )
        # Drain the detection chunks before switching phases.
        assert wait_for(lambda: manager.audio_queue.empty())
        time.sleep(0.1)

        # What the coordinator does on a confirmed interruption: stop TTS
        # (playback intake ends) and keep the turn continuous.
        manager.end_barge_in(3)
        manager.confirm_barge_in(3)
        assert pipeline.tail_active is True

        before = len(segmenter.calls)
        tail_chunks = [tone(320, amplitude=9000) for _ in range(20)]
        for index, chunk in enumerate(tail_chunks):
            manager.audio_queue.put(
                AudioChunk(
                    chunk, 16000, 16000, now + (12 + index) * 0.02
                )
            )
        assert wait_for(lambda: len(segmenter.calls) >= before + 20)
        # Reference exhaustion during the echo tail loses no audio, and the
        # samples arrive once, in order.
        assert segmenter.calls[-20:] == tail_chunks
        kinds = [item.kind for item in activities]
        assert kinds.count("barge_in") == 1
        assert kinds.count("started") == 1
        assert activities[0].utterance_id == activities[1].utterance_id
        assert len(manager._utterances) == 1
    finally:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()


# ----------------------------------------------------------- coordinator (C-F)
class _FakeASR:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model="fake-asr")
        self.microphone = SimpleNamespace(description="Fake Mic")
        self.dropped_chunks = 0
        self.input_states: list[bool] = []
        self.modes: list[str] = []
        self.on_final = None
        self.on_speech = None
        self.on_closed = None
        self.unavailable_listener = None
        self.barge_started: list[int] = []
        self.barge_stopped: list[int | None] = []
        self.barge_confirmed: list[int] = []
        self.tails_ended = 0
        self._stop = asyncio.Event()

    # -- manager contract --------------------------------------------------
    def set_listeners(self, *, on_final=None, on_speech=None, on_turn_closed=None, render_finals=None):
        self.on_final = on_final
        self.on_speech = on_speech
        self.on_closed = on_turn_closed

    def start(self) -> None:
        pass

    def run(self) -> None:
        time.sleep(0.05)

    def stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def set_input_enabled(self, enabled: bool) -> None:
        self.input_states.append(bool(enabled))

    def set_input_mode(self, mode) -> bool:
        self.modes.append(str(mode))
        return True

    def begin_barge_in(self, generation: int) -> bool:
        self.barge_started.append(generation)
        self.modes.append("BARGE_IN")
        return True

    def end_barge_in(self, generation=None) -> None:
        self.barge_stopped.append(generation)

    def confirm_barge_in(self, generation: int) -> bool:
        self.barge_confirmed.append(generation)
        return True

    def end_echo_tail(self) -> None:
        self.tails_ended += 1

    def barge_in_available(self) -> bool:
        return True

    def set_barge_in_unavailable_listener(self, listener) -> None:
        self.unavailable_listener = listener

    def barge_in_stats(self) -> dict:
        return {
            "candidates": 1,
            "confirmed": 1,
            "last_user_speech_to_confirm_ms": 210.0,
            "aec": {"valid_ratio": 0.99, "aec_p50_ms": 1.2},
        }

    # -- test-side injection ----------------------------------------------
    def speech(self, kind: str, uid: int, generation=None) -> None:
        self.on_speech(
            SpeechActivity(kind, uid, time.monotonic(), generation=generation)
        )

    def final(self, uid: int, text: str) -> None:
        self.on_final(FinalUtterance(uid, text, time.monotonic()))


class _FakeTTS:
    enabled = True

    def __init__(self) -> None:
        self.stops = 0

    def add_playback_listener(self, listener) -> None:
        pass

    def remove_playback_listener(self, listener) -> None:
        pass

    def stop(self, **kwargs) -> None:
        self.stops += 1


async def _make_coordinator(**kwargs):
    fake = _FakeASR()
    tts = _FakeTTS()
    cancelled: list[int] = []
    coordinator = VoiceCoordinator(
        manager_factory=lambda: fake,
        tts_manager=tts,
        cooldown_seconds=0.0,
        barge_in=kwargs.pop(
            "barge_in", BargeInConfig(enabled=True, echo_tail_ms=0)
        ),
        on_barge_in=cancelled.append,
        **kwargs,
    )
    ok, _message = await coordinator.enable()
    assert ok
    await settle()
    return coordinator, fake, tts, cancelled


async def test_c_barge_in_stops_tts_and_enters_recording() -> None:
    coordinator, fake, tts, cancelled = await _make_coordinator()
    try:
        token = coordinator.begin_turn()
        coordinator.on_reply_started(7, token=token)
        assert coordinator.state == VoiceState.SPEAKING
        assert fake.barge_started == [7]

        fake.speech("barge_in", uid=5, generation=7)
        await settle()
        assert tts.stops == 1
        assert cancelled == [7]
        assert coordinator.state == VoiceState.RECORDING
        assert fake.barge_stopped == [7]
        assert fake.barge_confirmed == [7]
        assert coordinator.barge_stats["confirmed"] == 1

        # The started notice that follows is not a second turn.
        fake.speech("started", uid=5)
        await settle()
        assert coordinator.state == VoiceState.RECORDING

        fake.final(5, "等一下，我想说别的")
        final = await asyncio.wait_for(coordinator.next_final(), 1.0)
        assert final.text == "等一下，我想说别的"
        assert coordinator.stats.finals_accepted == 1
    finally:
        await coordinator.shutdown()


async def test_f_stale_callbacks_cannot_restore_old_state() -> None:
    coordinator, fake, tts, canceled = await _make_coordinator()
    try:
        old_token = coordinator.begin_turn()
        coordinator.on_reply_started(7, token=old_token)
        fake.speech("barge_in", uid=5, generation=7)
        await settle()
        assert coordinator.state == VoiceState.RECORDING

        # Old playback finish and the old turn's late completion.
        coordinator.on_turn_finished({"delivered": True}, token=old_token)
        coordinator.on_turn_finished({"delivered": True})
        await settle()
        assert coordinator.state == VoiceState.RECORDING

        fake.final(5, "新的问题")
        assert (await asyncio.wait_for(coordinator.next_final(), 1.0)).text == "新的问题"

        new_token = coordinator.begin_turn()
        coordinator.on_reply_started(8, token=new_token)
        assert coordinator.state == VoiceState.SPEAKING
        # The interrupted turn unwinds late; it must not resume the mic.
        coordinator.on_turn_finished({"delivered": False}, token=old_token)
        await settle()
        assert coordinator.state == VoiceState.SPEAKING
        assert coordinator._expected_generation == 8
    finally:
        await coordinator.shutdown()


async def test_h_duplicate_barge_in_stops_once_and_one_turn() -> None:
    coordinator, fake, tts, canceled = await _make_coordinator()
    try:
        token = coordinator.begin_turn()
        coordinator.on_reply_started(7, token=token)
        fake.speech("barge_in", uid=5, generation=7)
        fake.speech("barge_in", uid=5, generation=7)
        fake.speech("started", uid=5)
        await settle()
        assert tts.stops == 1
        assert canceled == [7]
        assert coordinator.state == VoiceState.RECORDING

        fake.final(5, "一次")
        assert (await asyncio.wait_for(coordinator.next_final(), 1.0)).text == "一次"
        # A second, different utterance arriving while the reply is being
        # prepared must not start a parallel turn.
        fake.final(6, "一次")
        await settle()
        assert coordinator.stats.stale_finals >= 1
    finally:
        await coordinator.shutdown()


async def test_g_aec_unavailable_falls_back_and_reports_once() -> None:
    notices: list[str] = []
    coordinator, fake, tts, _canceled = await _make_coordinator(
        on_notice=notices.append
    )
    try:
        token = coordinator.begin_turn()
        coordinator.on_reply_started(7, token=token)
        assert coordinator.state == VoiceState.SPEAKING
        assert fake.unavailable_listener is not None
        fake.unavailable_listener("aec:RuntimeError")
        fake.unavailable_listener("aec:RuntimeError")
        await settle()
        assert coordinator.state == VoiceState.SPEAKING
        assert fake.modes[-1] == "MUTED"
        assert coordinator.barge_stats["fallbacks"] == 2
        # One user-visible notice even with repeated failures.
        assert sum("实时打断不可用" in text for text in notices) == 1
    finally:
        await coordinator.shutdown()


async def test_i_disabled_feature_never_arms_barge_in() -> None:
    coordinator, fake, tts, canceled = await _make_coordinator(
        barge_in=BargeInConfig(enabled=False)
    )
    try:
        token = coordinator.begin_turn()
        coordinator.on_reply_started(7, token=token)
        assert coordinator.state == VoiceState.SPEAKING
        assert fake.barge_started == []
        # Half-duplex: the mic stays paused while Ava speaks.
        assert fake.input_states[-1] is False
        assert coordinator.barge_in_status()["enabled"] is False
    finally:
        await coordinator.shutdown()


# ----------------------------------------------------- streaming cancellation
class _FakeStream:
    def __init__(self) -> None:
        self.closed = False
        self.fed: list[str] = []
        self.generation = 11

    def note_first_text(self) -> None:
        pass

    def feed(self, text: str) -> None:
        self.fed.append(text)

    def finish(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    async def __aiter__(self):
        yield SimpleNamespace(
            type="response_started",
            response_id="resp-1",
            conversation_id="conv-1",
            router_completed_at=time.monotonic(),
        )
        yield SimpleNamespace(
            type="response_delta",
            delta="这句话不应该继续",
            display_delta=None,
        )
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


class _FakeStreamTTS:
    enabled = True

    def __init__(self) -> None:
        self.cancelled = []
        self.stream = _FakeStream()

    def start_stream(self, *args, **kwargs):
        return self.stream

    def cancel_stream(self, stream) -> None:
        self.cancelled.append(stream)

    def stop(self, **kwargs) -> None:
        pass


class _Out:
    def __init__(self) -> None:
        self.text = ""

    async def write(self, text: str) -> None:
        self.text += text

    async def flush(self) -> None:
        pass


def test_tts_rechecks_generation_after_the_reference_listener() -> None:
    """A cancel between listener and write must not reach the speaker."""
    from companion.tts.manager import TTSManager

    class Player:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.writes.append(data)

    manager = TTSManager()
    player = Player()
    manager._player = player
    manager._started = True
    manager._generation = 1
    manager._config = SimpleNamespace(
        sample_rate=24000, sample_width=2, channels=1
    )
    seen: list[bytes] = []

    def listener(generation: int, frame: bytes, timestamp: float) -> None:
        seen.append(frame)
        if len(seen) == 3:
            # Simulates TTSManager.stop() on another thread while the
            # reference was being delivered.
            with manager._lock:
                manager._generation = 2

    manager.set_playback_pcm_listener(listener)
    manager._write_playback(1, b"\x01\x00" * (480 * 6))
    assert len(seen) == 3
    assert len(player.writes) == 2  # the superseded frame was never written


async def test_e_interrupted_stream_is_cancelled_and_closed() -> None:
    tts = _FakeStreamTTS()
    out = _Out()
    started: list[int] = []

    async def on_tts_started(generation: int) -> None:
        started.append(generation)

    rt = SimpleNamespace(
        orchestrator=SimpleNamespace(
            process_utterance_stream=lambda event: tts.stream
        )
    )
    event = SimpleNamespace(event_id="event-1")

    task = asyncio.create_task(
        _run_streaming_turn(
            rt,
            event,
            tts_manager=tts,
            submitted_at=time.monotonic(),
            output=out,
            on_tts_started=on_tts_started,
        )
    )
    await settle()
    assert started == [11]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The old reply's speech stream was cancelled and the generator closed, so
    # the orchestrator can persist CANCELLED and release its conversation lock.
    assert tts.cancelled == [tts.stream]
    assert tts.stream.closed is True


# ---------------------------------------------------------------- configuration
def test_timeline_offset_is_separate_from_the_aec_device_hint() -> None:
    cfg = load_barge_in_config({})
    assert cfg.timeline_offset_ms == 0.0
    assert cfg.aec_delay_ms == 0

    cfg = load_barge_in_config(
        {
            "AVA_BARGE_IN_AEC_DELAY_MS": "120",
            "AVA_BARGE_IN_TIMELINE_OFFSET_MS": "-35.5",
        }
    )
    # The device hint and the measured timeline calibration stay independent.
    assert cfg.aec_delay_ms == 120
    assert cfg.timeline_offset_ms == -35.5

    with pytest.raises(BargeInConfigError):
        load_barge_in_config({"AVA_BARGE_IN_TIMELINE_OFFSET_MS": "600"})
    with pytest.raises(BargeInConfigError):
        load_barge_in_config({"AVA_BARGE_IN_TIMELINE_OFFSET_MS": "abc"})
