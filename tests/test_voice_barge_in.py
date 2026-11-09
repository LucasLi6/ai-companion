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
from companion.voice.barge_in_config import BargeInConfig
from companion.voice.echo_canceller import (
    EchoCancellationPipeline,
    MicProcessingResult,
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


class PassThroughCanceller:
    """Models perfect AEC: the near signal survives unchanged."""

    def process(self, near: bytes, far: bytes) -> bytes:
        assert len(near) == len(far)
        return near


class SilenceCanceller:
    """Models perfect echo removal: only residual silence remains."""

    def process(self, near: bytes, far: bytes) -> bytes:
        assert len(near) == len(far)
        return b"\x00\x00" * (len(near) // 2)


class ExplodingCanceller:
    def process(self, near: bytes, far: bytes) -> bytes:
        raise RuntimeError("aec exploded")


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
    assert stale.reason in ("stale_reference", "ahead_reference")


def test_g_generation_mismatch_rejects_playback() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller()
    )
    assert pipeline.start(2)
    assert pipeline.push_playback(tone(480), 1.0, 3) is False
    assert pipeline.push_playback(tone(480), 1.0, 2) is True


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
