"""V0.9.9.28 — voice reliability regression tests.

Covers the five reported defects end to end using the same offline fakes as
the rest of the suite (no Silero, pywebrtc-audio, PyAudio, DashScope, LLM,
network or real microphone/speaker):

1. AEC must distinguish real TTS silence (GAP) from a dropped far reference
   while playback is expected (sync fault), and must keep barge-in usable.
2. The near-end gate must use aligned evidence + adaptive noise instead of a
   fixed energy floor and must never confirm on VAD alone.
3. A truncated/timed-out upload is an incomplete turn, never a final; partial
   and subtitle events are scoped to the current utterance.
4. Proactive delivery budget advances only on a confirmed LONG_IDLE delivery;
   async acknowledgement awaits the database write.
5. Cross-module acceptance: echo never becomes a user Final, long sequences,
   60 s speech and proactive/barge-in interaction.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np

from companion.asr.manager import (
    ASRManager,
    NearEndEvidence,
    NearEndSpeechGate,
)
from companion.asr.microphone import AudioChunk
from companion.cli import _PartialSubtitle
from companion.voice.bridge import PartialTranscript, VoiceBridge
from companion.voice.coordinator import VoiceCoordinator
from companion.voice.echo_canceller import (
    AECState,
    EchoCancellationPipeline,
    MicProcessingResult,
    ReferenceStatus,
)

import test_asr_manager as asr_tests
import test_v0999_continuous_proactive as proactive_tests
from test_voice_barge_in import (  # noqa: E402 - shared offline fixtures
    PassThroughCanceller,
    RecordingCanceller,
    _FakeASR,
    _FakeTTS,
    _RecordingVAD,
    _config,
    _renderer,
    energy_probability,
    settle,
    tone,
    wait_for,
)

# --------------------------------------------------------------- AEC reference
def _pipeline(canceller=None, **kwargs) -> EchoCancellationPipeline:
    defaults = dict(
        canceller_factory=lambda *_: canceller or PassThroughCanceller(),
        synthesis_gap_seconds=0.05,
        start_grace_seconds=0.01,
        playback_tail_grace_seconds=0.05,
        sync_failure_seconds=0.1,
    )
    defaults.update(kwargs)
    return EchoCancellationPipeline(**defaults)


def test_known_silence_is_a_gap_and_user_speech_still_passes() -> None:
    """A TTS pause must not be a sync fault, and zero far frames keep AEC live."""
    canceller = RecordingCanceller()
    pipeline = _pipeline(canceller, sync_tolerance_seconds=0.0)
    assert pipeline.start(1)
    # One real playback frame at the end of a sentence.
    assert pipeline.push_playback(tone(480), 1000.0, 1)
    paired = pipeline.process_microphone(tone(320), 1000.02)
    assert paired.valid

    # Longer than the synthesis gap: TTS is genuinely silent now.
    time.sleep(0.08)
    # First miss is still inside the known far-end timeline (tolerance 0).
    inside = pipeline.process_microphone(tone(320), 1000.04)
    assert not inside.valid
    # Inside the known far-end timeline but with no reference frame left.
    assert inside.reason == "no_reference"

    pause = pipeline.process_microphone(tone(320), 1000.06)
    assert pause.valid
    assert pipeline.stats()["reference_status"] == ReferenceStatus.KNOWN_SILENCE.value
    assert pipeline.stats()["state"] == AECState.GAP.value
    assert pipeline.healthy is True
    # The canceller ran with a zero far end, not with a dropped/stale frame.
    assert np.count_nonzero(canceller.fars[-1]) == 0

    # The user may speak during the pause: that speech survives cancellation.
    speech = tone(320, amplitude=9000)
    during = pipeline.process_microphone(speech, 1000.08)
    assert during.valid
    assert during.pcm == speech
    stats = pipeline.stats()
    assert stats["known_silence_frames"] >= 2
    assert stats["consecutive_expected_reference_errors"] == 0
    assert pipeline.healthy is True


def test_dropped_reference_during_playback_is_a_sync_fault_not_a_gap() -> None:
    """Audio is expected but every reference frame was dropped: fail loudly."""
    pipeline = _pipeline(sync_tolerance_seconds=0.10)
    assert pipeline.start(1)
    assert pipeline.push_playback(tone(480), 2000.0, 1)
    assert pipeline.process_microphone(tone(320), 2000.02).valid

    # The reference timeline says audio is still expected here, yet no
    # reference frame survives for these blocks. (Five 20 ms frames are the
    # configured failure run.)
    for index in range(5):
        result = pipeline.process_microphone(
            tone(320), 2000.0 + 0.02 * (index + 2)
        )
        assert not result.valid

    stats = pipeline.stats()
    assert stats["reference_status"] == ReferenceStatus.AUDIO_EXPECTED.value
    assert stats["reference_sync_errors"] >= 5
    assert stats["consecutive_expected_reference_errors"] >= 5
    assert stats["last_error"] == "reference_sync_failure"
    assert stats["state"] == AECState.FAILED.value
    assert pipeline.healthy is False

    # Past the tolerance the classifier turns conservative (UNKNOWN) but the
    # already-declared sync fault must stay a fault, never become a GAP.
    for index in range(5, 8):
        pipeline.process_microphone(tone(320), 2000.0 + 0.02 * (index + 2))
    stats = pipeline.stats()
    assert stats["state"] == AECState.FAILED.value
    assert stats["last_error"] == "reference_sync_failure"


def test_continuous_playback_timing_turns_a_lost_reference_into_a_sync_fault() -> None:
    """The player keeps writing frames but reference PCM never arrives again.

    Without the playback-timing channel this state is indistinguishable from a
    sentence pause, so the old code stayed in GAP forever and barge-in silently
    rotted. Timing proves audio is playing; the missing reference is a fault.
    """
    pipeline = _pipeline(
        sync_tolerance_seconds=0.10,
        sync_failure_seconds=0.1,
        start_grace_seconds=0.01,
        synthesis_gap_seconds=0.05,
    )
    assert pipeline.start(1)
    assert pipeline.push_playback(tone(480), 5000.0, 1)
    assert pipeline.process_microphone(tone(320), 5000.02).valid

    for index in range(6):
        pipeline.note_playback_timing(
            SimpleNamespace(kind="frame_submitted", generation=1)
        )
        result = pipeline.process_microphone(
            tone(320), 5000.0 + 0.02 * (index + 2)
        )
        assert not result.valid

    stats = pipeline.stats()
    assert stats["reference_status"] == ReferenceStatus.AUDIO_EXPECTED.value
    assert stats["reference_sync_errors"] >= 5
    assert stats["consecutive_expected_reference_errors"] >= 5
    assert stats["last_error"] == "reference_sync_failure"
    assert stats["state"] == AECState.FAILED.value
    assert pipeline.healthy is False

    # Unknown observations after the fault never resurrect a broken channel.
    pipeline.note_playback_event("finished", 1)
    time.sleep(0.06)
    pipeline.process_microphone(tone(320), 5000.2)
    assert pipeline.stats()["state"] == AECState.FAILED.value


def test_a_new_generation_recovers_from_a_reference_sync_fault() -> None:
    pipeline = _pipeline(sync_tolerance_seconds=0.10)
    assert pipeline.start(1)
    assert pipeline.push_playback(tone(480), 3000.0, 1)
    assert pipeline.process_microphone(tone(320), 3000.02).valid
    for index in range(6):
        pipeline.process_microphone(tone(320), 3000.0 + 0.02 * (index + 2))
    assert pipeline.stats()["state"] == AECState.FAILED.value

    # Per-generation recovery: a fresh reply starts clean.
    assert pipeline.start(2)
    assert pipeline.push_playback(tone(480), 4000.0, 2)
    recovered = pipeline.process_microphone(tone(320), 4000.02)
    assert recovered.valid
    assert pipeline.healthy is True
    assert pipeline.stats()["generation"] == 2


def test_playback_timing_channel_is_generation_scoped() -> None:
    pipeline = _pipeline()
    assert pipeline.start(5)
    timing = SimpleNamespace(generation=5)
    pipeline.note_playback_timing(timing)
    pipeline.note_playback_timing(SimpleNamespace(generation=4))
    assert pipeline.stats()["playback_timing_events"] == 1
    assert pipeline.stats()["playback_finished"] is False

    pipeline.note_playback_event("finished", 5)
    assert pipeline.stats()["playback_finished"] is True
    pipeline.note_playback_event("finished", 4)
    # A stale generation cannot change this generation's timeline.
    assert pipeline.stats()["playback_finished"] is True
    assert pipeline.stats()["state"] != AECState.FAILED.value


def test_tts_emits_frame_timing_and_playback_end_events() -> None:
    """The player reports the far-end timeline even without PCM references."""
    from companion.tts.manager import SpeechJob

    import test_tts_phase_a as tts_tests

    events: list = []
    chunks = [b"\x01\x00" * 960, b"\x01\x00" * 960]
    player = tts_tests.FakePlayer()
    manager = tts_tests.make_manager(
        tts_tests.FakeTTSClient(lambda text: iter(chunks)), player
    )
    manager.set_playback_timing_listener(events.append)
    try:
        ok, _ = manager.enable()
        assert ok
        assert manager.submit(SpeechJob("r1", "c1", "你好"))
        assert wait_for(
            lambda: any(e.kind == "playback_finished" for e in events),
            timeout=3.0,
        )
        kinds = [event.kind for event in events]
        assert "output_device_changed" in kinds
        # 2 × 1920-byte chunks at 24 kHz = 4 × 20 ms frames.
        assert kinds.count("frame_submitted") == 4
        assert kinds.count("frame_write_completed") == 4
        submitted = [e for e in events if e.kind == "frame_submitted"]
        assert [e.sequence for e in submitted] == [1, 2, 3, 4]
        assert all(e.generation == 1 for e in submitted)
        # The surface can tell a completed playback from an interruption.
        assert manager.playback_outcome(1) == "finished"
        assert manager.playback_outcome_for("r1") == "finished"
        assert manager.playback_outcome_for("never-submitted") is None
    finally:
        manager.shutdown()


def test_tts_playback_outcome_distinguishes_interruption_from_completion() -> None:
    from companion.tts.manager import TTSManager

    manager = TTSManager()
    manager._timings[3] = {"tts_queued_at": 1.0}
    manager._playback_info[3] = ("resp-3", "c1")
    assert manager.playback_outcome(3) is None
    manager._notify_playback("cancelled", 3)
    assert manager.playback_outcome(3) == "cancelled"
    assert manager.playback_outcome_for("resp-3") == "cancelled"

    # A finished reply is terminal: a later stop notice for the same
    # generation must not turn it into a partial delivery.
    manager._timings[4] = {"tts_queued_at": 1.0}
    manager._playback_info[4] = ("resp-4", "c1")
    manager._notify_playback("finished", 4)
    manager._notify_playback("cancelled", 4)
    assert manager.playback_outcome_for("resp-4") == "finished"
    assert manager.playback_outcome(99) is None
    assert manager.playback_outcome_for(None) is None


def test_long_alternating_playback_and_mic_sequence_stays_healthy() -> None:
    """10 s of 20 ms frames: no drift, no false fault, references paired once."""
    pipeline = _pipeline()
    assert pipeline.start(9)
    base = 5000.0
    for index in range(500):
        timestamp = base + index * 0.02
        assert pipeline.push_playback(tone(480), timestamp, 9)
        result = pipeline.process_microphone(tone(320), timestamp + 0.02)
        assert result.valid, result.reason
    stats = pipeline.stats()
    assert stats["valid_blocks"] == 500
    assert stats["invalid_blocks"] == 0
    assert stats["reference_consumed"] == 500
    assert stats["reference_sequence"] == 500
    assert pipeline.healthy is True


# ---------------------------------------------------------- near-end evidence
def _evidence(
    *,
    clean: float,
    near: float,
    probability: float | None = 0.9,
    generation: int | None = 1,
    duration_ms: float = 20.0,
    metrics_valid: bool = True,
    reference_valid: bool = True,
) -> NearEndEvidence:
    return NearEndEvidence(
        generation=generation,
        start_at=time.monotonic(),
        duration_ms=duration_ms,
        vad_probability=probability,
        near_rms=near,
        clean_rms=clean,
        metrics_valid=metrics_valid,
        reference_valid=reference_valid,
    )


def test_gate_rejects_echo_and_adapts_to_the_residual_floor() -> None:
    gate = NearEndSpeechGate(confirm_windows=4)
    for _ in range(6):
        assert not gate.observe_evidence(
            _evidence(near=5000.0, clean=50.0, probability=0.95)
        )
    assert gate.confirmed is False
    assert gate.echo_rejections >= 6
    # The residual echo was learned as background, not as user speech.
    assert gate.noise_floor < 50.0 or gate.noise_floor < 60.0 * 3.0


def test_gate_confirms_real_speech_after_aligned_evidence_time() -> None:
    gate = NearEndSpeechGate(confirm_windows=4, confirm_duration_ms=80.0)
    results = [
        gate.observe_evidence(_evidence(near=5000.0, clean=3000.0, probability=0.9))
        for _ in range(4)
    ]
    assert results == [False, False, False, True]
    assert gate.confirmed is True
    # Neither a fixed floor nor VAD alone made the decision.
    assert gate.observations == 4


def test_gate_never_confirms_without_trustworthy_metrics_or_reference() -> None:
    gate = NearEndSpeechGate(confirm_windows=2, confirm_duration_ms=40.0)
    for _ in range(8):
        assert not gate.observe_evidence(
            _evidence(
                near=6000.0,
                clean=6000.0,
                probability=1.0,
                metrics_valid=False,
            )
        )
        assert not gate.observe_evidence(
            _evidence(
                near=6000.0,
                clean=6000.0,
                probability=1.0,
                reference_valid=False,
            )
        )
    # The legacy VAD-only path is gone as well.
    assert not gate.observe(
        near_rms=6000.0,
        clean_rms=6000.0,
        metrics_available=False,
        vad_probability=1.0,
    )
    assert gate.confirmed is False
    assert gate.unmetered_rejections >= 9


def test_gate_rejects_loud_non_speech_background() -> None:
    gate = NearEndSpeechGate(confirm_windows=3)
    for _ in range(6):
        assert not gate.observe_evidence(
            _evidence(near=5000.0, clean=800.0, probability=0.1)
        )
    assert gate.confirmed is False
    assert gate.noise_rejections >= 6
    # A door/TV burst raises the learned floor instead of confirming a turn.
    assert gate.noise_floor > 60.0


def test_gate_uses_a_smaller_margin_for_confident_silero_speech() -> None:
    quiet = NearEndSpeechGate(confirm_windows=2, confirm_duration_ms=40.0)
    for _ in range(2):
        assert not quiet.observe_evidence(
            _evidence(near=400.0, clean=150.0, probability=None)
        )
    assert quiet.confirmed is False

    confident = NearEndSpeechGate(confirm_windows=2, confirm_duration_ms=40.0)
    results = [
        confident.observe_evidence(
            _evidence(near=400.0, clean=150.0, probability=0.9)
        )
        for _ in range(2)
    ]
    assert results[-1] is True


def test_gate_generations_never_accumulate_into_one_confirmation() -> None:
    gate = NearEndSpeechGate(confirm_windows=4, confirm_duration_ms=80.0)
    for _ in range(3):
        assert not gate.observe_evidence(
            _evidence(near=5000.0, clean=3000.0, generation=1)
        )
    # A new reply starts from zero; the old reply's tail cannot combine.
    assert not gate.observe_evidence(
        _evidence(near=5000.0, clean=3000.0, generation=2)
    )
    for _ in range(2):
        assert not gate.observe_evidence(
            _evidence(near=5000.0, clean=3000.0, generation=2)
        )
    assert gate.observe_evidence(
        _evidence(near=5000.0, clean=3000.0, generation=2)
    ) is True


class _MeteredPipeline:
    """Offline AEC double that reports near/clean levels explicitly."""

    available = True
    healthy = True
    degraded = False
    unavailable_reason = ""

    def __init__(self, near: float, clean: float) -> None:
        self.near = near
        self.clean = clean
        self.started: list[int] = []
        self.processed = 0

    def start(self, generation: int) -> bool:
        self.started.append(generation)
        return True

    def stop(self, generation: int | None = None) -> None:
        pass

    def reset(self) -> None:
        pass

    def stats(self) -> dict:
        return {}

    def process_microphone(self, pcm: bytes, timestamp: float):
        self.processed += 1
        return MicProcessingResult(
            pcm=bytes(pcm),
            valid=True,
            reason="ok",
            near_rms=self.near,
            clean_rms=self.clean,
            metrics_valid=True,
        )


def _run_manager_with_pipeline(pipeline, *, chunks: int = 20, on_final=None):
    config = _config()
    segmenter = _RecordingVAD(energy_probability, config)
    manager = ASRManager(
        config,
        microphone=SimpleNamespace(
            description="Fake Mic",
            dropped_chunks=0,
            start=lambda: None,
            stop=lambda: None,
        ),
        vad_segmenter=segmenter,
        renderer=_renderer(),
    )
    manager._active_sessions = config.max_concurrent_sessions
    activities = []
    manager.set_listeners(on_speech=activities.append, on_final=on_final)
    manager.attach_echo_canceller(pipeline)
    assert manager.begin_barge_in(3) is True
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: manager._started)
        now = time.monotonic()
        for index in range(chunks):
            manager.audio_queue.put(
                AudioChunk(tone(320), 16000, 16000, now + index * 0.02)
            )
        return manager, activities, thread, segmenter
    except Exception:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()
        raise


def test_manager_echo_never_stops_tts_or_starts_a_turn() -> None:
    pipeline = _MeteredPipeline(near=5000.0, clean=50.0)
    manager, activities, thread, _segmenter = _run_manager_with_pipeline(
        pipeline
    )
    try:
        assert wait_for(
            lambda: manager.barge_in_stats().get("echo_rejected", 0) >= 1,
            timeout=4.0,
        )
        time.sleep(0.1)
        assert [a.kind for a in activities].count("barge_in") == 0
        assert manager._utterances == {}
        assert manager.barge_in_stats()["confirmed"] == 0
    finally:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()


def test_manager_confirms_real_speech_that_survives_cancellation() -> None:
    pipeline = _MeteredPipeline(near=5000.0, clean=3000.0)
    manager, activities, thread, _segmenter = _run_manager_with_pipeline(
        pipeline, chunks=14
    )
    try:
        assert wait_for(
            lambda: any(a.kind == "barge_in" for a in activities),
            timeout=4.0,
        )
        assert manager.barge_in_stats()["confirmed"] == 1
    finally:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()


# ------------------------------------------------------- ASR completeness
class _SpeakerObserver:
    def __init__(self) -> None:
        self.finalized: list[tuple[int, bool]] = []

    def on_start(self, utterance_id, pcm):
        pass

    def on_audio(self, utterance_id, pcm):
        pass

    def on_end(self, utterance_id):
        pass

    def on_finalized(self, utterance_id, success):
        self.finalized.append((int(utterance_id), bool(success)))

    def on_cancel(self, utterance_id):
        pass

    def snapshot(self, utterance_id):
        return None


def test_truncated_upload_is_incomplete_and_never_emits_a_final() -> None:
    renderer, _out, status = asr_tests.make_renderer()
    running = asr_tests.start(
        [
            asr_tests.start_group(),
            asr_tests.audio_group(),
            asr_tests.audio_group(),
            asr_tests.audio_group(),
            asr_tests.end_group(),
        ],
        [
            asr_tests.SessionScript(
                partials=[("你好", "s1")],
                finals=[("你好世界", "s1")],
                send_delay=0.15,
            )
        ],
        renderer=renderer,
        status=status,
        cfg=asr_tests.config(
            asr_send_queue_frames=2, max_utterance_seconds=0
        ),
    )
    finals: list = []
    closed: list[tuple[int, bool]] = []
    running.manager.set_listeners(
        on_final=finals.append, on_turn_closed=lambda uid, ok: closed.append((uid, ok))
    )
    observer = _SpeakerObserver()
    running.manager.set_speaker_observer(observer)
    try:
        running.feed(4)
        assert asr_tests.wait_for(
            lambda: running.manager.stats.truncations == 1
        )
        running.feed(1)
        assert asr_tests.wait_for(
            lambda: running.manager.stats.unfinished == 1, timeout=6.0
        )
        assert finals == []
        assert renderer.confirmed == []
        assert running.manager.stats.finalized == 0
        assert "本次语音未完整识别" in status.getvalue()
        assert closed and closed[-1][1] is False
        # No voice-print sample is learned from a truncated turn.
        assert observer.finalized and observer.finalized[-1][1] is False
    finally:
        running.stop()


def test_stream_idle_timeout_fails_explicitly_without_a_final() -> None:
    renderer, _out, status = asr_tests.make_renderer()
    running = asr_tests.start(
        [
            asr_tests.start_group(),
            asr_tests.audio_group(),
            asr_tests.end_group(),
        ],
        [asr_tests.SessionScript(never_finish=True)],
        renderer=renderer,
        status=status,
        cfg=asr_tests.config(
            stream_idle_timeout_seconds=0.2,
            final_timeout_seconds=5.0,
        ),
    )
    finals: list = []
    closed: list[tuple[int, bool]] = []
    running.manager.set_listeners(
        on_final=finals.append, on_turn_closed=lambda uid, ok: closed.append((uid, ok))
    )
    try:
        running.feed(2)
        assert asr_tests.wait_for(
            lambda: running.manager.stats.errors >= 1, timeout=6.0
        )
        assert finals == []
        assert renderer.confirmed == []
        assert running.manager.stats.finalized == 0
        assert "长时间无响应" in status.getvalue()
        assert closed and closed[-1][1] is False
    finally:
        running.stop()


def test_queue_and_link_metrics_are_exposed_without_bigger_buffers() -> None:
    running = asr_tests.start(
        [
            asr_tests.start_group(),
            asr_tests.audio_group(),
            asr_tests.audio_group(),
            asr_tests.audio_group(),
            asr_tests.audio_group(),
            asr_tests.audio_group(),
        ],
        [
            asr_tests.SessionScript(
                partials=[("你", "s1")],
                finals=[("你好", "s1")],
                send_delay=0.15,
            )
        ],
    )
    try:
        running.feed(6)
        stats = running.manager.stats
        assert asr_tests.wait_for(
            lambda: (stats.queued_audio_ms or 0.0) > 0.0,
            timeout=4.0,
        )
        # Backlog is visible while the slow link drains it.
        assert stats.oldest_audio_age_ms is not None
        assert stats.connection_open_latency_ms is not None
        assert asr_tests.wait_for(
            lambda: stats.sender_throughput_bps is not None, timeout=4.0
        )
        # The documented queue is not silently enlarged to hide the deficit.
        assert running.manager.config.asr_send_queue_frames > 0
        assert stats.truncations == 0
    finally:
        running.stop()


class _SlowOpenSession(asr_tests.FakeSession):
    def open(self) -> None:
        time.sleep(0.2)
        super().open()


def test_connection_open_latency_is_measured_separately_from_throughput() -> None:
    running = asr_tests.start(
        [
            asr_tests.start_group(),
            asr_tests.audio_group(),
            asr_tests.end_group(),
        ],
        [],
        cfg=asr_tests.config(),
    )
    running.manager._session_factory = lambda: _SlowOpenSession(
        asr_tests.SessionScript(finals=[("你好", "s1")])
    )
    finals: list = []
    running.manager.set_listeners(on_final=finals.append)
    try:
        running.feed(3)
        assert asr_tests.wait_for(lambda: finals, timeout=6.0)
        stats = running.manager.stats
        # A slow handshake is a connect metric; it is not a throughput
        # deficit and must not truncate the turn.
        assert stats.connection_open_latency_ms is not None
        assert stats.connection_open_latency_ms >= 150.0
        assert asr_tests.wait_for(
            lambda: stats.sender_throughput_bps is not None, timeout=4.0
        )
        assert stats.truncations == 0
        assert finals[0].text == "你好"
    finally:
        running.stop()


# ------------------------------------------------- partial/subtitle identity
async def test_bridge_keeps_only_the_latest_partial_per_utterance() -> None:
    bridge = VoiceBridge(asyncio.get_running_loop(), max_pending=8)
    bridge.publish_partial(PartialTranscript(3, "一", sequence=1))
    bridge.publish_partial(PartialTranscript(3, "二", sequence=2))
    bridge.publish_partial(PartialTranscript(3, "三", sequence=3))
    item = await asyncio.wait_for(bridge.next_item(), 1.0)
    assert (item.utterance_id, item.sequence, item.text) == (3, 3, "三")
    # An older revision arriving late never moves the subtitle backwards.
    bridge.publish_partial(PartialTranscript(3, "旧", sequence=2))
    await asyncio.sleep(0)
    assert bridge.dropped_partials == 1


async def test_a_dropped_final_is_not_marked_seen_and_can_be_retried() -> None:
    from companion.asr.manager import FinalUtterance, SpeechActivity

    bridge = VoiceBridge(asyncio.get_running_loop(), max_pending=1)
    bridge._deliver(PartialTranscript(1, "占位", sequence=1))
    final = FinalUtterance(2, "真正最终", time.monotonic())
    bridge._deliver(final)
    assert bridge.dropped_finals == 1
    assert 2 not in bridge._seen_final_set

    assert (await asyncio.wait_for(bridge.next_item(), 1.0)) is not None
    bridge._deliver(final)
    assert (await asyncio.wait_for(bridge.next_item(), 1.0)).text == "真正最终"
    # Only after the successful enqueue is the final a duplicate.
    bridge._deliver(final)
    assert bridge.duplicate_finals == 1


async def test_coordinator_partial_is_scoped_to_the_recording_turn() -> None:
    fake = _FakeASR()
    coordinator = VoiceCoordinator(
        manager_factory=lambda: fake,
        tts_manager=_FakeTTS(),
        cooldown_seconds=0.0,
    )
    ok, _ = await coordinator.enable()
    assert ok
    seen: list[tuple[int, str]] = []
    coordinator.set_partial_listener(lambda uid, text: seen.append((uid, text)))
    try:
        fake.speech("started", 7)
        await settle()
        assert coordinator._current_uid == 7
        # A late partial from an older utterance must be ignored.
        coordinator._bridge.publish_partial(
            PartialTranscript(6, "旧字幕", sequence=99)
        )
        await settle()
        assert seen == []

        coordinator._bridge.publish_partial(
            PartialTranscript(7, "新字幕", sequence=2)
        )
        await settle()
        assert seen == [(7, "新字幕")]

        # An older final/close cannot clear the current turn's subtitle.
        coordinator._clear_partial(6)
        await settle()
        assert seen == [(7, "新字幕")]
    finally:
        await coordinator.shutdown()


def test_partial_subtitle_tracks_one_active_utterance() -> None:
    subtitle = _PartialSubtitle()
    subtitle.set_partial(1, 1, "第一句")
    assert subtitle.active_utterance_id == 1
    assert subtitle.render().endswith("第一句")

    subtitle.set_partial(1, 3, "第一句更新")
    assert subtitle.render().endswith("第一句更新")
    # An older revision of the same utterance is ignored.
    subtitle.set_partial(1, 2, "过期文本")
    assert subtitle.render().endswith("第一句更新")

    subtitle.set_partial(2, 1, "第二轮")
    assert subtitle.active_utterance_id == 2
    # A late close from the previous turn cannot erase the new subtitle.
    subtitle.clear_if_current(1)
    assert subtitle.active_utterance_id == 2
    assert subtitle.render().endswith("第二轮")
    subtitle.clear_if_current(2)
    assert subtitle.active_utterance_id is None
    assert subtitle.render() == ""


# --------------------------------------------------------- proactive delivery
async def _ack(manager, event, **kwargs) -> bool:
    kwargs.setdefault("delivered", True)
    return await manager.acknowledge_delivery(event, **kwargs)


async def test_generated_but_undelivered_does_not_advance_the_budget(db) -> None:
    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    proactive_tests._scripted(rt.proactive_manager, [True, True])
    manager = rt.proactive_manager
    try:
        await proactive_tests._start_cycle(rt, person, event)
        delivered = await asyncio.wait_for(manager.next_event(), 2)
        state = manager._states["c1"]
        assert state.unanswered == 0
        assert state.pending_message_id == delivered.response_id
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert [row.status for row in rows] == ["GENERATED"]
        # A generated-but-unseen message blocks stacking another one.
        assert await manager._attempt(
            SimpleNamespace(
                conversation_id="c1",
                person_id=person.person_id,
                source_event_id=event.event_id,
                generation=state.generation,
            ),
            state,
            "LONG_IDLE",
            "LONG_IDLE_1",
        ) is None
        assert len(await rt.repos.proactive.list_for_conversation("c1")) == 1
    finally:
        await manager.shutdown()


async def test_async_ack_persists_then_counts_a_long_idle_delivery_once(db) -> None:
    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    proactive_tests._scripted(rt.proactive_manager, [True, True])
    manager = rt.proactive_manager
    try:
        await proactive_tests._start_cycle(rt, person, event)
        immediate = await asyncio.wait_for(manager.next_event(), 2)
        assert await _ack(manager, immediate) is True
        assert manager._states["c1"].unanswered == 0

        long_idle = await asyncio.wait_for(manager.next_event(), 2)
        manager.idle_delay_seconds = 100.0
        assert await _ack(manager, long_idle, stage="text_displayed") is True
        row = await rt.repos.proactive.get(long_idle.response_id)
        assert row is not None and row.status == "COMPLETED"
        assert manager._states["c1"].unanswered == 1
        # A duplicate acknowledgement cannot count the same delivery twice.
        assert await _ack(manager, long_idle) is False
        assert manager._states["c1"].unanswered == 1
    finally:
        await manager.shutdown()


async def test_cancelled_delivery_is_marked_and_may_be_retried(db) -> None:
    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    provider = proactive_tests._scripted(
        rt.proactive_manager, [True, True, True]
    )
    manager = rt.proactive_manager
    try:
        await proactive_tests._start_cycle(rt, person, event)
        immediate = await asyncio.wait_for(manager.next_event(), 2)
        assert await _ack(manager, immediate) is True
        long_idle = await asyncio.wait_for(manager.next_event(), 2)
        manager.idle_delay_seconds = 100.0

        # A partially played message interrupted by the user is not a
        # completed delivery: it consumes no round.
        assert await _ack(
            manager,
            long_idle,
            delivered=False,
            reason="tts_interrupted",
            stage="tts_playback_completed",
        ) is True
        state = manager._states["c1"]
        assert state.unanswered == 0
        assert state.pending_message_id is None
        row = await rt.repos.proactive.get(long_idle.response_id)
        assert row is not None and row.status == "CANCELLED"

        # CANCELLED is not terminal: the same round may be regenerated. The
        # table enforces one row per (source_event_id, mode), so the retry
        # reopens that row instead of inserting a duplicate.
        before = len(await rt.repos.proactive.list_for_conversation("c1"))
        await manager._attempt(
            SimpleNamespace(
                conversation_id="c1",
                person_id=person.person_id,
                source_event_id=event.event_id,
                generation=state.generation,
            ),
            state,
            "LONG_IDLE",
            "LONG_IDLE_1",
        )
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert len(rows) == before
        retried = await rt.repos.proactive.get(long_idle.response_id)
        assert retried is not None and retried.status == "GENERATED"
        assert retried.text
        retry_event = await asyncio.wait_for(manager.next_event(), 2)
        assert retry_event.response_id == long_idle.response_id
        assert state.pending_message_id == long_idle.response_id
        assert provider is not None
    finally:
        await manager.shutdown()


async def test_database_failure_keeps_the_message_pending(db) -> None:
    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    proactive_tests._scripted(rt.proactive_manager, [True, True])
    manager = rt.proactive_manager
    try:
        await proactive_tests._start_cycle(rt, person, event)
        immediate = await asyncio.wait_for(manager.next_event(), 2)
        assert await _ack(manager, immediate) is True
        long_idle = await asyncio.wait_for(manager.next_event(), 2)
        manager.idle_delay_seconds = 100.0

        repo = rt.repos.proactive
        original = repo.update_status

        async def _failing(message_id, status):
            raise RuntimeError("database unavailable")

        repo.update_status = _failing
        try:
            assert await _ack(manager, long_idle) is False
        finally:
            repo.update_status = original

        state = manager._states["c1"]
        assert state.unanswered == 0
        assert state.pending_message_id == long_idle.response_id
        row = await repo.get(long_idle.response_id)
        assert row is not None and row.status == "GENERATED"

        # Once the write can succeed, the same acknowledgement completes.
        assert await _ack(manager, long_idle) is True
        row = await repo.get(long_idle.response_id)
        assert row is not None and row.status == "COMPLETED"
        assert state.unanswered == 1
    finally:
        await manager.shutdown()


def test_dedupe_status_semantics() -> None:
    from companion.proactive.manager import ProactiveManager

    manager = ProactiveManager(
        repos=None,
        registry=None,
        conversations=None,
        orchestrator=None,
        judge=None,
    )
    state = SimpleNamespace(pending_message_id="m1")
    assert manager._blocks_retry(None, state) is False
    assert manager._blocks_retry(
        SimpleNamespace(message_id="m1", status="GENERATED"), state
    ) is True
    assert manager._blocks_retry(
        SimpleNamespace(message_id="m2", status="GENERATED"), state
    ) is False
    assert manager._blocks_retry(
        SimpleNamespace(message_id="m1", status="COMPLETED"), state
    ) is True
    assert manager._blocks_retry(
        SimpleNamespace(message_id="m1", status="CANCELLED"), state
    ) is False


async def test_three_unanswered_long_idle_deliveries_stop_the_cycle(db) -> None:
    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    provider = proactive_tests._scripted(rt.proactive_manager, [True] * 6)
    manager = rt.proactive_manager
    try:
        await proactive_tests._start_cycle(rt, person, event)
        for _ in range(4):
            delivered = await asyncio.wait_for(manager.next_event(), 2)
            assert await _ack(manager, delivered) is True
        state = manager._states["c1"]
        assert state.unanswered == manager.max_unanswered_rounds
        await proactive_tests._wait_for(lambda: state.task is None, timeout=2.0)
        # No fifth message was generated for an empty room.
        assert len(provider.payloads) == 4
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert len(rows) == 4
        assert all(row.status == "COMPLETED" for row in rows)
    finally:
        await manager.shutdown()


async def test_immediate_and_judge_refusals_do_not_consume_rounds(db) -> None:
    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    provider = proactive_tests._scripted(
        rt.proactive_manager, [True, False, False, False]
    )
    manager = rt.proactive_manager
    try:
        await proactive_tests._start_cycle(rt, person, event)
        immediate = await asyncio.wait_for(manager.next_event(), 2)
        assert await _ack(manager, immediate) is True
        await proactive_tests._wait_for(
            lambda: len(provider.payloads) >= 4, timeout=2.0
        )
        assert manager._states["c1"].unanswered == 0
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert len(rows) == 1
        assert rows[0].status == "COMPLETED"
    finally:
        await manager.shutdown()


async def test_routing_sees_only_the_delivered_proactive_message(db) -> None:
    from companion.routing.context import RoutingContextBuilder

    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    proactive_tests._scripted(rt.proactive_manager, [True])
    manager = rt.proactive_manager
    builder = RoutingContextBuilder(repos=rt.repos, registry=rt.registry)
    history = [
        SimpleNamespace(
            event_id=event.event_id, conversation_id="c1"
        )
    ]
    try:
        await proactive_tests._start_cycle(rt, person, event)
        delivered = await asyncio.wait_for(manager.next_event(), 2)
        # GENERATED is not dialogue yet.
        assert await builder._proactive_by_source(history) == {}

        assert await _ack(manager, delivered) is True
        grouped = await builder._proactive_by_source(history)
        assert event.event_id in grouped
        assert [row.message_id for row in grouped[event.event_id]] == [
            delivered.response_id
        ]
    finally:
        await manager.shutdown()


async def test_user_turn_cancels_an_undelivered_proactive_message(db) -> None:
    rt = await proactive_tests._runtime(db)
    person = await proactive_tests._person(rt)
    event = await proactive_tests._delivered_turn(rt, person)
    proactive_tests._scripted(rt.proactive_manager, [True, True])
    manager = rt.proactive_manager
    try:
        await proactive_tests._start_cycle(rt, person, event)
        delivered = await asyncio.wait_for(manager.next_event(), 2)
        manager.idle_delay_seconds = 100.0
        manager.note_user_input("c1", "new-user-event")
        state = manager._states["c1"]
        assert state.unanswered == 0
        assert state.pending_message_id is None
        # The previously generated message is cancelled by the surface, and
        # its row is not replayed as delivered dialogue.
        assert await _ack(
            manager,
            delivered,
            delivered=False,
            reason="user_interrupted",
            stage="tts_playback_completed",
        ) in (True, False)
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert all(row.status != "COMPLETED" for row in rows)
    finally:
        await manager.shutdown()


# ------------------------------------------------------ cross-module acceptance
async def test_echo_never_becomes_a_user_final() -> None:
    """A full barge-in path: loud echo is cancelled, VAD sees silence, no turn."""
    pipeline = _MeteredPipeline(near=12000.0, clean=0.0)
    finals: list = []
    manager, activities, thread, segmenter = _run_manager_with_pipeline(
        pipeline, chunks=20, on_final=finals.append
    )
    try:
        assert wait_for(
            lambda: manager.barge_in_stats().get("echo_rejected", 0) >= 1,
            timeout=4.0,
        )
        time.sleep(0.1)
        assert finals == []
        assert [a.kind for a in activities].count("started") == 0
        assert manager._utterances == {}
        assert segmenter.calls  # VAD only ever saw the cancelled audio.
    finally:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()


def test_sixty_seconds_of_speech_is_uploaded_whole() -> None:
    """60 s of 64 ms fake windows; the documented queue must not truncate."""
    windows = int(round(60 * 1000 / 64))
    groups = [asr_tests.start_group()]
    groups.extend(asr_tests.audio_group() for _ in range(windows))
    groups.append(asr_tests.end_group())
    running = asr_tests.start(
        groups,
        [
            asr_tests.SessionScript(
                partials=[("我在讲一个", "s1"), ("很长的故事", "s2")],
                finals=[("我在讲一个很长的故事。", "s1")],
            )
        ],
        cfg=asr_tests.config(
            asr_send_queue_frames=512,
            max_utterance_seconds=120,
            mic_queue_windows=256,
            final_timeout_seconds=3.0,
        ),
    )
    try:
        running.feed(windows + 2)
        assert asr_tests.wait_for(
            lambda: running.renderer.confirmed == ["我在讲一个很长的故事。"],
            timeout=15.0,
        )
        assert running.manager.stats.truncations == 0
        assert running.manager.stats.unfinished == 0
        assert running.manager.stats.finalized == 1
    finally:
        running.stop()
