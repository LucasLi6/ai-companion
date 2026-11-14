"""V0.9.9.27 regression tests — AEC lifecycle, self-recognition, ASR, time.

Each test reproduces one of the four reported user-visible defects against the
offline fakes used across this suite (no Silero, PyAudio, DashScope, LLM or
real pywebrtc-audio needed).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from companion.asr.manager import (
    ASRManager,
    FinalUtterance,
    InputMode,
    NearEndSpeechGate,
    SpeechActivity,
    _Utterance,
)
from companion.voice.barge_in_config import BargeInConfig
from companion.voice.coordinator import VoiceCoordinator, VoiceState
from companion.voice.echo_canceller import (
    AECState,
    EchoCancellationPipeline,
    MicProcessingResult,
)

from test_voice_barge_in import (  # noqa: E402 - shared offline fixtures
    PassThroughCanceller,
    SilenceCanceller,
    _FakeASR,
    _FakeMic,
    _FakeTTS,
    _config,
    _renderer,
    _TailPipeline,
    settle,
    tone,
)


# ------------------------------------------------ Part 4: AEC gap vs failure
def test_tts_silence_between_sentences_is_a_gap_not_a_failure() -> None:
    """The original bug: sentence pauses permanently degraded AEC.

    A real reply is spoken sentence by sentence; between sentences the player
    pushes no far-end frames for a moment. That must be reported as ``GAP``
    and must never flip the channel to degraded/failed.
    """
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller(),
        fallback_after_invalid_seconds=0.05,
    )
    assert pipeline.start(1)
    assert pipeline.push_playback(tone(480), 10.0, 1)
    assert pipeline.process_microphone(tone(320), 10.0).valid
    assert pipeline.state is AECState.ACTIVE

    # TTS sentence pause: no new far-end frames, microphone keeps running.
    time.sleep(0.09)
    gap = pipeline.process_microphone(tone(320), 10.2)
    assert not gap.valid
    assert pipeline.state is AECState.GAP
    assert pipeline.healthy is True
    assert pipeline.degraded is False

    # The next sentence's playback resumes; detection recovers by itself.
    assert pipeline.push_playback(tone(480), 10.32, 1)
    assert pipeline.process_microphone(tone(320), 10.32).valid
    assert pipeline.state is AECState.ACTIVE


def test_gap_never_escalates_to_degraded_while_the_generation_is_active() -> None:
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller(),
        fallback_after_invalid_seconds=0.01,
    )
    assert pipeline.start(2)
    assert pipeline.push_playback(tone(480), 20.0, 2)
    assert pipeline.process_microphone(tone(320), 20.0).valid
    time.sleep(0.05)
    for index in range(5):
        pipeline.process_microphone(tone(320), 20.1 + index * 0.02)
    assert pipeline.healthy is True
    assert pipeline.stats()["state"] in ("GAP", "ACTIVE")


def test_each_generation_recovers_after_a_previous_failure() -> None:
    """One broken reply must not disable barge-in for every later reply."""
    calls = {"count": 0}

    class Flaky:
        def process(self, near, far):
            calls["count"] += 1
            if calls["count"] <= 3:
                raise RuntimeError("boom")
            return near.copy()

    pipeline = EchoCancellationPipeline(canceller_factory=lambda *_: Flaky())
    assert pipeline.start(1)
    for index in range(3):
        assert pipeline.push_playback(tone(480), 30.0 + index * 0.02, 1)
        assert not pipeline.process_microphone(
            tone(320), 30.0 + index * 0.02
        ).valid
    assert pipeline.healthy is False

    # A new generation rebuilds the canceller and clears the failure flags.
    assert pipeline.start(2) is True
    assert pipeline.healthy is True
    assert pipeline.push_playback(tone(480), 40.0, 2)
    assert pipeline.process_microphone(tone(320), 40.0).valid


def test_previous_generation_playback_cannot_degrade_a_fresh_generation() -> None:
    """Lifetime playback totals must not count as this generation's playback.

    The old guard read ``_playback_frames > 0`` (whole-process total). After a
    first reply had spoken anything, a fresh generation that had not received
    a single far-end frame yet looked like playback with a missing reference,
    so it degraded to half-duplex before its TTS even started.
    """
    pipeline = EchoCancellationPipeline(
        canceller_factory=lambda *_: PassThroughCanceller(),
        fallback_after_invalid_seconds=0.02,
    )
    assert pipeline.start(1)
    assert pipeline.push_playback(tone(480), 50.0, 1)
    assert pipeline.process_microphone(tone(320), 50.0).valid
    assert pipeline.stats()["generation_playback_frames"] == 1
    pipeline.stop(1)

    assert pipeline.start(2) is True
    # The lifetime diagnostic survives; the per-generation counter is reset.
    assert pipeline.stats()["playback_frames"] >= 1
    assert pipeline.stats()["generation_playback_frames"] == 0
    pipeline.stop(2)

    # No far-end frame ever arrives in generation 2. The microphone keeps
    # producing blocks; after the fallback window this is still not evidence
    # that generation 2's AEC is broken. (The window is clamped to 0.1 s.)
    assert not pipeline.process_microphone(tone(320), 60.0).valid
    time.sleep(0.15)
    result = pipeline.process_microphone(tone(320), 60.02)
    assert not result.valid
    assert pipeline.degraded is False
    assert pipeline.healthy is True


def test_previous_generation_aec_errors_cannot_fail_a_fresh_generation() -> None:
    """Only this generation's error count may declare it failed."""
    script = ["raise"] * 4 + ["ok"] * 4

    class Scripted:
        def process(self, near, far):
            action = script.pop(0) if script else "ok"
            if action == "raise":
                raise RuntimeError("boom")
            return near.copy()

    pipeline = EchoCancellationPipeline(canceller_factory=lambda *_: Scripted())
    assert pipeline.start(1)
    for index in range(3):
        assert pipeline.push_playback(tone(480), 70.0 + index * 0.02, 1)
        assert not pipeline.process_microphone(
            tone(320), 70.0 + index * 0.02
        ).valid
    stats = pipeline.stats()
    assert stats["aec_errors"] == 3
    assert stats["generation_aec_errors"] == 3
    assert pipeline.healthy is False

    # Generation 2 starts clean. Its first error (the process lifetime's
    # fourth) must not immediately reproduce generation 1's failure.
    assert pipeline.start(2) is True
    assert pipeline.stats()["generation_aec_errors"] == 0
    assert pipeline.healthy is True
    assert pipeline.push_playback(tone(480), 80.0, 2)
    assert not pipeline.process_microphone(tone(320), 80.0).valid
    stats = pipeline.stats()
    assert stats["aec_errors"] == 4
    assert stats["generation_aec_errors"] == 1
    assert pipeline.healthy is True
    assert stats["state"] == "ACTIVE"

    # And the generation keeps working once its AEC recovers.
    assert pipeline.push_playback(tone(480), 80.02, 2)
    assert pipeline.process_microphone(tone(320), 80.02).valid


async def test_aec_failure_in_one_reply_does_not_disable_the_next_reply() -> None:
    """The coordinator used to keep ``_barge_disabled_reason`` forever."""
    fake = _FakeASR()
    coordinator = VoiceCoordinator(
        manager_factory=lambda: fake,
        tts_manager=_FakeTTS(),
        cooldown_seconds=0.0,
        barge_in=BargeInConfig(enabled=True, echo_tail_ms=0),
    )
    ok, _ = await coordinator.enable()
    assert ok
    try:
        token = coordinator.begin_turn()
        coordinator.on_reply_started(1, token=token)
        fake.unavailable_listener("aec:RuntimeError")
        await settle()
        assert coordinator.barge_in_status()["fallback_reason"]

        # The next reply tries again instead of inheriting the fallback.
        coordinator._handle_playback(
            SimpleNamespace(generation=1, kind="finished")
        )
        coordinator.on_turn_finished({"delivered": True}, token=token)
        await settle(rounds=10)
        assert coordinator.state == VoiceState.LISTENING

        token2 = coordinator.begin_turn()
        coordinator.on_reply_started(2, token=token2)
        await settle()
        assert fake.barge_started[-1] == 2
        assert coordinator._barge_armed is True
    finally:
        await coordinator.shutdown()


# ------------------------------------- Part 2: own TTS must not self-confirm
def test_near_end_gate_rejects_cancelled_echo() -> None:
    gate = NearEndSpeechGate(confirm_windows=6)
    # Loud near end, but cancellation leaves almost nothing: Ava's own echo.
    for _ in range(20):
        assert not gate.observe(near_rms=9000.0, clean_rms=120.0)
    assert gate.confirmed is False


def test_near_end_gate_confirms_real_user_speech() -> None:
    gate = NearEndSpeechGate(confirm_windows=6)
    results = [
        gate.observe(near_rms=9000.0, clean_rms=6000.0) for _ in range(6)
    ]
    assert results[:5] == [False] * 5
    assert results[5] is True
    assert gate.confirmed is True


def test_near_end_gate_resets_on_a_clean_window() -> None:
    gate = NearEndSpeechGate(confirm_windows=3)
    gate.observe(near_rms=9000.0, clean_rms=6000.0)
    gate.observe(near_rms=9000.0, clean_rms=6000.0)
    gate.observe(near_rms=9000.0, clean_rms=50.0)
    assert gate.confirmed is False


def test_manager_never_creates_a_turn_from_unconfirmed_echo() -> None:
    """The exact self-recognition bug: Ava's echo became a user utterance."""
    pipeline = _EchoOnlyPipeline()
    manager = ASRManager(
        _config(),
        microphone=_FakeMic(),
        vad_segmenter=_RecordingSegmenter(),
        renderer=_renderer(),
    )
    activities: list[SpeechActivity] = []
    finals: list[FinalUtterance] = []
    manager.set_listeners(
        on_speech=activities.append, on_final=finals.append
    )
    manager._active_sessions = manager.config.max_concurrent_sessions
    manager.attach_echo_canceller(pipeline)
    assert manager.begin_barge_in(4) is True
    manager._barge_in_detecting = True

    # The VAD found speech in the cancelled audio; the gate has only seen
    # echo-like chunks, so this is Ava, not the user.
    from companion.asr.manager import _SpeechStarted

    manager._handle(_SpeechStarted(b"pcm", time.monotonic(), barge_in=True))
    assert activities == []
    assert finals == []
    assert manager._utterances == {}
    assert manager._current is None


def test_manager_creates_a_turn_after_real_near_end_speech() -> None:
    pipeline = _UserSpeechPipeline()
    manager = ASRManager(
        _config(),
        microphone=_FakeMic(),
        vad_segmenter=_RecordingSegmenter(),
        renderer=_renderer(),
    )
    activities: list[SpeechActivity] = []
    manager.set_listeners(on_speech=activities.append)
    manager._active_sessions = manager.config.max_concurrent_sessions
    manager.attach_echo_canceller(pipeline)
    assert manager.begin_barge_in(4) is True
    manager._barge_in_detecting = True

    from companion.asr.microphone import AudioChunk

    for index in range(6):
        pipeline.valid = True
        manager._process_barge_in_audio(
            AudioChunk(tone(320), 16000, 16000, time.monotonic())
        )
    from companion.asr.manager import _SpeechStarted

    manager._handle(_SpeechStarted(b"pcm", time.monotonic(), barge_in=True))
    kinds = [activity.kind for activity in activities]
    assert kinds[:2] == ["barge_in", "started"]
    assert len(manager._utterances) == 1


class _RecordingSegmenter:
    def process(self, pcm: bytes):
        return []

    def reset(self) -> None:
        pass


class _EchoOnlyPipeline(_TailPipeline):
    """Valid AEC output whose clean level collapses (pure echo)."""

    def process_microphone(self, pcm: bytes, timestamp: float):
        return MicProcessingResult(
            pcm=bytes(pcm),
            valid=True,
            reason="ok",
            near_rms=8000.0,
            clean_rms=60.0,
        )


class _UserSpeechPipeline(_TailPipeline):
    """Valid AEC output where the user's voice survives cancellation."""

    def process_microphone(self, pcm: bytes, timestamp: float):
        return MicProcessingResult(
            pcm=bytes(pcm),
            valid=True,
            reason="ok",
            near_rms=9000.0,
            clean_rms=6500.0,
        )


# ------------------------------------- Part 2 end-to-end through the VAD loop
def test_vad_loop_rejects_echo_and_still_hears_the_next_real_speech() -> None:
    import threading

    from companion.asr.microphone import AudioChunk
    from companion.asr.vad import VADSegmenter

    from test_voice_barge_in import _RecordingVAD, energy_probability, wait_for

    class SwitchablePipeline(_TailPipeline):
        echo = True

        def process_microphone(self, pcm: bytes, timestamp: float):
            return MicProcessingResult(
                pcm=bytes(pcm),
                valid=True,
                reason="ok",
                near_rms=8000.0,
                clean_rms=60.0 if self.echo else 6500.0,
            )

    config = _config()
    pipeline = SwitchablePipeline()
    manager = ASRManager(
        config,
        microphone=_FakeMic(),
        vad_segmenter=_RecordingVAD(energy_probability, config),
        renderer=_renderer(),
    )
    activities: list[SpeechActivity] = []
    manager.set_listeners(on_speech=activities.append)
    manager._active_sessions = config.max_concurrent_sessions
    manager.attach_echo_canceller(pipeline)
    assert manager.begin_barge_in(6) is True
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: manager._started)
        now = time.monotonic()
        for index in range(14):
            manager.audio_queue.put(
                AudioChunk(tone(320), 16000, 16000, now + index * 0.02)
            )
        assert wait_for(lambda: manager.barge_in_stats().get("echo_rejected"))
        assert activities == []
        assert manager._utterances == {}

        # The user now really speaks: cancellation cannot remove the voice.
        pipeline.echo = False
        for index in range(14, 40):
            manager.audio_queue.put(
                AudioChunk(tone(320), 16000, 16000, now + index * 0.02)
            )
        assert wait_for(
            lambda: any(item.kind == "barge_in" for item in activities)
        )
        kinds = [item.kind for item in activities]
        assert kinds[:2] == ["barge_in", "started"]
        assert len(manager._utterances) == 1
    finally:
        manager.stop()
        thread.join(timeout=4.0)
        manager.shutdown()


# ------------------------------------ Part 3: ASR partials while speaking
def test_partial_reaches_the_surface_when_the_embedder_renders_finals() -> None:
    """The formal REPL showed nothing until the final result arrived."""
    from test_asr_manager import (
        SessionScript,
        config,
        end_group,
        start,
        start_group,
        wait_for,
    )

    running = start(
        [start_group(), end_group()],
        [SessionScript(partials=[("你好", "s1")], finals=[("你好世界", "s1")])],
        cfg=config(),
    )
    partials: list[tuple[int, str]] = []
    finals: list[FinalUtterance] = []
    running.manager.set_listeners(
        on_final=finals.append, render_finals=False
    )
    running.manager.set_partial_listener(
        lambda event: partials.append((event.utterance_id, event.text))
    )
    try:
        running.feed(1)
        assert wait_for(lambda: partials == [(1, "你好")])
        running.feed(1)
        assert wait_for(lambda: [item.text for item in finals] == ["你好世界"])
    finally:
        running.stop()


async def test_final_clears_the_live_partial_on_the_surface() -> None:
    from companion.voice.bridge import PartialTranscript

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
        coordinator._bridge.publish_partial(
            PartialTranscript(7, "我想问", sequence=2, at=time.monotonic())
        )
        await settle()
        assert seen == [(7, "我想问")]

        # A delayed older revision must not overwrite the newer text.
        coordinator._bridge.publish_partial(
            PartialTranscript(7, "我", sequence=1, at=time.monotonic())
        )
        await settle()
        assert seen == [(7, "我想问")]

        # The final result replaces the provisional line; the display must
        # never keep showing stale text after the turn is committed.
        fake.final(7, "我想问一下时间")
        await settle()
        assert seen[-1] == (7, "")
        final = await asyncio.wait_for(coordinator.next_final(), 1.0)
        assert final.text == "我想问一下时间"
    finally:
        await coordinator.shutdown()


async def test_partial_event_crosses_the_bridge_with_its_sequence() -> None:
    """The manager callback used to take two args while the bridge took one."""
    from companion.voice.bridge import VoiceBridge

    bridge = VoiceBridge(asyncio.get_running_loop())
    manager = ASRManager(
        _config(),
        microphone=_FakeMic(),
        vad_segmenter=_RecordingSegmenter(),
        renderer=_renderer(),
    )
    manager.set_partial_listener(bridge.publish_partial)
    utterance = _Utterance(3, 8)
    manager._emit_partial(utterance, "你好")
    first = await asyncio.wait_for(bridge.next_item(), 1.0)
    manager._emit_partial(utterance, "你好世界")
    second = await asyncio.wait_for(bridge.next_item(), 1.0)
    assert (first.utterance_id, first.text, first.sequence) == (3, "你好", 1)
    assert (second.text, second.sequence) == ("你好世界", 2)


def test_partial_subtitle_hides_when_empty() -> None:
    from companion.cli import _PartialSubtitle

    class _Session:
        bottom_toolbar = None

        class app:  # noqa: N801 - stand-in for prompt_toolkit's app
            is_running = False

    session = _Session()
    subtitle = _PartialSubtitle()
    subtitle.attach(session)
    assert session.bottom_toolbar is None

    subtitle.set("  你好\n世界  ")
    assert session.bottom_toolbar is not None
    assert subtitle.render() == "🎙 你好 世界"

    subtitle.clear()
    assert session.bottom_toolbar is None
    assert subtitle.render() == ""


# --------------------------- Part 3: a long statement is not a lost turn
def test_send_queue_holds_a_whole_documented_utterance() -> None:
    from companion.asr.config import VAD_WINDOW_MS

    frames = int(120 * 1000 / VAD_WINDOW_MS) + 1
    assert (
        _Utterance(1, 512, max_utterance_seconds=120).send_queue.maxsize
        == frames
    )
    assert _Utterance(2, 512).send_queue.maxsize == 512


def test_long_upload_backpressure_truncates_instead_of_losing_the_turn() -> None:
    """A truncated upload is an explicit incomplete turn, never a success.

    V0.9.9.27 kept the recognized prefix and finalized it as a normal user
    message even though later audio was never uploaded. V0.9.9.28 forbids
    that: the turn must close as incomplete (no FinalUtterance, no Router
    text, no success mark), while the recognized prefix stays visible only
    as an explicitly unsubmitted reference line.
    """
    import test_asr_manager as asr_tests

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
        cfg=asr_tests.config(asr_send_queue_frames=2, max_utterance_seconds=0),
    )
    try:
        running.feed(4)
        assert asr_tests.wait_for(
            lambda: running.manager.stats.truncations == 1
        )
        running.feed(1)
        assert asr_tests.wait_for(
            lambda: running.manager.stats.unfinished == 1, timeout=6.0
        )
        text = status.getvalue()
        assert "本次语音未完整识别" in text
        # The prefix is shown for reference only and explicitly marked as
        # not submitted; it must never become a confirmed final.
        assert "已识别部分（未提交）：你好世界" in text
        assert renderer.confirmed == []
        assert running.manager.stats.finalized == 0
        # Truncation is an incomplete turn, not an abort/worker failure.
        assert running.manager.stats.aborted == 0
    finally:
        running.stop()


@pytest.mark.parametrize("seconds", [20, 45, 90], ids=["20s", "45s", "90s"])
def test_long_statement_uploads_whole_without_a_lost_turn(seconds: int) -> None:
    """The pre-V0.9.9.27 buffer held ~16 s; a 20 s statement overflowed it.

    Simulated here with 32 ms frames fed faster than the fake session sends
    them, which is what a slow upload link does in real life. The documented
    120 s buffer must absorb the whole statement, the final must commit, and
    nothing may be truncated or aborted. 20 s is the old boundary; 45 s and
    90 s check the same path well past it.
    """
    import test_asr_manager as asr_tests

    frames = seconds * 1000 // 32  # 32 ms Silero windows
    running = asr_tests.start(
        [
            asr_tests.start_group(),
            *[asr_tests.audio_group() for _ in range(frames)],
            asr_tests.end_group(),
        ],
        [
            asr_tests.SessionScript(
                partials=[("我想说", "s1")],
                finals=[("我想说一段比较长的话", "s1")],
                send_delay=0.001,
            )
        ],
        cfg=asr_tests.config(max_utterance_seconds=120),
    )
    try:
        running.feed(frames + 2)  # speech start + frames + speech end
        assert asr_tests.wait_for(
            lambda: running.renderer.confirmed
            == ["我想说一段比较长的话"],
            timeout=10.0,
        )
        session = running.factory.created[0]
        # Every fed byte must reach the server, not just enough to finalize.
        assert asr_tests.wait_for(
            lambda: session.sent_bytes == 2048 * (frames + 1),
            timeout=6.0,
        )
        assert running.manager.stats.truncations == 0
        assert running.manager.stats.aborted == 0
        assert running.manager.stats.finalized == 1
        assert len(running.factory.created) == 1
    finally:
        running.stop()


# ------------------------------------- Part 1: time awareness / continuity
def test_continuity_boundaries_are_pure_and_ordered() -> None:
    from companion.runtime_context.temporal import classify_continuity

    cases = [
        (None, "NEW"),
        (0, "ACTIVE"),
        (599, "ACTIVE"),
        (600, "SHORT_PAUSE"),
        (2 * 3600 - 1, "SHORT_PAUSE"),
        (2 * 3600, "EXTENDED_PAUSE"),
        (6 * 3600 - 1, "EXTENDED_PAUSE"),
        (6 * 3600, "RETURNING"),
        (24 * 3600 - 1, "RETURNING"),
        (24 * 3600, "NEW_DAY"),
        (3 * 86400, "NEW_DAY"),
    ]
    for gap, expected in cases:
        assert classify_continuity(gap).value == expected, gap


def test_runtime_block_shows_user_time_and_the_new_day_stage() -> None:
    """The Speaker used to see only when Ava herself last replied."""
    from datetime import datetime, timedelta, timezone

    from companion.runtime_context.assembler import compile_runtime_context
    from companion.runtime_context.snapshot import RuntimeSnapshot

    now = datetime(2026, 11, 13, 1, 20, tzinfo=timezone.utc)
    heard = now - timedelta(hours=26)
    snapshot = RuntimeSnapshot(
        snapshot_id="rts_e1",
        event_id="evt_1",
        conversation_id="c1",
        speaker_person_id="person_1",
        speaker_label="Alice",
        received_at_utc=now,
        occurred_at_utc=now,
        timezone="Asia/Shanghai",
        timezone_source="user_config",
        utc_offset_seconds=8 * 3600,
        local_time="2026-11-13 09:20",
        last_heard_at_utc=heard,
        last_heard_gap_seconds=26 * 3600,
        last_interaction_at_utc=heard,
        last_interaction_gap_seconds=26 * 3600,
        continuity="NEW_DAY",
    )
    block = compile_runtime_context(snapshot, now=now)
    assert "last_user_message=" in block
    assert "continuity=NEW_DAY" in block
    assert "新的交流阶段" in block

    # An ordinary exchange stays compact: no extra continuity explanation.
    close = snapshot.model_copy(
        update={
            "continuity": "ACTIVE",
            "last_heard_gap_seconds": 30,
            "last_interaction_gap_seconds": 20,
        }
    )
    compact = compile_runtime_context(close, now=now)
    assert "last_user_message=" in compact
    assert "continuity=" not in compact


async def test_proactive_snapshot_measures_silence_at_generation_time(db) -> None:
    """The proactive turn used to inherit the old source event's clock."""
    from datetime import datetime, timedelta, timezone

    from test_v06_runtime_context import CapturingClient, _event, _runtime

    rt, person = await _runtime(db, CapturingClient())
    await rt.repos.conversation.get_or_create("c1")
    source = _event(
        person.person_id,
        "昨天的机器人电机问题",
        at=datetime.now(timezone.utc) - timedelta(hours=26),
    )
    await rt.repos.utterance.save(source)
    moment = datetime.now(timezone.utc)
    snapshot = await rt.orchestrator.runtime_snapshot_builder.build(
        source,
        allow_external_ai=True,
        persist=False,
        received_at=moment,
        temporal_now=moment,
    )
    # The person's own last message — the source event itself — not the one
    # before it, and not Ava's proactive speech.
    assert snapshot.last_heard_at_utc == source.occurred_at
    assert snapshot.last_heard_gap_seconds >= 25 * 3600
    assert snapshot.continuity == "NEW_DAY"
    # Nothing was written: the source event keeps its original timing.
    assert await rt.repos.runtime_snapshot.get(source.event_id) is None


def test_response_prompt_separates_short_and_long_breaks() -> None:
    from companion.prompts.response import SYSTEM_PROMPT

    assert "新的交流阶段" in SYSTEM_PROMPT
    assert "隔了数小时" in SYSTEM_PROMPT
    assert "不要因为间隔长就必定说" in SYSTEM_PROMPT


# ------------------------------ Part 2: proactive must not be re-recognized
async def _proactive_runtime(db):
    from test_v0999_continuous_proactive import (
        _delivered_turn,
        _person,
        _runtime,
        _scripted,
        _start_cycle,
    )

    rt = await _runtime(db)
    person = await _person(rt)
    return rt, person, _delivered_turn, _scripted, _start_cycle


async def test_generated_proactive_is_not_completed_before_delivery(db):
    """Generated is not delivered; the row stays GENERATED until the ack."""
    rt, person, _delivered_turn, _scripted, _start_cycle = (
        await _proactive_runtime(db)
    )
    _scripted(rt.proactive_manager, [True])
    manager = rt.proactive_manager
    try:
        event = await _delivered_turn(rt, person)
        await _start_cycle(rt, person, event)
        message = await asyncio.wait_for(manager.next_event(), 2)
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert len(rows) == 1
        assert rows[0].message_id == message.response_id
        assert rows[0].status == "GENERATED"

        assert manager.note_proactive_delivered(message) is True
        await settle(rounds=10)
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert rows[0].status == "COMPLETED"
    finally:
        await manager.shutdown()


async def test_undelivered_proactive_is_never_replayed_as_history(db):
    from companion.domain.enums import EventSource
    from companion.domain.models import make_utterance_event
    from companion.routing.context import RoutingContextBuilder

    rt, person, _delivered_turn, _scripted, _start_cycle = (
        await _proactive_runtime(db)
    )
    _scripted(rt.proactive_manager, [True])
    manager = rt.proactive_manager
    try:
        first = await _delivered_turn(rt, person)
        await _start_cycle(rt, person, first)
        message = await asyncio.wait_for(manager.next_event(), 2)

        second = make_utterance_event(
            conversation_id="c1",
            text="你刚才说什么？",
            addressee_hint="agent",
            speaker_person_id=person.person_id,
            source=EventSource.CLI,
        )
        await rt.repos.utterance.save(second)
        builder = RoutingContextBuilder(repos=rt.repos, registry=rt.registry)
        history = await rt.repos.utterance.recent_context(
            "c1", exclude_event_id=second.event_id, limit=8
        )
        # Generated, not yet delivered: invisible to the router, so Ava can
        # never treat her own unsent line as something she already said.
        assert await builder._proactive_by_source(history) == {}

        # The surface really delivered it. Only now is it dialogue history.
        assert manager.note_proactive_delivered(message) is True
        await settle(rounds=20)
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert rows[-1].status == "COMPLETED"
        grouped = await builder._proactive_by_source(history)
        delivered = [
            row
            for rows_ in grouped.values()
            for row in rows_
            if row.status == "COMPLETED"
        ]
        assert delivered
        turns = builder._interleave(
            history,
            await rt.repos.analysis.get_actions_for_events(
                [item.event_id for item in history]
            ),
            {},
            grouped,
        )
        assistant_texts = [
            turn["text"] for turn in turns if turn["role"] == "assistant"
        ]
        assert delivered[-1].text in assistant_texts
    finally:
        await manager.shutdown()


async def test_no_second_message_while_delivery_is_pending(db):
    """Repeated generation is what made Ava answer herself."""
    from test_v0999_continuous_proactive import MODE_IMMEDIATE

    rt, person, _delivered_turn, _scripted, _start_cycle = (
        await _proactive_runtime(db)
    )
    _scripted(rt.proactive_manager, [True, True, True])
    manager = rt.proactive_manager
    try:
        event = await _delivered_turn(rt, person)
        await _start_cycle(rt, person, event)
        first = await asyncio.wait_for(manager.next_event(), 2)
        # No acknowledgement: the next round must not generate another one.
        await asyncio.sleep(0.2)
        assert manager._states["c1"].pending_message_id == first.response_id
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert len(rows) == 1
        assert rows[0].mode == MODE_IMMEDIATE
    finally:
        await manager.shutdown()
