"""Offline tests for the V0.9.9.20 voice input integration.

Fake ASR manager, fake TTS manager, fake proactive manager and a fake turn
runner keep every test network-free and deterministic. The real Silero,
DashScope, PyAudio and prompt_toolkit objects are never constructed here.
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from companion.asr.config import ASRConfig
from companion.asr.manager import ASRManager, FinalUtterance, SpeechActivity
from companion.asr.microphone import AudioChunk
from companion.asr.terminal import TerminalRenderer
from companion.asr.vad import SPEECH_ENDED, SPEECH_STARTED, VADEvent
from companion.cli import handle_user_message
from companion.domain.enums import EventSource
from companion.tts.manager import PlaybackEvent, TTSManager
from companion.tts.config import TTSConfig
from companion.voice import VoiceCoordinator, VoiceState


# --------------------------------------------------------------------- helpers
async def settle(rounds: int = 3) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0.01)


def wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeASR:
    """ASR manager contract used by the coordinator, driven by the test."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(model="fake-asr")
        self.microphone = SimpleNamespace(description="Fake Mic")
        self.started = 0
        self.stopped = 0
        self.shutdowns = 0
        self.input_states: list[bool] = []
        self.on_final = None
        self.on_speech = None
        self.on_closed = None
        self.render_finals = True
        self._stop = threading.Event()
        self._running = threading.Event()

    def set_listeners(
        self, *, on_final=None, on_speech=None, on_turn_closed=None, render_finals=None
    ) -> None:
        self.on_final = on_final
        self.on_speech = on_speech
        self.on_closed = on_turn_closed
        if render_finals is not None:
            self.render_finals = render_finals

    def start(self) -> None:
        self.started += 1

    def run(self) -> None:
        self._running.set()
        self._stop.wait(5.0)

    def stop(self) -> None:
        self.stopped += 1
        self._stop.set()

    def shutdown(self) -> None:
        self.shutdowns += 1
        self._stop.set()

    def set_input_enabled(self, enabled: bool) -> None:
        self.input_states.append(bool(enabled))

    # -- test-side event injection (standing in for ASR worker threads) -----
    def speech(self, uid: int) -> None:
        self.on_speech(SpeechActivity("started", uid, time.monotonic()))

    def final(self, uid: int, text: str) -> None:
        self.on_final(FinalUtterance(uid, text, time.monotonic()))

    def closed(self, uid: int, ok: bool) -> None:
        self.on_closed(uid, ok)


class FakeTTS:
    enabled = True

    def __init__(self) -> None:
        self.listeners = []
        self.stops = 0

    def add_playback_listener(self, listener) -> None:
        self.listeners.append(listener)

    def remove_playback_listener(self, listener) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def stop(self, **kwargs) -> None:
        self.stops += 1

    def emit(self, kind: str, generation: int) -> None:
        event = PlaybackEvent(
            kind=kind,
            generation=generation,
            response_id="resp",
            conversation_id="conv",
            at=time.monotonic(),
        )
        for listener in list(self.listeners):
            listener(event)


class FakeProactive:
    def __init__(self) -> None:
        self.notes = []
        self.delivered = []
        self.cancels = 0

    def note_user_input(self, conversation_id, source_event_id) -> None:
        self.notes.append((conversation_id, source_event_id))

    def on_response_delivered(self, **kwargs) -> None:
        self.delivered.append(kwargs)

    def cancel_cycle(self, conversation_id) -> None:
        self.cancels += 1


class FakeRuntime:
    def __init__(self) -> None:
        self.proactive_manager = FakeProactive()


async def make_coordinator(
    *,
    tts=None,
    cooldown: float = 0.0,
    factory=None,
    on_speech_started=None,
    notices=None,
):
    fake = FakeASR()
    coordinator = VoiceCoordinator(
        manager_factory=factory or (lambda: fake),
        tts_manager=tts,
        cooldown_seconds=cooldown,
        on_speech_started=on_speech_started,
        on_notice=(notices.append if notices is not None else None),
    )
    return coordinator, fake


async def accepted_turn(coordinator, fake, text: str, uid: int = 1):
    fake.speech(uid)
    await settle(1)
    fake.final(uid, text)
    await settle()
    return await asyncio.wait_for(coordinator.next_final(), 1.0)


# ------------------------------------------------------- ASR callback contract
class _Mic:
    description = "Fake Mic"
    dropped_chunks = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _Segmenter:
    def __init__(self, groups: list[list[VADEvent]]) -> None:
        self.groups = list(groups)

    def process(self, pcm: bytes) -> list[VADEvent]:
        return self.groups.pop(0) if self.groups else []

    def reset(self) -> None:
        pass


class _Session:
    def __init__(self, finals: list[tuple[str, str]]) -> None:
        self.finals = finals
        self._stopped = threading.Event()

    def open(self) -> None:
        pass

    def send_audio(self, pcm: bytes) -> None:
        pass

    def stop(self) -> None:
        self._stopped.set()

    def events(self):
        from companion.asr.client import ASREvent

        yield ASREvent("sentence", "临", "s1", False)
        yield ASREvent("sentence", "临时", "s1", False)
        self._stopped.wait(2.0)
        for text, sid in self.finals:
            yield ASREvent("sentence", text, sid, True)

    def close(self) -> None:
        self._stopped.set()


def _asr_config() -> ASRConfig:
    return ASRConfig(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        timeout_seconds=2.0,
        final_timeout_seconds=1.0,
    )


def test_partials_never_fire_the_external_final_callback() -> None:
    finals: list[FinalUtterance] = []
    closed: list[tuple[int, bool]] = []
    out = io.StringIO()
    renderer = TerminalRenderer(
        stream=out, status_stream=io.StringIO(), is_tty=False
    )
    segmenter = _Segmenter(
        [
            [VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)],
            [VADEvent(SPEECH_ENDED)],
        ]
    )
    manager = ASRManager(
        _asr_config(),
        microphone=_Mic(),
        vad_segmenter=segmenter,
        session_factory=lambda: _Session([("你好世界", "s1")]),
        renderer=renderer,
        on_final=finals.append,
        on_turn_closed=lambda uid, ok: closed.append((uid, ok)),
    )
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    try:
        manager.audio_queue.put(
            AudioChunk(b"\x00\x00" * 512, 16000, 16000, time.monotonic()),
            timeout=2.0,
        )
        # Partials arrive while the turn is still open: no callback yet.
        time.sleep(0.15)
        assert finals == []
        manager.audio_queue.put(
            AudioChunk(b"\x00\x00" * 512, 16000, 16000, time.monotonic()),
            timeout=2.0,
        )
        assert wait_for(lambda: len(finals) == 1)
        assert finals[0].text == "你好世界"
        assert closed == [(finals[0].utterance_id, True)]
    finally:
        manager.stop()
        thread.join(timeout=3.0)
        manager.shutdown()
    assert "临时" not in out.getvalue()
    assert "你好世界" in out.getvalue()


# --------------------------------------------------------------- shared handler
async def test_voice_and_keyboard_share_one_handler_and_event_source() -> None:
    calls: list[tuple[EventSource, str]] = []

    async def runner(rt, event, **kwargs):
        calls.append((event.source, event.text))
        return {"delivered": True, "status": "COMPLETED", "action": "RESPOND"}

    rt = FakeRuntime()
    for text, source in (("键盘输入", EventSource.CLI), ("voz 语音", EventSource.VOICE)):
        await handle_user_message(
            rt,
            text=text,
            source=source,
            submitted_at=time.monotonic(),
            conversation_id="default",
            person_id="p1",
            addressee="agent",
            turn_runner=runner,
        )

    assert calls == [
        (EventSource.CLI, "键盘输入"),
        (EventSource.VOICE, "voz 语音"),
    ]
    assert len(rt.proactive_manager.notes) == 2
    assert len(rt.proactive_manager.delivered) == 2


async def test_mixed_language_text_is_forwarded_unchanged() -> None:
    seen: list[str] = []

    async def runner(rt, event, **kwargs):
        seen.append(event.text)
        return {"delivered": False}

    rt = FakeRuntime()
    mixed = "今天 weather 不错, tá bom! 明天见。"
    await handle_user_message(
        rt,
        text=mixed,
        source=EventSource.VOICE,
        submitted_at=time.monotonic(),
        conversation_id="default",
        person_id="p1",
        addressee="agent",
        turn_runner=runner,
    )
    assert seen == [mixed]


# --------------------------------------------------------------- coordinator
async def test_asr_final_is_accepted_exactly_once() -> None:
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    try:
        fake.speech(1)
        await settle(1)
        fake.final(1, "你好")
        fake.final(1, "你好")  # duplicate callback
        await settle()
        final = await asyncio.wait_for(coordinator.next_final(), 1.0)
        assert final.text == "你好"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coordinator.next_final(), 0.05)
        assert coordinator.stats.finals_accepted == 1
    finally:
        await coordinator.shutdown()


async def test_partial_events_cannot_start_a_turn() -> None:
    # The bridge has no partial entry point at all; only finals are accepted.
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    try:
        # A final arriving while not recording (Ava thinking/speaking/idle) is
        # stale, never a new user message.
        fake.final(9, "不该出现的消息")
        await settle()
        assert coordinator.stats.stale_finals == 1
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coordinator.next_final(), 0.05)
    finally:
        await coordinator.shutdown()


async def test_mic_is_paused_while_ava_replies() -> None:
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    try:
        final = await accepted_turn(coordinator, fake, "第一句")
        assert coordinator.state == VoiceState.THINKING
        assert fake.input_states[-1] is False

        # A result produced while THINKING (e.g. Ava's own audio) is dropped.
        fake.final(2, "第二句")
        await settle()
        assert coordinator.stats.stale_finals == 1
        assert final.text == "第一句"
    finally:
        await coordinator.shutdown()


async def test_listening_resumes_only_after_real_playback_completion() -> None:
    tts = FakeTTS()
    coordinator, fake = await make_coordinator(tts=tts)
    ok, _ = await coordinator.enable()
    assert ok
    try:
        final = await accepted_turn(coordinator, fake, "你好")
        coordinator.begin_turn()
        coordinator.on_reply_started(7)
        await settle(1)
        assert coordinator.state == VoiceState.SPEAKING

        coordinator.on_turn_finished(
            {"delivered": True, "status": "COMPLETED", "action": "RESPOND"}
        )
        await settle()
        # Text is done, but this reply's audio has not finished.
        assert coordinator.state == VoiceState.SPEAKING
        assert fake.input_states[-1] is False

        tts.emit("finished", 7)
        await settle()
        assert coordinator.state == VoiceState.LISTENING
        assert fake.input_states[-1] is True
        assert final.text == "你好"
    finally:
        await coordinator.shutdown()


async def test_cooldown_delays_resume_after_playback() -> None:
    tts = FakeTTS()
    coordinator, fake = await make_coordinator(tts=tts, cooldown=0.2)
    ok, _ = await coordinator.enable()
    assert ok
    try:
        await accepted_turn(coordinator, fake, "你好")
        coordinator.begin_turn()
        coordinator.on_reply_started(3)
        coordinator.on_turn_finished({"delivered": True})
        tts.emit("finished", 3)
        await settle(2)
        assert coordinator.state == VoiceState.SPEAKING
        await asyncio.sleep(0.25)
        assert coordinator.state == VoiceState.LISTENING
    finally:
        await coordinator.shutdown()


async def test_tts_off_resumes_after_the_text_reply() -> None:
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    try:
        await accepted_turn(coordinator, fake, "你好")
        coordinator.begin_turn()
        coordinator.on_turn_finished(
            {"delivered": True, "status": "COMPLETED", "action": "RESPOND"}
        )
        await settle()
        assert coordinator.state == VoiceState.LISTENING
        assert fake.input_states[-1] is True
    finally:
        await coordinator.shutdown()


async def test_asr_start_failure_never_breaks_text_chat() -> None:
    def factory():
        raise RuntimeError("ASR 配置缺失: DASHSCOPE_API_KEY")

    coordinator = VoiceCoordinator(
        manager_factory=factory, tts_manager=None, cooldown_seconds=0.0
    )
    ok, message = await coordinator.enable()
    assert not ok
    assert "DASHSCOPE_API_KEY" in message
    assert coordinator.state == VoiceState.DISABLED

    # The shared text path is independent of microphone availability.
    seen = []

    async def runner(rt, event, **kwargs):
        seen.append(event.text)
        return {"delivered": False}

    await handle_user_message(
        FakeRuntime(),
        text="仍然可以打字",
        source=EventSource.CLI,
        submitted_at=time.monotonic(),
        conversation_id="default",
        person_id="p1",
        addressee="agent",
        turn_runner=runner,
    )
    assert seen == ["仍然可以打字"]
    await coordinator.shutdown()


async def test_wait_or_ignore_does_not_permanently_mute_the_mic() -> None:
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    try:
        await accepted_turn(coordinator, fake, "你好")
        coordinator.begin_turn()
        coordinator.on_turn_finished({"delivered": False})
        await settle()
        assert coordinator.state == VoiceState.LISTENING
        # A later utterance is still recognized.
        final = await accepted_turn(coordinator, fake, "第二句", uid=2)
        assert final.text == "第二句"
    finally:
        await coordinator.shutdown()


async def test_proactive_is_blocked_while_the_user_records() -> None:
    paused: list[int] = []
    coordinator, fake = await make_coordinator(
        on_speech_started=lambda: paused.append(1)
    )
    ok, _ = await coordinator.enable()
    assert ok
    try:
        assert coordinator.should_deliver_proactive()
        fake.speech(1)
        await settle()
        assert coordinator.state == VoiceState.RECORDING
        assert paused == [1]
        assert not coordinator.should_deliver_proactive()

        await asyncio.sleep(0)  # final already queued next
        fake.final(1, "你好")
        await settle()
        await coordinator.next_final()
        coordinator.begin_turn()
        coordinator.on_turn_finished({"delivered": False})
        await settle()
        assert coordinator.should_deliver_proactive()
    finally:
        await coordinator.shutdown()


async def test_stale_tts_generation_cannot_resume_the_mic() -> None:
    tts = FakeTTS()
    coordinator, fake = await make_coordinator(tts=tts)
    ok, _ = await coordinator.enable()
    assert ok
    try:
        await accepted_turn(coordinator, fake, "你好")
        coordinator.begin_turn()
        coordinator.on_reply_started(2)
        coordinator.on_turn_finished({"delivered": True})
        tts.emit("finished", 1)  # older reply
        await settle()
        assert coordinator.state == VoiceState.SPEAKING
        tts.emit("finished", 2)
        await settle()
        assert coordinator.state == VoiceState.LISTENING
    finally:
        await coordinator.shutdown()


async def test_voice_and_keyboard_inputs_are_processed_serially() -> None:
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    turns: list[tuple[EventSource, str]] = []

    async def runner(rt, event, **kwargs):
        turns.append((event.source, event.text))
        await asyncio.sleep(0)
        return {"delivered": False}

    rt = FakeRuntime()
    try:
        fake.speech(1)
        await settle(1)
        fake.final(1, "语音输入")
        await settle()
        # A keyboard line can be accepted while the voice final is queued.
        coordinator.begin_turn()
        final = await asyncio.wait_for(coordinator.next_final(), 1.0)
        await handle_user_message(
            rt,
            text=final.text,
            source=EventSource.VOICE,
            submitted_at=time.monotonic(),
            conversation_id="default",
            person_id="p1",
            addressee="agent",
            turn_runner=runner,
        )
        coordinator.on_turn_finished({"delivered": False})
        await handle_user_message(
            rt,
            text="键盘输入",
            source=EventSource.CLI,
            submitted_at=time.monotonic(),
            conversation_id="default",
            person_id="p1",
            addressee="agent",
            turn_runner=runner,
        )
        assert turns == [
            (EventSource.VOICE, "语音输入"),
            (EventSource.CLI, "键盘输入"),
        ]
    finally:
        await coordinator.shutdown()


async def test_mic_disconnect_recovers_to_listening() -> None:
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    try:
        fake.speech(1)
        await settle()
        assert coordinator.state == VoiceState.RECORDING
        # Bluetooth/device loss aborts the turn without a final.
        fake.closed(1, False)
        await settle()
        assert coordinator.state == VoiceState.LISTENING
        assert fake.input_states[-1] is True
        final = await accepted_turn(coordinator, fake, "恢复后", uid=2)
        assert final.text == "恢复后"
    finally:
        await coordinator.shutdown()


async def test_fifty_consecutive_turns_leave_no_threads_or_duplicates() -> None:
    coordinator, fake = await make_coordinator()
    ok, _ = await coordinator.enable()
    assert ok
    try:
        seen: list[str] = []
        for index in range(50):
            final = await accepted_turn(
                coordinator, fake, f"第{index}句", uid=index + 1
            )
            seen.append(final.text)
            coordinator.begin_turn()
            coordinator.on_turn_finished({"delivered": False})
            await settle(1)
            assert coordinator.state == VoiceState.LISTENING
        assert seen == [f"第{i}句" for i in range(50)]
        assert coordinator.stats.finals_accepted == 50
        assert coordinator._finals.empty()
        assert fake.started == 1
        runners = [
            t for t in threading.enumerate() if t.name == "ava-voice-asr"
        ]
        assert len(runners) == 1
    finally:
        await coordinator.shutdown()
    assert not [
        t for t in threading.enumerate() if t.name == "ava-voice-asr"
    ]


# ---------------------------------------------------- TTS playback notification
class _TtsSession:
    _END = object()

    def __init__(self) -> None:
        self._audio: "queue.Queue" = queue.Queue()

    def start(self):
        return self

    def send_text(self, text: str) -> None:
        self._audio.put(b"\x01\x00")

    def flush(self) -> None:
        pass

    def audio_chunks(self):
        while True:
            item = self._audio.get(timeout=5)
            if item is self._END:
                return
            yield item

    def stop(self) -> None:
        self._audio.put(self._END)

    def close(self) -> None:
        self._audio.put(self._END)


class _TtsClient:
    def __init__(self) -> None:
        self.sessions = []

    def open_stream(self):
        session = _TtsSession()
        session.start()
        self.sessions.append(session)
        return session

    def stream_pcm(self, text):
        raise AssertionError("must use the streaming path")
        yield b""

    def close(self) -> None:
        pass


class _TtsPlayer:
    def __init__(self) -> None:
        self.writes = []

    def open(self) -> None:
        pass

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    def interrupt(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_tts_playback_notice_carries_response_and_generation() -> None:
    config = TTSConfig(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        segment_min_chars=1,
        segment_max_chars=60,
        instruction="test",
    )
    manager = TTSManager(
        config_loader=lambda: config,
        client_factory=lambda cfg: _TtsClient(),
        player_factory=lambda cfg: _TtsPlayer(),
    )
    events: list[PlaybackEvent] = []
    manager.start()
    manager.add_playback_listener(events.append)
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        assert stream is not None
        stream.feed("你好。")
        stream.finish()
        assert wait_for(lambda: any(e.kind == "finished" for e in events))
        finished = [e for e in events if e.kind == "finished"][-1]
        assert finished.generation == stream.generation
        assert finished.response_id == "resp-1"
        assert finished.conversation_id == "conv-1"
    finally:
        manager.shutdown()
