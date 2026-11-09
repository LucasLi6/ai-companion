"""V0.9.9.21: continuous proactive scheduling + proactive speech.

Offline only. These tests pin the behaviours introduced by the continuous
proactive feature: adaptive intervals that slow down but never stop, stable
per-round persistence IDs, real TTS submission, deferral (never silent drop)
while Ava is speaking, mic half-duplex during proactive playback, and clean
shutdown without orphan tasks or timers.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from companion.asr.manager import FinalUtterance, SpeechActivity
from companion.cli import (
    _deliver_proactive_voice,
    _render_proactive_event,
)
from companion.core.response_events import ResponseEvent
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.proactive.judge import ProactiveJudge
from companion.proactive.manager import (
    ACTIVITY_HIGH,
    ACTIVITY_INTERVALS,
    ACTIVITY_LOW,
    MODE_IMMEDIATE,
    MODE_LONG_IDLE,
    ProactiveManager,
    _Cycle,
)
from companion.providers.deepseek_client import ChatResponse
from companion.providers.deepseek_proactive import (
    ScriptedProactiveJudgeProvider,
)
from companion.runtime import build_runtime
from companion.tts.manager import PlaybackEvent
from companion.voice import VoiceCoordinator, VoiceState


# --------------------------------------------------------------------- offline
class FakeDeepSeekClient:
    """Offline client: normal reply text, silent judge JSON by default."""

    def __init__(self, response_text: str = "正常回复。") -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(
            {
                "request_type": request_type,
                "messages": messages,
                "event_id": event_id,
            }
        )
        content = (
            self.response_text
            if request_type == "response"
            else json.dumps(
                {"should_initiate": False, "reason": "scripted"},
                ensure_ascii=False,
            )
        )
        return ChatResponse(
            content=content,
            model="fake-model",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _runtime(db, client=None):
    return build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client or FakeDeepSeekClient(),
        with_worker=False,
        router_mode="legacy",
    )


async def _person(rt, name="Proactive"):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _scripted(manager, decisions):
    provider = ScriptedProactiveJudgeProvider(list(decisions))
    manager.judge = ProactiveJudge(provider)
    manager.idle_delay_seconds = 0.02
    return provider


async def _delivered_turn(rt, person, text="我在研究机器人控制", conversation="c1"):
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.CLI,
    )
    report = await rt.orchestrator.process_utterance(event)
    assert report.status in ("COMPLETED", "WAIT", "IGNORE")
    return event


async def _start_cycle(rt, person, event, conversation="c1"):
    rt.proactive_manager.on_response_delivered(
        conversation_id=conversation,
        person_id=person.person_id,
        source_event_id=event.event_id,
    )


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met in time")


def _event(
    *,
    conversation="c1",
    source="evt",
    message="pmsg-1",
    text="你好呀，我刚刚想到一件事。",
    display=None,
) -> ResponseEvent:
    return ResponseEvent(
        type="response_completed",
        event_id=source,
        conversation_id=conversation,
        response_id=message,
        sequence=0,
        status="COMPLETED",
        text=text,
        display_text=display,
        report_status="COMPLETED",
        action="RESPOND",
    )


class RecordingTTS:
    """Minimal TTSManager contract: records SpeechJobs, drives playback."""

    enabled = True

    def __init__(self) -> None:
        self.generation = 0
        self.jobs = []
        self.listeners = []
        self.stops = 0

    def add_playback_listener(self, listener) -> None:
        self.listeners.append(listener)

    def remove_playback_listener(self, listener) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def submit(self, job) -> bool:
        self.jobs.append(job)
        self.generation += 1
        return True

    def stop(self, **kwargs) -> None:
        self.stops += 1

    def emit(self, kind: str, generation=None) -> None:
        event = PlaybackEvent(
            kind=kind,
            generation=self.generation if generation is None else generation,
            response_id="pmsg-1",
            conversation_id="c1",
            at=time.monotonic(),
        )
        for listener in list(self.listeners):
            listener(event)


class FakeManager:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.delivered = []
        self.cancelled = []
        self.user_notes = []

    async def can_deliver(self, event) -> bool:
        return self.allowed

    def note_proactive_delivered(self, event) -> None:
        self.delivered.append(event)

    def note_proactive_cancelled(self, event, reason) -> None:
        self.cancelled.append((event, reason))

    def note_user_input(self, conversation_id, source_event_id) -> None:
        self.user_notes.append((conversation_id, source_event_id))


class FakeASR:
    """ASR manager contract used by VoiceCoordinator."""

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
        self._stop = threading.Event()

    def set_listeners(
        self,
        *,
        on_final=None,
        on_speech=None,
        on_turn_closed=None,
        render_finals=None,
    ) -> None:
        self.on_final = on_final
        self.on_speech = on_speech
        self.on_closed = on_turn_closed

    def start(self) -> None:
        self.started += 1

    def run(self) -> None:
        self._stop.wait(5.0)

    def stop(self) -> None:
        self.stopped += 1
        self._stop.set()

    def shutdown(self) -> None:
        self.shutdowns += 1
        self._stop.set()

    def set_input_enabled(self, enabled: bool) -> None:
        self.input_states.append(bool(enabled))

    def speech(self, uid: int) -> None:
        self.on_speech(SpeechActivity("started", uid, time.monotonic()))

    def final(self, uid: int, text: str) -> None:
        self.on_final(FinalUtterance(uid, text, time.monotonic()))

    def closed(self, uid: int, ok: bool) -> None:
        self.on_closed(uid, ok)


async def _coordinator(tts):
    fake = FakeASR()
    coordinator = VoiceCoordinator(
        manager_factory=lambda: fake,
        tts_manager=tts,
        cooldown_seconds=0.0,
    )
    ok, _ = await coordinator.enable()
    assert ok
    return coordinator, fake


async def _settle(rounds: int = 4) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0.01)


# --------------------------------------------------- continuous scheduling (1)
async def test_silence_keeps_producing_adaptive_judgments(db):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person)
    provider = _scripted(rt.proactive_manager, [])
    manager = rt.proactive_manager
    try:
        await _start_cycle(rt, person, event)
        await _wait_for(lambda: len(provider.payloads) >= 4, timeout=2.0)
        modes = [p.mode for p in provider.payloads]
        assert modes[0] == MODE_IMMEDIATE
        assert all(mode == MODE_LONG_IDLE for mode in modes[1:])
        # Still scheduled: silence never permanently stops the loop.
        assert manager._states["c1"].task is not None
        assert not manager._states["c1"].task.done()
    finally:
        await manager.shutdown()


# ------------------------------------------- second LONG_IDLE still judges (2)
async def test_second_long_idle_round_can_still_deliver(db):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person)
    provider = _scripted(rt.proactive_manager, [False, False, True])
    manager = rt.proactive_manager
    try:
        await _start_cycle(rt, person, event)
        delivered = await asyncio.wait_for(manager.next_event(), 2)
        assert delivered is not None
        assert delivered.text == "正常回复。"
        assert [p.mode for p in provider.payloads[:3]] == [
            MODE_IMMEDIATE,
            MODE_LONG_IDLE,
            MODE_LONG_IDLE,
        ]
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert [row.mode for row in rows] == [f"{MODE_LONG_IDLE}_2"]
    finally:
        await manager.shutdown()


# --------------------------------------- multiple persisted rounds, no crash (3)
async def test_multiple_rounds_persist_with_distinct_storage_modes(db):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person)
    _scripted(rt.proactive_manager, [True, True, True])
    manager = rt.proactive_manager
    try:
        await _start_cycle(rt, person, event)
        for _ in range(3):
            assert await asyncio.wait_for(manager.next_event(), 2) is not None
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert sorted(row.mode for row in rows) == sorted(
            [MODE_IMMEDIATE, f"{MODE_LONG_IDLE}_1", f"{MODE_LONG_IDLE}_2"]
        )
        assert len({row.message_id for row in rows}) == 3
        # A later attempt for an already-persisted round is a no-op even if
        # the judge would say yes again.
        assert await manager._attempt(
            _Cycle("c1", person.person_id, event.event_id, 1),
            manager._states["c1"],
            MODE_LONG_IDLE,
            f"{MODE_LONG_IDLE}_1",
        ) is None
        assert len(await rt.repos.proactive.list_for_conversation("c1")) == 3
    finally:
        await manager.shutdown()


# ----------------------------------------------- proactive speech reaches TTS (4)
async def test_proactive_message_submits_display_text_to_tts(capsys):
    tts = RecordingTTS()
    coordinator = SimpleNamespace(
        calls=[], on_reply_started=lambda generation: coordinator.calls.append(generation)
    )
    event = _event(display="[soft] 你好呀，在想什么呢？")

    rendered = await _render_proactive_event(
        event, None, tts_manager=tts, coordinator=coordinator
    )

    assert rendered is True
    assert "assistant> [soft] 你好呀，在想什么呢？" in capsys.readouterr().out
    assert len(tts.jobs) == 1
    assert tts.jobs[0].text == "[soft] 你好呀，在想什么呢？"
    assert tts.jobs[0].response_id == event.response_id
    assert tts.jobs[0].conversation_id == event.conversation_id
    # The coordinator is told about the real generation that is now speaking.
    assert coordinator.calls == [tts.generation]

    # Without display_text the canonical text is spoken.
    plain = _event(message="pmsg-2", text="那本书我读完了。", display=None)
    await _render_proactive_event(plain, None, tts_manager=tts)
    assert tts.jobs[-1].text == "那本书我读完了。"


# --------------------------- mic is paused during proactive playback, no ASR (5)
async def test_proactive_speech_pauses_mic_and_never_hears_itself():
    tts = RecordingTTS()
    manager = FakeManager()
    coordinator, fake = await _coordinator(tts)
    event = _event()
    try:
        task = asyncio.create_task(
            _deliver_proactive_voice(
                event, manager=manager, coordinator=coordinator, tts_manager=tts
            )
        )
        await _wait_for(lambda: bool(tts.jobs), timeout=2.0)
        assert coordinator.state == VoiceState.SPEAKING
        assert fake.input_states[-1] is False
        assert manager.delivered == []

        # Ava's own audio is never accepted: input is paused and the
        # coordinator is SPEAKING, so any stray ASR result is stale.
        fake.speech(7)
        await _settle(2)
        fake.final(7, "你好呀，我刚刚想到一件事。")
        await _settle(2)
        assert coordinator.stats.stale_finals >= 1
        assert coordinator._finals.empty()

        tts.emit("finished")
        assert await asyncio.wait_for(task, 2) is True
        assert coordinator.state == VoiceState.LISTENING
        assert fake.input_states[-1] is True
        assert manager.delivered == [event]

        # Normal voice interaction still works afterwards.
        fake.speech(8)
        await _settle(2)
        fake.final(8, "下一句")
        await _settle(2)
        final = await asyncio.wait_for(coordinator.next_final(), 1.0)
        assert final.text == "下一句"
    finally:
        await coordinator.shutdown()


# ----------------------------------- a user turn invalidates the old cycle (6)
async def test_new_user_input_cancels_continuous_cycle(db):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person)
    provider = _scripted(rt.proactive_manager, [False, True])
    manager = rt.proactive_manager
    try:
        await _start_cycle(rt, person, event)
        await _wait_for(lambda: provider.payloads)
        manager.note_user_input("c1", "new-event")
        assert manager._states["c1"].task is None
        await asyncio.sleep(0.15)
        assert [p.mode for p in provider.payloads] == [MODE_IMMEDIATE]
        assert await rt.repos.proactive.list_for_conversation("c1") == []
        assert manager.poll_event() is None
    finally:
        await manager.shutdown()


async def test_deferred_proactive_is_cancelled_when_user_takes_over():
    tts = RecordingTTS()
    manager = FakeManager()
    coordinator, fake = await _coordinator(tts)
    event = _event()
    try:
        # A reply is in progress: the proactive message waits instead of
        # being dropped or interrupting.
        coordinator.begin_turn()
        task = asyncio.create_task(
            _deliver_proactive_voice(
                event, manager=manager, coordinator=coordinator, tts_manager=tts
            )
        )
        await _settle(3)
        assert not tts.jobs
        assert not task.done()

        # The user starts a new turn; the re-check rejects the stale message.
        manager.allowed = False
        coordinator.on_turn_finished({"delivered": False})
        assert await asyncio.wait_for(task, 2) is False
        assert tts.jobs == []
        assert manager.delivered == []
        assert manager.cancelled == [(event, "stale_before_delivery")]
        assert coordinator.state == VoiceState.LISTENING
    finally:
        await coordinator.shutdown()


# -------------------------------------- slow-down without permanent silence (7)
def test_activity_profiles_slow_down_but_never_stop():
    high = ProactiveManager(
        repos=None,
        registry=None,
        conversations=None,
        orchestrator=None,
        judge=None,
        activity="high",
    )
    assert high.interval_plan() == ACTIVITY_INTERVALS[ACTIVITY_HIGH]
    assert high._interval_bounds(0) == (20, 40)
    assert high._interval_bounds(1) == (45, 90)
    assert high._interval_bounds(2) == (120, 300)
    assert high._interval_bounds(3) == (180, 300)
    # Repeated unanswered rounds settle at 5-10 minutes and never stop.
    assert high._interval_bounds(4) == (300, 600)
    assert high._interval_bounds(99) == (300, 600)

    normal = ProactiveManager(
        repos=None,
        registry=None,
        conversations=None,
        orchestrator=None,
        judge=None,
        activity="normal",
    )
    assert normal._interval_bounds(0) == (120, 240)
    assert normal._interval_bounds(99) == (300, 600)

    low = ProactiveManager(
        repos=None,
        registry=None,
        conversations=None,
        orchestrator=None,
        judge=None,
        activity="low",
    )
    assert low._interval_bounds(0) == (300, 600)
    assert low._interval_bounds(99) == (600, 900)

    for unanswered in (0, 1, 2, 3, 4, 50):
        low_bound, high_bound = high._interval_bounds(unanswered)
        delay = high._next_idle_delay(unanswered)
        assert low_bound <= delay <= high_bound

    assert ACTIVITY_INTERVALS[ACTIVITY_LOW][-1] == (600, 900)


def test_invalid_activity_falls_back_to_high(monkeypatch):
    monkeypatch.setenv("AVA_PROACTIVE_ACTIVITY", "turbo")
    manager = ProactiveManager(
        repos=None,
        registry=None,
        conversations=None,
        orchestrator=None,
        judge=None,
    )
    assert manager.activity == ACTIVITY_HIGH


async def test_real_acks_advance_the_unanswered_interval(db):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person)
    _scripted(rt.proactive_manager, [True, True])
    manager = rt.proactive_manager
    assert manager.activity == ACTIVITY_HIGH
    try:
        await _start_cycle(rt, person, event)
        immediate = await asyncio.wait_for(manager.next_event(), 2)
        assert manager.note_proactive_delivered(immediate) is True
        # An immediate remark is visible but does not count as an unanswered
        # long-idle round.
        assert manager._states["c1"].unanswered == 0

        long_idle = await asyncio.wait_for(manager.next_event(), 2)
        assert manager.note_proactive_delivered(long_idle) is True
        assert manager._states["c1"].unanswered == 1
        assert manager._interval_bounds(1) == (45, 90)
        # A duplicate acknowledgement cannot slow it twice.
        assert manager.note_proactive_delivered(long_idle) is False
        assert manager._states["c1"].unanswered == 1
    finally:
        await manager.shutdown()


# ------------------------------- explicit departure / quiet / refusal stops (8)
@pytest.mark.parametrize(
    "text",
    ["我先去睡了，晚安", "我不想再聊了", "闭嘴，别说了"],
)
async def test_explicit_stop_prevents_any_round(db, text):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person, text=text)
    provider = _scripted(rt.proactive_manager, [True, True, True])
    manager = rt.proactive_manager
    try:
        await _start_cycle(rt, person, event)
        await asyncio.sleep(0.15)
        assert provider.payloads == []
        assert await rt.repos.proactive.list_for_conversation("c1") == []
        assert manager._states["c1"].task is None
    finally:
        await manager.shutdown()


async def test_quiet_mode_prevents_any_round(db):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person)
    provider = _scripted(rt.proactive_manager, [True, True])
    manager = rt.proactive_manager
    try:
        await rt.conversations.set_quiet_mode("c1", True)
        await _start_cycle(rt, person, event)
        await asyncio.sleep(0.15)
        assert provider.payloads == []
        assert await rt.repos.proactive.list_for_conversation("c1") == []
    finally:
        await manager.shutdown()


# ------------------------------ proactive does not break normal interaction (9)
async def test_ack_then_new_turn_starts_a_fresh_cycle(db):
    rt = await _runtime(db)
    person = await _person(rt)
    first = await _delivered_turn(rt, person)
    _scripted(rt.proactive_manager, [True, False, True])
    manager = rt.proactive_manager
    try:
        await _start_cycle(rt, person, first)
        proactive = await asyncio.wait_for(manager.next_event(), 2)
        assert manager.note_proactive_delivered(proactive) is True
        # A duplicate acknowledgement never double-counts.
        assert manager.note_proactive_delivered(proactive) is False

        second = await _delivered_turn(rt, person, "那控制频率呢？")
        await _start_cycle(rt, person, second)
        next_event = await asyncio.wait_for(manager.next_event(), 2)
        assert next_event is not None
        assert next_event.event_id == second.event_id
        rows = await rt.repos.proactive.list_for_conversation("c1")
        assert len(rows) == 2
    finally:
        await manager.shutdown()


# ------------------------------------- shutdown leaves no timers or tasks (10)
async def test_shutdown_cancels_cycle_and_audio_lifecycle():
    tts = RecordingTTS()
    manager = FakeManager()
    coordinator, fake = await _coordinator(tts)
    task = None
    try:
        coordinator.begin_turn()
        task = asyncio.create_task(
            _deliver_proactive_voice(
                _event(), manager=manager, coordinator=coordinator, tts_manager=tts
            )
        )
        await _settle(2)
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(manager.cancelled) == 1
        assert manager.cancelled[0][1] == "interrupted_by_user"

        await coordinator.shutdown()
        assert coordinator._resume_task is None
        assert coordinator._consume_task is None
        assert fake.stopped >= 1
        assert fake.shutdowns >= 1
        assert coordinator._tts_registered is False
        assert tts.listeners == []
        assert manager.delivered == []
    finally:
        if task is not None and not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        if coordinator.state != VoiceState.DISABLED:
            await coordinator.shutdown()


async def test_manager_shutdown_leaves_no_background_cycle(db):
    rt = await _runtime(db)
    person = await _person(rt)
    event = await _delivered_turn(rt, person)
    _scripted(rt.proactive_manager, [])
    manager = rt.proactive_manager
    await _start_cycle(rt, person, event)
    await _wait_for(lambda: manager._states["c1"].task is not None)
    cycle_task = manager._states["c1"].task
    await manager.shutdown()
    assert cycle_task.cancelled() or cycle_task.done()
    assert manager._states["c1"].task is None
    assert manager._states["c1"].next_attempt_at is None
    assert await manager.next_event() is None
