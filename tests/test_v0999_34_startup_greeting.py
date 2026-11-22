"""V0.9.9.34 — one memory-aware startup greeting.

The greeting reuses the proactive context/prompt/provider path but is
non-persisted and runs before ASR listening starts. These tests are offline:
a fake DeepSeek client, fake TTS manager and fake coordinator only.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from companion.cli import (
    STARTUP_GREETING_ENV,
    _enable_after_startup_greeting,
    _play_startup_greeting,
    _run_startup_greeting,
    _startup_greeting_enabled,
)
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime


class _GreetingClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    async def complete(self, messages, *, request_type, event_id=None, **kwargs):
        self.calls.append(
            {"request_type": request_type, "messages": messages}
        )
        return ChatResponse(
            content=self.text,
            model="fake",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _person(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("c1")
    return person


# --------------------------------------------------------------- generation
async def test_greeting_works_without_history_or_memories(db) -> None:
    client = _GreetingClient("刚刚安静下来，像是一天还没完全开始。")
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
    )
    person = await _person(rt)

    result = await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    assert result is not None
    assert result["text"] == "刚刚安静下来，像是一天还没完全开始。"
    assert len(client.calls) == 1
    system = client.calls[0]["messages"][0]["content"]
    assert "【启动开场白】" in system
    assert "【本轮生成模式：主动发言】" not in system

    # The synthetic source event is never persisted or rendered as a user
    # turn, so no conversation history was created.
    turns = await rt.repos.utterance.recent_context(
        "c1", exclude_event_id=None, limit=20
    )
    assert turns == []


async def test_opening_parses_voice_expression_without_a_recovery_call(db) -> None:
    client = _GreetingClient(
        "VOICE=excited\nTEXT=[giggles] 今天的光让人想慢一点。"
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
        speech_expression_enabled=True,
    )
    person = await _person(rt)

    result = await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    assert result is not None
    assert result["text"] == "今天的光让人想慢一点。"
    assert "[excited]" in result["display_text"]
    assert len(client.calls) == 1  # never a second generation


# ------------------------------------------------------------- CLI ordering
class _FakeTTS:
    def __init__(self, *, enabled: bool = True, emit: bool = True) -> None:
        self.enabled = enabled
        self.emit = emit
        self.generation = 0
        self.listeners: list = []
        self.jobs: list = []
        self.stop_calls = 0
        self.finished = threading.Event()

    def add_playback_listener(self, listener) -> None:
        self.listeners.append(listener)

    def remove_playback_listener(self, listener) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def submit(self, job) -> bool:
        assert self.enabled, "disabled TTS must never be submitted to"
        self.jobs.append(job)
        if self.emit:
            def _play() -> None:
                time.sleep(0.05)
                self.finished.set()
                for listener in list(self.listeners):
                    listener(
                        SimpleNamespace(
                            kind="finished", response_id=job.response_id
                        )
                    )

            threading.Thread(target=_play, daemon=True).start()
        return True

    def stop(self, *, wait: bool = False) -> None:
        self.stop_calls += 1


class _Output:
    def __init__(self) -> None:
        self.lines: list[str] = []

    async def write(self, text: str) -> None:
        self.lines.append(text)


class _FakeOrchestrator:
    def __init__(self, text: str = "你好呀。", *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0
        self.proactive_reply_calls = 0

    async def generate_startup_greeting(self, **_kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("speaker unavailable")
        return {"text": self.text, "display_text": None}

    async def generate_proactive_reply(self, **_kwargs):
        self.proactive_reply_calls += 1
        raise AssertionError("startup greeting must not use proactive delivery")


class _Coordinator:
    def __init__(self, tts: _FakeTTS) -> None:
        self.tts = tts
        self.enabled_after_playback = False
        self.enable_calls = 0

    async def enable(self):
        self.enable_calls += 1
        self.enabled_after_playback = self.tts.finished.is_set()
        return True, "mic on"


async def _enable(tts: _FakeTTS, orch: _FakeOrchestrator) -> tuple:
    coordinator = _Coordinator(tts)
    output = _Output()
    result = await _enable_after_startup_greeting(
        coordinator,
        rt=SimpleNamespace(orchestrator=orch),
        conversation_id="c1",
        person_id="p1",
        tts_manager=tts,
        output=output,
    )
    return coordinator, output, result


async def test_asr_enable_waits_for_the_greeting_playback() -> None:
    tts = _FakeTTS()
    orch = _FakeOrchestrator("你好呀。")

    coordinator, output, (ok, _message) = await _enable(tts, orch)

    assert ok is True
    assert orch.calls == 1
    assert len(tts.jobs) == 1
    assert tts.jobs[0].response_id.startswith("startup-greeting-")
    assert output.lines == ["assistant> 你好呀。\n"]
    assert coordinator.enable_calls == 1
    assert coordinator.enabled_after_playback is True
    assert orch.proactive_reply_calls == 0


async def test_disabled_tts_displays_without_waiting() -> None:
    tts = _FakeTTS(enabled=False)
    orch = _FakeOrchestrator("早上好。")

    coordinator, output, _result = await _enable(tts, orch)

    assert output.lines == ["assistant> 早上好。\n"]
    assert tts.jobs == []
    assert tts.finished.is_set() is False
    assert coordinator.enable_calls == 1


async def test_speaker_failure_skips_greeting_and_still_starts() -> None:
    tts = _FakeTTS()
    orch = _FakeOrchestrator(fail=True)

    coordinator, output, _result = await _enable(tts, orch)

    assert orch.calls == 1  # one attempt, no retry
    assert output.lines == []
    assert tts.jobs == []
    assert coordinator.enable_calls == 1


async def test_playback_timeout_cancels_and_returns() -> None:
    tts = _FakeTTS(emit=False)
    started = time.monotonic()

    outcome = await _play_startup_greeting(
        tts, text="你好。", conversation_id="c1", timeout=0.05
    )

    assert outcome == "timeout"
    assert tts.stop_calls == 1
    assert tts.listeners == []
    assert time.monotonic() - started < 2.0


async def test_greeting_is_generated_at_most_once_per_startup() -> None:
    tts = _FakeTTS()
    orch = _FakeOrchestrator("只有一次。")

    await _enable(tts, orch)

    assert orch.calls == 1


async def test_greeting_switch_off_restores_old_startup() -> None:
    assert _startup_greeting_enabled({}) is True
    assert _startup_greeting_enabled({STARTUP_GREETING_ENV: "true"}) is True
    for raw in ("0", "false", "FALSE", "no", "off"):
        assert _startup_greeting_enabled({STARTUP_GREETING_ENV: raw}) is False


async def test_run_greeting_contains_all_failures() -> None:
    class _BrokenOutput:
        async def write(self, _text: str) -> None:
            raise RuntimeError("surface gone")

    orch = _FakeOrchestrator("你好。")
    await _run_startup_greeting(
        SimpleNamespace(orchestrator=orch),
        conversation_id="c1",
        person_id="p1",
        tts_manager=_FakeTTS(enabled=False),
        output=_BrokenOutput(),
    )
    assert orch.calls == 1
