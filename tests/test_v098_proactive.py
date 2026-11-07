"""V0.9.8 key checks, updated for the V0.9.9 dual-stage lifecycle.

These tests are deliberately small and offline. They pin the behaviours the
feature must not get wrong: the immediate stage runs without waiting for
background jobs, new user input invalidates stale work, NO is silent and YES
speaks exactly once per stage, and the proactive message is saved/read as real
assistant history without scheduling another check.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pytest

from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    JobType,
    ReasonCode,
    ReplyNecessity,
)
from companion.domain.models import (
    CognitiveResult,
    make_utterance_event,
    utcnow,
)
from companion.proactive.judge import (
    ProactiveJudge,
    ProactiveJudgeDecision,
)
from companion.prompts.proactive import PROACTIVE_JUDGE_SYSTEM_PROMPT
from companion.prompts.response import build_proactive_response_messages
from companion.providers.deepseek_client import ChatResponse
from companion.providers.deepseek_proactive import ScriptedProactiveJudgeProvider
from companion.proactive.manager import ProactiveManager
from companion.runtime import build_runtime


class FakeDeepSeekClient:
    """Offline client: judge JSON by default, normal reply for response calls."""

    def __init__(self, response_text="正常回复。", judge=None):
        self.response_text = response_text
        self.judge = judge or {"should_initiate": False, "reason": "自然结束"}
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
            else json.dumps(self.judge, ensure_ascii=False)
        )
        return ChatResponse(
            content=content,
            model="fake-model",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _runtime(db, client):
    return build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
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


def _event(person_id, text="我在研究机器人控制", conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.CLI,
    )


async def _delivered_normal_turn(rt, person, text="我在研究机器人控制"):
    event = _event(person.person_id, text)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    return event


def _scripted_manager(rt, decisions):
    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    manager.judge = ProactiveJudge(ScriptedProactiveJudgeProvider(decisions))
    return manager


async def test_judge_prompt_is_a_silent_yes_no_judgment():
    assert '"should_initiate"' in PROACTIVE_JUDGE_SYSTEM_PROMPT
    assert "不要先生成候选回复再判断" in PROACTIVE_JUDGE_SYSTEM_PROMPT
    assert "沉默是正常交流的一部分" in PROACTIVE_JUDGE_SYSTEM_PROMPT


async def test_judge_no_is_silent_and_yes_speaks_exactly_once(db):
    client = FakeDeepSeekClient()
    rt = await _runtime(db, client)
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")

    first = await _delivered_normal_turn(rt, person)
    manager = _scripted_manager(rt, [False])
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    await asyncio.sleep(0.15)
    assert manager.poll_event() is None
    assert await rt.repos.proactive.get_for_source_event(first.event_id) is None

    second = await _delivered_normal_turn(rt, person, "那控制频率呢？")
    # IMMEDIATE says yes, LONG_IDLE says no: exactly one message.
    manager.judge = ProactiveJudge(ScriptedProactiveJudgeProvider([True, False]))
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=second.event_id,
    )
    delivered = await asyncio.wait_for(manager.next_event(), 2)
    assert delivered.type == "response_completed"
    assert delivered.status == "COMPLETED"
    assert delivered.text == "正常回复。"

    # A duplicate delivery event cannot create a second proactive message.
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=second.event_id,
    )
    await asyncio.sleep(0.1)
    assert manager.poll_event() is None
    rows = await rt.repos.proactive.list_for_conversation("c1")
    assert len(rows) == 1
    assert rows[0].origin == "AVA_INITIATED"
    await manager.shutdown()


async def test_proactive_message_becomes_real_history_and_schedules_no_next_check(db):
    client = FakeDeepSeekClient(response_text="正常回复。")
    rt = await _runtime(db, client)
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    first = await _delivered_normal_turn(rt, person)
    manager = _scripted_manager(rt, [True, False])
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    delivered = await asyncio.wait_for(manager.next_event(), 2)
    proactive_text = delivered.text

    later = _event(person.person_id, "继续说", conversation="c1")
    await rt.repos.utterance.save(later)
    context = await rt.context_builder.build(later, [])
    assert proactive_text in [turn.text for turn in context.turns]
    assert any(
        turn.role == "assistant" and turn.text == proactive_text
        for turn in context.turns
    )

    # The manager has no self-scheduling: nothing more arrives without a new
    # delivered user-triggered reply.
    await asyncio.sleep(0.15)
    assert manager.poll_event() is None
    assert len(await rt.repos.proactive.list_for_conversation("c1")) == 1
    await manager.shutdown()


async def test_new_user_input_invalidates_a_pending_check(db):
    client = FakeDeepSeekClient()
    rt = await _runtime(db, client)
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    event = await _delivered_normal_turn(rt, person)

    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.3
    judge = ScriptedProactiveJudgeProvider([False, True])
    manager.judge = ProactiveJudge(judge)
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=event.event_id,
    )
    # Let the immediate stage run (it declines) so the idle wait is the only
    # remaining work when the user speaks again.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while not judge.payloads and loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert [payload.mode for payload in judge.payloads] == ["IMMEDIATE"]

    new_event = _event(person.person_id, "等等，我换个问题", conversation="c1")
    await rt.repos.utterance.save(new_event)
    manager.note_user_input("c1", new_event.event_id)

    await asyncio.sleep(0.5)
    # The long-idle judge never runs after the user came back.
    assert [payload.mode for payload in judge.payloads] == ["IMMEDIATE"]
    assert manager.poll_event() is None
    assert await rt.repos.proactive.get_for_source_event(event.event_id) is None
    await manager.shutdown()


async def test_new_user_input_during_generation_discards_the_reply(db, monkeypatch):
    client = FakeDeepSeekClient()
    rt = await _runtime(db, client)
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    event = await _delivered_normal_turn(rt, person)
    manager = _scripted_manager(rt, [True])

    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_generation(*, source_event, person_id, mode="IMMEDIATE"):
        started.set()
        await release.wait()
        return {
            "text": "这是一条已经过时的主动消息。",
            "finish_reason": "stop",
            "model": "fake",
        }

    monkeypatch.setattr(
        rt.orchestrator, "generate_proactive_reply", blocked_generation
    )
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=event.event_id,
    )
    await asyncio.wait_for(started.wait(), 2)

    newer = _event(person.person_id, "我插一句", conversation="c1")
    await rt.repos.utterance.save(newer)
    manager.note_user_input("c1", newer.event_id)
    release.set()

    await asyncio.sleep(0.2)
    assert manager.poll_event() is None
    assert await rt.repos.proactive.get_for_source_event(event.event_id) is None
    await manager.shutdown()


async def test_immediate_stage_does_not_observe_background_jobs(db):
    client = FakeDeepSeekClient()
    rt = await _runtime(db, client)
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    event = await _delivered_normal_turn(rt, person)
    manager = rt.proactive_manager
    # A long idle wait makes any observation of background jobs visible as
    # extra latency before the immediate judge runs.
    manager.idle_delay_seconds = 2.0
    judge = ScriptedProactiveJudgeProvider([False])
    manager.judge = ProactiveJudge(judge)

    await rt.repos.job.enqueue(
        JobType.TURN_ANALYSIS,
        f"turn_analysis:{event.event_id}",
        {"event_id": event.event_id},
        next_run_at=utcnow() + timedelta(seconds=60),
    )
    await rt.repos.job.enqueue(
        JobType.MEMORY_CONSOLIDATION,
        f"memory:{event.event_id}",
        {"event_id": event.event_id},
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=event.event_id,
    )
    deadline = started + 2.0
    while not judge.payloads and loop.time() < deadline:
        await asyncio.sleep(0.01)
    elapsed = loop.time() - started

    assert judge.payloads, "the immediate judge must run"
    assert judge.payloads[0].mode == "IMMEDIATE"
    assert elapsed < 1.0, f"immediate check took {elapsed:.3f}s"
    await manager.shutdown()


async def test_proactive_generation_uses_no_synthetic_user_message(db):
    client = FakeDeepSeekClient(response_text="正常回复。")
    rt = await _runtime(db, client)
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    event = await _delivered_normal_turn(rt, person)

    context = await rt.orchestrator.build_proactive_context(
        source_event=event, person_id=person.person_id
    )
    analysis = CognitiveResult(
        event_id=event.event_id,
        meaning="proactive",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        reply_necessity=ReplyNecessity.USEFUL,
    )
    messages = build_proactive_response_messages(context, analysis)

    assert messages[0]["role"] == "system"
    assert "本轮生成模式：主动发言" in messages[0]["content"]
    assert messages[-1]["role"] == "assistant"
    contents = "\n".join(str(message.get("content", "")) for message in messages)
    assert "请你现在主动说一句话" not in contents
    user_texts = [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]
    # The real triggering turn appears once; no invented current user turn.
    assert sum("我在研究机器人控制" in text for text in user_texts) == 1
