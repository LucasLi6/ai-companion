"""Phase 3 — conversation context builder tests.

History is inserted directly into the repositories with explicit
``occurred_at`` timestamps so ordering is deterministic (no reliance on
wall-clock microsecond timing).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from companion.cognition.context_builder import ContextBuilder
from companion.cognition.models import CognitiveTurn
from companion.domain.enums import (
    AddresseeHint,
    CognitiveAction,
    EventSource,
    ProcessingStatus,
    ReasonCode,
)
from companion.domain.models import AgentAction, make_utterance_event, new_id, utcnow

CONV = "default"


def _at(minutes: int) -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


async def _save_turn(rt, person_id, minutes, text, to, response=None):
    await rt.conversations.get_or_create(CONV)
    event_id = new_id("evt")
    event = make_utterance_event(
        conversation_id=CONV,
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
        occurred_at=_at(minutes),
    )
    event = event.model_copy(update={"processing_status": ProcessingStatus.COMPLETED})
    await rt.repos.utterance.save(event)
    if response is not None:
        await rt.repos.analysis.save_action(
            AgentAction(
                action_id=new_id("act"),
                event_id=event_id,
                action_type=CognitiveAction.RESPOND,
                response_text=response,
                reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
                status="COMPLETED",
                created_at=utcnow(),
            )
        )
    return event_id


@pytest.fixture
def builder(rt):
    return ContextBuilder(repos=rt.repos, registry=rt.registry)


async def _current_event(person_id, minutes=100):
    return make_utterance_event(
        conversation_id=CONV,
        text="现在这句话",
        addressee_hint=AddresseeHint.AGENT,
        speaker_person_id=person_id,
        source=EventSource.TEST,
        occurred_at=_at(minutes),
    )


# -- basics -----------------------------------------------------------------


async def test_build_empty_history(builder, register, rt):
    person = await register("Alice")
    event = await _current_event(person.person_id)
    ctx = await builder.build(event, memories=[{"memory_id": "m1"}])
    assert ctx.turns == []
    assert ctx.current_event_id == event.event_id
    assert ctx.current_text == "现在这句话"
    assert ctx.current_addressee_hint == "agent"
    assert ctx.current_speaker_label == "Alice"
    assert ctx.memories == [{"memory_id": "m1"}]


async def test_build_excludes_current_event(builder, register, rt):
    person = await register("Alice")
    # The current event is persisted (as in the orchestrator flow) before the
    # builder runs; it must be excluded from the history it builds.
    event = await _current_event(person.person_id, minutes=100)
    await rt.conversations.get_or_create(CONV)
    await rt.repos.utterance.save(
        event.model_copy(update={"processing_status": ProcessingStatus.COMPLETED})
    )
    await _save_turn(rt, person.person_id, 10, "早前的话", "agent", response="回复")
    ctx = await builder.build(event, memories=[])
    texts = [t.text for t in ctx.turns]
    assert "现在这句话" not in texts
    assert texts == ["早前的话", "回复"]


async def test_interleaves_human_and_assistant(builder, register, rt):
    person = await register("Alice")
    await _save_turn(rt, person.person_id, 1, "你好", "agent", response="你好！")
    await _save_turn(rt, person.person_id, 2, "自言自语", "human")  # IGNORE -> no assistant
    await _save_turn(rt, person.person_id, 3, "再来一句", "agent", response="收到")
    event = await _current_event(person.person_id, minutes=100)
    ctx = await builder.build(event, memories=[])

    assert [t.role for t in ctx.turns] == [
        "human",
        "assistant",
        "human",
        "human",
        "assistant",
    ]
    assert [t.text for t in ctx.turns] == ["你好", "你好！", "自言自语", "再来一句", "收到"]
    # assistant turns carry the human event id and no person id
    assistants = [t for t in ctx.turns if t.role == "assistant"]
    assert all(t.speaker_person_id is None for t in assistants)
    assert all(t.speaker_label == "assistant" for t in assistants)


async def test_labels_multiple_speakers(builder, register, rt):
    alice = await register("Alice")
    bob = await register("Bob")
    await _save_turn(rt, alice.person_id, 1, "我是Alice", "agent", response="嗨Alice")
    await _save_turn(rt, bob.person_id, 2, "我是Bob", "agent", response="嗨Bob")
    event = await _current_event(alice.person_id, minutes=100)
    ctx = await builder.build(event, memories=[])

    human_turns = [t for t in ctx.turns if t.role == "human"]
    assert [t.speaker_label for t in human_turns] == ["Alice", "Bob"]
    assert [t.speaker_person_id for t in human_turns] == [
        alice.person_id,
        bob.person_id,
    ]


async def test_addressee_hint_preserved(builder, register, rt):
    person = await register("Alice")
    await _save_turn(rt, person.person_id, 1, "对他说", "human")
    event = await _current_event(person.person_id, minutes=100)
    ctx = await builder.build(event, memories=[])
    assert ctx.turns[0].addressee_hint == "human"


# -- truncation -------------------------------------------------------------


async def test_truncates_turns_keeps_most_recent(builder, register, rt):
    person = await register("Alice")
    for i in range(1, 6):
        await _save_turn(rt, person.person_id, i, f"消息{i}", "agent", response=f"回{i}")
    b = ContextBuilder(
        repos=rt.repos, registry=rt.registry, max_turns=4, max_chars=None
    )
    event = await _current_event(person.person_id, minutes=100)
    ctx = await b.build(event, memories=[])
    texts = [t.text for t in ctx.turns]
    # 4 most-recent turns: 消息4, 回4, 消息5, 回5
    assert texts == ["消息4", "回4", "消息5", "回5"]


async def test_truncates_chars_drops_oldest(builder, register, rt):
    person = await register("Alice")
    await _save_turn(rt, person.person_id, 1, "AAAAAAAAAA", "agent")
    await _save_turn(rt, person.person_id, 2, "BBBBBBBBBB", "agent")
    b = ContextBuilder(
        repos=rt.repos, registry=rt.registry, max_turns=None, max_chars=15
    )
    event = await _current_event(person.person_id, minutes=100)
    ctx = await b.build(event, memories=[])
    # "AAAAAAAAAA" (10) + "BBBBBBBBBB" (10) = 20 > 15 -> drop oldest "AAAAAAAAAA"
    assert [t.text for t in ctx.turns] == ["BBBBBBBBBB"]


# -- batch action lookup ----------------------------------------------------


async def test_get_actions_for_events_empty(rt):
    assert await rt.repos.analysis.get_actions_for_events([]) == {}


async def test_get_actions_for_events_batches(rt, register):
    person = await register("Alice")
    e1 = await _save_turn(rt, person.person_id, 1, "a", "agent", response="resp1")
    e2 = await _save_turn(rt, person.person_id, 2, "b", "human")  # no response
    actions = await rt.repos.analysis.get_actions_for_events([e1, e2])
    assert set(actions.keys()) == {e1}
    assert actions[e1].response_text == "resp1"
