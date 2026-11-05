"""Routing context is minimal, authorized and never retrieves long-term memory."""

from __future__ import annotations

import pytest

from companion.cognition.models import CognitiveContext
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.routing.context import RoutingContextBuilder, build_speaker_context
from companion.routing.models import (
    RouteAction,
    RouteAddressee,
    ValidatedRouteDecision,
)
from companion.domain.enums import ReplyNecessity
from companion.tools.registry import build_default_tool_registry


class _ExplodingMemory:
    async def retrieve(self, *args, **kwargs):
        raise AssertionError("routing context must not call memory.retrieve")


@pytest.mark.asyncio
async def test_routing_context_contains_no_memories_and_no_persona(rt, register):
    person = await register("Lucas", external_ai=True)
    await rt.conversations.get_or_create("conv")
    previous = make_utterance_event(
        conversation_id="conv",
        text="earlier message",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(previous)
    current = make_utterance_event(
        conversation_id="conv",
        text="current message",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )

    builder = RoutingContextBuilder(
        repos=rt.repos,
        registry=rt.registry,
        tool_registry=build_default_tool_registry(include_web=False),
    )
    conversation = await rt.repos.conversation.get("conv")
    view = await builder.build(current, conversation)

    payload = view.as_prompt_dict()
    assert payload["current_event"]["text"] == "current message"
    assert payload["current_event"]["trusted_addressee_hint"] == "agent"
    assert "memories" not in payload
    assert "personality" not in payload
    assert "character_snapshot" not in payload
    assert previous.event_id in view.visible_event_ids
    assert view.authorized_history_texts == ["earlier message"]
    assert any(
        spec["name"] == "memory.search" for spec in view.available_tool_specs
    )
    assert not hasattr(builder, "memory")


@pytest.mark.asyncio
async def test_speaker_context_reuses_base_and_labels_route_data(rt, register):
    person = await register("Lucas", external_ai=True)
    current = make_utterance_event(
        conversation_id="conv",
        text="current message",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    builder = RoutingContextBuilder(repos=rt.repos, registry=rt.registry)
    view = await builder.build(current, await rt.conversations.get_or_create("conv"))
    route = ValidatedRouteDecision(
        action=RouteAction.RESPOND,
        reply_necessity=ReplyNecessity.USEFUL,
        addressee_assessment=RouteAddressee.AGENT,
        reason="semantic reason",
        tool_requests=[],
    )
    base = CognitiveContext(
        conversation_id="conv",
        current_event_id=current.event_id,
        current_speaker_id=person.person_id,
        current_speaker_label="Lucas",
        current_text=current.text,
        current_addressee_hint="agent",
        memories=[{"memory_id": "mem1", "content": "x"}],
    )
    speaker = build_speaker_context(
        current,
        view,
        route,
        [{"tool_name": "runtime.current_time", "status": "SUCCESS", "data": {}}],
        base=base,
    )
    assert speaker is not base
    assert speaker.memories == base.memories
    assert speaker.current_text == current.text
    assert speaker.route_context["action"] == "RESPOND"
    assert speaker.route_context["reason"] == "semantic reason"
    assert speaker.tool_results[0]["tool_name"] == "runtime.current_time"


# -- dialogue-unit truncation ------------------------------------------------


def _turn(event_id, role, text, speaker=None):
    return {
        "event_id": event_id,
        "role": role,
        "text": text,
        "speaker_person_id": speaker,
        "speaker_label": speaker or "assistant",
    }


def test_truncation_drops_whole_units_and_never_an_orphan_assistant():
    from companion.routing.context import truncate_dialogue_turns

    turns = [
        _turn("h1", "human", "A" * 100, "p1"),
        _turn("a1", "assistant", "R" * 100),
        _turn("h2", "human", "B" * 100, "p1"),
        _turn("a2", "assistant", "S" * 100),
        _turn("h3", "human", "C" * 100, "p2"),
    ]

    kept = truncate_dialogue_turns(turns, 250)

    assert [turn["event_id"] for turn in kept] == ["h3"]
    assert kept[0]["role"] == "human"
    assert all(
        turn["role"] == "human"
        for turn in kept
        if kept.index(turn) == 0
    )


def test_truncation_keeps_full_units_in_order_for_interleaved_speakers():
    from companion.routing.context import truncate_dialogue_turns

    turns = [
        _turn("h1", "human", "A" * 120, "p1"),
        _turn("a1", "assistant", "R" * 120),
        _turn("h2", "human", "B" * 120, "p2"),
        _turn("a2", "assistant", "S" * 120),
        _turn("h3", "human", "C" * 120, "p1"),
    ]

    kept = truncate_dialogue_turns(turns, 400)

    assert [turn["event_id"] for turn in kept] == ["h2", "a2", "h3"]
    assert [turn["role"] for turn in kept] == ["human", "assistant", "human"]
    assert kept[0]["speaker_person_id"] == "p2"
    assert kept[2]["speaker_person_id"] == "p1"
    # The assistant reply stays attached to the question it answered.
    assert kept[1]["role"] == "assistant"
    assert kept[1]["text"] == "S" * 120


def test_orphan_assistant_and_oversized_latest_unit_are_handled():
    from companion.routing.context import truncate_dialogue_turns

    orphan = [_turn("a0", "assistant", "hello")]
    assert truncate_dialogue_turns(orphan, 10) == []

    oversized = [
        _turn("h1", "human", "A" * 10, "p1"),
        _turn("a1", "assistant", "R" * 10),
        _turn("h2", "human", "B" * 1000, "p2"),
        _turn("a2", "assistant", "S" * 1000),
    ]
    kept = truncate_dialogue_turns(oversized, 50)
    # The most recent unit survives even though it alone exceeds the soft cap.
    assert [turn["event_id"] for turn in kept] == ["h2", "a2"]


@pytest.mark.asyncio
async def test_builder_only_exposes_the_final_visible_history(rt, register):
    alice = await register("Alice", external_ai=True)
    bob = await register("Bob", external_ai=True)
    await rt.conversations.get_or_create("conv")

    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    history = []
    for index, (person, text, reply) in enumerate(
        [
            (alice, "A" * 120, "R" * 120),
            (bob, "B" * 120, "S" * 120),
            (alice, "C" * 120, "T" * 120),
            (bob, "D" * 120, None),
        ]
    ):
        event = make_utterance_event(
            conversation_id="conv",
            text=text,
            addressee_hint="agent",
            speaker_person_id=person.person_id,
            source=EventSource.TEST,
            occurred_at=start + timedelta(seconds=index),
            event_id=f"evt_{index}",
        )
        await rt.repos.utterance.save(event)
        if reply is not None:
            await rt.db.execute(
                "INSERT INTO agent_actions(action_id, event_id, action_type,"
                " response_text, reason_code, status, created_at)"
                " VALUES (?, ?, 'RESPOND', ?, 'direct_agent_address',"
                " 'COMPLETED', ?)",
                (f"act_{index}", event.event_id, reply, start.isoformat()),
            )
        history.append(event)

    current = make_utterance_event(
        conversation_id="conv",
        text="current message",
        addressee_hint="agent",
        speaker_person_id=bob.person_id,
        source=EventSource.TEST,
        occurred_at=start + timedelta(seconds=10),
        event_id="evt_current",
    )
    await rt.repos.utterance.save(current)

    builder = RoutingContextBuilder(
        repos=rt.repos,
        registry=rt.registry,
        max_chars=400,
    )
    view = await builder.build(current, await rt.repos.conversation.get("conv"))

    # The last unit is a human turn (no action), the one before it is a full
    # human+assistant pair. The oldest pair was dropped whole.
    assert [turn["role"] for turn in view.recent_turns] == [
        "human",
        "assistant",
        "human",
    ]
    assert view.recent_turns[0]["text"] == "C" * 120
    assert view.recent_turns[1]["text"] == "T" * 120
    assert view.recent_turns[2]["text"] == "D" * 120
    assert view.recent_turns[2]["speaker_person_id"] == bob.person_id
    assert view.visible_event_ids == ["evt_2", "evt_2", "evt_3"]
    assert view.authorized_history_texts == [
        "C" * 120,
        "T" * 120,
        "D" * 120,
    ]
    assert "A" * 120 not in view.authorized_history_texts
    assert "R" * 120 not in view.authorized_history_texts
