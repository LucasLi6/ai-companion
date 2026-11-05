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
