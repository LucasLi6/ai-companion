"""Validator tests: untrusted model payloads become locally valid decisions."""

from __future__ import annotations

import pytest

from companion.domain.enums import ReplyNecessity
from companion.routing.errors import RouterValidationError
from companion.routing.models import (
    RawRouteDecision,
    RouteAction,
    RouteAddressee,
    RoutingContext,
    TurnState,
)
from companion.routing.validator import RoutingValidator
from companion.tools.registry import build_default_tool_registry


def context(**overrides) -> RoutingContext:
    data = {
        "conversation_id": "conv1",
        "current_event_id": "evt1",
        "current_speaker_id": "p1",
        "current_speaker_label": "Lucas",
        "current_text": "please look this up",
        "trusted_addressee_hint": "agent",
        "visible_event_ids": ["evt0"],
        "authorized_history_texts": ["earlier line"],
    }
    data.update(overrides)
    return RoutingContext(**data)


def raw(**overrides) -> RawRouteDecision:
    data = {
        "action": "RESPOND",
        "reply_necessity": "USEFUL",
        "addressee_assessment": "AGENT",
        "input_semantics": "COMMUNICATIVE",
        "reason": "needs a visible reply",
    }
    data.update(overrides)
    return RawRouteDecision.model_validate(data)


def test_valid_decision_normalizes_arguments():
    registry = build_default_tool_registry(include_web=False)
    decision = RoutingValidator().validate(
        raw(
            tool_requests=[
                {
                    "tool_name": "memory.search",
                    "arguments": {"query": "project", "limit": 3},
                    "purpose": "check a fact",
                    "required": True,
                }
            ]
        ),
        context(),
        registry,
    )
    assert decision.tool_requests[0].arguments == {"query": "project", "limit": 3}
    assert decision.action == RouteAction.RESPOND
    assert decision.provider == "deepseek"


def test_unregistered_tool_and_invalid_arguments_are_rejected():
    registry = build_default_tool_registry(include_web=False)
    validator = RoutingValidator()
    with pytest.raises(RouterValidationError):
        validator.validate(
            raw(tool_requests=[{"tool_name": "sql.execute", "arguments": {}}]),
            context(),
            registry,
        )
    with pytest.raises(RouterValidationError):
        validator.validate(
            raw(
                tool_requests=[
                    {"tool_name": "memory.search", "arguments": {"query": ""}}
                ]
            ),
            context(),
            registry,
        )


def test_tool_budget_is_enforced():
    registry = build_default_tool_registry(include_web=False)
    requests = [
        {"tool_name": "memory.search", "arguments": {"query": str(i)}}
        for i in range(5)
    ]
    with pytest.raises(RouterValidationError):
        RoutingValidator(max_tool_calls=4).validate(
            raw(tool_requests=requests), context(), registry
        )


def test_evidence_must_come_from_authorized_visible_text():
    registry = build_default_tool_registry(include_web=False)
    decision = RoutingValidator().validate(
        raw(
            evidence=[
                {"event_id": "evt1", "quote": "look this up"},
                {"event_id": "evt0", "quote": "earlier line"},
                {"event_id": "evt0", "quote": "not in the transcript"},
                {"event_id": "evt_hidden", "quote": "secret"},
                {"event_id": None, "quote": "look"},
            ]
        ),
        context(),
        registry,
    )
    assert [(item.event_id, item.quote) for item in decision.evidence] == [
        ("evt1", "look this up"),
        ("evt0", "earlier line"),
        (None, "look"),
    ]


def test_invalid_optional_signals_become_none():
    registry = build_default_tool_registry(include_web=False)
    decision = RoutingValidator().validate(
        raw(
            affective_signal={
                "event_type": "not-a-real-type",
                "confidence": 0.9,
                "evidence_span": "look",
            },
            pragmatic={"primary_act": "NOT_A_REAL_ACT"},
            scene_proposal={"topic_change": "MAYBE"},
        ),
        context(),
        registry,
    )
    assert decision.affective_signal is None
    assert decision.pragmatic is None
    assert decision.scene_proposal is None


def test_silence_flag_only_applies_to_ignore():
    registry = build_default_tool_registry(include_web=False)
    validator = RoutingValidator()
    kept = validator.validate(
        raw(action="IGNORE", silence_requested=True), context(), registry
    )
    assert kept.silence_requested is True
    dropped = validator.validate(
        raw(action="WAIT", silence_requested=True), context(), registry
    )
    assert dropped.silence_requested is False


def test_followup_plan_may_not_change_the_reply_decision():
    registry = build_default_tool_registry(include_web=False)
    validator = RoutingValidator()
    previous = validator.validate(raw(), context(), registry)
    same = validator.validate_followup(
        raw(
            tool_requests=[
                {"tool_name": "history.recent", "arguments": {"limit": 3}}
            ]
        ),
        context(),
        registry,
        previous,
    )
    assert same.action == previous.action

    with pytest.raises(RouterValidationError):
        validator.validate_followup(
            raw(
                action="WAIT",
                reply_necessity="UNNECESSARY",
                addressee_assessment="UNKNOWN",
            ),
            context(),
            registry,
            previous,
        )
