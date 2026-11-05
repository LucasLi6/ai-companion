"""Model-level contract tests for the Adaptive Router."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from companion.domain.enums import CognitiveAction, ReasonCode, ReplyNecessity
from companion.routing.adapter import route_reason_code, route_to_cognitive_result
from companion.routing.models import (
    MAX_TOOL_REQUESTS,
    ROUTER_SCHEMA_VERSION,
    EvidenceReference,
    ProcessingIntent,
    RawRouteDecision,
    RouteAction,
    RouteAddressee,
    ToolRequest,
    ToolStatus,
    TurnState,
    ValidatedRouteDecision,
)


def raw(**overrides) -> RawRouteDecision:
    data = {
        "schema_version": ROUTER_SCHEMA_VERSION,
        "action": "WAIT",
        "reply_necessity": "OPTIONAL",
        "addressee_assessment": "UNKNOWN",
    }
    data.update(overrides)
    return RawRouteDecision.model_validate(data)


def test_enum_surface_is_stable():
    assert [item.value for item in RouteAction] == ["RESPOND", "WAIT", "IGNORE"]
    assert [item.value for item in ToolStatus] == [
        "SUCCESS",
        "EMPTY",
        "DENIED",
        "DISABLED",
        "TIMEOUT",
        "ERROR",
    ]
    assert "MEMORY_CORRECTION" in {item.value for item in ProcessingIntent}
    assert "FOLLOW_UP" in {item.value for item in TurnState}
    assert {item.value for item in RouteAddressee} == {"AGENT", "HUMAN", "UNKNOWN"}


def test_unknown_fields_and_schema_versions_are_rejected():
    with pytest.raises(ValidationError):
        RawRouteDecision.model_validate(
            {
                "action": "RESPOND",
                "reply_necessity": "REQUIRED",
                "addressee_assessment": "AGENT",
                "keyword_rule": "ignore this",
            }
        )
    with pytest.raises(ValidationError):
        raw(schema_version=999)


def test_tool_requests_are_bounded_and_json_only():
    with pytest.raises(ValidationError):
        raw(
            tool_requests=[
                {"tool_name": "memory.search", "arguments": {"query": str(i)}}
                for i in range(MAX_TOOL_REQUESTS + 1)
            ]
        )
    with pytest.raises(ValidationError):
        ToolRequest(tool_name="memory.search", arguments={"query": math.nan})


def test_intents_are_deduplicated_and_reason_is_compact():
    decision = raw(
        processing_intents=["MEMORY_SAVE", "NONE", "MEMORY_SAVE", "PRIVACY_REQUEST"],
        reason="  a\n b\t c " + ("x" * 900),
    )
    assert decision.processing_intents == [
        ProcessingIntent.MEMORY_SAVE,
        ProcessingIntent.PRIVACY_REQUEST,
    ]
    assert "\n" not in decision.reason and len(decision.reason) <= 500


def test_validated_decision_core_fingerprint_detects_changes():
    base = dict(
        schema_version=ROUTER_SCHEMA_VERSION,
        action=RouteAction.RESPOND,
        reply_necessity=ReplyNecessity.REQUIRED,
        addressee_assessment=RouteAddressee.AGENT,
    )
    first = ValidatedRouteDecision(**base)
    second = ValidatedRouteDecision(**{**base, "action": RouteAction.WAIT})
    assert first.core_fingerprint() == first.core_fingerprint()
    assert first.core_fingerprint() != second.core_fingerprint()


def test_evidence_reference_is_bounded():
    evidence = EvidenceReference(quote="x" * 600)
    assert len(evidence.quote) == 512


def _validated(**overrides) -> ValidatedRouteDecision:
    data = {
        "action": RouteAction.RESPOND,
        "reply_necessity": ReplyNecessity.USEFUL,
        "addressee_assessment": RouteAddressee.AGENT,
        "turn_state": TurnState.CONTINUING,
    }
    data.update(overrides)
    return ValidatedRouteDecision(**data)


def test_adapter_maps_only_validated_semantic_enums():
    assert route_reason_code(
        _validated(addressee_assessment=RouteAddressee.AGENT)
    ) == ReasonCode.DIRECT_AGENT_ADDRESS
    assert route_reason_code(
        _validated(
            addressee_assessment=RouteAddressee.AGENT,
            turn_state=TurnState.FOLLOW_UP,
        )
    ) == ReasonCode.FOLLOW_UP_TO_AGENT
    assert route_reason_code(
        _validated(action=RouteAction.WAIT, addressee_assessment=RouteAddressee.HUMAN)
    ) == ReasonCode.CONVERSATION_BETWEEN_HUMANS
    assert route_reason_code(
        _validated(action=RouteAction.IGNORE, silence_requested=True)
    ) == ReasonCode.EXPLICIT_SILENCE_REQUEST


def test_adapter_builds_trusted_cognitive_result_and_memory_refs():
    event = SimpleNamespace(event_id="evt_1")
    memory_result = SimpleNamespace(
        tool_name="memory.search",
        status=ToolStatus.SUCCESS,
        source_refs=["mem_1", "mem_2"],
    )
    web_result = SimpleNamespace(
        tool_name="web.search",
        status=ToolStatus.SUCCESS,
        source_refs=["https://example.invalid/x"],
    )
    route = _validated(reason="semantic reason")
    result = route_to_cognitive_result(
        event, route, tool_results=[memory_result, web_result]
    )
    assert result.event_id == "evt_1"
    assert result.action == CognitiveAction.RESPOND
    assert result.provider == "semantic_router"
    assert result.meaning == "semantic reason"
    assert result.memory_candidates == []
    assert result.referenced_memory_ids == ["mem_1", "mem_2"]
    assert result.reply_necessity == ReplyNecessity.USEFUL
