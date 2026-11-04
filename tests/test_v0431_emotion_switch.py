"""V0.4.3.1: the conversation-transition master switch is authoritative.

``conversation_transition_enabled`` must gate frame changes, same-frame
developments, old-affect suppression and the repair tension adjustment. An
independently validated appraisal rule remains a separate mechanism and still
fires while conversation transitions are off. Time-decay mode must not decide
whether a conversation event can move affect.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.emotion.models import AppraisalRule, EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG, EmotionDynamicsConfig
from companion.emotion.transition import (
    EVENT,
    FRAME_CHANGE,
    NONE,
    SAME_FRAME_EVENT,
    build_conversational_affect,
    transition_affect,
)
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _config(**updates) -> EmotionDynamicsConfig:
    return DEFAULT_CONFIG.model_copy(update=updates)


def _pragmatic(**updates):
    payload = {
        "primary_act": "COMPLAINT",
        "secondary_acts": [],
        "interaction_frame": "UNKNOWN",
        "refers_to_previous_turn": False,
        "antecedent_event_id": None,
        "setup_payoff_role": "NONE",
        "literal_interpretation": "",
        "implied_interpretation": "",
        "topic_transition": "CONTINUE",
        "confidence": 0.9,
        "evidence_span": "你",
    }
    payload.update(updates)
    return type("P", (), payload)()


def _state(**updates) -> EmotionState:
    payload = {
        "emotion": "SERIOUS",
        "intensity": 0.3,
        "activations": {"SERIOUS": 0.3},
        "relationship_tension": 0.25,
        "updated_at": NOW,
    }
    payload.update(updates)
    return EmotionState(**payload)


def _rule(emotion="JOY", intensity=0.5) -> AppraisalRule:
    return AppraisalRule(
        rule_id="rule_direct_praise",
        agent_id="agent_1",
        name="direct_praise_joy",
        trigger_type="semantic_event",
        pattern={"event_type": "direct_praise"},
        emotion=emotion,
        intensity=intensity,
        priority=60,
        status="ACTIVE",
        source="system",
        created_at=NOW,
        updated_at=NOW,
    )


def _frame_affect(current="CONFLICT", previous="CASUAL", **updates):
    return build_conversational_affect(
        pragmatic=_pragmatic(
            interaction_frame=current, confidence=0.9, **updates
        ),
        previous_frame=previous,
    )


def test_master_switch_off_blocks_a_frame_change_entirely():
    before = _state()
    result = transition_affect(
        before,
        _frame_affect("CONFLICT", "CASUAL"),
        config=_config(conversation_transition_enabled=False),
        now=NOW,
    )

    assert result.changed is False
    assert result.cause.code == NONE
    assert result.state.activations == before.activations
    assert result.state.emotion == "SERIOUS"
    assert result.state.relationship_tension == before.relationship_tension


def test_master_switch_off_blocks_same_frame_progress():
    before = _state(emotion="JOY", intensity=0.2, activations={"JOY": 0.2})
    affect = build_conversational_affect(
        pragmatic=_pragmatic(
            interaction_frame="PLAYFUL",
            previous_frame="PLAYFUL",
            setup_payoff_role="PAYOFF",
            refers_to_previous_turn=True,
            confidence=0.9,
        ),
        previous_frame="PLAYFUL",
    )
    result = transition_affect(
        before,
        affect,
        config=_config(conversation_transition_enabled=False),
        now=NOW,
    )

    assert result.changed is False
    assert result.state.activations == {"JOY": 0.2}
    assert result.state.relationship_tension == before.relationship_tension


def test_master_switch_off_still_applies_a_validated_appraisal_rule():
    before = _state(
        emotion="FRUSTRATION",
        intensity=0.5,
        activations={"FRUSTRATION": 0.5},
    )
    result = transition_affect(
        before,
        _frame_affect("CONFLICT", "CASUAL"),
        rule=_rule("JOY", 0.6),
        config=_config(conversation_transition_enabled=False),
        now=NOW,
    )

    assert result.changed is True
    assert result.cause.code == EVENT
    assert result.cause.rule_name == "direct_praise_joy"
    assert result.state.activations["JOY"] > 0.0
    # The frame is not applied, so no frustration boost or tension change
    # comes from the conversation layer.
    assert result.state.relationship_tension == before.relationship_tension


def test_conversation_transition_survives_time_decay_off():
    result = transition_affect(
        _state(),
        _frame_affect("CONFLICT", "CASUAL"),
        config=_config(time_decay_mode="OFF"),
        now=NOW,
    )

    assert result.changed is True
    assert result.cause.code == FRAME_CHANGE
    assert result.state.activations["FRUSTRATION"] > 0.0
    assert result.state.relationship_tension > 0.25


def test_conversation_transition_survives_auxiliary_time_decay():
    result = transition_affect(
        _state(),
        _frame_affect("PLAYFUL", "CASUAL"),
        config=_config(time_decay_mode="AUXILIARY"),
        now=NOW,
    )

    assert result.changed is True
    assert result.cause.code == FRAME_CHANGE
    assert result.state.activations["JOY"] > 0.0


def test_same_frame_repetition_does_not_accumulate_without_evidence():
    affect = build_conversational_affect(
        pragmatic=_pragmatic(
            interaction_frame="PLAYFUL",
            setup_payoff_role="CONTINUATION",
            refers_to_previous_turn=True,
            confidence=0.9,
        ),
        previous_frame="PLAYFUL",
    )
    before = _state(emotion="JOY", intensity=0.2, activations={"JOY": 0.2})
    result = transition_affect(before, affect, config=DEFAULT_CONFIG, now=NOW)

    assert result.changed is False
    assert result.state.activations == {"JOY": 0.2}


class _FakeClient:
    def __init__(self, payload):
        self.payload = json.dumps(payload, ensure_ascii=False)
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        return ChatResponse(
            content=self.payload,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


def _event(person_id, text):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def _person(rt):
    return await rt.registry.register(
        "Ada",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


async def test_orchestrator_off_blocks_the_frame_but_keeps_the_rule(db):
    client = _FakeClient(
        {
            "meaning": "对方在表达强烈不满，同时又说了一句夸奖",
            "action": "RESPOND",
            "reason_code": "direct_agent_address",
            "referenced_memory_ids": [],
            "memory_candidates": [],
            "affective_signal": {
                "event_type": "direct_praise",
                "subject": "speaker",
                "target": "AGENT",
                "social_intent": "SINCERE",
                "factuality": "asserted",
                "confidence": 0.95,
                "evidence_span": "谢谢你",
            },
            "pragmatic": {
                "primary_act": "COMPLAINT",
                "secondary_acts": [],
                "interaction_frame": "CONFLICT",
                "refers_to_previous_turn": False,
                "antecedent_event_id": None,
                "setup_payoff_role": "NONE",
                "literal_interpretation": "在生气",
                "implied_interpretation": "在生气",
                "topic_transition": "SHIFT",
                "confidence": 0.9,
                "evidence_span": "你到底",
            },
        }
    )
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    off = _config(conversation_transition_enabled=False)
    rt.emotion_service.config = off
    rt.emotion_service.appraisal.config = off
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "你到底想怎样，不过还是谢谢你")
    )
    assert report.status == "COMPLETED"

    agent_id = await _agent_id(rt)
    state = await rt.emotion_service.get_state(agent_id, person.person_id)
    # The validated praise rule fired...
    assert state.activations.get("JOY", 0.0) > 0.0
    # ...but the CONFLICT frame did not add frustration or tension.
    assert state.activations.get("FRUSTRATION", 0.0) == 0.0
    assert state.relationship_tension == 0.0

    events = await rt.emotion_service.repo.list_events(agent_id)
    frame_events = [e for e in events if e["appraisal_rule_id"] is None]
    assert frame_events == []
    assert any(e["appraisal_rule_id"] for e in events)
