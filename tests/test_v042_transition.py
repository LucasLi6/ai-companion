"""V0.4.2 Step 3: conversation-driven affect.

V0.4.1 decayed emotion correctly but let a stale, strong activation keep the
expressed label through an unrelated turn. These tests pin the V0.4.2 policy:
the validated conversational event is the primary cause of change; scene
relevance and frame changes suppress old affect; time decay is auxiliary; and a
clear new event can take over the expressed emotion. Nothing here can change an
action.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from companion.domain.enums import EventSource
from companion.domain.models import (
    AffectiveSignal,
    PragmaticAnalysis,
    make_utterance_event,
)
from companion.emotion.appraisal import (
    APPLIED,
    FRAME_TRANSITION,
    AppraisalEngine,
)
from companion.emotion.dynamics import decay
from companion.emotion.models import AppraisalRule, EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG, EmotionDynamicsConfig
from companion.emotion.transition import (
    EVENT,
    FRAME_CHANGE,
    REPAIR,
    build_conversational_affect,
    transition_affect,
)
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _config(**overrides) -> EmotionDynamicsConfig:
    return EmotionDynamicsConfig(**overrides)


def _rule(name="direct_praise_joy", emotion="JOY", intensity=0.4):
    return AppraisalRule(
        rule_id=f"rule_{name}",
        agent_id="agent_1",
        name=name,
        trigger_type="semantic_event",
        pattern={},
        emotion=emotion,
        intensity=intensity,
        priority=40,
        status="ACTIVE",
        source="system",
        created_at=NOW,
        updated_at=NOW,
    )


def _pragmatic(**overrides):
    payload = {
        "primary_act": "INFORMATION",
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
    payload.update(overrides)
    return PragmaticAnalysis(**payload)


def _praise_signal(text="你真厉害"):
    return AffectiveSignal(
        event_type="direct_praise",
        target="AGENT",
        social_intent="SINCERE",
        factuality="asserted",
        confidence=0.95,
        evidence_span="真厉害",
    )


def _stale_frustration(value=0.80):
    return EmotionState(
        valence=-0.48,
        arousal=0.56,
        intensity=value,
        emotion="FRUSTRATION",
        activations={"FRUSTRATION": value},
        updated_at=NOW,
        mood_valence=-0.2,
        mood_arousal=0.3,
        mood_intensity=0.2,
        mood_emotion="FRUSTRATION",
        mood_updated_at=NOW,
    )


def _appraise(text, signal, *, affect=None, state=None, config=DEFAULT_CONFIG):
    return AppraisalEngine(config=config).appraise(
        text,
        rules=[_rule()],
        signal=signal,
        current_state=state or EmotionState(updated_at=NOW),
        now=NOW,
        addressee_hint="agent",
        config=config,
        conversational_affect=affect,
    )


# -- event is the primary cause; relevance suppresses stale affect ----------


def test_topic_shift_lets_a_new_event_replace_stale_emotion():
    text = "你真厉害"
    affect = build_conversational_affect(
        pragmatic=_pragmatic(topic_transition="SHIFT"),
        signal=_praise_signal(text),
        previous_frame="UNKNOWN",
    )
    result = _appraise(
        text, _praise_signal(text), affect=affect, state=_stale_frustration()
    )

    assert result.diagnostic == APPLIED
    assert result.transition_cause == EVENT
    assert result.frame_from == "UNKNOWN"
    assert result.scene_relevance < 0.5
    # The old frustration is no longer the strongest activation.
    assert result.emotion == "JOY"
    assert result.state.activations["FRUSTRATION"] < 0.4


def test_time_decay_mode_does_not_disable_conversation_transition():
    text = "你真厉害"
    affect = build_conversational_affect(
        pragmatic=_pragmatic(topic_transition="SHIFT"),
        signal=_praise_signal(text),
    )
    result = _appraise(
        text,
        _praise_signal(text),
        affect=affect,
        state=_stale_frustration(),
        config=_config(time_decay_mode="LEGACY"),
    )

    # V0.4.3: time decay and conversation-driven change are separate
    # mechanisms. LEGACY keeps old decay maths, but a clearly new, relevant
    # event still re-ranks the stale label.
    assert result.emotion == "JOY"
    assert result.state.activations["JOY"] > result.state.activations.get(
        "FRUSTRATION", 0.0
    )


def test_continuing_topic_keeps_relevance_high():
    affect = build_conversational_affect(
        pragmatic=_pragmatic(topic_transition="CONTINUE"),
        signal=_praise_signal(),
    )
    assert affect.scene_relevance() >= 0.9


def test_a_strong_new_event_is_not_locked_out_by_a_stale_maximum():
    text = "你这个没用的东西"
    insult = AffectiveSignal(
        event_type="direct_insult",
        target="AGENT",
        social_intent="HOSTILE",
        factuality="asserted",
        confidence=0.95,
        evidence_span="没用的东西",
    )
    stale_joy = EmotionState(
        valence=0.5,
        arousal=0.4,
        intensity=0.80,
        emotion="JOY",
        activations={"JOY": 0.80},
        updated_at=NOW,
    )
    affect = build_conversational_affect(
        pragmatic=_pragmatic(
            topic_transition="SHIFT",
            primary_act="COMPLAINT",
            interaction_frame="CONFLICT",
            previous_frame="CASUAL",
        ),
        signal=insult,
        previous_frame="CASUAL",
    )
    result = AppraisalEngine(config=DEFAULT_CONFIG).appraise(
        text,
        rules=[_rule("direct_insult", "FRUSTRATION", 0.55)],
        signal=insult,
        current_state=stale_joy,
        now=NOW,
        addressee_hint="agent",
        config=DEFAULT_CONFIG,
        conversational_affect=affect,
    )
    assert result.emotion == "FRUSTRATION"
    assert result.state.relationship_tension > 0.0


# -- frame changes can move affect on their own ----------------------------


def test_frame_change_alone_produces_a_transition_without_a_rule():
    affect = build_conversational_affect(
        pragmatic=_pragmatic(
            primary_act="COMPLAINT",
            interaction_frame="CONFLICT",
            topic_transition="SHIFT",
        ),
        previous_frame="CASUAL",
    )
    result = _appraise("你到底想怎样", signal=None, affect=affect)

    assert result.diagnostic == FRAME_TRANSITION
    assert result.trigger_rule_id is None
    assert result.transitioned is True
    assert result.transition_cause == FRAME_CHANGE
    assert result.frame_from == "CASUAL"
    assert result.frame_to == "CONFLICT"
    assert result.emotion == "FRUSTRATION"
    assert result.state.intensity > 0.0
    assert result.state.relationship_tension > 0.0


def test_repair_frame_lowers_tension():
    state = EmotionState(
        valence=-0.2,
        arousal=0.50,
        intensity=0.30,
        emotion="FRUSTRATION",
        activations={"FRUSTRATION": 0.30},
        relationship_tension=0.60,
        tension_updated_at=NOW,
        updated_at=NOW,
    )
    affect = build_conversational_affect(
        pragmatic=_pragmatic(
            primary_act="APOLOGY",
            interaction_frame="REPAIR",
            topic_transition="CONTINUE",
        ),
        previous_frame="CONFLICT",
    )
    transition = transition_affect(
        state, affect, profile=None, traits=None, now=NOW, config=DEFAULT_CONFIG
    )
    assert transition.changed
    assert transition.cause.code == REPAIR
    assert transition.state.relationship_tension < state.relationship_tension


def test_unchanged_frame_and_no_event_is_a_true_no_change():
    affect = build_conversational_affect(
        pragmatic=_pragmatic(interaction_frame="CASUAL"),
        previous_frame="CASUAL",
    )
    transition = transition_affect(
        EmotionState(updated_at=NOW), affect, now=NOW, config=DEFAULT_CONFIG
    )
    assert not transition.changed
    assert transition.cause.code == "NONE"


# -- time decay is auxiliary ------------------------------------------------


def test_off_mode_never_decays_but_auxiliary_does():
    state = _stale_frustration()
    later = NOW + timedelta(minutes=30)
    off = decay(state, now=later, config=_config(time_decay_mode="OFF"))
    aux = decay(state, now=later, config=DEFAULT_CONFIG)

    assert off is state
    assert "FRUSTRATION" not in aux.activations
    assert aux.emotion == "NEUTRAL"


# -- full runtime: pragmatic frame reaches persisted affect -----------------


class _FakeClient:
    def __init__(self, payload: dict, *, response_text="我在。"):
        self.payload = {
            "meaning": "用户在表达不满",
            "action": "RESPOND",
            "reason_code": "direct_agent_address",
            "referenced_memory_ids": [],
            "memory_candidates": [],
            **payload,
        }
        self.response_text = response_text
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        content = (
            json.dumps(self.payload, ensure_ascii=False)
            if request_type == "analysis"
            else self.response_text
        )
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
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


def _event(person_id, text):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


async def test_pragmatic_conflict_changes_emotion_through_the_real_runtime(db):
    text = "你到底想怎样"
    client = _FakeClient(
        {
            "pragmatic": {
                "primary_act": "COMPLAINT",
                "secondary_acts": [],
                "interaction_frame": "CONFLICT",
                "refers_to_previous_turn": False,
                "antecedent_event_id": None,
                "setup_payoff_role": "NONE",
                "literal_interpretation": "在质问我",
                "implied_interpretation": "不耐烦了",
                "topic_transition": "SHIFT",
                "confidence": 0.9,
                "evidence_span": "你到底想怎样",
            }
        }
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(_event(person.person_id, text))

    assert report.status == "COMPLETED"
    agent_id = await _agent_id(rt)
    state = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert state.emotion == "FRUSTRATION"
    assert state.relationship_tension > 0.0

    events = await rt.emotion_service.repo.list_events(agent_id)
    frame_events = [e for e in events if e["appraisal_rule_id"] is None]
    assert len(frame_events) == 1
    assert frame_events[0]["appraisal_reason"] == "frame_conflict"
    assert frame_events[0]["emotion"] == "FRUSTRATION"

    snapshot = await rt.scene_manager.scene_repo.load("c1", person.person_id)
    assert snapshot is not None
    assert snapshot.state.interaction.frame == "CONFLICT"


async def test_a_validated_event_still_outranks_the_frame_hint(db):
    """One model turn cannot both name an event and have the frame bypass it."""
    text = "你这个没用的东西"
    client = _FakeClient(
        {
            "affective_signal": {
                "event_type": "direct_insult",
                "subject": "speaker",
                "target": "AGENT",
                "social_intent": "HOSTILE",
                "factuality": "asserted",
                "confidence": 0.95,
                "evidence_span": "没用的东西",
            },
            "pragmatic": {
                "primary_act": "REPAIR",
                "secondary_acts": [],
                "interaction_frame": "REPAIR",
                "topic_transition": "CONTINUE",
                "confidence": 0.9,
                "evidence_span": "没用的东西",
            },
        }
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)

    await rt.orchestrator.process_utterance(_event(person.person_id, text))

    agent_id = await _agent_id(rt)
    state = await rt.emotion_service.get_state(agent_id, person.person_id)
    # An event-rule turn must not be turned into a calm repair by a label.
    assert state.relationship_tension > 0.0
    events = await rt.emotion_service.repo.list_events(agent_id)
    assert events[0]["event_type"] == "direct_insult"
    assert events[0]["appraisal_rule_id"] is not None or events[0][
        "appraisal_reason"
    ] == "direct_insult"
