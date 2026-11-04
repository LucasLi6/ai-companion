"""V0.4.3: affect is a compact state, not a numeric style command.

The conversation owns the change (a real event, a frame shift, or a validated
same-frame development); time decay is a separate cleanup and never disables
conversational updating. These tests pin the composite activation gate, the
same-frame evidence rule and the admin-configured dispositions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from companion.emotion.dynamics import apply_emotional_stimulus
from companion.emotion.expression_policy import (
    compile_affect_context,
    compile_expression_influence,
)
from companion.emotion.models import AppraisalRule, EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG, EmotionDynamicsConfig
from companion.emotion.transition import (
    NONE,
    SAME_FRAME_EVENT,
    build_conversational_affect,
    transition_affect,
)

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _state(**updates) -> EmotionState:
    payload = {
        "emotion": "NEUTRAL",
        "intensity": 0.0,
        "updated_at": NOW,
    }
    payload.update(updates)
    return EmotionState(**payload)


def _config(**updates) -> EmotionDynamicsConfig:
    return DEFAULT_CONFIG.model_copy(update=updates)


def test_compact_affect_is_a_state_description_without_numbers():
    text = compile_affect_context(
        _state(
            emotion="CONCERN",
            intensity=0.5,
            mood_emotion="SADNESS",
            mood_intensity=0.4,
            relationship_tension=0.2,
        ),
        interaction_frame="SUPPORT",
    )

    assert "此刻情绪偏向关切" in text
    assert "心情底色偏向低落" in text
    assert "还残留一点紧张" in text
    assert "本轮对话气氛是" in text
    # The description is state-only; the generic "don't perform" rule lives
    # in the trusted system prompt and is not repeated every turn.
    assert "这是角色当下的内在状态" not in text
    assert "真实的身体感受" not in text
    assert "0." not in text
    assert "directness" not in text


def test_affect_context_empty_when_neutral_and_no_frame():
    assert compile_affect_context(_state()) == ""


def test_frame_change_wording_is_distinct_from_steady_frame():
    state = _state(emotion="JOY", intensity=0.4)

    changed = compile_affect_context(
        state, interaction_frame="PLAYFUL", frame_changed=True
    )
    steady = compile_affect_context(
        state, interaction_frame="PLAYFUL", frame_changed=False
    )
    assert "本轮气氛刚刚转为" in changed
    assert "本轮对话气氛是" in steady


def test_composite_gate_lets_mood_or_tension_through_a_quiet_acute_layer():
    quiet_acute_low_mood = _state(
        emotion="NEUTRAL",
        intensity=0.10,
        mood_emotion="SADNESS",
        mood_intensity=0.5,
    )
    influence = compile_expression_influence(
        quiet_acute_low_mood, config=DEFAULT_CONFIG
    )
    # The old single 0.20 acute threshold returned zeros here.
    assert influence.is_active()
    assert influence.source_emotion == "NEUTRAL"

    tension_only = _state(
        intensity=0.05, relationship_tension=0.6, activations={}
    )
    assert compile_expression_influence(
        tension_only, config=DEFAULT_CONFIG
    ).is_active()


def test_off_mode_still_applies_a_conversation_frame_change():
    off = _config(time_decay_mode="OFF")
    affect = build_conversational_affect(
        previous_frame="CASUAL",
        pragmatic=type(
            "P",
            (),
            {
                "primary_act": "TEASING",
                "interaction_frame": "PLAYFUL",
                "refers_to_previous_turn": False,
                "setup_payoff_role": "NONE",
                "topic_transition": "CONTINUE",
                "confidence": 0.9,
            },
        )(),
    )

    result = transition_affect(
        _state(emotion="SERIOUS", intensity=0.3),
        affect,
        config=off,
        now=NOW,
    )

    assert result.changed is True
    assert result.cause.code == "FRAME_CHANGE"
    assert result.state.activations["JOY"] > 0.0


def test_same_frame_repetition_alone_changes_nothing():
    affect = build_conversational_affect(
        previous_frame="PLAYFUL",
        pragmatic=type(
            "P",
            (),
            {
                "primary_act": "TEASING",
                "interaction_frame": "PLAYFUL",
                "refers_to_previous_turn": False,
                "setup_payoff_role": "CONTINUATION",
                "topic_transition": "CONTINUE",
                "confidence": 0.9,
            },
        )(),
    )
    before = _state(emotion="JOY", intensity=0.3, activations={"JOY": 0.3})

    result = transition_affect(before, affect, config=DEFAULT_CONFIG, now=NOW)

    assert result.changed is False
    assert result.cause.code == NONE


def test_same_frame_development_with_evidence_moves_affect():
    affect = build_conversational_affect(
        previous_frame="PLAYFUL",
        pragmatic=type(
            "P",
            (),
            {
                "primary_act": "TEASING",
                "interaction_frame": "PLAYFUL",
                "refers_to_previous_turn": True,
                "setup_payoff_role": "PAYOFF",
                "topic_transition": "CONTINUE",
                "confidence": 0.9,
            },
        )(),
    )
    before = _state(emotion="JOY", intensity=0.1, activations={"JOY": 0.1})

    result = transition_affect(before, affect, config=DEFAULT_CONFIG, now=NOW)

    assert result.changed is True
    assert result.cause.code == SAME_FRAME_EVENT
    assert result.cause.reason == "same_frame_progress"
    assert result.state.activations["JOY"] > before.activations["JOY"]


def _rule(name, *, emotion="JOY", intensity=0.5) -> AppraisalRule:
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


def _apply(dispositions):
    return apply_emotional_stimulus(
        _state(updated_at=NOW),
        _rule("playful_teasing", emotion="JOY", intensity=0.5),
        None,
        None,
        NOW,
        config=DEFAULT_CONFIG,
        dispositions=dispositions,
    )


def test_teasing_tolerance_scales_the_reaction():
    low = _apply({"tolerance_for_teasing": 0.0})
    high = _apply({"tolerance_for_teasing": 1.0})

    assert high.activations["JOY"] < low.activations["JOY"]


def test_reconciliation_openness_scales_tension_repair():
    before = _state(
        updated_at=NOW,
        emotion="RELIEF",
        intensity=0.2,
        relationship_tension=0.8,
    )
    slow = apply_emotional_stimulus(
        before,
        _rule("agent_apology", emotion="RELIEF", intensity=0.3),
        None,
        None,
        NOW,
        config=DEFAULT_CONFIG,
        dispositions={"reconciliation_openness": 0.0},
    )
    fast = apply_emotional_stimulus(
        before,
        _rule("agent_apology", emotion="RELIEF", intensity=0.3),
        None,
        None,
        NOW,
        config=DEFAULT_CONFIG,
        dispositions={"reconciliation_openness": 1.0},
    )

    assert fast.relationship_tension < slow.relationship_tension
