"""V0.4.1 numerical dynamics: saturation, layered decay, and 12 time points.

All values are computed from the shipped defaults, not from a running model.
The time-point test injects the service clock, so it is deterministic and does
not sleep.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from companion.domain.models import AffectiveSignal, new_id
from companion.emotion.dynamics import (
    apply_emotional_stimulus,
    decay,
    derive_acute,
)
from companion.emotion.models import AppraisalRule, EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG


def _rule(name="direct_insult", *, intensity=0.30, emotion="FRUSTRATION"):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return AppraisalRule(
        rule_id="rule_test",
        agent_id="agent_test",
        name=name,
        trigger_type="event",
        pattern={},
        emotion=emotion,
        valence_delta=0.0,
        arousal_delta=0.0,
        intensity=intensity,
        priority=1.0,
        status="ACTIVE",
        source="system",
        created_at=now,
        updated_at=now,
    )


def _neutral(now):
    return EmotionState(updated_at=now)


def test_saturating_accumulation_matches_documented_example():
    """0.30 → 0.51 → 0.657 under three equal stimuli."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = _neutral(now)
    rule = _rule(intensity=0.30)
    values = []
    for index in range(3):
        state = apply_emotional_stimulus(
            state,
            rule,
            None,
            None,
            now + timedelta(seconds=index),
            config=DEFAULT_CONFIG,
        )
        values.append(state.activations["FRUSTRATION"])
    assert values == pytest.approx([0.30, 0.51, 0.657], abs=1e-9)


def test_acute_cap_is_never_exceeded():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = _neutral(now)
    rule = _rule(intensity=1.0)
    for index in range(20):
        state = apply_emotional_stimulus(
            state,
            rule,
            None,
            None,
            now + timedelta(seconds=index),
            config=DEFAULT_CONFIG,
        )
    assert state.activations["FRUSTRATION"] <= DEFAULT_CONFIG.acute_intensity_cap
    assert state.intensity <= DEFAULT_CONFIG.acute_intensity_cap


def test_same_input_is_deterministic():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rule = _rule(intensity=0.55)
    a = apply_emotional_stimulus(_neutral(now), rule, None, None, now, config=DEFAULT_CONFIG)
    b = apply_emotional_stimulus(_neutral(now), rule, None, None, now, config=DEFAULT_CONFIG)
    assert a == b


def test_opposite_valence_mitigates_instead_of_adding():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    joy = _rule(name="direct_praise_joy", intensity=0.6, emotion="JOY")
    insult = _rule(intensity=0.6, emotion="FRUSTRATION")
    state = apply_emotional_stimulus(_neutral(now), joy, None, None, now, config=DEFAULT_CONFIG)
    assert state.activations["JOY"] == pytest.approx(0.6)
    state = apply_emotional_stimulus(
        state, insult, None, None, now + timedelta(seconds=1), config=DEFAULT_CONFIG
    )
    assert state.activations["JOY"] == pytest.approx(0.6 * 0.4)
    # The new, opposite activation is added from its own zero baseline.
    assert state.activations["FRUSTRATION"] == pytest.approx(0.6)


def test_per_category_half_lives_make_curiosity_fade_before_sadness():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = EmotionState(
        activations={"CURIOSITY": 0.8, "SADNESS": 0.8},
        updated_at=now,
    )
    later = decay(
        state, now=now + timedelta(seconds=600), config=DEFAULT_CONFIG
    )
    assert later.activations["SADNESS"] > later.activations["CURIOSITY"]
    assert later.activations["CURIOSITY"] == pytest.approx(
        0.8 * 2 ** (-600 / 180), abs=1e-9
    )
    assert later.activations["SADNESS"] == pytest.approx(
        0.8 * 2 ** (-600 / 900), abs=1e-9
    )


def test_tension_decays_on_its_own_slower_clock():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = EmotionState(
        activations={"FRUSTRATION": 0.6},
        mood_intensity=0.12,
        mood_valence=-0.07,
        mood_updated_at=now,
        relationship_tension=0.6,
        tension_updated_at=now,
        updated_at=now,
    )
    later = decay(
        state, now=now + timedelta(seconds=1800), config=DEFAULT_CONFIG
    )
    # FRUSTRATION's 300 s half-life leaves ~0.0094 at 30 minutes, below the
    # neutral threshold, so it is cleaned out while mood and tension remain.
    assert "FRUSTRATION" not in later.activations
    assert later.intensity == 0.0
    assert later.mood_intensity == pytest.approx(0.06)
    assert later.relationship_tension == pytest.approx(0.6 * 2 ** (-1800 / 2700))


def test_clock_rollback_never_amplifies_emotion():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = EmotionState(activations={"JOY": 0.5}, updated_at=now)
    earlier = decay(
        state, now=now - timedelta(hours=1), config=DEFAULT_CONFIG
    )
    assert earlier.intensity == pytest.approx(0.5)


def test_derive_acute_label_is_the_strongest_activation():
    emotion, valence, arousal, intensity, clean = derive_acute(
        {"JOY": 0.2, "CONCERN": 0.9}
    )
    assert emotion == "CONCERN"
    assert intensity == pytest.approx(0.85)  # capped
    assert valence < 0
    assert clean == {"JOY": 0.2, "CONCERN": 0.85}


async def test_twelve_time_points_decay_three_layers(rt, register):
    """0s, 10s, 30s, 1m, 3m, 5m, 10m, 15m, 30m, 60m, 120m, 24h."""
    person = await register("Mira")
    manifest = await rt.agent_service.repo.get_manifest()
    agent_id = manifest["agent_id"]

    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    current = [start]
    rt.emotion_service.clock = lambda: current[0]

    text = "你真蠢"
    signal = AffectiveSignal(
        event_type="direct_insult",
        target="AGENT",
        social_intent="HOSTILE",
        subject="speaker",
        factuality="asserted",
        confidence=0.95,
        evidence_span=text,
    )
    seeded = await rt.emotion_service.process_turn(
        agent_id,
        person.person_id,
        text=text,
        signal=signal,
        source_event_id=new_id("evt"),
        addressee_hint="agent",
    )
    assert seeded.activations["FRUSTRATION"] > 0.5
    assert seeded.relationship_tension > 0.4
    assert seeded.mood_intensity > 0.0

    initial_acute = seeded.activations["FRUSTRATION"]
    initial_mood = seeded.mood_intensity
    initial_tension = seeded.relationship_tension

    offsets = [0, 10, 30, 60, 180, 300, 600, 900, 1800, 3600, 7200, 86400]
    rows = []
    for offset in offsets:
        current[0] = start + timedelta(seconds=offset)
        state = await rt.emotion_service.get_state(agent_id, person.person_id)
        rows.append(
            (
                offset,
                state.intensity,
                state.mood_intensity,
                state.relationship_tension,
            )
        )

    # Monotonic non-increase at every layer.
    for index in range(1, len(rows)):
        assert rows[index][1] <= rows[index - 1][1] + 1e-12
        assert rows[index][2] <= rows[index - 1][2] + 1e-12
        assert rows[index][3] <= rows[index - 1][3] + 1e-12

    # Exact values at chosen points (per-category and layer clocks).
    assert rows[0][1] == pytest.approx(initial_acute, abs=1e-9)
    assert rows[3][1] == pytest.approx(initial_acute * 2 ** (-60 / 300), abs=1e-9)
    assert rows[5][1] == pytest.approx(initial_acute * 2 ** (-300 / 300), abs=1e-9)
    assert rows[8][2] == pytest.approx(initial_mood * 2 ** (-1800 / 1800), abs=1e-9)
    assert rows[8][3] == pytest.approx(initial_tension * 2 ** (-1800 / 2700), abs=1e-9)
    assert rows[10][3] == pytest.approx(initial_tension * 2 ** (-7200 / 2700), abs=1e-9)

    # Acute is gone by 30 minutes; mood survives it; tension outlives mood.
    assert rows[8][1] == 0.0
    assert rows[8][2] > 0.0
    assert rows[8][3] > 0.0
    assert rows[8][3] > rows[8][2]
    # Everything has returned to baseline after a day.
    assert rows[-1][1] == 0.0
    assert rows[-1][2] == 0.0
    assert rows[-1][3] == 0.0


async def test_reapplying_same_source_event_is_idempotent(rt, register):
    person = await register("Noor")
    agent_id = (await rt.agent_service.repo.get_manifest())["agent_id"]
    text = "你真是个废物"
    signal = AffectiveSignal(
        event_type="direct_insult",
        target="AGENT",
        social_intent="HOSTILE",
        subject="speaker",
        factuality="asserted",
        confidence=0.95,
        evidence_span=text,
    )
    source = new_id("evt")
    first = await rt.emotion_service.process_turn(
        agent_id,
        person.person_id,
        text=text,
        signal=signal,
        source_event_id=source,
        addressee_hint="agent",
    )
    second = await rt.emotion_service.process_turn(
        agent_id,
        person.person_id,
        text=text,
        signal=signal,
        source_event_id=source,
        addressee_hint="agent",
    )
    # The duplicate turn is a no-op: it returns the decayed snapshot from the
    # receipt path rather than applying the insult a second time.
    assert second.intensity <= first.intensity + 1e-12
    assert second.relationship_tension <= first.relationship_tension + 1e-12
    assert second.intensity == pytest.approx(first.intensity, abs=1e-3)
    assert second.relationship_tension == pytest.approx(
        first.relationship_tension, abs=1e-3
    )
    receipt = await rt.emotion_service.repo.get_receipt(source)
    assert receipt is not None
    assert receipt["outcome"] == "APPLIED"
