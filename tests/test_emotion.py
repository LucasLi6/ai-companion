"""V0.3 emotion framework: appraisal, decay, persistence, speech planning.

All tests are offline and deterministic. Emotion is the companion's own
simulated affect — nothing here claims anything about a human's real feelings.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from companion.emotion.dynamics import decay, decay_factor, label_for
from companion.emotion.models import EmotionState
from companion.emotion.speech import SpeechPlanner


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


# -- defaults / profile ------------------------------------------------------


async def test_ensure_defaults_seeds_profile_and_rules(rt):
    agent_id = await _agent_id(rt)
    profile = await rt.emotion_service.ensure_defaults(agent_id)
    assert profile.agent_id == agent_id
    rules = await rt.emotion_service.list_rules(agent_id)
    assert len(rules) == 5
    assert all(r.source == "system" for r in rules)


async def test_ensure_defaults_idempotent(rt):
    agent_id = await _agent_id(rt)
    await rt.emotion_service.ensure_defaults(agent_id)
    await rt.emotion_service.ensure_defaults(agent_id)
    rules = await rt.emotion_service.list_rules(agent_id)
    assert len(rules) == 5


# -- appraisal ---------------------------------------------------------------


async def _rules(rt):
    return await rt.emotion_service.list_rules(await _agent_id(rt))


async def test_appraise_milestone_joy(rt):
    rules = await _rules(rt)
    result = rt.emotion_service.appraisal.appraise(
        "我成功了！项目完成了", rules=rules
    )
    assert result.emotion == "JOY"
    assert result.reason == "milestone_joy"
    assert result.trigger_rule_id is not None


async def test_appraise_no_rule_holds_state(rt):
    rules = await _rules(rt)
    current = EmotionState(
        valence=0.3, arousal=0.2, intensity=0.4, emotion="JOY"
    )
    result = rt.emotion_service.appraisal.appraise(
        "随便聊聊", rules=rules, current_state=current
    )
    assert result.emotion == "JOY"
    assert result.valence == 0.3
    assert result.reason == "no_rule_matched"


async def test_appraise_priority_urgent_over_milestone(rt):
    rules = await _rules(rt)
    result = rt.emotion_service.appraisal.appraise(
        "紧急！我成功了", rules=rules
    )
    assert result.emotion == "CONCERN"
    assert result.reason == "urgent_concern"


# -- decay / labeling --------------------------------------------------------


def test_decay_factor_exact():
    assert decay_factor(3600, 3600) == pytest.approx(0.5)
    assert decay_factor(7200, 3600) == pytest.approx(0.25)
    assert decay_factor(0, 3600) == 1.0


def test_decay_returns_neutral_below_threshold():
    now = datetime.now(timezone.utc)
    state = EmotionState(
        valence=0.5,
        arousal=0.5,
        intensity=0.01,  # below the neutral floor
        emotion="JOY",
        updated_at=now,
        half_life_seconds=3600,
    )
    out = decay(state, now=now)
    assert out.emotion == "NEUTRAL"
    assert out.intensity == 0.0


def test_decay_drifts_toward_baseline():
    now = datetime.now(timezone.utc)
    state = EmotionState(
        valence=0.4, arousal=0.6, intensity=0.8, emotion="JOY",
        updated_at=now, half_life_seconds=3600,
    )
    # No elapsed time: factor is 1.0, so the state is unchanged.
    out = decay(state, now=now, baseline_valence=0.0, baseline_arousal=0.0)
    assert out.valence == pytest.approx(0.4)


def test_label_for_thresholds():
    assert label_for(0.0, 0.0, 0.0) == "NEUTRAL"
    assert label_for(0.5, 0.3, 0.5) == "JOY"
    assert label_for(0.5, 0.7, 0.5) == "EXCITEMENT"
    assert label_for(-0.5, 0.1, 0.5) == "SADNESS"
    assert label_for(-0.5, 0.7, 0.5) == "FRUSTRATION"
    assert label_for(-0.1, 0.3, 0.5) == "CONCERN"


# -- state persistence -------------------------------------------------------


async def test_get_state_no_snapshot_returns_baseline(rt):
    agent_id = await _agent_id(rt)
    state = await rt.emotion_service.get_state(agent_id, "person_none")
    assert state.emotion == "NEUTRAL"
    assert state.intensity == 0.0


async def test_apply_result_persists_event_and_snapshot(rt):
    agent_id = await _agent_id(rt)
    rules = await _rules(rt)
    result = rt.emotion_service.appraisal.appraise("我成功了", rules=rules)
    state = await rt.emotion_service.apply_result(
        agent_id, "person_x", result, source_event_id="evt_x", persist=True
    )
    assert state.emotion == "JOY"
    assert await rt.emotion_service.repo.has_event_for_source("evt_x")
    snap = await rt.emotion_service.repo.get_snapshot(agent_id, "person_x")
    assert snap is not None
    assert snap["emotion"] == "JOY"


async def test_apply_result_idempotent_by_source(rt):
    agent_id = await _agent_id(rt)
    rules = await _rules(rt)
    result = rt.emotion_service.appraisal.appraise("我成功了", rules=rules)
    await rt.emotion_service.apply_result(
        agent_id, "person_y", result, source_event_id="evt_y", persist=True
    )
    events_before = len(
        await rt.emotion_service.repo.list_events(agent_id)
    )
    # Re-processing the same source must not apply the shift twice.
    await rt.emotion_service.apply_result(
        agent_id, "person_y", result, source_event_id="evt_y", persist=True
    )
    events_after = len(await rt.emotion_service.repo.list_events(agent_id))
    assert events_after == events_before


async def test_apply_result_persist_false_no_writes(rt):
    agent_id = await _agent_id(rt)
    rules = await _rules(rt)
    result = rt.emotion_service.appraisal.appraise("我成功了", rules=rules)
    await rt.emotion_service.apply_result(
        agent_id, "person_z", result, source_event_id="evt_z", persist=False
    )
    assert not await rt.emotion_service.repo.has_event_for_source("evt_z")
    assert await rt.emotion_service.repo.get_snapshot(agent_id, "person_z") is None


# -- profile / rules / speech ------------------------------------------------


async def test_update_profile_clamps_and_audits(rt):
    agent_id = await _agent_id(rt)
    await rt.emotion_service.ensure_defaults(agent_id)
    before = await rt.emotion_service.get_profile(agent_id)
    updated = await rt.emotion_service.update_profile(
        agent_id, baseline_valence=5.0, default_half_life_seconds=0
    )
    assert updated.baseline_valence == 1.0  # clamped to [-1, 1]
    assert updated.default_half_life_seconds == 1  # clamped to >= 1
    assert updated.revision == before.revision + 1


async def test_propose_rule_creates_disabled(rt):
    agent_id = await _agent_id(rt)
    rule = await rt.emotion_service.propose_rule(
        agent_id,
        name="learned_rule",
        trigger_type="utterance",
        pattern={"keywords": ["测试"]},
        emotion="JOY",
        valence_delta=0.2,
        arousal_delta=0.1,
        intensity=0.3,
        evidence_ref="memory_123",
    )
    assert rule.status == "DISABLED"
    assert rule.source == "reflection"


async def test_set_rule_status_audited(rt):
    agent_id = await _agent_id(rt)
    rules = await rt.emotion_service.list_rules(agent_id)
    target = rules[0]
    changed = await rt.emotion_service.set_rule_status(
        agent_id, target.rule_id, "DISABLED"
    )
    assert changed is True
    disabled = await rt.emotion_service.repo.get_rule(agent_id, target.rule_id)
    assert disabled.status == "DISABLED"


def test_speech_plan_rate_bounds_and_instruction():
    planner = SpeechPlanner()
    plan = planner.plan(
        state=EmotionState(emotion="JOY", intensity=0.5), voice_profile=None, text=""
    )
    assert plan.emotion == "JOY"
    assert plan.delivery_instruction
    assert plan.speaking_rate == pytest.approx(1.05)
    # A high base rate is clamped into the safe range.
    fast = planner.plan(
        state=EmotionState(emotion="EXCITEMENT", intensity=0.7),
        voice_profile={"speaking_rate": 2.0, "voice_profile_id": "v1"},
        text="",
    )
    assert fast.speaking_rate <= 1.6
    assert fast.voice_profile_id == "v1"
