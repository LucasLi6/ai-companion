"""V0.3.1 semantic appraisal: signal gating, legacy guards, dynamics.

All offline. The model's ``AffectiveSignal`` is only a hint; these tests pin
the local rules that decide whether it is allowed to move the companion's
simulated affect at all.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from companion.cognition.models import CognitiveContext
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ProcessingStatus,
    ReasonCode,
)
from companion.domain.models import (
    AffectiveSignal,
    CognitiveResult,
    make_utterance_event,
    validate_affective_signal,
)
from companion.emotion.dynamics import apply_emotional_stimulus, reaction_gain
from companion.emotion.models import AppraisalRule, EmotionProfile, EmotionState


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


async def _rules(rt):
    return await rt.emotion_service.list_rules(await _agent_id(rt))


def _signal(event_type, evidence, *, subject="speaker", factuality="asserted", confidence=0.9):
    return AffectiveSignal(
        event_type=event_type,
        subject=subject,
        factuality=factuality,
        confidence=confidence,
        evidence_span=evidence,
    )


SEMANTIC_CASES = [
    ("achievement", "我升职了", "JOY", "milestone_joy", {}),
    ("setback", "项目失败了", "CONCERN", "setback_concern", {}),
    ("gratitude", "谢谢你", "JOY", "explicit_gratitude", {}),
    ("urgency", "紧急情况", "CONCERN", "urgent_concern", {}),
    ("question", "这是为什么？", "CURIOSITY", "curiosity_question", {}),
]


@pytest.mark.parametrize(
    "event_type,text,emotion,reason,extra", SEMANTIC_CASES
)
async def test_semantic_signal_maps_to_named_rule(
    rt, event_type, text, emotion, reason, extra
):
    result = rt.emotion_service.appraisal.appraise(
        text,
        rules=await _rules(rt),
        signal=_signal(event_type, text, **extra),
    )
    assert result.emotion == emotion
    assert result.reason == reason
    assert result.trigger_rule_id is not None


@pytest.mark.parametrize(
    "signal_kwargs",
    [
        {"factuality": "negated"},
        {"factuality": "hypothetical"},
        {"factuality": "quoted"},
        {"factuality": "uncertain"},
        {"subject": "other"},
        {"subject": "agent"},
        {"confidence": 0.74},
    ],
)
async def test_semantic_signal_not_asserted_by_speaker_is_ignored(
    rt, signal_kwargs
):
    text = "我升职了"
    result = rt.emotion_service.appraisal.appraise(
        text,
        rules=await _rules(rt),
        signal=_signal("achievement", text, **signal_kwargs),
    )
    assert result.emotion == "NEUTRAL"
    assert result.reason == "signal_not_actionable"
    assert result.trigger_rule_id is None


async def test_semantic_signal_requires_evidence_in_text(rt):
    result = rt.emotion_service.appraisal.appraise(
        "我升职了",
        rules=await _rules(rt),
        signal=_signal("achievement", "完全不同的话"),
    )
    assert result.emotion == "NEUTRAL"
    assert result.reason == "signal_not_actionable"


async def test_semantic_signal_none_holds_state(rt):
    text = "我升职了"
    result = rt.emotion_service.appraisal.appraise(
        text,
        rules=await _rules(rt),
        signal=_signal("none", None),
    )
    assert result.reason == "signal_not_actionable"
    assert result.emotion == "NEUTRAL"


async def test_semantic_respects_disabled_rule(rt):
    agent_id = await _agent_id(rt)
    rules = await _rules(rt)
    joy = next(r for r in rules if r.name == "milestone_joy")
    await rt.emotion_service.set_rule_status(agent_id, joy.rule_id, "DISABLED")
    text = "我升职了"
    result = rt.emotion_service.appraisal.appraise(
        text,
        rules=await _rules(rt),
        signal=_signal("achievement", text),
    )
    assert result.emotion == "NEUTRAL"
    assert result.reason == "no_semantic_rule"


# -- local validation --------------------------------------------------------


def test_validate_signal_rejects_bad_payloads():
    text = "我升职了"
    good = {
        "event_type": "achievement",
        "subject": "speaker",
        "factuality": "asserted",
        "confidence": 0.9,
        "evidence_span": "升职",
    }
    assert validate_affective_signal(good, text) is not None
    for bad in (
        {**good, "event_type": "made_up"},
        {**good, "subject": "everyone"},
        {**good, "factuality": "definitely"},
        {**good, "confidence": float("nan")},
        {**good, "confidence": float("inf")},
        {**good, "confidence": -0.1},
        {**good, "evidence_span": "不在原文里"},
        {**good, "evidence_span": ""},
        {**good, "evidence_span": "x" * 513},
        "not a dict",
    ):
        assert validate_affective_signal(bad, text) is None


def test_validate_signal_none_event_needs_no_evidence():
    signal = validate_affective_signal(
        {
            "event_type": "none",
            "subject": "speaker",
            "factuality": "asserted",
            "confidence": 0.9,
        },
        "普通的一句话",
    )
    assert signal is not None and signal.event_type == "none"


# -- legacy keyword guards ---------------------------------------------------


LEGACY_GUARD_CASES = [
    "没有成功",
    "还没完成",
    "如果成功",
    "要是成功了就好了",
    "他说他成功了",
    "他成功了",
    "她今天很开心",
    '他引用"成功了"',
    "引用“成功了”",
    "不开心",
    "我开心吗？",
    "为什么觉得我开心？",
    "不想再失败了",
    "如果真的失败了",
]


@pytest.mark.parametrize("text", LEGACY_GUARD_CASES)
async def test_legacy_keywords_do_not_claim_joy(rt, text):
    result = rt.emotion_service.appraisal.appraise(
        text, rules=await _rules(rt)
    )
    assert result.reason != "milestone_joy"
    assert result.emotion != "JOY"


async def test_legacy_guard_holds_current_state(rt):
    current = EmotionState(
        valence=0.4, arousal=0.2, intensity=0.5, emotion="JOY"
    )
    result = rt.emotion_service.appraisal.appraise(
        "没有成功", rules=await _rules(rt), current_state=current
    )
    assert result.emotion == "JOY"
    assert result.valence == 0.4
    assert result.intensity == 0.5


# -- memory rules need a verified association --------------------------------


async def test_memory_kind_alone_does_not_match(rt):
    rule = AppraisalRule(
        rule_id="r1",
        agent_id="a",
        name="memory_rule",
        trigger_type="memory",
        pattern={"kind": "PERSONAL_FACT"},
        emotion="JOY",
        valence_delta=0.3,
        arousal_delta=0.1,
        intensity=0.5,
        priority=50,
        status="ACTIVE",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    memories = [
        {"memory_id": "m1", "kind": "PERSONAL_FACT", "status": "ACTIVE"}
    ]
    result = rt.emotion_service.appraisal.appraise(
        "随便聊聊", rules=[rule], memories=memories
    )
    assert result.reason == "no_rule_matched"
    assert result.matched_memory_id is None


async def test_memory_rule_requires_listed_active_memory(rt):
    rule = AppraisalRule(
        rule_id="r2",
        agent_id="a",
        name="memory_rule",
        trigger_type="memory",
        pattern={"memory_ids": ["m1"]},
        emotion="JOY",
        valence_delta=0.3,
        arousal_delta=0.1,
        intensity=0.5,
        priority=50,
        status="ACTIVE",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    memories = [{"memory_id": "m1", "status": "ACTIVE"}]
    result = rt.emotion_service.appraisal.appraise(
        "随便聊聊", rules=[rule], memories=memories
    )
    assert result.matched_memory_id == "m1"


# -- pure dynamics -----------------------------------------------------------


def _rule(emotion="JOY", intensity=0.5, valence=0.4, arousal=0.2):
    now = datetime.now(timezone.utc)
    return AppraisalRule(
        rule_id="r",
        agent_id="a",
        name="n",
        trigger_type="utterance",
        pattern={},
        emotion=emotion,
        valence_delta=valence,
        arousal_delta=arousal,
        intensity=intensity,
        created_at=now,
        updated_at=now,
    )


def test_reaction_gain_maps_responsiveness():
    now = datetime.now(timezone.utc)
    profile = EmotionProfile(
        agent_id="a",
        responsiveness=1.0,
        created_at=now,
        updated_at=now,
    )
    gain, empathy = reaction_gain(profile, {"empathy": 100})
    assert gain == pytest.approx(2.0)
    assert empathy == pytest.approx(1.0)
    quiet = profile.model_copy(update={"responsiveness": 0.25})
    assert reaction_gain(quiet, None)[0] == pytest.approx(0.5)


def test_apply_emotional_stimulus_formula_and_no_input_mutation():
    now = datetime.now(timezone.utc)
    profile = EmotionProfile(
        agent_id="a",
        responsiveness=0.5,
        created_at=now,
        updated_at=now,
    )
    current = EmotionState(
        valence=0.2, arousal=0.1, intensity=0.0, emotion="NEUTRAL",
        updated_at=now, half_life_seconds=3600.0,
    )
    state = apply_emotional_stimulus(current, _rule(), profile, {}, now)
    # Saturated accumulation: 0 + (1-0)*0.5, then the category target
    # (JOY = +0.65 valence / 0.45 arousal) scales the activation.
    assert state.valence == pytest.approx(0.325)
    assert state.arousal == pytest.approx(0.225)
    assert state.intensity == pytest.approx(0.5)
    assert state.emotion == "JOY"
    assert state.updated_at == now
    # Frozen dataclass: the original was not modified.
    assert current.valence == pytest.approx(0.2)
    assert current.emotion == "NEUTRAL"


def test_apply_stimulus_retains_stronger_current_intensity():
    now = datetime.now(timezone.utc)
    current = EmotionState(
        valence=0.1, arousal=0.1, intensity=0.8, emotion="EXCITEMENT",
        updated_at=now, half_life_seconds=3600.0,
    )
    state = apply_emotional_stimulus(
        current, _rule(intensity=0.3), None, None, now
    )
    # The 0.8 EXCITEMENT activation remains dominant; the weaker JOY event is
    # added alongside it rather than replacing it.
    assert state.intensity == pytest.approx(0.8)
    assert state.emotion == "EXCITEMENT"
    assert state.activations["JOY"] == pytest.approx(0.3)


def test_apply_stimulus_caps_acute_intensity_and_bounds_summary():
    now = datetime.now(timezone.utc)
    current = EmotionState(
        valence=0.95, arousal=0.95, intensity=0.0, emotion="JOY",
        updated_at=now, half_life_seconds=3600.0,
    )
    profile = EmotionProfile(
        agent_id="a", responsiveness=1.0, created_at=now, updated_at=now
    )
    state = apply_emotional_stimulus(
        current, _rule(valence=0.4, arousal=0.2, intensity=1.0), profile,
        {"empathy": 0}, now,
    )
    # Gain 2 × intensity 1.0 would be 2.0; the acute cap is 0.85, and the
    # summary is derived from the category target rather than unbounded sums.
    assert state.intensity == pytest.approx(0.85)
    assert state.valence == pytest.approx(0.65 * 0.85)
    assert state.arousal == pytest.approx(0.45 * 0.85)
    assert -1.0 <= state.valence <= 1.0
    assert 0.0 <= state.arousal <= 1.0


def test_speech_rate_scales_with_intensity():
    from companion.emotion.speech import SpeechPlanner

    planner = SpeechPlanner()
    low = planner.plan(state=EmotionState(emotion="JOY", intensity=0.0))
    medium = planner.plan(state=EmotionState(emotion="JOY", intensity=0.5))
    high = planner.plan(state=EmotionState(emotion="JOY", intensity=1.0))
    assert low.speaking_rate == pytest.approx(1.0)
    assert medium.speaking_rate == pytest.approx(1.05)
    assert high.speaking_rate == pytest.approx(1.10)
    assert low.delivery_instruction != medium.delivery_instruction
    assert high.delivery_instruction != medium.delivery_instruction
    assert "轻快" in high.delivery_instruction
    # Never a first-person claim about feeling; only delivery guidance.
    assert "我现在非常开心" not in high.delivery_instruction

    clamped_low = planner.plan(
        state=EmotionState(emotion="SADNESS", intensity=1.0),
        voice_profile={"speaking_rate": 0.1, "voice_profile_id": "v1"},
    )
    clamped_high = planner.plan(
        state=EmotionState(emotion="EXCITEMENT", intensity=1.0),
        voice_profile={"speaking_rate": 2.0, "voice_profile_id": "v1"},
    )
    assert clamped_low.speaking_rate == pytest.approx(0.6)
    assert clamped_high.speaking_rate == pytest.approx(1.6)
    assert clamped_high.voice_profile_id == "v1"


def test_empathy_softens_only_concern_and_sadness():
    now = datetime.now(timezone.utc)
    cold = {"empathy": 0}
    warm = {"empathy": 100}
    current = EmotionState(
        valence=0.0, arousal=0.0, intensity=0.0, emotion="NEUTRAL",
        updated_at=now, half_life_seconds=3600.0,
    )
    concern_cold = apply_emotional_stimulus(
        current, _rule("CONCERN", valence=-0.4), None, cold, now
    )
    concern_warm = apply_emotional_stimulus(
        current, _rule("CONCERN", valence=-0.4), None, warm, now
    )
    joy_cold = apply_emotional_stimulus(
        current, _rule("JOY", valence=-0.4), None, cold, now
    )
    joy_warm = apply_emotional_stimulus(
        current, _rule("JOY", valence=-0.4), None, warm, now
    )
    assert concern_warm.valence < concern_cold.valence
    assert joy_warm.valence == pytest.approx(joy_cold.valence)


# -- persistence + orchestration ordering ------------------------------------


async def test_affective_signal_roundtrips_analysis(rt, register):
    person = await register("Alice")
    await rt.conversations.get_or_create("c1")
    event = make_utterance_event(
        conversation_id="c1",
        text="我升职了",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    signal = _signal("achievement", "升职")
    analysis = CognitiveResult(
        event_id=event.event_id,
        meaning="speaker reports a promotion",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        provider="fake",
        affective_signal=signal,
    )
    await rt.repos.analysis.save_analysis(analysis, "ana_sig")

    loaded = await rt.repos.analysis.get_analysis(event.event_id)
    assert loaded is not None
    assert loaded.affective_signal == signal

    await rt.db.execute(
        "UPDATE cognitive_analyses SET affective_signal_json = 'not-json'"
        " WHERE analysis_id = ?",
        ("ana_sig",),
    )
    loaded = await rt.repos.analysis.get_analysis(event.event_id)
    assert loaded is not None and loaded.affective_signal is None


class _SignalCognitive:
    def __init__(self):
        self.calls = 0

    async def analyze(self, event, context, memories):
        self.calls += 1
        return CognitiveResult(
            event_id=event.event_id,
            meaning="achievement asserted",
            action=CognitiveAction.RESPOND,
            reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
            provider="fake",
            affective_signal=_signal("achievement", "成功"),
        )


async def test_orchestrator_retry_reuses_analysis_and_receipt(rt, register):
    person = await register("Alice")
    spy = _SignalCognitive()
    rt.orchestrator.cognitive = spy
    event = make_utterance_event(
        conversation_id="c2",
        text="我成功了",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "COMPLETED"
    assert spy.calls == 1

    agent_id = await _agent_id(rt)
    events = await rt.emotion_service.repo.list_events(agent_id)
    assert len(events) == 1
    assert events[0]["emotion"] == "JOY"

    # Simulate a retry of the same durable event: the persisted analysis and
    # the emotion receipt must both be reused, with no second model call.
    await rt.repos.utterance.update_status(
        event.event_id, ProcessingStatus.PROCESSING
    )
    second = await rt.orchestrator.process_utterance(event)
    assert second.status == "COMPLETED"
    assert spy.calls == 1
    assert len(await rt.emotion_service.repo.list_events(agent_id)) == 1


async def test_analysis_failure_does_not_change_emotion(rt, register):
    person = await register("Alice")

    class _Broken:
        async def analyze(self, event, context, memories):
            raise RuntimeError("provider down")

    rt.orchestrator.cognitive = _Broken()
    event = make_utterance_event(
        conversation_id="c3",
        text="我成功了",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "FAILED"
    agent_id = await _agent_id(rt)
    assert await rt.emotion_service.repo.list_events(agent_id) == []


async def test_emotion_uses_privacy_filtered_memories(rt, register, say):
    """The emotion layer never sees memories the provider context dropped."""
    person = await register("Alice")
    await rt.memory_service.create(
        kind="PERSONAL_FACT",
        content="Alice 的私密偏好",
        access=rt_admin(),
        subjects=[("PERSON", person.person_id)],
    )

    captured: dict = {}

    class _SpyEmotion:
        async def process_turn(self, agent_id, person_id, **kwargs):
            captured.update(kwargs)
            return EmotionState(emotion="NEUTRAL", intensity=0.0)

        async def plan_speech(self, agent_id, *, state, text=""):
            from companion.emotion.models import SpeechStylePlan

            return SpeechStylePlan()

    class _FilteringBuilder:
        async def build(self, event, memories):
            return CognitiveContext(
                conversation_id=event.conversation_id,
                current_event_id=event.event_id,
                current_text=event.text,
                memories=[],  # privacy gateway dropped everything
            )

    rt.orchestrator.context_builder = _FilteringBuilder()
    rt.orchestrator.emotion_service = _SpyEmotion()
    await say(person.person_id, "agent", "聊点什么")
    assert captured["memories"] == []


def rt_admin():
    from companion.memory.models import AccessContext

    return AccessContext("admin", is_admin=True)
