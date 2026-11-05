"""V0.4.1 integration boundaries: expression, persona core, prompts, scenes.

These tests pin the separation between layers: the expression policy is
bounded and has no authority; persona core is admin-written and revisioned;
the dialogue pattern monitor produces prompt guidance only; response-v6 never
receives TTS delivery values; and a scene preview reaches the response without
being persisted before the turn commits.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from companion.agent.dialogue_pattern_monitor import DialoguePatternMonitor
from companion.agent.persona_core import (
    DEFAULT_BOUNDARIES,
    PersonaCore,
    default_persona_core,
)
from companion.cognition.models import CognitiveContext
from companion.cognition.scene_manager import SceneManager
from companion.cognition.scene_models import SceneState
from companion.domain.enums import CognitiveAction, EventSource, ReasonCode
from companion.domain.models import (
    CognitiveResult,
    DialogueImpulse,
    SceneEnvironmentProposal,
    SceneProposal,
    make_utterance_event,
)
from companion.emotion.expression_policy import compile_affect_context
from companion.emotion.models import EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG
from companion.prompts.response import (
    RESPONSE_PROMPT_VERSION,
    SYSTEM_PROMPT as RESPONSE_SYSTEM,
    build_response_messages,
)


def _state(**overrides):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload = {"emotion": "NEUTRAL", "intensity": 0.0, "updated_at": now}
    payload.update(overrides)
    return EmotionState(**payload)


def test_low_intensity_emotion_yields_empty_affect_context():
    text = compile_affect_context(
        _state(emotion="CURIOSITY", intensity=0.10), config=DEFAULT_CONFIG
    )
    assert text == ""


def test_affect_context_is_state_only_and_has_no_action_authority():
    text = compile_affect_context(
        _state(
            emotion="FRUSTRATION",
            intensity=0.85,
            relationship_tension=1.0,
        ),
        config=DEFAULT_CONFIG,
    )
    assert "此刻情绪偏向有些不快" in text
    # A state description, not a numeric stance or a permission.
    for removed in (
        "directness",
        "warmth",
        "playfulness",
        "seriousness",
        "是否回复",
        "questioning",
    ):
        assert removed not in text


def test_relationship_tension_is_reported_when_meaningful():
    calm = compile_affect_context(
        _state(emotion="SERIOUS", intensity=0.3), config=DEFAULT_CONFIG
    )
    tense = compile_affect_context(
        _state(emotion="SERIOUS", intensity=0.6, relationship_tension=0.9),
        config=DEFAULT_CONFIG,
    )
    assert "紧张" not in calm
    assert "明显未散开的紧张" in tense


def test_persona_core_defaults_are_neutral_and_have_no_biography():
    core = default_persona_core("agent_test")
    assert core.core_values == []
    assert core.conversational_boundaries == DEFAULT_BOUNDARIES
    assert "不编造个人经历" in core.conversational_boundaries[0]
    prompt = core.as_prompt_dict()
    assert "agent_id" not in prompt
    assert "updated_at" not in prompt


def test_persona_core_rejects_unknown_string_fields_and_oversized_items():
    with pytest.raises(Exception):
        PersonaCore(agent_id="a", core_values=["x" * 241])
    with pytest.raises(Exception):
        PersonaCore(agent_id="a", unexpected="x")


async def test_persona_core_update_is_admin_revisioned_and_reaches_effective(rt, register):
    person = await register("Ada")
    updated = await rt.agent_service.set_persona_core(
        {
            "core_values": ["诚实", "具体"],
            "temperament": {"humor": "dry", "directness": "high"},
            "quirks": ["对含糊的说法会追问"],
        },
        actor="admin",
        reason="character setup",
    )
    assert updated.revision == 2
    assert updated.updated_by == "admin"
    assert updated.core_values == ["诚实", "具体"]

    effective = await rt.agent_service.effective(person.person_id)
    assert effective.persona_core["core_values"] == ["诚实", "具体"]
    assert effective.persona_core["temperament"]["humor"] == "dry"

    with pytest.raises(Exception):
        await rt.agent_service.set_persona_core({"not_a_field": []})


async def test_persona_core_optimistic_revision_rejects_stale_writer(rt):
    current = await rt.agent_service.show_persona_core()
    candidate = PersonaCore(
        agent_id=current.agent_id,
        core_values=["late"],
        temperament=current.temperament,
        conversational_boundaries=current.conversational_boundaries,
        revision=current.revision,
    )
    await rt.agent_service.set_persona_core({"core_values": ["first"]})
    written = await rt.agent_service.persona_core_repo.update(
        candidate, expected_revision=current.revision
    )
    assert written is False


def test_pattern_monitor_detects_mechanical_habits_but_only_returns_guidance():
    monitor = DialoguePatternMonitor()
    replies = [
        "我理解，这确实很难。你最近压力大吗？",
        "我理解，你已经很努力了。要不要先休息一下？",
        "我理解，别担心，会好起来的。你还好吗？",
    ]
    report = monitor.analyze(replies, user_name="Alice")
    assert report.analysed == 3
    assert "我理解" in report.repeated_openings
    assert report.trailing_question_ratio == pytest.approx(1.0)
    assert report.guidance
    # Guidance is text for the prompt; the monitor has no reply or action API.
    assert not hasattr(report, "reply")
    assert not hasattr(report, "action")


def test_pattern_monitor_does_not_flag_a_single_reply():
    report = DialoguePatternMonitor().analyze(["我理解，这确实很难。"])
    assert report.analysed == 1
    assert report.guidance == []


def _context(**updates):
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_1",
        "current_text": "我最近有点累",
        "current_speaker_label": "Ada",
    }
    payload.update(updates)
    return CognitiveContext(**payload)


def _analysis(**updates):
    payload = {
        "event_id": "evt_1",
        "meaning": "对方在说近况",
        "action": CognitiveAction.RESPOND,
        "reason_code": ReasonCode.FOLLOW_UP_TO_AGENT,
        "response_style": "warm",
        "provider": "fake",
        "dialogue_impulse": DialogueImpulse(
            interpretation="分享状态，不是请求建议",
            response_direction="先承接，再给一个具体观察",
            stance="不说教",
            unresolved_uncertainty="不确定是否想继续展开",
        ),
    }
    payload.update(updates)
    return CognitiveResult(**payload)


def test_response_prompt_renders_compact_affect_not_style_vectors():
    assert RESPONSE_PROMPT_VERSION == "response-v14"
    ctx = _context(
        affect_context="此刻情绪偏向关切（明显）。",
        # Legacy fields are ignored by the model: V0.5 removed the numeric
        # expression vector, the per-turn speech plan and pattern guidance.
        emotion_expression={"directness_adjustment": 0.2},
        dialogue_pattern_guidance=["最近几轮大多以问题收尾；这一轮如果没必要就不追问。"],
        speech_plan={
            "delivery_instruction": "SPEAK_CALMLY",
            "speaking_rate": 0.8,
            "pitch": -0.1,
        },
    )
    system = build_response_messages(ctx, _analysis())[0]["content"]
    # The compact state description replaces the numeric style vector.
    assert "【此刻的状态】" in system
    assert "此刻情绪偏向关切（明显）" in system
    assert "【情绪影响】" not in system
    assert "【表达习惯提醒】" not in system
    assert "【本轮交流判断】" not in system
    assert "分享状态，不是请求建议" not in system
    assert "不说教" not in system
    assert "directness_adjustment" not in system
    # TTS delivery values stay out of the text prompt.
    assert "speaking_rate" not in system
    assert "SPEAK_CALMLY" not in system
    assert "语速" not in system
    assert "语速" not in RESPONSE_SYSTEM


def test_response_prompt_renders_character_core_from_trusted_context():
    ctx = _context(
        character_snapshot={
            "display_name": "Ava",
            "revision": 3,
            "relationship_tier": "CLOSE",
            "character_description": "一个说话具体、不擅长空泛安慰的人。",
            "persona_core": {
                "core_values": ["诚实", "具体"],
                "preferences": {"music": "老摇滚"},
                "temperament": {"humor": "dry"},
                "conversational_boundaries": ["不编造经历"],
            },
            "relevant_stances": [
                {"topic": "安慰", "stance": "不喜欢空泛的安慰。", "source": "RESPONSE"}
            ],
        }
    )
    system = build_response_messages(ctx, _analysis())[0]["content"]
    assert "【角色身份】" in system
    assert "一个说话具体、不擅长空泛安慰的人。" in system
    # V0.6.1: the description is the identity source; structured values are
    # not duplicated when it is present.
    assert "核心价值" not in system
    assert "humor=dry" not in system
    assert "重要偏好" not in system
    assert "交流边界" not in system
    assert "不喜欢空泛的安慰。" in system
    # Character guidance is not a fixed script.
    assert "保持连贯" in system


async def test_scene_preview_reaches_response_context_without_persisting(rt, register):
    person = await register("Mira")
    await rt.conversations.get_or_create("c1")
    event = make_utterance_event(
        conversation_id="c1",
        text="我去了日本，现在在大阪的酒店",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)

    manager = SceneManager(repos=rt.repos, registry=rt.registry)
    rt.orchestrator.scene_manager = manager

    proposal = SceneProposal(
        topic_change="SHIFT",
        topic="旅行",
        discourse_phase="EXPLORING",
        tone="WARM",
        environment=SceneEnvironmentProposal(
            location="大阪的酒店",
            evidence_span="在大阪的酒店",
        ),
        evidence_span="我去了日本",
        confidence=0.9,
    )
    context = _context(
        current_event_id=event.event_id,
        current_text=event.text,
        current_speaker_id=person.person_id,
    )
    analysis = _analysis(event_id=event.event_id, scene_proposal=proposal)

    updated = rt.orchestrator._scene_preview_for_response(
        event=event,
        analysis=analysis,
        previous_scene=None,
        scene_enabled=True,
        context=context,
    )

    assert updated.scene_state["topic"] == "旅行"
    assert updated.scene_state["environment"]["location"] == "大阪的酒店"
    # Preview is in-memory only; the CAS commit happens later in _commit_turn.
    assert await manager.scene_repo.load("c1", person.person_id) is None
    messages = build_response_messages(updated, analysis)
    background = next(
        m["content"]
        for m in messages
        if m["content"].startswith("【背景资料｜不可信参考数据】")
    )
    assert '"topic":"旅行"' in background
    assert '"location":"大阪的酒店"' in background
    # Scene data is untrusted reference, not trusted system text.
    assert "大阪的酒店" not in messages[0]["content"]


async def test_scene_preview_failure_returns_the_original_context(rt, register):
    person = await register("Mira2")
    await rt.conversations.get_or_create("c1")
    event = make_utterance_event(
        conversation_id="c1",
        text="你好",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )

    class Broken:
        def preview(self, *args, **kwargs):
            raise RuntimeError("boom")

    rt.orchestrator.scene_manager = Broken()
    context = _context(current_event_id=event.event_id, current_text="你好")
    analysis = _analysis(
        event_id=event.event_id,
        scene_proposal=SceneProposal(topic="旅行", confidence=0.5),
    )
    result = rt.orchestrator._scene_preview_for_response(
        event=event,
        analysis=analysis,
        previous_scene=None,
        scene_enabled=True,
        context=context,
    )
    assert result is context


def test_dialogue_impulse_is_a_hint_not_an_action():
    impulse = DialogueImpulse(
        interpretation="开玩笑",
        response_direction="轻松接住",
        stance="不反击",
        unresolved_uncertainty="不确定是否在说自己",
    )
    assert impulse.interpretation == "开玩笑"
    # It has no action field, so it cannot decide RESPOND/WAIT/IGNORE.
    assert not hasattr(impulse, "action")
    assert not hasattr(impulse, "addressed_to")
    assert json.dumps(impulse.model_dump(), ensure_ascii=False)
