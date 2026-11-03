"""V0.4 — deterministic personality/style compilation and trusted examples.

The compiler is not a second persona store: it only projects the existing
effective Agent Core into expressive guidance. These tests pin the precedence
(current explicit request > stored preference > trait threshold), the absence
of forced jokes/psychoanalysis, and the person-scoped style-example boundary.
"""

from __future__ import annotations

import pytest

from companion.agent.dialogue_style import (
    StyleThresholds,
    compile_dialogue_style,
)
from companion.agent.models import EffectiveAgentContext
from companion.agent.personality_compiler import PersonalityCompiler
from companion.agent.style_examples import StyleExampleError, StyleExampleRepository


def _effective(**traits):
    base = {
        "formality": 50,
        "assertiveness": 45,
        "verbosity": 45,
        "humor": 50,
        "playfulness": 50,
        "warmth": 60,
        "curiosity": 60,
        "empathy": 65,
        "directness": 50,
    }
    base.update(traits)
    return EffectiveAgentContext(
        agent_id="agent_test",
        display_name="Ava",
        pronouns=None,
        description="A companion",
        person_id="person_1",
        familiarity="CLOSE",
        traits=base,
    )


def test_trait_thresholds_compile_into_bands():
    style = compile_dialogue_style(
        _effective(
            formality=10,
            assertiveness=90,
            verbosity=10,
            humor=90,
            curiosity=10,
            empathy=90,
        )
    )
    assert style.formality == "LOW"
    assert style.directness == "HIGH"
    assert style.typical_length == "SHORT"
    assert style.humor_tendency == "FREQUENT_WHEN_FITTING"
    assert style.questioning_style == "RARELY"
    assert style.empathy_expression == "HIGH"


@pytest.mark.parametrize("value,band", [(0, "LOW"), (34, "LOW"), (35, "MODERATE"), (69, "MODERATE"), (70, "HIGH"), (100, "HIGH")])
def test_band_boundaries_are_stable(value, band):
    assert StyleThresholds().band(value) == band


def test_current_request_outranks_stored_preference():
    style = compile_dialogue_style(
        _effective(verbosity=90),
        memories=["用户喜欢详细回复"],
        current_text="这次简短一点",
    )
    assert style.typical_length == "SHORT"
    assert "current_request_short" in style.explicit_request
    assert "prefers_detail" in style.personal_preferences


def test_stored_preferences_do_not_override_current_request():
    style = compile_dialogue_style(
        _effective(verbosity=10),
        memories=["用户喜欢详细回复"],
        current_text="简短说",
    )
    assert style.typical_length == "SHORT"


def test_stored_preference_for_no_name_is_respected():
    style = compile_dialogue_style(
        _effective(), memories=["不要叫我名字"]
    )
    assert style.name_addressing == "NEVER"
    assert "avoid_name_address" in style.personal_preferences


def test_current_request_no_humor_and_no_questions():
    style = compile_dialogue_style(
        _effective(humor=90, curiosity=90),
        current_text="严肃一点，别开玩笑，也别追问",
    )
    assert style.humor_tendency == "OCCASIONAL"
    assert style.questioning_style == "RARELY"


def test_current_request_no_psychoanalysis_lowers_empathy_expression():
    style = compile_dialogue_style(
        _effective(empathy=95), current_text="不要分析我"
    )
    assert style.empathy_expression == "LOW"
    assert "current_request_no_analysis" in style.explicit_request
    assert not any("可以共情" in line for line in style.guidance)


def test_high_humor_never_becomes_an_obligation_to_joke():
    style = compile_dialogue_style(_effective(humor=100))
    joined = "\n".join(style.guidance)
    assert "不要为了幽默硬讲笑话" in joined
    assert "必须讲笑话" not in joined


def test_high_empathy_never_becomes_diagnosis():
    style = compile_dialogue_style(_effective(empathy=100))
    joined = "\n".join(style.guidance)
    assert "不要做心理诊断" in joined
    assert "贴标签" in joined


def test_style_guidance_rejects_customer_service_tone():
    joined = "\n".join(compile_dialogue_style(_effective()).guidance)
    assert "客户" in joined or "客服" in joined
    assert "模板" in joined


def test_personality_uses_existing_identity_without_new_sources():
    effective = _effective()
    style = compile_dialogue_style(effective)
    profile = PersonalityCompiler().compile(effective, style=style)
    assert profile["agent_id"] == effective.agent_id
    assert profile["display_name"] == "Ava"
    assert profile["relationship_tier"] == "CLOSE"
    assert profile["traits"] == dict(effective.traits)
    # Nothing that could silently override permissions or identity.
    assert "allow_external_ai" not in profile
    assert "permissions" not in profile
    assert "identity_override" not in profile


def test_style_examples_are_capped_and_marked_as_references():
    effective = _effective()
    style = compile_dialogue_style(effective)
    examples = [
        {
            "scenario": f"s{i}",
            "user_text": f"u{i}",
            "ideal_response": f"r{i}",
        }
        for i in range(5)
    ]
    profile = PersonalityCompiler().compile(
        effective, style=style, style_examples=examples
    )
    assert len(profile["style_examples"]) == 3
    assert all(
        item["role"] == "STYLE_REFERENCE_NOT_HISTORY"
        for item in profile["style_examples"]
    )
    assert "不是固定回复" in profile["style_examples_policy"]


def test_pragmatic_guidance_is_not_a_script():
    compiler = PersonalityCompiler()
    lines = compiler.pragmatic_guidance(
        scene={
            "topic": "旅行",
            "open_loops": [{"topic": "工作"}],
            "environment": {"location": "家里", "other_people_present": "NO"},
        },
        current_addressee_hint="human",
        quiet_mode=True,
    )
    joined = "\n".join(lines)
    assert "当前话题：旅行" in joined
    assert "不要为了显得记得而强行提起" in joined
    assert "不要把它当成现在的确定事实" in joined
    assert "保持简短" in joined
    assert "由本地策略决定" in joined
    # Reminders, not canned replies.
    assert "你可以说" not in joined
    assert "回复：" not in joined


def test_pragmatic_guidance_for_pending_followup():
    record = type("Record", (), {"pending_followup": True})()
    lines = PersonalityCompiler().pragmatic_guidance(action_history=[record])
    joined = "\n".join(lines)
    assert "不要每次恢复说话都道歉" in joined
    assert "只有对方现在主动追问时才解释" in joined


def test_pragmatic_guidance_silent_without_scene_or_history():
    lines = PersonalityCompiler().pragmatic_guidance()
    assert lines == ["只回应本轮当前说话者；不要替别人说话或假装听见了未提供的对话。"]


# -- trusted style examples -------------------------------------------------


async def _people(rt):
    alice = await rt.registry.register(
        "Alice", "owner", allow_processing=True, allow_persistence=True
    )
    bob = await rt.registry.register(
        "Bob", "friend", allow_processing=True, allow_persistence=True
    )
    return alice.person_id, bob.person_id


async def test_style_examples_are_person_scoped(db, rt):
    repo = StyleExampleRepository(db)
    alice, bob = await _people(rt)
    await repo.add(
        agent_id="agent_test",
        scenario="技术",
        user_text="这个 bug 怎么查？",
        ideal_response="先看复现路径。",
        person_id=alice,
    )
    await repo.add(
        agent_id="agent_test",
        scenario="通用",
        user_text="你好",
        ideal_response="在。",
        person_id=None,
    )
    a_examples = await repo.list_for_prompt(agent_id="agent_test", person_id=alice)
    b_examples = await repo.list_for_prompt(agent_id="agent_test", person_id=bob)
    assert {e["scenario"] for e in a_examples} == {"技术", "通用"}
    assert {e["scenario"] for e in b_examples} == {"通用"}


async def test_disabled_examples_are_not_prompted(db, rt):
    repo = StyleExampleRepository(db)
    alice, _ = await _people(rt)
    row = await repo.add(
        agent_id="agent_test",
        scenario="技术",
        user_text="问题",
        ideal_response="回答",
        person_id=alice,
    )
    await repo.disable(row["example_id"])
    assert await repo.list_for_prompt(agent_id="agent_test", person_id=alice) == []
    listed = await repo.list(agent_id="agent_test", include_disabled=True)
    assert listed[0]["status"] == "DISABLED"


async def test_style_example_prompt_limit_is_three(db, rt):
    repo = StyleExampleRepository(db)
    alice, _ = await _people(rt)
    for index in range(6):
        await repo.add(
            agent_id="agent_test",
            scenario=f"s{index}",
            user_text=f"u{index}",
            ideal_response=f"r{index}",
            person_id=alice,
        )
    assert len(await repo.list_for_prompt(agent_id="agent_test", person_id=alice)) == 3
    assert len(
        await repo.list_for_prompt(
            agent_id="agent_test", person_id=alice, limit=999
        )
    ) == 3


@pytest.mark.parametrize(
    "scenario,user_text,ideal_response",
    [("", "u", "r"), ("s", "", "r"), ("s", "u", "  ")],
)
async def test_empty_style_example_is_refused(db, scenario, user_text, ideal_response):
    repo = StyleExampleRepository(db)
    with pytest.raises(StyleExampleError):
        await repo.add(
            agent_id="agent_test",
            scenario=scenario,
            user_text=user_text,
            ideal_response=ideal_response,
        )
