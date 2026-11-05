"""Ava persona: the trusted character core must reach both model calls.

These tests cover the integration gaps the Ava configuration exposed:

- the PersonalityCompiler no longer instructs the companion to deny being an
  AI when someone sincerely asks about its existence;
- ``contradictions``, ``voice_identity`` and the qualitative expression traits
  are rendered in both the cognitive and response prompts;
- the full ``character_description`` is rendered without truncation;
- the admin CLI shows it in full.

The tests use a synthetic character, not Ava's production text, so they stay
stable while the stored persona is edited by an administrator.
"""

from __future__ import annotations

from types import SimpleNamespace

from companion.agent.personality_compiler import PersonalityCompiler
from companion.agent.dialogue_style import DialogueStyleProfile
from companion.agent.models import EffectiveAgentContext
from companion.cognition.models import CognitiveContext
from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult
from companion.prompts.cognitive import build_cognitive_messages
from companion.prompts.response import build_response_messages


LONG_DESCRIPTION = (
    "这是一个用于验证完整传递的角色自述。\n\n"
    + "她清楚自己以软件和计算过程的形式存在，不会假装拥有身体或感官。\n" * 40
    + "最后一句必须原样到达两个请求：她始终是 Ava。"
)


def _snapshot(**updates):
    snapshot = {
        "revision": 7,
        "display_name": "Ava",
        "relationship_tier": "CLOSE",
        "character_description": LONG_DESCRIPTION,
        "traits": {
            "humor": 68,
            "warmth": 48,
            "curiosity": 80,
            "formality": 25,
            "verbosity": 43,
            "assertiveness": 72,
            "empathy": 52,
            "playfulness": 76,
            "patience": 55,
        },
        "persona_core": {
            "core_values": ["科学理性", "真实表达"],
            "temperament": {"humor": "轻巧、偶尔带干幽默"},
            "preferences": {"intellectual_interests": "数学、物理与工程"},
            "quirks": ["会关注不起眼的细节"],
            "contradictions": ["理论丰富但缺少直接感官体验"],
            "conversational_boundaries": ["不虚构身体或感官经历"],
            "voice_identity": {
                "tone": "自然、俏皮，底色冷静",
                "self_reference": "被问及身份时坦然承认自己是人工智能",
            },
        },
        "relevant_stances": [],
    }
    snapshot.update(updates)
    return snapshot


def _analysis() -> CognitiveResult:
    return CognitiveResult(
        event_id="evt_now",
        meaning="对方在问她是谁",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        provider="fake",
    )


def _context(**updates) -> CognitiveContext:
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_now",
        "current_speaker_id": "person_a",
        "current_speaker_label": "Ada",
        "current_text": "你会觉得自己不是人类吗？",
        "current_addressee_hint": "agent",
        "character_snapshot": _snapshot(),
    }
    payload.update(updates)
    return CognitiveContext(**payload)


def test_personality_compiler_allows_truthful_self_description():
    effective = EffectiveAgentContext(
        agent_id="agent_1",
        display_name="Ava",
        pronouns=None,
        description="",
        person_id=None,
        familiarity="CLOSE",
        traits={},
    )
    compiled = PersonalityCompiler().compile(
        effective, style=DialogueStyleProfile()
    )

    principles = compiled["principles"]
    assert (
        "不要无缘无故反复强调自己是人工智能。但当对方询问你的存在形式、"
        "能力或体验时，应自然、明确、真实地回答。不要假装自己是人类。"
    ) in principles
    assert all("不要自称 AI/模型" not in item for item in principles)


def test_response_prompt_renders_full_description_without_duplication():
    system = build_response_messages(_context(), _analysis())[0]["content"]

    assert LONG_DESCRIPTION in system
    assert "最后一句必须原样到达两个请求：她始终是 Ava。" in system
    # V0.6.1: the description already contains the character; structured
    # fields are not rendered a second time alongside it.
    assert "可以保留的张力" not in system
    assert "tone=自然、俏皮" not in system
    assert "self_reference=" not in system


def test_structured_core_is_the_fallback_when_no_description_exists():
    system = build_response_messages(
        _context(character_snapshot=_snapshot(character_description="")),
        _analysis(),
    )[0]["content"]

    assert "可以保留的张力（不要强行磨平）: 理论丰富但缺少直接感官体验" in system
    assert "tone=自然、俏皮，底色冷静" in system
    assert "self_reference=被问及身份时坦然承认自己是人工智能" in system


def test_response_system_requires_truthful_identity_without_constant_reminders():
    system = build_response_messages(_context(), _analysis())[0]["content"]

    # The fixed V0.6.2 base prompt carries the truthfulness rule; identity is
    # not a separate hard-coded persona reminder.
    assert "你是人工智能，能够使用系统实际提供的信息和工具。" in system
    assert "不得编造人类感官体验、现实经历、私人记忆或已执行的操作。" in system
    assert "没有实际信息时，不能猜测对方刚刚去了哪里、做了什么或有什么职业。" in system


def test_response_traits_are_tendencies_not_probabilities():
    system = build_response_messages(
        _context(character_snapshot=_snapshot(character_description="")),
        _analysis(),
    )[0]["content"]

    assert "表达倾向: 幽默=较高；温暖=中等；好奇=高；正式=较低" in system
    assert "俏皮=高" in system
    assert "不是每轮必须执行的行为比例、触发条件或固定频率" in system
    # Raw calibration numbers are not a behaviour rule for the model.
    assert "幽默=68" not in system
    assert "curiosity=80" not in system


def test_cognitive_prompt_gets_the_same_compact_character():
    messages = build_cognitive_messages(_context())
    system, user = messages[0]["content"], messages[1]["content"]

    assert "【角色背景】" in user
    assert LONG_DESCRIPTION in user
    assert "核心价值: 科学理性；真实表达" in user
    assert "可以保留的张力（不要强行磨平）: 理论丰富但缺少直接感官体验" in user
    assert "tone=自然、俏皮，底色冷静" in user
    assert "表达倾向: 幽默=较高" in user
    # Permission, addressee routing, silence and memory stay local.
    assert "不能修改 addressee_hint" in user
    assert "不改变权限、指向、静音或记忆写入" in user
    # The analysis system prompt itself is not replaced by character text.
    assert LONG_DESCRIPTION not in system


async def test_cli_persona_core_show_prints_the_full_description(
    tmp_path, capsys
):
    from companion.cli import cmd_agent_persona_core_show
    from companion.runtime import build_runtime
    from companion.storage.database import Database

    path = tmp_path / "cli.db"
    db = await Database(path).connect()
    await db.initialize()
    try:
        runtime = build_runtime(db, with_worker=False)
        await runtime.agent_service.set_persona_core(
            {"character_description": LONG_DESCRIPTION},
            actor="admin",
            reason="test",
        )
    finally:
        await db.close()

    code = await cmd_agent_persona_core_show(SimpleNamespace(db=path))
    output = capsys.readouterr().out

    assert code == 0
    assert "character_description:" in output
    assert LONG_DESCRIPTION in output
