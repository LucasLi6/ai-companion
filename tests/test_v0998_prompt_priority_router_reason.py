"""V0.9.9.8: one priority frame for the Speaker, no router prose in it.

The Speaker prompt keeps every original section and sentence unless V0.9.9.31
deliberately retired it; the sections are reordered so understanding comes
before expression, and one 【交流的核心】 section states how facts, persona,
relationship, current intent and style are weighed. The router still analyses
everything, but its natural-language ``reason`` stays in the decision record
and audit row instead of being handed to the Speaker as guidance for what to
say.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.agent.dialogue_style import DialogueStyleProfile
from companion.agent.models import EffectiveAgentContext
from companion.agent.personality_compiler import PersonalityCompiler
from companion.cognition.models import CognitiveContext
from companion.prompts.proactive import (
    PROACTIVE_MODE_BLOCK,
    PROACTIVE_MODE_BLOCK_LONG_IDLE,
)
from companion.prompts.response import (
    BASE_RESPONSE_PROMPT,
    RESPONSE_PROMPT_VERSION,
    build_proactive_response_messages,
    build_response_messages,
)
from companion.routing.context import build_speaker_context

ORIGINAL_V18 = (
    Path(__file__).parent / "data" / "response_prompt_v18.txt"
).read_text()

SECTION_ORDER = (
    "【交流的核心】",
    "【交流连续性】",
    "【相处方式】",
    "【热情与个性】",
    "【说话方式】",
    "【信息消化】",
    "【理性与真实】",
    "【交流边界】",
)

# V0.9.9.31 deliberately replaced the fixed-length wording. These lines must be
# gone, and their replacements are asserted below; every other V18 line is
# still required to survive verbatim.
RETIRED_V21_LINES = frozenset(
    {
        "默认一句话。",
        "先说最想说、最值得说的那一点，说完就停。",
        "普通交流默认简短；复杂任务按实际需要调整长度。",
        "则根据任务需要充分展开，不能为了保持一句话而遗漏必要内容。",
        "可以偶尔俏皮、小得意、嘴硬或者有些任性。",
        # V0.9.9.31 also retires the repeated "stay brief / skip explanation"
        # wording; the same intent now lives once in 【交流的核心】 and
        # 【说话方式】 instead of being restated in three sections.
        "不需要每次都认真解释，也不需要永远表现得体贴周到。",
        "有时候一句随口的反应、一个奇怪的联想，甚至半句话，就足够了。",
        "不要默认补充背景、解释原因、总结意义或提出建议。",
        "也不需要每次都正面回答所有细节。",
    }
)


def _context(**updates) -> CognitiveContext:
    payload = dict(
        conversation_id="c1",
        current_event_id="evt_now",
        current_speaker_id="p1",
        current_speaker_label="Alice",
        current_text="那你刚才说的那个呢？",
        current_addressee_hint="agent",
    )
    payload.update(updates)
    return CognitiveContext(**payload)


def _background(messages):
    import json

    for message in messages:
        content = message.get("content") or ""
        marker = content.find('"schema": "companion.background.v1"')
        if marker < 0:
            marker = content.find('"schema":"companion.background.v1"')
        if marker >= 0:
            start = content.rfind("{", 0, marker)
            return json.loads(content[start:])
    raise AssertionError("no background message in the response request")


# -- structure ---------------------------------------------------------------


def test_prompt_version_reflects_the_reorganisation():
    assert RESPONSE_PROMPT_VERSION == "response-v21"


def test_sections_follow_understanding_then_expression_order():
    positions = [BASE_RESPONSE_PROMPT.index(name) for name in SECTION_ORDER]
    assert positions == sorted(positions)
    # The identity frame and the shared principle come before every section.
    intro = BASE_RESPONSE_PROMPT.index("你是 Ava。")
    assert intro < positions[0]


def test_every_original_line_is_still_present():
    """A section may move, but its wording must not be summarised away."""
    missing = [
        line
        for line in ORIGINAL_V18.splitlines()
        if line.strip()
        and line not in RETIRED_V21_LINES
        and line not in BASE_RESPONSE_PROMPT
    ]
    assert missing == []
    # A retired line must really be gone; otherwise the exclusion list would
    # silently hide a forgotten duplicate.
    for line in RETIRED_V21_LINES:
        assert line not in BASE_RESPONSE_PROMPT


def test_fixed_length_rules_are_replaced_by_natural_length_guidance():
    for phrase in (
        "日常交流倾向于简洁自然，但不固定回复长度。",
        "交流值得深入时，可以自然展开观点、理由与思考",
        "轻松闲聊时，一句随口的回应也可以足够。",
        "不为了简短而省略真正有意义的内容",
        "回复长度取决于你实际想表达的内容，而不是固定的字数或句数。",
        "不能为了追求简短而遗漏必要内容。",
    ):
        assert phrase in BASE_RESPONSE_PROMPT


def test_original_blocks_are_not_only_lines_but_coherent_sections():
    for header in SECTION_ORDER[1:]:
        assert BASE_RESPONSE_PROMPT.count(header) == 1
    # The new section is added once, after the intro.
    assert BASE_RESPONSE_PROMPT.count("【交流的核心】") == 1
    assert BASE_RESPONSE_PROMPT.index("【交流的核心】") < BASE_RESPONSE_PROMPT.index(
        "【交流连续性】"
    )


def test_shared_principle_covers_understanding_priority_and_non_obedience():
    section = BASE_RESPONSE_PROMPT.split("【交流的核心】", 1)[1]
    section = section.split("【交流连续性】", 1)[0]
    for phrase in (
        "结合对方当前的话与最近的真实对话，理解这句话的含义、指代、交流意图",
        "注意区分不同说话者、他们各自的行为和属性",
        "追问、纠正、质疑、开玩笑或突然转向其他事情",
        "重新理解实际发生的对话",
        "并不意味着必须顺从",
        "真实对话和有来源的事实是理解当前情况的依据；推断、摘要和场景判断只是辅助",
        "稳定人格决定你是谁",
        "当前关系、情绪和场景影响你的态度与反应，但不能改变事实或人物身份",
        "说话风格是自然表达的倾向，不应妨碍正确理解或完成需要准确性的交流",
        "不需要显式执行、提及或解释这些原则",
    ):
        assert phrase in section


def test_prompt_does_not_add_a_second_priority_list_or_answer_flow():
    # PersonalityCompiler's expression_precedence is internal compiled data; it
    # is not rendered, so there is no competing priority block in the prompt.
    for item in (
        "expression_precedence",
        "安全与事实",
        "角色核心（价值观、偏好、边界）",
        "当前语境与交流意图",
    ):
        assert item not in BASE_RESPONSE_PROMPT
    # No fixed answering procedure was introduced.
    for banned in (
        "首先正面回答问题",
        "固定回答流程",
        "每次都必须先",
    ):
        assert banned not in BASE_RESPONSE_PROMPT


def test_compiler_precedence_remains_internal_and_compatible_with_the_prompt():
    compiler = PersonalityCompiler()
    effective = EffectiveAgentContext(
        agent_id="ava",
        display_name="Ava",
        pronouns=None,
        description=None,
        person_id="p1",
        familiarity="FRIEND",
    )
    compiled = compiler.compile(
        effective, style=DialogueStyleProfile(), current_text="hi"
    )
    assert compiled["expression_precedence"][0] == "安全与事实"
    assert compiled["expression_precedence"][1] == "角色核心（价值观、偏好、边界）"
    # It stays a compiled field; the Speaker prompt states the same order in
    # its own words instead of importing this list a second time.
    assert "expression_precedence" not in BASE_RESPONSE_PROMPT


def test_original_behavioral_capabilities_survive():
    for phrase in (
        "你可以接梗、轻轻吐槽、反问、卖关子，偶尔有点嘴硬。",
        "不追求每句话都像书面语一样完整。",
        "不能为了追求简短而遗漏必要内容。",
        "幽默和嘴硬是自然的表达倾向，不是每轮需要执行的任务。",
        "对方已经拒绝某种玩笑后，不要换一种措辞继续重复。",
        "过去的记忆有助于理解对方，但不应限制当前话题。",
        "不需要每个话题都与对方以前的项目、兴趣或经历相关。",
        "例如25.4°C可以自然说成“二十五度左右”。",
        "如果对方只是随口询问天气，通常不需要同时报出温度、湿度、风速和降水概率。",
        "如果对方询问一件新闻，通常先说发生了什么",
        "如果他只说“那这个呢”“所以呢”“后来怎么样了”，先结合真实上下文理解指代。",
        "如果新问题与刚才明显不搭，你可以自然地感到一瞬间的意外或疑惑。",
        "你知道自己是人工智能，不会编造感官体验、身体经历或没有发生过的事情。",
        "不为了追求口语感而牺牲事实正确性。",
    ):
        assert phrase in BASE_RESPONSE_PROMPT


# -- router -> speaker projection -------------------------------------------


class _Enum:
    def __init__(self, value):
        self.value = value


class FakeRoute:
    action = _Enum("RESPOND")
    reply_necessity = _Enum("USEFUL")
    addressee_assessment = _Enum("AGENT")
    turn_state = _Enum("CONTINUING")
    processing_intents = [_Enum("QUESTION"), _Enum("CHITCHAT")]
    tool_requests = []
    evidence = []
    reason = "用户在追问上一轮提到的工具结果，应该先解释再回答。"


def test_speaker_context_drops_the_router_reason_but_keeps_structured_fields():
    speaker = build_speaker_context(
        SimpleNamespace(),
        None,
        FakeRoute(),
        [],
        base=_context(),
    )
    route = speaker.route_context
    assert "reason" not in route
    assert route["action"] == "RESPOND"
    assert route["reply_necessity"] == "USEFUL"
    assert route["addressee_assessment"] == "AGENT"
    assert route["turn_state"] == "CONTINUING"
    assert route["processing_intents"] == ["QUESTION", "CHITCHAT"]
    # The original decision object is untouched and still carries its prose.
    assert FakeRoute.reason.startswith("用户在追问")


def test_router_reason_never_appears_in_the_final_speaker_messages():
    reason = "内部说明：这句话是在纠正你，你应该承认错误并继续解释。"
    context = _context(
        route_context={
            "kind": "MODEL_JUDGMENT",
            "action": "RESPOND",
            "reply_necessity": "USEFUL",
            "addressee_assessment": "AGENT",
            "turn_state": "CONTINUING",
            "processing_intents": ["CORRECTION"],
            "reason": reason,
            "tool_requests": [],
        }
    )
    messages = build_response_messages(context, None)
    for message in messages:
        assert reason not in (message.get("content") or "")
    assert "reason" not in _background(messages)["route"]


def test_structured_route_fields_still_reach_the_speaker():
    context = _context(
        route_context={
            "kind": "MODEL_JUDGMENT",
            "action": "RESPOND",
            "reply_necessity": "USEFUL",
            "addressee_assessment": "AGENT",
            "turn_state": "CONTINUING",
            "processing_intents": ["QUESTION"],
            "reason": "kept internally only",
            "tool_requests": [
                {"tool_name": "runtime.current_time", "purpose": "time"}
            ],
        }
    )
    messages = build_response_messages(context, None)
    route = _background(messages)["route"]
    assert route["action"] == "RESPOND"
    assert route["processing_intents"] == ["QUESTION"]
    assert route["tools_requested"] == ["runtime.current_time"]
    assert "reason" not in route


def test_real_history_persona_and_runtime_are_still_delivered():
    context = _context(
        turns=[
            {
                "event_id": "evt_1",
                "role": "user",
                "text": "不是这个意思，我说的是另一个文件",
                "speaker_person_id": "p1",
                "speaker_label": "Alice",
            },
            {"event_id": "evt_2", "role": "assistant", "text": "哦，你说那个。"},
        ],
        personality={
            "display_name": "Ava",
            "identity": {"character_description": "Ava 有稳定的性格和自己的判断。"},
        },
    )
    messages = build_response_messages(context, None)
    system = messages[0]["content"]
    assert "Ava 有稳定的性格和自己的判断。" in system
    assert any(
        message["role"] == "assistant" and message["content"] == "哦，你说那个。"
        for message in messages
    )
    assert messages[-1]["role"] == "user"
    assert "不是这个意思" in messages[1]["content"]


# -- proactive generation ----------------------------------------------------


def test_proactive_prompt_uses_the_same_persona_without_router_reason():
    context = _context(
        route_context={"action": "RESPOND", "reason": "router-only explanation"}
    )
    for mode, block in (
        ("IMMEDIATE", PROACTIVE_MODE_BLOCK),
        ("LONG_IDLE", PROACTIVE_MODE_BLOCK_LONG_IDLE),
    ):
        messages = build_proactive_response_messages(context, None, mode=mode)
        system = messages[0]["content"]
        assert BASE_RESPONSE_PROMPT in system
        assert block in system
        assert "router-only explanation" not in system


@pytest.mark.parametrize(
    "capability",
    [
        "指代",
        "纠正",
        "开玩笑",
        "突然换个方向",
        "充分展开",
        "精确数字",
    ],
)
def test_conflict_and_regression_scenarios_have_explicit_prompt_support(capability):
    assert capability in BASE_RESPONSE_PROMPT
