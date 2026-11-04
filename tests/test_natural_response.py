"""V0.4 — prompt-level guarantees for natural, continuous dialogue.

These are offline assertions over the actual rendered messages. They do not
claim a model will always obey; they pin that the prompt gives it the scene,
the compiled style and the style references, while explicitly forbidding
fixed replies, forced questions, invented environment details and a second
"customer service" persona.
"""

from __future__ import annotations

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult
from companion.prompts.cognitive import (
    COGNITIVE_PROMPT_VERSION,
    SYSTEM_PROMPT as COGNITIVE_SYSTEM,
    build_cognitive_messages,
)
from companion.prompts.response import (
    RESPONSE_PROMPT_VERSION,
    SYSTEM_PROMPT as RESPONSE_SYSTEM,
    build_response_messages,
)


def _analysis():
    return CognitiveResult(
        event_id="evt_current",
        meaning="对方在继续聊旅行",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.FOLLOW_UP_TO_AGENT,
        response_style="warm",
        provider="fake",
    )


def _context(**overrides):
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_current",
        "current_speaker_id": "person_1",
        "current_speaker_label": "Alice",
        "current_text": "后来我还去了大阪",
        "turns": [
            CognitiveTurn(
                event_id="evt_prev",
                role="human",
                speaker_person_id="person_1",
                speaker_label="Alice",
                text="我去了日本",
            )
        ],
        "scene_state": {
            "topic": "旅行",
            "topic_stack": ["工作"],
            "mode": "CONVERSATION",
            "discourse_phase": "DEEPENING",
            "tone": "WARM",
            "open_loops": [{"topic": "工作", "summary": "工作的事还没讲完"}],
            "environment": {
                "location": "UNKNOWN",
                "other_people_present": "UNKNOWN",
                "provenance": "UNKNOWN",
            },
        },
        "dialogue_style": {
            "formality": "LOW",
            "directness": "BALANCED",
            "typical_length": "SHORT",
            "humor_tendency": "WHEN_NATURAL",
            "warmth_expression": "MODERATE",
            "questioning_style": "WHEN_USEFUL",
            "explicit_request": "current_request_short",
            "guidance": [
                "先回应眼前这句话，不要机械复述对方或背诵人设。",
                "不要每轮都用反问或追问推进对话。",
            ],
        },
        "personality": {
            "display_name": "Ava",
            "relationship_tier": "CLOSE",
            "principles": ["保持同一个人格：语气、价值观和边界跨轮次稳定，不随机换人设。"],
        },
        "style_examples": [
            {
                "scenario": "技术讨论",
                "user_text": "这个 bug 怎么查？",
                "ideal_response": "先看复现路径，再看最近改了什么。",
            }
        ],
        "pragmatic_guidance": [
            "当前话题：旅行。先顺着语境回应，不要突然换题。"
        ],
    }
    payload.update(overrides)
    return CognitiveContext(**payload)


def test_prompt_versions_are_bumped():
    assert COGNITIVE_PROMPT_VERSION == "cognitive-v8"
    assert RESPONSE_PROMPT_VERSION == "response-v7"


def test_cognitive_prompt_requests_scene_proposal_but_keeps_action_local():
    messages = build_cognitive_messages(_context())
    user = messages[1]["content"]
    assert '"scene_proposal"' in COGNITIVE_SYSTEM
    assert "【当前场景】" in user
    assert "话题: 旅行" in user
    assert "话题栈: 工作" in user
    assert "未完成: 工作 / 工作的事还没讲完" in user
    assert "场景判断只影响理解与措辞，不改变 action" in COGNITIVE_SYSTEM
    assert "不要输出 scene_id、loop_id、person_id、event_id" in COGNITIVE_SYSTEM
    assert "不能根据语气、时间或想象推断房间、地点、旁人" in COGNITIVE_SYSTEM


def test_cognitive_prompt_keeps_nonresponse_and_memory_rules():
    assert "WAIT 不是“以后一定补答”" in COGNITIVE_SYSTEM
    assert "上一轮沉默不会自动授权当前回复" in COGNITIVE_SYSTEM
    assert "memory_candidates 只记录当前说话者明确、持久、值得记住的陈述" in COGNITIVE_SYSTEM
    assert "不要决定权限、静音开关或是否保存" in COGNITIVE_SYSTEM


def test_response_prompt_renders_scene_and_style_as_guidance():
    user = build_response_messages(_context(), _analysis())[1]["content"]
    assert "【人格准则】" in user
    assert "【当前场景】" in user
    assert "【对话风格】" in user
    assert "【对话衔接提醒】" in user
    assert "【风格参考】" in user
    assert "话题: 旅行" in user
    assert "本轮明确要求: current_request_short" in user


def test_response_prompt_puts_persona_before_scene_and_emotion():
    ctx = _context(
        emotion_expression={
            "source_emotion": "CURIOSITY",
            "intensity": 0.4,
            "questioning_adjustment": 0.2,
        }
    )
    user = build_response_messages(ctx, _analysis())[1]["content"]
    assert (
        user.index("【角色核心】")
        < user.index("【当前场景】")
        < user.index("【情绪影响】")
    )


def test_response_prompt_does_not_forward_cozy_style_or_direction():
    user = build_response_messages(_context(), _analysis())[1]["content"]
    # ``response_style`` is a legacy label; it must not become a cozy script.
    assert "想要的回应风格" not in user
    assert "可考虑的方向" not in user


def test_response_prompt_forbids_fixed_replies_and_category_templates():
    assert "没有固定回复" in RESPONSE_SYSTEM
    assert "不要按ANSWER/SUPPORT/PLAYFUL 等分类套模板" in RESPONSE_SYSTEM
    user = build_response_messages(_context(), _analysis())[1]["content"]
    assert "不是真实历史，不是固定回复" in user


def test_response_prompt_forbids_forced_followup_questions():
    assert "不要每轮以问题结尾" in RESPONSE_SYSTEM
    assert "安慰、追问、总结都不是默认动作" in RESPONSE_SYSTEM
    user = build_response_messages(_context(), _analysis())[1]["content"]
    assert "不要每轮都用反问或追问推进对话" in user


def test_response_prompt_forbids_unsupported_environment_claims():
    assert "没有证据的物理环境" in RESPONSE_SYSTEM
    assert "不要补全" in RESPONSE_SYSTEM
    user = build_response_messages(_context(), _analysis())[1]["content"]
    assert "没有证据的环境信息保持未知，不要自行补全" in user


def test_response_prompt_keeps_personality_stable_without_self_intro():
    assert "保持同一个人" in RESPONSE_SYSTEM
    assert "不要每轮自我介绍" in RESPONSE_SYSTEM
    assert "可以有不同意见、具体偏好" in RESPONSE_SYSTEM


def test_response_prompt_keeps_truthful_memory_and_silence_rules():
    assert "PENDING_REVIEW 必须说还需要确认" in RESPONSE_SYSTEM
    assert "不要编造心理活动" in RESPONSE_SYSTEM
    assert "只有对方明确追问过去的沉默时" in RESPONSE_SYSTEM


def test_response_prompt_has_no_hard_coded_persona_lines():
    """Ava is not encoded here: the name comes only from trusted context."""
    assert "Ava" not in RESPONSE_SYSTEM
    assert "日本好玩吗" not in RESPONSE_SYSTEM
    assert "我喜欢" not in RESPONSE_SYSTEM


def test_response_prompt_is_plain_text_only():
    assert "只输出回复正文" in RESPONSE_SYSTEM
    assert "不要 JSON" in RESPONSE_SYSTEM
    assert "不要 Markdown" in RESPONSE_SYSTEM


def test_scene_unknown_is_rendered_as_unknown_not_invented():
    ctx = _context(
        scene_state={
            "topic": "UNKNOWN",
            "environment": {
                "location": "UNKNOWN",
                "other_people_present": "UNKNOWN",
                "provenance": "UNKNOWN",
            },
        }
    )
    user = build_response_messages(ctx, _analysis())[1]["content"]
    assert "地点=UNKNOWN" in user
    assert "旁人在场=UNKNOWN" in user
    assert "家里" not in user
