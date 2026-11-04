"""V0.5 — response prompt strategy and autonomous stopping.

The response call owns its length. There is no ``stop`` sentinel, no paragraph
split, no regex deletion and no post-generation polish call: DeepSeek finishes
at the natural end of the reply. These tests pin that contract directly.
"""

from __future__ import annotations

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult
from companion.prompts.response import (
    BASE_RESPONSE_PROMPT,
    RESPONSE_PROMPT_VERSION,
    SOCIAL_GENERATION_GUIDANCE,
    TASK_GENERATION_GUIDANCE,
    build_response_messages,
)
from companion.providers.deepseek_response import DeepSeekResponseProvider
from companion.providers.deepseek_client import ChatResponse


class RecordingClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def complete(self, messages, *, request_type, event_id=None, json_mode=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "request_type": request_type,
                "event_id": event_id,
                "json_mode": json_mode,
                "kwargs": kwargs,
            }
        )
        return ChatResponse(
            content=self.content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


def _turn(event_id, role, text, *, speaker="Alice", person_id="person_a"):
    return CognitiveTurn(
        event_id=event_id,
        role=role,
        speaker_person_id=None if role == "assistant" else person_id,
        speaker_label="Ava" if role == "assistant" else speaker,
        text=text,
    )


def _context(**updates):
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_now",
        "current_speaker_id": "person_a",
        "current_speaker_label": "Alice",
        "current_text": "今天有点累。",
        "current_addressee_hint": "agent",
        "turns": [
            _turn("evt_h1", "human", "今天有点累。"),
        ],
    }
    payload.update(updates)
    return CognitiveContext(**payload)


def _analysis():
    return CognitiveResult(
        event_id="evt_now",
        meaning="fake analysis",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        provider="fake",
    )


def _system(messages):
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


# -- no truncation, no stop sentinel ----------------------------------------


async def test_two_paragraph_reply_is_returned_unchanged():
    client = RecordingClient("这是第一段。\n\n这是第二段。")
    provider = DeepSeekResponseProvider(client)
    context = _context()

    text = await provider.generate(_event(), _analysis(), context)

    assert text == "这是第一段。\n\n这是第二段。"


async def test_multiline_task_reply_is_returned_unchanged():
    answer = "第一步：检查接线。\n\n第二步：查看串口配置。\n\n第三步：确认中断。\n"
    client = RecordingClient(answer)
    provider = DeepSeekResponseProvider(client)
    context = _context().model_copy(update={"reply_mode": "TASK"})

    text = await provider.generate(_event(), _analysis(), context)

    assert text == answer.strip()


async def test_no_stop_parameter_is_sent():
    client = RecordingClient("好的。")
    provider = DeepSeekResponseProvider(client)

    await provider.generate(_event(), _analysis(), _context())

    call = client.calls[0]
    assert "stop" not in call["kwargs"]
    assert call["kwargs"] == {}
    assert call["request_type"] == "response"


async def test_exactly_one_model_call_per_generation():
    client = RecordingClient("第一段。\n\n第二段。")
    provider = DeepSeekResponseProvider(client)

    await provider.generate(_event(), _analysis(), _context())

    assert len(client.calls) == 1


# -- prompt structure --------------------------------------------------------


def test_prompt_version_is_v11():
    assert RESPONSE_PROMPT_VERSION == "response-v11"


def test_base_prompt_has_generation_goal_and_stopping_rule():
    assert "【生成目标】" in BASE_RESPONSE_PROMPT
    assert "交流已经到达自然结束点，就停止" in BASE_RESPONSE_PROMPT
    assert "你不是客服" in BASE_RESPONSE_PROMPT
    assert "只输出回复正文" in BASE_RESPONSE_PROMPT


def test_social_is_the_default_generation_mode():
    system = _system(build_response_messages(_context(), _analysis()))

    assert SOCIAL_GENERATION_GUIDANCE in system
    assert "【当前交流模式：SOCIAL】" in system
    assert "约1–3句话" in system
    assert "完成一次自然回应后直接结束" in system
    assert TASK_GENERATION_GUIDANCE not in system


def test_task_mode_switches_guidance_and_allows_detail():
    context = _context().model_copy(update={"reply_mode": "TASK"})
    system = _system(build_response_messages(context, _analysis()))

    assert TASK_GENERATION_GUIDANCE in system
    assert "【当前交流模式：TASK】" in system
    assert "允许多段回答、代码、步骤" in system
    assert SOCIAL_GENERATION_GUIDANCE not in system


def test_social_guidance_is_not_a_hard_sentence_cap():
    assert "不是硬性限制" in SOCIAL_GENERATION_GUIDANCE
    assert "需要说得更多时可以继续" in SOCIAL_GENERATION_GUIDANCE


def test_style_examples_are_fictional_references_not_history():
    examples = [
        {
            "scenario": "日常闲聊",
            "user_text": "我今天差点迟到。",
            "ideal_response": "差点？那就是没迟到。先算你赢。",
        }
    ]
    context = _context(style_examples=examples)
    messages = build_response_messages(context, _analysis())
    system = _system(messages)

    assert "以下是管理员审核的虚构对话" in system
    assert "它们不是你与当前用户实际发生过的历史" in system
    assert "如果与正式 Persona Core 冲突" in system
    assert "场景：日常闲聊" in system
    assert "用户：我今天差点迟到。" in system
    assert "差点？那就是没迟到。先算你赢。" in system

    # A reference may never become an assistant turn or real user history.
    assistant_contents = [
        message["content"]
        for message in messages
        if message["role"] == "assistant"
    ]
    assert all("差点？那就是没迟到" not in content for content in assistant_contents)
    user_contents = [
        message["content"] for message in messages if message["role"] == "user"
    ]
    assert all("我今天差点迟到" not in content for content in user_contents)


def test_at_most_three_style_examples_reach_the_prompt():
    examples = [
        {
            "scenario": f"场景{i}",
            "user_text": f"用户{i}",
            "ideal_response": f"回应{i}",
        }
        for i in range(6)
    ]
    system = _system(
        build_response_messages(_context(style_examples=examples), _analysis())
    )

    assert "场景0" in system
    assert "场景2" in system
    assert "场景3" not in system


def test_no_style_section_when_no_examples_are_selected():
    system = _system(build_response_messages(_context(), _analysis()))
    assert "交流表达参考" not in system


def _event():
    from companion.domain.enums import EventSource
    from companion.domain.models import make_utterance_event

    return make_utterance_event(
        conversation_id="c1",
        text="今天有点累。",
        addressee_hint="agent",
        speaker_person_id="person_a",
        source=EventSource.TEST,
    )
