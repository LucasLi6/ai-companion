"""V0.5 — the response prompt is one unified speaking-style contract.

The response call owns its length: there is no ``stop`` sentinel, no paragraph
split, no regex deletion and no post-generation polish call. V0.5 removed the
SOCIAL/TASK mode switch and the per-turn style-reference block, so the system
message now carries exactly one 【说话方式】 contract regardless of topic. These
tests pin that contract directly.
"""

from __future__ import annotations

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult
from companion.prompts.response import (
    BASE_RESPONSE_PROMPT,
    RESPONSE_PROMPT_VERSION,
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

    text = await provider.generate(_event(), _analysis(), _context())

    assert text == "这是第一段。\n\n这是第二段。"


async def test_multiline_reply_is_returned_unchanged():
    answer = "第一步：检查接线。\n\n第二步：查看串口配置。\n\n第三步：确认中断。\n"
    client = RecordingClient(answer)
    provider = DeepSeekResponseProvider(client)

    text = await provider.generate(_event(), _analysis(), _context())

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


def test_prompt_version_is_v12():
    assert RESPONSE_PROMPT_VERSION == "response-v12"


def test_base_prompt_has_generation_goal_and_stopping_rule():
    assert "【生成目标】" in BASE_RESPONSE_PROMPT
    assert "交流已经到达自然结束点，就停止" in BASE_RESPONSE_PROMPT
    assert "你不是客服" in BASE_RESPONSE_PROMPT
    assert "只输出回复正文" in BASE_RESPONSE_PROMPT


def test_speaking_style_is_one_unified_block_not_mode_switched():
    system = _system(build_response_messages(_context(), _analysis()))

    assert "【说话方式】" in system
    assert "像日常聊天一样说话" in system
    assert "默认简短" in system
    # No per-turn SOCIAL/TASK strategy is injected any more.
    assert "【当前交流模式：SOCIAL】" not in system
    assert "【当前交流模式：TASK】" not in system
    assert "当前交流模式" not in system


def test_shortness_is_guidance_not_a_hard_cap():
    assert "这只是表达倾向，不是硬性限制" in BASE_RESPONSE_PROMPT
    # Task detail lives in the same block rather than a second guidance string.
    assert "允许多段回答" in BASE_RESPONSE_PROMPT
    assert "详细解释" in BASE_RESPONSE_PROMPT


def test_style_examples_are_not_injected():
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

    assert "交流表达参考" not in system
    assert "以下是管理员审核的虚构对话" not in system
    assert "差点？那就是没迟到" not in system
    # The old reference can never appear as history either.
    assert all("差点？那就是没迟到" not in m["content"] for m in messages)


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
