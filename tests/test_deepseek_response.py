"""Phase 6 — DeepSeek response provider (fake client, no network)."""

from __future__ import annotations

import pytest

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.cognition.privacy import ExternalAIAuthorization
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ReasonCode,
)
from companion.domain.models import CognitiveResult, make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.providers.deepseek_response import (
    DeepSeekResponseProvider,
    ResponsePermissionError,
)


class FakeDeepSeekClient:
    def __init__(self, content):
        self._content = content
        self.calls = []

    async def complete(
        self,
        messages,
        *,
        request_type,
        event_id=None,
        json_mode=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "request_type": request_type,
                "event_id": event_id,
                "json_mode": json_mode,
            }
        )
        return ChatResponse(
            content=self._content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


class StubGateway:
    def __init__(self, allowed):
        self.allowed = allowed

    async def authorize_event(self, person_id):
        return ExternalAIAuthorization(
            self.allowed,
            reason="authorized" if self.allowed else "external_ai_not_granted",
        )


def _event(text="你好"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id="person_1",
        source=EventSource.TEST,
    )


def _analysis(event, meaning="用户在问好", style="warm"):
    return CognitiveResult(
        event_id=event.event_id,
        meaning=meaning,
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        response_style=style,
        provider="deepseek",
    )


def _context(event, turns=None, memories=None):
    return CognitiveContext(
        conversation_id=event.conversation_id,
        current_event_id=event.event_id,
        current_speaker_id=event.speaker_person_id,
        current_speaker_label="Alice",
        current_text=event.text,
        current_addressee_hint="agent",
        turns=turns or [],
        memories=memories or [],
    )


async def test_generate_returns_text():
    client = FakeDeepSeekClient("你好！很高兴见到你。")
    provider = DeepSeekResponseProvider(client)
    event = _event()
    text = await provider.generate(event, _analysis(event), _context(event))
    assert text == "你好！很高兴见到你。"


async def test_generate_uses_response_request_type():
    client = FakeDeepSeekClient("好的")
    provider = DeepSeekResponseProvider(client)
    event = _event()
    await provider.generate(event, _analysis(event), _context(event))
    call = client.calls[0]
    assert call["request_type"] == "response"
    assert call["json_mode"] is False
    assert call["event_id"] == event.event_id


async def test_generate_strips_whitespace():
    client = FakeDeepSeekClient("   收到。\n")
    provider = DeepSeekResponseProvider(client)
    event = _event()
    text = await provider.generate(event, _analysis(event), _context(event))
    assert text == "收到。"


async def test_prompt_includes_context_and_analysis():
    turns = [
        CognitiveTurn(
            event_id="evt_old",
            role="human",
            speaker_person_id="person_1",
            speaker_label="Alice",
            addressee_hint="agent",
            text="我在学做饭",
        ),
        CognitiveTurn(
            event_id="evt_old",
            role="assistant",
            speaker_label="assistant",
            text="加油！",
        ),
    ]
    memories = [{"memory_id": "mem_1", "kind": "PERSONAL_FACT", "content": "Alice 喜欢川菜"}]
    client = FakeDeepSeekClient("好呀")
    provider = DeepSeekResponseProvider(client)
    event = _event(text="今晚吃什么好？")
    await provider.generate(
        event,
        _analysis(event, meaning="用户在征求晚饭建议"),
        _context(event, turns=turns, memories=memories),
    )
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "我在学做饭" in user_msg
    assert "加油！" in user_msg
    assert "Alice 喜欢川菜" in user_msg
    assert "今晚吃什么好？" in user_msg
    assert "用户在征求晚饭建议" in user_msg
    # memory ids are not shown to the response model
    assert "mem_1" not in user_msg


async def test_permission_denied_raises():
    client = FakeDeepSeekClient("should not happen")
    provider = DeepSeekResponseProvider(client, privacy_gateway=StubGateway(False))
    event = _event()
    with pytest.raises(ResponsePermissionError):
        await provider.generate(event, _analysis(event), _context(event))
    assert client.calls == []


async def test_authorized_gate_allows_call():
    client = FakeDeepSeekClient("你好")
    provider = DeepSeekResponseProvider(client, privacy_gateway=StubGateway(True))
    event = _event()
    await provider.generate(event, _analysis(event), _context(event))
    assert len(client.calls) == 1
