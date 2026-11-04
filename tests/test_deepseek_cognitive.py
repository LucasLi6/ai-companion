"""Phase 5 — DeepSeek cognitive provider (fake client, no network).

Covers: strict JSON parsing, trust-boundary sanitisation (ids come from the
program, not the model), memory-id validation, the explicit-human guard, the
one-repair-retry policy (opt-in, off by default), and the privacy gate
short-circuit.
"""

from __future__ import annotations

import json

import pytest

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.cognition.privacy import ExternalAIAuthorization
from companion.domain.enums import (
    AddresseeHint,
    CognitiveAction,
    EventSource,
    MemoryOperation,
    ReasonCode,
)
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.providers.deepseek_cognitive import (
    CognitiveValidationError,
    DeepSeekCognitiveProvider,
)


# -- fakes ------------------------------------------------------------------


class FakeDeepSeekClient:
    def __init__(self, contents):
        self._contents = list(contents)
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
        item = self._contents.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResponse(
            content=item,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


class StubGateway:
    def __init__(self, allowed: bool):
        self.allowed = allowed

    async def authorize_event(self, person_id):
        return ExternalAIAuthorization(
            self.allowed,
            reason="authorized" if self.allowed else "external_ai_not_granted",
        )


def _event(addressee="agent", speaker_id="person_1", text="你好，今天怎么样？"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint=addressee,
        speaker_person_id=speaker_id,
        source=EventSource.TEST,
    )


def _context(event, turns=None, memories=None):
    return CognitiveContext(
        conversation_id=event.conversation_id,
        current_event_id=event.event_id,
        current_speaker_id=event.speaker_person_id,
        current_speaker_label="Alice",
        current_text=event.text,
        current_addressee_hint=event.addressee_hint.value,
        turns=turns or [],
        memories=memories
        if memories is not None
        else [{"memory_id": "mem_1", "kind": "PERSONAL_FACT", "content": "用户喜欢猫"}],
    )


def _valid(**overrides):
    payload = {
        "meaning": "用户在问好",
        "topic": "greeting",
        "addressee": "agent",
        "action": "RESPOND",
        "reason_code": "direct_agent_address",
        "referenced_memory_ids": ["mem_1"],
        "memory_candidates": [{"content": "用户叫小明", "operation": "ADD"}],
        "response_style": "warm",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


# -- happy path -------------------------------------------------------------


async def test_analyze_returns_trusted_result():
    client = FakeDeepSeekClient([_valid()])
    provider = DeepSeekCognitiveProvider(client)
    event = _event()
    result = await provider.analyze(event, _context(event), [])

    assert result.event_id == event.event_id
    assert result.provider == "deepseek"
    assert result.action == CognitiveAction.RESPOND
    assert result.reason_code == ReasonCode.DIRECT_AGENT_ADDRESS
    assert result.referenced_memory_ids == ["mem_1"]
    assert result.response_style == "warm"
    assert len(result.memory_candidates) == 1
    cand = result.memory_candidates[0]
    assert cand.source_event_id == event.event_id
    assert cand.subject_person_id == event.speaker_person_id
    assert cand.operation == MemoryOperation.ADD
    assert client.calls[0]["request_type"] == "analysis"
    assert client.calls[0]["json_mode"] is True


async def test_analyze_strips_markdown_fence():
    client = FakeDeepSeekClient(["```json\n" + _valid() + "\n```"])
    provider = DeepSeekCognitiveProvider(client)
    event = _event()
    result = await provider.analyze(event, _context(event), [])
    assert result.action == CognitiveAction.RESPOND


async def test_analyze_drops_fabricated_memory_ids():
    client = FakeDeepSeekClient(
        [_valid(referenced_memory_ids=["mem_1", "mem_fake"])]
    )
    provider = DeepSeekCognitiveProvider(client)
    event = _event()
    result = await provider.analyze(event, _context(event), [])
    assert result.referenced_memory_ids == ["mem_1"]


async def test_model_ids_are_ignored():
    payload = json.loads(_valid())
    payload["event_id"] = "evt_evil"
    payload["subject_person_id"] = "person_evil"
    payload["memory_candidates"][0]["candidate_id"] = "cand_evil"
    payload["memory_candidates"][0]["source_event_id"] = "evt_evil"
    payload["memory_candidates"][0]["subject_person_id"] = "person_evil"
    client = FakeDeepSeekClient([json.dumps(payload, ensure_ascii=False)])
    provider = DeepSeekCognitiveProvider(client)
    event = _event(speaker_id="person_real")
    result = await provider.analyze(event, _context(event), [])

    assert result.event_id == event.event_id
    cand = result.memory_candidates[0]
    assert cand.candidate_id != "cand_evil"
    assert cand.source_event_id == event.event_id
    assert cand.subject_person_id == "person_real"


# -- explicit human guard ---------------------------------------------------


async def test_explicit_human_addressee_never_responds():
    client = FakeDeepSeekClient([_valid(action="RESPOND")])
    provider = DeepSeekCognitiveProvider(client)
    event = _event(addressee="human")
    result = await provider.analyze(event, _context(event), [])
    assert result.action == CognitiveAction.IGNORE
    assert result.reason_code == ReasonCode.DIRECT_HUMAN_ADDRESS


# -- repair retry (explicit opt-in since V0.4.3) ----------------------------


async def test_repair_is_off_by_default_one_cognitive_call_per_turn():
    client = FakeDeepSeekClient(["not json at all", _valid()])
    provider = DeepSeekCognitiveProvider(client)
    event = _event()
    with pytest.raises(CognitiveValidationError):
        await provider.analyze(event, _context(event), [])
    assert len(client.calls) == 1


async def test_repair_retry_succeeds_when_explicitly_enabled():
    client = FakeDeepSeekClient(["not json at all", _valid()])
    provider = DeepSeekCognitiveProvider(client, repair_enabled=True)
    event = _event()
    result = await provider.analyze(event, _context(event), [])
    assert result.action == CognitiveAction.RESPOND
    assert len(client.calls) == 2
    # repair turn asks for corrected JSON
    repair_messages = client.calls[1]["messages"]
    assert "重新输出" in repair_messages[-1]["content"]


async def test_repair_retry_fails_after_one_attempt():
    client = FakeDeepSeekClient(["bad", "still bad"])
    provider = DeepSeekCognitiveProvider(client, repair_enabled=True)
    event = _event()
    with pytest.raises(CognitiveValidationError):
        await provider.analyze(event, _context(event), [])
    assert len(client.calls) == 2  # exactly one repair


async def test_invalid_action_triggers_repair_when_enabled():
    client = FakeDeepSeekClient([_valid(action="SHOUT"), _valid()])
    provider = DeepSeekCognitiveProvider(client, repair_enabled=True)
    event = _event()
    result = await provider.analyze(event, _context(event), [])
    assert result.action == CognitiveAction.RESPOND
    assert len(client.calls) == 2


# -- privacy gate -----------------------------------------------------------


async def test_permission_denied_makes_no_network_call():
    client = FakeDeepSeekClient([_valid()])
    provider = DeepSeekCognitiveProvider(client, privacy_gateway=StubGateway(False))
    event = _event()
    result = await provider.analyze(event, _context(event), [])
    assert client.calls == []
    assert result.action == CognitiveAction.WAIT
    assert result.reason_code == ReasonCode.PERMISSION_DENIED
    assert result.provider == "deepseek"


async def test_authorized_gate_allows_network_call():
    client = FakeDeepSeekClient([_valid()])
    provider = DeepSeekCognitiveProvider(client, privacy_gateway=StubGateway(True))
    event = _event()
    await provider.analyze(event, _context(event), [])
    assert len(client.calls) == 1


# -- prompt assembly --------------------------------------------------------


async def test_prompt_includes_history_and_memories():
    turns = [
        CognitiveTurn(
            event_id="evt_old",
            role="human",
            speaker_person_id="person_1",
            speaker_label="Alice",
            addressee_hint="agent",
            text="我昨天买了咖啡豆",
        ),
        CognitiveTurn(
            event_id="evt_old",
            role="assistant",
            speaker_label="assistant",
            text="听起来不错！",
        ),
    ]
    client = FakeDeepSeekClient([_valid()])
    provider = DeepSeekCognitiveProvider(client)
    event = _event(text="你觉得哪种好？")
    await provider.analyze(event, _context(event, turns=turns), [])

    user_msg = client.calls[0]["messages"][1]["content"]
    assert "我昨天买了咖啡豆" in user_msg
    assert "听起来不错！" in user_msg
    assert "mem_1" in user_msg
    assert "你觉得哪种好？" in user_msg
    assert "Alice" in user_msg
