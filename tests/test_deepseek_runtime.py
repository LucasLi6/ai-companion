"""Phase 7 — runtime wiring for the real DeepSeek providers.

Everything here uses a fake client: no network, no key, no cost. The tests
pin down the two safety properties that matter most at this layer:

- ``provider="deepseek"`` without a key raises instead of silently mocking.
- a real cognitive provider can never be paired with a mock response provider
  (or vice versa) unless the caller opts in explicitly.

They also cover the end-to-end pipeline: privacy gate -> context builder ->
cognitive provider -> policy -> response provider -> persistence.
"""

from __future__ import annotations

import json

import pytest

from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ReasonCode,
)
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse, DeepSeekConfigError
from companion.providers.deepseek_cognitive import DeepSeekCognitiveProvider
from companion.providers.deepseek_response import DeepSeekResponseProvider
from companion.providers.mock import MockCognitiveProvider, MockResponseProvider
from companion.runtime import build_runtime


class FakeDeepSeekClient:
    """Routes analysis to valid JSON and response to plain text."""

    def __init__(self, *, meaning="用户在打招呼", response_text="你好！很高兴见到你。"):
        self.analysis = json.dumps(
            {
                "meaning": meaning,
                "action": "RESPOND",
                "reason_code": "direct_agent_address",
                "referenced_memory_ids": [],
                "memory_candidates": [],
                "response_style": "warm",
            },
            ensure_ascii=False,
        )
        self.response_text = response_text
        self.calls: list[dict] = []

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
                "request_type": request_type,
                "messages": messages,
                "event_id": event_id,
                "json_mode": json_mode,
            }
        )
        content = self.analysis if request_type == "analysis" else self.response_text
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _person(rt, *, external_ai=True, persistence=True):
    return await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=persistence,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
    )


def _event(person_id, text="你好", *, to="agent", conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


# -- configuration safety ----------------------------------------------------


async def test_mock_is_default_and_unchanged(db):
    rt = build_runtime(db)
    assert rt.provider == "mock"
    assert rt.context_builder is None
    assert rt.privacy_gateway is None
    assert isinstance(rt.cognitive, MockCognitiveProvider)
    assert isinstance(rt.response, MockResponseProvider)


async def test_deepseek_wires_both_real_providers(db):
    client = FakeDeepSeekClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    assert rt.provider == "deepseek"
    assert isinstance(rt.cognitive, DeepSeekCognitiveProvider)
    assert isinstance(rt.response, DeepSeekResponseProvider)
    assert rt.context_builder is not None
    assert rt.privacy_gateway is not None


async def test_deepseek_without_key_raises_instead_of_mocking(db, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(DeepSeekConfigError):
        build_runtime(db, provider="deepseek", router_mode="legacy")


async def test_mixed_real_cognitive_mock_response_is_rejected(db):
    with pytest.raises(ValueError):
        build_runtime(
            db,
            provider="deepseek",
            deepseek_client=FakeDeepSeekClient(),
            response=MockResponseProvider(),
            router_mode="legacy",
)


async def test_mixed_mock_cognitive_real_response_is_rejected(db):
    with pytest.raises(ValueError):
        build_runtime(
            db,
            provider="deepseek",
            deepseek_client=FakeDeepSeekClient(),
            cognitive=MockCognitiveProvider(),
            router_mode="legacy",
)


async def test_mixed_allowed_only_when_explicit(db):
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=FakeDeepSeekClient(),
        cognitive=MockCognitiveProvider(),
        allow_mixed_providers=True,
        router_mode="legacy",
)
    assert isinstance(rt.cognitive, MockCognitiveProvider)
    assert isinstance(rt.response, DeepSeekResponseProvider)


async def test_unknown_provider_is_rejected(db):
    with pytest.raises(ValueError):
        build_runtime(db, provider="not-a-provider")


# -- end-to-end pipeline -----------------------------------------------------


async def test_deepseek_responds_through_the_real_pipeline(db):
    client = FakeDeepSeekClient(response_text="你好！很高兴见到你。")
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)
    event = _event(person.person_id)

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    assert report.reason_code == ReasonCode.DIRECT_AGENT_ADDRESS
    assert report.response_text == "你好！很高兴见到你。"
    # V0.6 Fast Path: a clearly addressed direct turn makes exactly one model
    # call (the response). No cognitive round-trip is needed for the reply.
    assert [c["request_type"] for c in client.calls] == ["response"]

    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.provider == "local_fast_path"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None and action.response_text == "你好！很高兴见到你。"


async def test_deepseek_denies_network_without_external_ai(db):
    client = FakeDeepSeekClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt, external_ai=False)
    event = _event(person.person_id)

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.WAIT
    assert report.reason_code == ReasonCode.PERMISSION_DENIED
    assert report.response_text is None
    assert client.calls == []  # nothing ever left the machine
    # The valid local cognition result is still retained.
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.reason_code == ReasonCode.PERMISSION_DENIED


async def test_deepseek_quiet_mode_suppresses_only_the_response(db):
    client = FakeDeepSeekClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)
    await rt.conversations.set_quiet_mode("c1", True)

    report = await rt.orchestrator.process_utterance(_event(person.person_id))

    assert report.action == CognitiveAction.WAIT
    assert report.response_text is None
    # Fast Path: quiet mode is a local rule, so nothing is sent at all. The
    # local decision record is still persisted for audit.
    assert [c["request_type"] for c in client.calls] == []
    analysis = await rt.repos.analysis.get_analysis(report.event_id)
    assert analysis is not None and analysis.provider == "local_fast_path"


async def test_deepseek_second_turn_sees_prior_history(db):
    client = FakeDeepSeekClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)

    await rt.orchestrator.process_utterance(
        _event(person.person_id, "我在学做饭")
    )
    await rt.orchestrator.process_utterance(
        _event(person.person_id, "今晚吃什么好？")
    )

    # Fast Path: calls are 0 response#1, 1 response#2. The second response
    # request sees the prior human turn and the assistant reply to it.
    second_prompt = "\n".join(
        m["content"] for m in client.calls[1]["messages"]
    )
    assert "我在学做饭" in second_prompt
    assert "你好！很高兴见到你。" in second_prompt  # the assistant turn


async def test_deepseek_non_persistence_calls_ai_but_writes_nothing(db):
    client = FakeDeepSeekClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt, external_ai=True, persistence=False)
    event = _event(person.person_id)

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    assert report.response_text == "你好！很高兴见到你。"
    assert len(client.calls) == 1  # Fast Path: response only
    # No durable record of this interaction.
    assert await rt.repos.utterance.get(event.event_id) is None
    assert await rt.repos.analysis.get_analysis(event.event_id) is None
    assert await rt.repos.analysis.get_action(event.event_id) is None
