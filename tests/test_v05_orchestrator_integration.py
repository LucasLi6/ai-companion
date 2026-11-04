"""V0.5 — the real pipeline makes exactly two calls and injects no extra routing.

These tests drive the actual Orchestrator over the real database and the real
privacy-filtered ContextBuilder, with fake providers (no network). They pin:

- the default generation chain no longer classifies SOCIAL/TASK or retrieves
  style references per turn;
- the cognitive call happens exactly once and never sees generation guidance;
- a multi-paragraph reply survives the whole pipeline unchanged;
- the real DeepSeek response request carries the one unified 【说话方式】 block.
"""

from __future__ import annotations

import json

from companion.domain.enums import CognitiveAction, EventSource, ReasonCode
from companion.domain.models import (
    CognitiveResult,
    PragmaticAnalysis,
    make_utterance_event,
)
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime


class FakeCognitive:
    def __init__(self, pragmatic):
        self.pragmatic = pragmatic
        self.calls = 0
        self.contexts = []

    async def analyze(self, event, context, memories):
        self.calls += 1
        self.contexts.append(context)
        return CognitiveResult(
            event_id=event.event_id,
            meaning="fake analysis",
            action=CognitiveAction.RESPOND,
            reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
            response_style="warm",
            provider="fake",
            pragmatic=self.pragmatic,
        )


class FakeResponse:
    def __init__(self, text="好的。"):
        self.text = text
        self.calls = 0
        self.contexts = []

    async def generate(self, event, analysis, context):
        self.calls += 1
        self.contexts.append(context)
        return self.text


class FailingStyleRepo:
    async def list_relevant_for_prompt(self, **kwargs):
        raise RuntimeError("style store unavailable")


class FakeDeepSeekClient:
    """Real providers, fake transport: analysis JSON plus plain response text."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(
            {
                "request_type": request_type,
                "messages": messages,
                "event_id": event_id,
                "json_mode": json_mode,
            }
        )
        content = (
            json.dumps(
                {
                    "meaning": "用户在说话",
                    "action": "RESPOND",
                    "reason_code": "direct_agent_address",
                    "referenced_memory_ids": [],
                    "memory_candidates": [],
                    "response_style": "warm",
                }
            )
            if request_type == "analysis"
            else self.response_text
        )
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _runtime(db, *, pragmatic, response_text="好的。"):
    cognitive = FakeCognitive(pragmatic)
    response = FakeResponse(response_text)
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=object(),
        cognitive=cognitive,
        response=response,
        allow_mixed_providers=True,
    )
    await rt.agent_service.ensure_defaults()
    person = await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("c1")
    return rt, cognitive, response, person


def _event_for(person_id, text):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def test_social_turn_has_no_mode_routing_or_style_retrieval(db):
    pragmatic = PragmaticAnalysis(
        primary_act="SELF_DISCLOSURE",
        interaction_frame="SUPPORT",
        confidence=0.9,
    )
    rt, cognitive, response, person = await _runtime(
        db,
        pragmatic=pragmatic,
        response_text="差点？那就是没迟到。先算你赢。\n\n今天别卡点出门就行。",
    )

    report = await rt.orchestrator.process_utterance(
        _event_for(person.person_id, "我今天差点迟到。")
    )

    assert report.status == "COMPLETED"
    assert cognitive.calls == 1
    assert response.calls == 1
    # The removed fields are gone from the model entirely.
    assert not hasattr(response.contexts[-1], "reply_mode")
    assert not hasattr(response.contexts[-1], "style_examples")
    assert not any("REPLY_MODE" in step for step in report.steps)
    assert not any("STYLE" in step for step in report.steps)


async def test_task_turn_uses_the_same_single_generation_path(db):
    pragmatic = PragmaticAnalysis(
        primary_act="REQUEST",
        interaction_frame="TASK",
        confidence=0.9,
    )
    rt, cognitive, response, person = await _runtime(
        db,
        pragmatic=pragmatic,
        response_text="第一步，先看误差。\n\n第二步，再看积分。",
    )

    report = await rt.orchestrator.process_utterance(
        _event_for(person.person_id, "帮我解释一下 PID 控制，从头开始。")
    )

    assert report.status == "COMPLETED"
    assert cognitive.calls == 1
    assert response.calls == 1
    assert not hasattr(response.contexts[-1], "reply_mode")


async def test_cognitive_call_never_receives_generation_guidance(db):
    pragmatic = PragmaticAnalysis(
        primary_act="SELF_DISCLOSURE",
        interaction_frame="SUPPORT",
        confidence=0.9,
    )
    rt, cognitive, response, person = await _runtime(db, pragmatic=pragmatic)

    await rt.orchestrator.process_utterance(
        _event_for(person.person_id, "今天有点累。")
    )

    cognitive_context = cognitive.contexts[0]
    assert not hasattr(cognitive_context, "style_examples")
    assert not hasattr(cognitive_context, "reply_mode")


async def test_two_calls_per_turn_and_no_post_generation_truncation(db):
    pragmatic = PragmaticAnalysis(
        primary_act="SELF_DISCLOSURE",
        interaction_frame="SUPPORT",
        confidence=0.9,
    )
    reply = "这是第一段。\n\n这是第二段。"
    rt, cognitive, response, person = await _runtime(
        db, pragmatic=pragmatic, response_text=reply
    )

    report = await rt.orchestrator.process_utterance(
        _event_for(person.person_id, "今天有点累。")
    )

    assert cognitive.calls == 1
    assert response.calls == 1
    assert report.response_text == reply


async def test_style_store_failure_cannot_affect_the_reply(db):
    # The default chain never touches the style store; a broken one is inert.
    pragmatic = PragmaticAnalysis(
        primary_act="SELF_DISCLOSURE",
        interaction_frame="SUPPORT",
        confidence=0.9,
    )
    rt, cognitive, response, person = await _runtime(
        db, pragmatic=pragmatic, response_text="嗯，我在。"
    )
    rt.orchestrator.style_repo = FailingStyleRepo()

    report = await rt.orchestrator.process_utterance(
        _event_for(person.person_id, "今天有点累。")
    )

    assert report.status == "COMPLETED"
    assert report.response_text == "嗯，我在。"


async def test_real_deepseek_providers_receive_the_unified_speaking_block(db):
    reply = "这是第一段。\n\n这是第二段。"
    client = FakeDeepSeekClient(reply)
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    await rt.agent_service.ensure_defaults()
    person = await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("c1")

    report = await rt.orchestrator.process_utterance(
        _event_for(person.person_id, "帮我解释一下 PID 控制，从头开始。")
    )

    assert report.status == "COMPLETED"
    assert report.response_text == reply
    assert [call["request_type"] for call in client.calls] == [
        "analysis",
        "response",
    ]
    system = client.calls[1]["messages"][0]["content"]
    assert "【说话方式】" in system
    assert "当前交流模式" not in system
    assert "交流表达参考" not in system
    # The cognitive request never carries response-only material.
    cognitive_system = client.calls[0]["messages"][0]["content"]
    assert "【说话方式】" not in cognitive_system
    assert "当前交流模式" not in cognitive_system
