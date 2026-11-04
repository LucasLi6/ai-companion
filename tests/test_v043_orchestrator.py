"""V0.4.3: pipeline order, one cognitive call, and retry recovery.

The character layer must not add a second model request or a third "judge".
A successful RESPOND turn makes exactly one cognitive call and one response
call; a failed turn stays retryable and reuses the persisted analysis.
"""

from __future__ import annotations

import json

from companion.domain.enums import CognitiveAction, EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.prompts.response import build_response_messages
from companion.runtime import build_runtime


class _FakeClient:
    def __init__(self, *, analysis=None, response_text="好。", fail_response=False):
        self.analysis = json.dumps(
            analysis
            or {
                "meaning": "用户在说话",
                "action": "RESPOND",
                "reason_code": "direct_agent_address",
                "referenced_memory_ids": [],
                "memory_candidates": [],
            },
            ensure_ascii=False,
        )
        self.response_text = response_text
        self.fail_response = fail_response
        self.calls = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        if request_type == "analysis":
            content = self.analysis
        else:
            if self.fail_response:
                self.fail_response = False
                raise RuntimeError("upstream response failure")
            content = self.response_text
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def response_requests(self):
        return [
            call["messages"]
            for call in self.calls
            if call["request_type"] == "response"
        ]


async def _person(rt):
    return await rt.registry.register(
        "Ada",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _event(person_id, text, *, to="agent"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def test_successful_respond_turn_is_exactly_two_requests(db):
    client = _FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "你好")
    )

    assert report.status == "COMPLETED"
    assert [call["request_type"] for call in client.calls] == [
        "analysis",
        "response",
    ]
    # Native response request shape, not a flattened transcript.
    messages = client.response_requests()[0]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "你好"}


async def test_wait_turn_makes_no_response_request(db):
    client = _FakeClient(
        analysis={
            "meaning": "对方只是在自言自语",
            "action": "WAIT",
            "reason_code": "response_not_necessary",
            "referenced_memory_ids": [],
            "memory_candidates": [],
        }
    )
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "今天天气不错")
    )

    assert report.action == CognitiveAction.WAIT
    assert [call["request_type"] for call in client.calls] == ["analysis"]


async def test_response_failure_is_retryable_without_a_second_analysis(db):
    client = _FakeClient(fail_response=True)
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    event = _event(person.person_id, "帮我看个东西")

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "FAILED"
    assert [call["request_type"] for call in client.calls] == [
        "analysis",
        "response",
    ]

    second = await rt.orchestrator.process_utterance(event)
    assert second.status == "COMPLETED"
    # The persisted analysis is reused: no second cognitive request.
    assert [call["request_type"] for call in client.calls] == [
        "analysis",
        "response",
        "response",
    ]
    assert second.response_text == "好。"


async def test_explicit_human_addressee_is_not_sent_to_the_model(db):
    client = _FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "Alice，你帮我拿一下", to="human")
    )

    assert report.action == CognitiveAction.IGNORE
    assert report.response_text is None
    assert [call["request_type"] for call in client.calls] == ["analysis"]


async def test_response_context_is_built_from_the_same_snapshot(db):
    client = _FakeClient(response_text="嗯，我在。")
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    await rt.agent_service.set_persona_core(
        {"character_description": "一个说话很直接的人。"}, actor="admin"
    )

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "在吗")
    )
    assert report.status == "COMPLETED"

    # Both requests were built from the trusted per-turn context; the response
    # system carries the same character description the analysis saw.
    analysis_request = json.dumps(
        client.calls[0]["messages"], ensure_ascii=False
    )
    response_system = client.response_requests()[0][0]["content"]
    assert "一个说话很直接的人。" in analysis_request
    assert "一个说话很直接的人。" in response_system
