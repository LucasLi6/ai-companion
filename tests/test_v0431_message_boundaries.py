"""V0.4.3.1: real message boundaries for consecutive user inputs.

Chat Completions accepts consecutive ``user`` messages, so the builder must not
merge them. Each event keeps its own boundary, speaker label and order. A past
WAIT/IGNORE has no utterance, so it must never appear as an assistant message —
it belongs only in the untrusted action-history background data.
"""

from __future__ import annotations

import json

import pytest

from companion.cognition.models import (
    ActionHistoryRecord,
    CognitiveContext,
    CognitiveTurn,
)
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.prompts.response import build_response_messages
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime


# -- pure builder cases ------------------------------------------------------


def _ctx(turns, **overrides):
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_current",
        "current_speaker_id": "person_a",
        "current_speaker_label": "Alice",
        "current_text": "所以呢？",
        "current_addressee_hint": "agent",
        "turns": turns,
    }
    payload.update(overrides)
    return CognitiveContext(**payload)


def _human(event_id, text, *, person="person_a", label="Alice"):
    return CognitiveTurn(
        event_id=event_id,
        role="human",
        speaker_person_id=person,
        speaker_label=label,
        text=text,
    )


def _assistant(event_id, text):
    return CognitiveTurn(
        event_id=event_id,
        role="assistant",
        speaker_label="assistant",
        text=text,
    )


def _analysis_json():
    return {
        "event_id": "evt_current",
        "meaning": "对方在继续追问",
        "action": "RESPOND",
        "reason_code": "follow_up_to_agent",
        "response_style": "plain",
        "provider": "fake",
    }


def _background(messages):
    for message in messages:
        if message["role"] == "user" and message["content"].startswith(
            "【背景资料"
        ):
            return message["content"]
    return None


def test_consecutive_user_messages_then_assistant_are_not_merged():
    messages = build_response_messages(
        _ctx(
            [
                _human("e1", "第一句"),
                _human("e2", "第二句"),
                _assistant("e2", "回答第二句"),
            ]
        ),
        build_analysis(),
    )
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert [m["content"] for m in messages[1:]] == [
        "第一句",
        "第二句",
        "回答第二句",
        "所以呢？",
    ]


def test_three_consecutive_user_messages_keep_three_boundaries():
    messages = build_response_messages(
        _ctx(
            [
                _human("e1", "第一句"),
                _human("e2", "第二句"),
                _human("e3", "第三句"),
            ]
        ),
        build_analysis(),
    )
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "user",
        "user",
        "user",
    ]
    assert [m["content"] for m in messages[1:]] == [
        "第一句",
        "第二句",
        "第三句",
        "所以呢？",
    ]


def test_two_speakers_keep_their_labels_and_separate_events():
    ctx = _ctx(
        [
            _human("e1", "我先说", label="Alice"),
            _human("e2", "我也说", person="person_b", label="Bob"),
        ],
        current_speaker_id="person_a",
        current_speaker_label="Alice",
    )
    messages = build_response_messages(ctx, build_analysis())
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "user",
        "user",
    ]
    assert messages[1]["content"] == "[Alice → unknown] 我先说"
    assert messages[2]["content"] == "[Bob → unknown] 我也说"
    assert messages[3]["content"] == "[Alice → Companion] 所以呢？"
    assert "多人群聊" in messages[0]["content"]


def test_speaker_change_across_an_assistant_turn_keeps_order_and_labels():
    ctx = _ctx(
        [
            _human("e1", "你们在聊什么？", label="Alice"),
            _assistant("e1", "在聊旅行。"),
            _human("e2", "我也想加入。", person="person_b", label="Bob"),
        ],
        current_speaker_id="person_a",
        current_speaker_label="Alice",
    )
    messages = build_response_messages(ctx, build_analysis())
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert messages[1]["content"] == "[Alice → unknown] 你们在聊什么？"
    assert messages[2]["content"] == "在聊旅行。"
    assert messages[3]["content"] == "[Bob → unknown] 我也想加入。"
    assert messages[4]["content"] == "[Alice → Companion] 所以呢？"


@pytest.mark.parametrize("action", ["WAIT", "IGNORE"])
def test_past_silence_is_background_data_not_an_assistant_message(action):
    ctx = _ctx(
        [_human("e1", "你听见了吗？")],
        action_history=[
            ActionHistoryRecord(
                event_id="e1",
                final_action=action,
                cause_code="AMBIGUOUS_ADDRESSEE",
                cause_source="MODEL_ASSESSMENT",
                pending_followup=False,
            )
        ],
    )
    messages = build_response_messages(ctx, build_analysis())
    # V0.6.1 sets the past-action budget to zero, so the WAIT/IGNORE produces
    # no background message and no assistant utterance.
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert _background(messages) is None
    assert "LOCAL_ACTION_HISTORY" not in messages[0]["content"]
    assert messages[-1]["content"] == "所以呢？"


def build_analysis():
    from companion.domain.models import CognitiveResult

    return CognitiveResult(**_analysis_json())


# -- real orchestrator cases -------------------------------------------------


class _ScriptedClient:
    def __init__(self, analyses, response_text="在。"):
        self.analyses = list(analyses)
        self.response_text = response_text
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        if request_type == "analysis":
            content = json.dumps(self.analyses.pop(0), ensure_ascii=False)
        else:
            content = self.response_text
        return ChatResponse(
            content=content,
            model="fake",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def response_messages(self) -> list[dict]:
        return [
            call["messages"]
            for call in self.calls
            if call["request_type"] == "response"
        ][-1]


def _payload(action="RESPOND", reason="direct_agent_address"):
    return {
        "meaning": "用户在说话",
        "action": action,
        "reason_code": reason,
        "reply_necessity": "REQUIRED" if action == "RESPOND" else "OPTIONAL",
        "response_style": "plain",
    }


def _event(person_id, text, *, event_id, conversation="c1", to="agent"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


async def _person(rt, name="Alice"):
    return await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        ("WAIT", "ambiguous_addressee"),
        ("IGNORE", "social_response_not_needed"),
    ],
)
async def test_orchestrator_keeps_two_user_turns_and_no_fake_assistant(
    db, action, reason
):
    client = _ScriptedClient(
        [
            _payload(action, reason),
            _payload("RESPOND", "direct_agent_address"),
        ]
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    alice = await _person(rt)
    first = _event(alice.person_id, "嗯。", event_id="evt_first")
    second = _event(alice.person_id, "你到底在不在？", event_id="evt_second")

    r1 = await rt.orchestrator.process_utterance(first)
    r2 = await rt.orchestrator.process_utterance(second)
    assert r1.action.value == action
    assert r2.action.value == "RESPOND"

    messages = client.response_messages()
    assert messages[0]["role"] == "system"
    assert messages[-2] == {"role": "user", "content": "嗯。"}
    assert messages[-1] == {"role": "user", "content": "你到底在不在？"}
    assert all(message["role"] != "assistant" for message in messages)
    assert _background(messages) is None


async def test_orchestrator_labels_two_speakers_without_merging(db):
    client = _ScriptedClient(
        [
            _payload("WAIT", "ambiguous_addressee"),
            _payload("RESPOND", "direct_agent_address"),
        ]
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")

    await rt.orchestrator.process_utterance(
        _event(alice.person_id, "有人吗？", event_id="evt_a", to="unknown")
    )
    await rt.orchestrator.process_utterance(
        _event(bob.person_id, "你好。", event_id="evt_b", to="agent")
    )

    messages = client.response_messages()
    assert messages[-2] == {"role": "user", "content": "[Alice → unknown] 有人吗？"}
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("[Bob → ")
    assert messages[-1]["content"].endswith("] 你好。")
    assert all(message["role"] != "assistant" for message in messages)
    assert "多人群聊" in messages[0]["content"]
