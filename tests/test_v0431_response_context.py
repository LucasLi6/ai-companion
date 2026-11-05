"""V0.4.3.1: the response prompt keeps only what the reply needs.

``analysis.meaning`` is a model inference and is removed from the response
prompt, while policy, appraisal, scene and memory code still consume the
analysis locally. Unknown scene fields and neutral affect no longer fill the
prompt with noise; the emotion state and speech (TTS) plan remain intact.
"""

from __future__ import annotations

import json

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.domain.enums import EventSource
from companion.domain.models import CognitiveResult, make_utterance_event
from companion.prompts.response import build_response_messages
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

SECRET = "模型推断：对方其实在试探你是不是真人，这句话绝不该出现在回复提示里。"


def _analysis(**overrides):
    payload = {
        "event_id": "evt_current",
        "meaning": SECRET,
        "action": "RESPOND",
        "reason_code": "follow_up_to_agent",
        "response_style": "plain",
        "provider": "fake",
    }
    payload.update(overrides)
    return CognitiveResult(**payload)


def _ctx(**overrides):
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_current",
        "current_speaker_id": "person_a",
        "current_speaker_label": "Alice",
        "current_text": "后来我还去了大阪",
        "current_addressee_hint": "agent",
        "turns": [
            CognitiveTurn(
                event_id="evt_prev",
                role="human",
                speaker_person_id="person_a",
                speaker_label="Alice",
                text="我去了日本",
            )
        ],
    }
    payload.update(overrides)
    return CognitiveContext(**payload)


def _background(messages):
    for message in messages:
        if message["role"] == "user" and message["content"].startswith(
            "【背景资料"
        ):
            return message["content"]
    return None


# -- pure builder ------------------------------------------------------------


def test_analysis_meaning_is_never_rendered_but_history_is():
    messages = build_response_messages(_ctx(), _analysis())
    joined = "\n".join(message["content"] for message in messages)
    assert SECRET not in joined
    assert "我去了日本" in joined
    assert "后来我还去了大阪" in joined


def test_unknown_scene_produces_no_background_and_no_placeholder_fields():
    scene = {
        "topic": "UNKNOWN",
        "mode": "UNKNOWN",
        "discourse_phase": "UNCERTAIN",
        "tone": "UNCERTAIN",
        "open_loops": [],
        "interaction": {"frame": "UNKNOWN", "phase": "UNKNOWN"},
        "environment": {
            "location": "UNKNOWN",
            "other_people_present": "UNKNOWN",
            "provenance": "UNKNOWN",
        },
    }
    messages = build_response_messages(_ctx(scene_state=scene), _analysis())
    # system + the real history turn + the current turn. No background block:
    # there is nothing but UNKNOWN placeholders to render.
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    joined = "\n".join(message["content"] for message in messages)
    assert "UNKNOWN" not in joined
    assert "UNCERTAIN" not in joined
    assert "地点" not in joined
    assert "旁人在场" not in joined


def test_unknown_scene_is_omitted_even_when_memories_make_a_background():
    messages = build_response_messages(
        _ctx(
            scene_state={
                "topic": "UNKNOWN",
                "environment": {
                    "location": "UNKNOWN",
                    "other_people_present": "UNKNOWN",
                    "provenance": "UNKNOWN",
                },
            },
            memories=[{"kind": "EPISODE", "content": "她养了一只猫"}],
        ),
        _analysis(),
    )
    background = _background(messages)
    assert background is not None
    assert "她养了一只猫" in background
    assert '"scene"' not in background
    assert "UNKNOWN" not in background


def test_real_open_loop_and_tension_survive():
    messages = build_response_messages(
        _ctx(
            scene_state={
                "topic": "旅行",
                "open_loops": [{"topic": "工作", "summary": "工作的事还没讲完"}],
            },
            affect_context="两人之间还残留一点紧张。",
        ),
        _analysis(),
    )
    background = _background(messages)
    assert background is not None
    assert "工作的事还没讲完" in background
    assert "两人之间还残留一点紧张。" in messages[0]["content"]


def test_no_affect_context_means_no_affect_line():
    messages = build_response_messages(_ctx(), _analysis())
    assert "【此刻的状态】" not in messages[0]["content"]
    assert "情绪比较平稳" not in messages[0]["content"]
    assert "模拟情绪" not in messages[0]["content"]


# -- real orchestrator -------------------------------------------------------


class _FakeClient:
    def __init__(self, analysis, response_text="好。"):
        self.payload = json.dumps(analysis, ensure_ascii=False)
        self.response_text = response_text
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        return ChatResponse(
            content=self.payload if request_type == "analysis" else self.response_text,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def count(self, request_type: str) -> int:
        return sum(1 for call in self.calls if call["request_type"] == request_type)

    def response_messages(self) -> list[dict]:
        return [
            call["messages"]
            for call in self.calls
            if call["request_type"] == "response"
        ][-1]


def _event(person_id, text, *, event_id="evt_1"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


def _payload(**overrides):
    payload = {
        "meaning": SECRET,
        "action": "RESPOND",
        "reason_code": "direct_agent_address",
        "referenced_memory_ids": [],
        "memory_candidates": [],
    }
    payload.update(overrides)
    return payload


async def _person(rt):
    return await rt.registry.register(
        "Ada",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


async def test_real_response_request_never_contains_the_inference(db):
    client = _FakeClient(_payload())
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "后来我还去了大阪")
    )
    assert report.status == "COMPLETED"

    joined = "\n".join(
        message["content"] for message in client.response_messages()
    )
    assert SECRET not in joined
    assert "后来我还去了大阪" in joined


async def test_analysis_action_still_drives_local_policy(db):
    client = _FakeClient(_payload(action="WAIT", reason_code="ambiguous_addressee"))
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "嗯。")
    )
    assert report.status == "COMPLETED"
    assert report.action.value == "WAIT"
    # WAIT is decided locally from the analysis; no response provider call.
    assert client.count("analysis") == 1
    assert client.count("response") == 0


async def test_scene_proposal_is_still_applied_even_without_meaning(db):
    client = _FakeClient(
        _payload(
            scene_proposal={
                "topic_change": "SHIFT",
                "topic": "旅行",
                "mode": "CONVERSATION",
                "discourse_phase": "EXPLORING",
                "tone": "WARM",
                "environment": None,
                "open_loop": None,
                "resolve_loop_topics": [],
                "confidence": 0.9,
                "evidence_span": "旅行",
            }
        )
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "我们聊聊旅行吧")
    )
    assert report.status == "COMPLETED"

    record = await rt.scene_manager.scene_repo.load("c1", person.person_id)
    assert record is not None
    assert record.state.topic == "旅行"

    background = _background(client.response_messages())
    assert background is not None
    assert '"topic":"旅行"' in background


async def test_neutral_turn_has_no_filler_but_keeps_state_and_tts(db):
    client = _FakeClient(_payload())
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "你好")
    )
    assert report.status == "COMPLETED"

    system = client.response_messages()[0]["content"]
    assert "【此刻的状态】" not in system
    assert "情绪比较平稳" not in system

    # The state and the delivery plan still exist; only the prompt filler is
    # gone.
    agent_id = await _agent_id(rt)
    state = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert state is not None
    plan = await rt.emotion_service.plan_speech(
        agent_id, state=state, text="你好"
    )
    assert plan is not None
    assert 0.5 <= plan.speaking_rate <= 1.6
