"""V0.4.2 Step 2: pragmatic understanding and the interaction frame.

The model's pragmatic hint is a suggestion. These tests pin the trusted side:
an unsupported act or non-verbatim evidence is dropped, a fabricated
antecedent id never survives, the frame is merged locally, and the response
request sees the validated reading without it becoming an action.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.cognition.scene_manager import SceneManager
from companion.cognition.scene_models import SceneState
from companion.cognition.scene_repository import SceneSnapshotRecord
from companion.domain.enums import EventSource
from companion.domain.models import (
    CognitiveAction,
    PragmaticAnalysis,
    ReasonCode,
    SceneProposal,
    make_utterance_event,
    validate_pragmatic_analysis,
)
from companion.providers.deepseek_client import ChatResponse
from companion.providers.deepseek_cognitive import DeepSeekCognitiveProvider
from companion.runtime import build_runtime


class _FakeClient:
    """Returns a mutable analysis payload and a fixed response text."""

    def __init__(self, payload: dict, *, response_text="嗯，接着说。"):
        self.payload = {
            "meaning": "用户在说话",
            "action": "RESPOND",
            "reason_code": "direct_agent_address",
            "referenced_memory_ids": [],
            "memory_candidates": [],
            "response_style": "warm",
            **payload,
        }
        self.response_text = response_text
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(
            {
                "request_type": request_type,
                "messages": messages,
                "event_id": event_id,
            }
        )
        content = (
            json.dumps(self.payload, ensure_ascii=False)
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

    def response_requests(self) -> list[str]:
        return [
            "\n".join(m["content"] for m in c["messages"])
            for c in self.calls
            if c["request_type"] == "response"
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


def _event(person_id, text, *, conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


def _raw(**updates):
    payload = {
        "primary_act": "QUESTION",
        "secondary_acts": ["TEASING"],
        "interaction_frame": "PLAYFUL",
        "refers_to_previous_turn": True,
        "antecedent_event_id": "evt_prev",
        "setup_payoff_role": "SETUP",
        "literal_interpretation": "在问我刚才那句是什么意思",
        "implied_interpretation": "想让我接住这个梗",
        "topic_transition": "CONTINUE",
        "confidence": 0.9,
        "evidence_span": "你刚才那句",
    }
    payload.update(updates)
    return payload


# -- local validation -------------------------------------------------------


def test_pragmatic_analysis_validates_fields_and_evidence():
    text = "你刚才那句是什么意思？"
    analysis = validate_pragmatic_analysis(
        _raw(), text, allowed_antecedent_ids={"evt_prev"}
    )
    assert analysis is not None
    assert analysis.primary_act == "QUESTION"
    assert analysis.secondary_acts == ["TEASING"]
    assert analysis.interaction_frame == "PLAYFUL"
    assert analysis.antecedent_event_id == "evt_prev"
    assert analysis.setup_payoff_role == "SETUP"

    # Evidence that is not verbatim, or an unknown primary act, makes the whole
    # hint unusable instead of being guessed.
    assert validate_pragmatic_analysis(_raw(evidence_span="编造"), text) is None
    assert validate_pragmatic_analysis(
        _raw(primary_act="PSYCHOANALYZE"), text
    ) is None
    assert validate_pragmatic_analysis(
        _raw(evidence_span=None), text
    ) is None


def test_fabricated_antecedent_is_dropped_but_the_reading_survives():
    text = "你刚才那句是什么意思？"
    analysis = validate_pragmatic_analysis(
        _raw(antecedent_event_id="evt_invented"),
        text,
        allowed_antecedent_ids={"evt_real"},
    )
    assert analysis is not None
    assert analysis.antecedent_event_id is None
    assert analysis.refers_to_previous_turn is True
    # The hint has no action field at all.
    assert not hasattr(analysis, "action")
    assert not hasattr(analysis, "addressed_to")


async def test_provider_drops_antecedent_outside_the_visible_context(db):
    rt = build_runtime(db, provider="mock")
    person = await _person(rt)
    current = _event(person.person_id, "你刚才那句是什么意思？")
    context = CognitiveContext(
        conversation_id="c1",
        current_event_id=current.event_id,
        current_speaker_id=person.person_id,
        current_speaker_label="Ada",
        current_text=current.text,
        current_addressee_hint="agent",
        turns=[
            CognitiveTurn(
                event_id="evt_visible",
                role="human",
                speaker_person_id=person.person_id,
                speaker_label="Ada",
                text="我给你讲个冷笑话",
            )
        ],
    )
    provider = DeepSeekCognitiveProvider(_FakeClient({"pragmatic": _raw()}))
    result = await provider.analyze(current, context, [])

    assert result.pragmatic is not None
    assert result.pragmatic.antecedent_event_id is None
    assert result.pragmatic.primary_act == "QUESTION"


async def test_provider_accepts_a_visible_antecedent(db):
    rt = build_runtime(db, provider="mock")
    person = await _person(rt)
    current = _event(person.person_id, "你刚才那句是什么意思？")
    context = CognitiveContext(
        conversation_id="c1",
        current_event_id=current.event_id,
        current_speaker_id=person.person_id,
        current_speaker_label="Ada",
        current_text=current.text,
        current_addressee_hint="agent",
        turns=[
            CognitiveTurn(
                event_id="evt_previous",
                role="human",
                speaker_person_id=person.person_id,
                speaker_label="Ada",
                text="我给你讲个冷笑话",
            )
        ],
    )
    provider = DeepSeekCognitiveProvider(
        _FakeClient({"pragmatic": _raw(antecedent_event_id="evt_previous")})
    )
    result = await provider.analyze(current, context, [])

    assert result.pragmatic is not None
    assert result.pragmatic.antecedent_event_id == "evt_previous"


# -- the frame is local, bounded and persistent ----------------------------


def _record(state, revision=0):
    return SceneSnapshotRecord(
        conversation_id=state.conversation_id,
        person_id=state.person_id,
        revision=revision,
        state=state,
        source_event_ids=list(state.source_event_ids),
        last_committed_event_id=state.last_committed_event_id,
        updated_at=datetime.now(timezone.utc),
    )


def test_scene_manager_merges_frame_and_tracks_a_setup_payoff(rt):
    manager = SceneManager(repos=rt.repos)
    event = _event("person_1", "你刚才那句是什么意思？")
    first = PragmaticAnalysis(
        **_raw(
            primary_act="TEASING",
            secondary_acts=[],
            interaction_frame="PLAYFUL",
            setup_payoff_role="SETUP",
            literal_interpretation="我要讲个冷笑话",
            implied_interpretation="等一个捧场",
        )
    )

    state = manager.preview(
        None,
        SceneProposal(discourse_phase="EXPLORING", confidence=0.7),
        event,
        pragmatic=first,
    )
    assert state.interaction.frame == "PLAYFUL"
    assert state.interaction.phase == "ONGOING"
    assert state.interaction.last_social_act == "TEASING"
    assert state.interaction.open_setup == "等一个捧场"

    second_event = _event("person_1", "哈哈，接住了")
    second = PragmaticAnalysis(
        **{
            **_raw(
                primary_act="INFORMATION",
                secondary_acts=[],
                interaction_frame="CASUAL",
                setup_payoff_role="PAYOFF",
                literal_interpretation="笑",
                implied_interpretation="",
                evidence_span="接住了",
            ),
        }
    )
    state2 = manager.preview(
        _record(state, revision=0),
        None,
        second_event,
        pragmatic=second,
    )
    assert state2.interaction.frame == "CASUAL"
    assert state2.interaction.open_setup is None
    assert state2.interaction.revision > state.interaction.revision
    assert state2.prompt_dict()["interaction"]["frame"] == "CASUAL"


def test_low_confidence_pragmatic_does_not_move_the_frame(rt):
    manager = SceneManager(repos=rt.repos)
    event = _event("person_1", "随便聊聊")
    low = PragmaticAnalysis(**_raw(confidence=0.2, interaction_frame="CONFLICT"))
    state = manager.preview(None, None, event, pragmatic=low)
    assert state.interaction.frame == "UNKNOWN"


# -- full runtime: validated reading reaches the response request -----------


async def test_pragmatic_reaches_response_and_interaction_scene(db):
    client = _FakeClient({})
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)

    first = _event(person.person_id, "我给你讲个冷笑话")
    first_report = await rt.orchestrator.process_utterance(first)
    assert first_report.status == "COMPLETED"

    client.payload["pragmatic"] = _raw(
        antecedent_event_id=first.event_id,
        evidence_span="你刚才那句",
    )
    second = _event(person.person_id, "你刚才那句是什么意思？")
    second_report = await rt.orchestrator.process_utterance(second)
    assert second_report.status == "COMPLETED"

    request = client.response_requests()[-1]
    # V0.4.3: the pragmatic reading is a context hint for the cognitive layer,
    # not a labelled instruction block dumped into the response prompt.
    assert "【语用理解】" not in request
    assert "主要行为: QUESTION" not in request
    # The validated frame survives only as untrusted scene data, not as a
    # labelled pragmatic instruction block in the system message.
    assert '"frame":"PLAYFUL"' in request
    assert "【当前场景】" not in request

    snapshot = await rt.scene_manager.scene_repo.load("c1", person.person_id)
    assert snapshot is not None
    assert snapshot.state.interaction.frame == "PLAYFUL"
    assert snapshot.state.interaction.open_setup == "想让我接住这个梗"
    assert snapshot.state.last_committed_event_id == second.event_id


async def test_pragmatic_does_not_override_the_local_action(db):
    client = _FakeClient(
        {
            "action": "WAIT",
            "reason_code": "response_not_necessary",
            "pragmatic": _raw(primary_act="REQUEST", interaction_frame="SERIOUS"),
        }
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)
    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "你刚才那句是什么意思？")
    )
    assert report.action == CognitiveAction.WAIT
    # WAIT means no response generation; the pragmatic hint cannot force one.
    assert [c["request_type"] for c in client.calls] == ["analysis"]
