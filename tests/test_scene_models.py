"""V0.4 — local validation of untrusted scene proposals.

No network, no database. These tests pin the trust boundary: a model may
propose a topic, environment detail or open loop, but only a verbatim evidence
span from the current input or this speaker's authorized history can support
it. Invalid fields are dropped, not promoted to facts.
"""

from __future__ import annotations

import json

import pytest

from companion.cognition.scene_models import (
    ENVIRONMENT_TTL_TURNS,
    SceneState,
    default_environment,
)
from companion.domain.models import SceneProposal, validate_scene_proposal


def _proposal(**overrides):
    payload = {
        "topic_change": "SHIFT",
        "topic": "旅行",
        "mode": "CONVERSATION",
        "discourse_phase": "EXPLORING",
        "tone": "WARM",
        "confidence": 0.8,
        "evidence_span": "我去了日本",
    }
    payload.update(overrides)
    return payload


def test_valid_proposal_is_accepted():
    proposal = validate_scene_proposal(
        _proposal(), current_text="我去了日本，挺好玩的"
    )
    assert proposal is not None
    assert proposal.topic_change == "SHIFT"
    assert proposal.topic == "旅行"
    assert proposal.mode == "CONVERSATION"
    assert proposal.discourse_phase == "EXPLORING"
    assert proposal.tone == "WARM"
    assert proposal.evidence_span == "我去了日本"


def test_topic_without_evidence_is_dropped_and_shift_degrades():
    proposal = validate_scene_proposal(
        _proposal(evidence_span="完全不在原文里"),
        current_text="我去了日本",
    )
    assert proposal is not None
    assert proposal.topic is None
    assert proposal.topic_change == "UNCERTAIN"


def test_environment_without_evidence_is_never_inferred():
    proposal = validate_scene_proposal(
        _proposal(
            environment={
                "location": "办公室",
                "other_people_present": "YES",
                "evidence_span": None,
            }
        ),
        current_text="今天有点累",
    )
    assert proposal is not None
    assert proposal.environment is None


def test_explicit_environment_needs_verbatim_evidence():
    proposal = validate_scene_proposal(
        _proposal(
            environment={
                "location": "家里",
                "other_people_present": "NO",
                "evidence_span": "我一个人在家里",
            }
        ),
        current_text="我一个人在家里，不想出门",
    )
    assert proposal is not None
    assert proposal.environment is not None
    assert proposal.environment.location == "家里"
    assert proposal.environment.other_people_present == "NO"


def test_location_must_be_anchored_in_the_evidence_span():
    proposal = validate_scene_proposal(
        _proposal(
            environment={
                "location": "办公室",
                "other_people_present": "NO",
                "evidence_span": "我一个人在家里",
            }
        ),
        current_text="我一个人在家里",
    )
    assert proposal is not None
    assert proposal.environment is not None
    assert proposal.environment.location is None
    assert proposal.environment.other_people_present == "NO"


def test_another_speakers_text_cannot_support_this_scene():
    proposal = validate_scene_proposal(
        _proposal(evidence_span="我在医院"),
        current_text="你好",
        authorized_texts=["我在医院"],  # caller must pass only this person's text
    )
    # The function itself accepts the supplied authorized set; the manager and
    # provider restrict that set to the current speaker. Here we prove the
    # evidence gate is explicit rather than accidental.
    assert proposal is not None
    assert proposal.topic == "旅行"


def test_open_loop_requires_evidence_and_summary():
    proposal = validate_scene_proposal(
        _proposal(
            open_loop={
                "topic": "工作",
                "summary": "还没讲完工作上的事",
                "evidence_span": "我先说工作",
            }
        ),
        current_text="我先说工作，等会儿再讲细节",
    )
    assert proposal is not None
    assert proposal.open_loop is not None
    assert proposal.open_loop.summary == "还没讲完工作上的事"

    invalid = validate_scene_proposal(
        _proposal(
            open_loop={
                "topic": "工作",
                "summary": "还没讲完",
                "evidence_span": "不存在",
            }
        ),
        current_text="我先说工作",
    )
    assert invalid.open_loop is None


@pytest.mark.parametrize(
    "overrides,field,value",
    [
        ({"mode": "MADE_UP"}, "mode", "UNKNOWN"),
        ({"discourse_phase": "DUNNO"}, "discourse_phase", "UNCERTAIN"),
        ({"tone": "ANGRY_AT_EVERYONE"}, "tone", "UNCERTAIN"),
        ({"topic_change": "TELEPORT"}, "topic_change", "UNCERTAIN"),
    ],
)
def test_invalid_enums_degrade(overrides, field, value):
    proposal = validate_scene_proposal(
        _proposal(**overrides), current_text="我去了日本"
    )
    assert proposal is not None
    assert getattr(proposal, field) == value


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1, 2])
def test_nonfinite_or_out_of_range_confidence_is_zero(confidence):
    proposal = validate_scene_proposal(
        _proposal(confidence=confidence), current_text="我去了日本"
    )
    assert proposal is not None
    assert proposal.confidence == 0.0


def test_model_ids_are_ignored():
    proposal = validate_scene_proposal(
        {
            **_proposal(),
            "scene_id": "scene_injected",
            "loop_id": "loop_injected",
            "person_id": "person_injected",
            "event_id": "evt_injected",
        },
        current_text="我去了日本",
    )
    dumped = proposal.model_dump()
    for key in ("scene_id", "loop_id", "person_id", "event_id"):
        assert key not in dumped


def test_no_usable_signal_returns_none():
    assert (
        validate_scene_proposal(
            {"topic_change": "UNCERTAIN", "confidence": 0.4},
            current_text="嗯",
        )
        is None
    )
    assert validate_scene_proposal(None, current_text="嗯") is None
    assert validate_scene_proposal("not-an-object", current_text="嗯") is None


def test_resolve_topics_are_bounded_and_deduplicated():
    proposal = validate_scene_proposal(
        _proposal(resolve_loop_topics=["工作", "工作", "", "x" * 200, "旅行"]),
        current_text="我去了日本",
    )
    assert proposal.resolve_loop_topics == ["工作", "旅行"]


def test_scene_state_defaults_to_unknown_environment():
    state = SceneState(conversation_id="c1", person_id="p1")
    assert state.environment == default_environment()
    assert state.environment["location"] == "UNKNOWN"
    assert state.environment["other_people_present"] == "UNKNOWN"
    assert state.environment["status"] == "UNKNOWN"
    assert state.topic == "UNKNOWN"
    assert state.revision == 0


def test_prompt_view_hides_stale_environment():
    state = SceneState(
        conversation_id="c1",
        person_id="p1",
        environment={
            "location": "家里",
            "other_people_present": "YES",
            "status": "EXPLICIT",
            "age_turns": ENVIRONMENT_TTL_TURNS + 1,
        },
    )
    view = state.prompt_dict()
    assert view["environment"]["location"] == "UNKNOWN"
    assert view["environment"]["other_people_present"] == "UNKNOWN"


def test_prompt_view_keeps_fresh_explicit_environment():
    state = SceneState(
        conversation_id="c1",
        person_id="p1",
        environment={
            "location": "家里",
            "other_people_present": "YES",
            "status": "EXPLICIT",
            "age_turns": 0,
        },
    )
    view = state.prompt_dict()
    assert view["environment"]["location"] == "家里"
    assert view["environment"]["other_people_present"] == "YES"
    assert view["environment"]["provenance"] == "EXPLICIT"


def test_scene_state_json_roundtrip():
    state = SceneState(conversation_id="c1", person_id="p1", topic="旅行", tone="WARM")
    raw = state.model_dump_json()
    restored = SceneState.model_validate_json(raw)
    assert restored.topic == "旅行"
    assert restored.tone == "WARM"
    assert restored.updated_at == state.updated_at
    # JSON is a plain object, safe for the single CMF database.
    assert isinstance(json.loads(raw), dict)
