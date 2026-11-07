"""V0.9.7.1 Sections C/D: trustworthy evidence and non-travelling history.

Only this person's own persisted messages are evidence. Ava's replies are
context, never a self-report. A review sees only turns persisted *before* its
own event, so a later correction cannot leak backwards into an earlier one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileEvidenceBundle,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider


def _at(hour, minute):
    return datetime(2026, 1, 1, hour, minute, tzinfo=timezone.utc)


async def _person(rt, name="Evidence"):
    person = await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.profile_service.ensure_profile(person.person_id, actor="test")
    return person


async def _event(
    rt,
    person,
    text,
    conversation="c1",
    *,
    occurred_at=None,
    speaker_person_id="__default__",
):
    await rt.conversations.get_or_create(conversation)
    speaker = (
        person.person_id
        if speaker_person_id == "__default__"
        else speaker_person_id
    )
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=speaker,
        source=EventSource.TEST,
        occurred_at=occurred_at,
    )
    await rt.repos.utterance.save(event)
    return event


def _source(rt):
    return ProfileReviewContextSource(
        rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
    )


def _engine(rt, proposal, *, quotes=None):
    decision = ProfileReviewDecision(
        person_id="ignored",
        base_revision=0,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=[proposal],
    )
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(decision),
        _source(rt),
    )


def _proposal(field_key, content, *, events=None, quotes=None):
    return ProfileChangeProposal(
        operation="ADD",
        field_key=field_key,
        proposed_content=content,
        evidence_event_ids=events or [],
        evidence_quotes=quotes or [],
        confidence=0.99,
    )


# -- history order -----------------------------------------------------------


async def test_a_review_never_sees_later_turns(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我正在做机器狗。", occurred_at=_at(10, 0))
    second = await _event(rt, person, "我不做了。", occurred_at=_at(10, 1))
    third = await _event(rt, person, "算了，我继续做。", occurred_at=_at(10, 2))

    bundle = await _source(rt).load(
        person_id=person.person_id, source_event_id=first.event_id
    )
    assert bundle.current_turn == "我正在做机器狗。"
    assert bundle.eligible_user_evidence == [
        {
            "event_id": first.event_id,
            "speaker_person_id": person.person_id,
            "text": "我正在做机器狗。",
            "role": "user",
        }
    ]
    assert first.event_id in bundle.allowed_event_ids
    assert second.event_id not in bundle.allowed_event_ids
    assert third.event_id not in bundle.allowed_event_ids
    assert all("我不做了" not in item["text"] for item in bundle.recent_context)
    assert all("继续做" not in item["text"] for item in bundle.recent_context)

    latest = await _source(rt).load(
        person_id=person.person_id, source_event_id=third.event_id
    )
    assert [item["text"] for item in latest.recent_context] == [
        "我正在做机器狗。",
        "我不做了。",
    ]
    assert [item["event_id"] for item in latest.eligible_user_evidence] == [
        third.event_id,
        first.event_id,
        second.event_id,
    ]


async def test_same_timestamp_turns_use_persisted_order_not_id_text(rt):
    person = await _person(rt)
    stamp = _at(12, 0)
    first = await _event(rt, person, "第一条。", occurred_at=stamp)
    second = await _event(rt, person, "第二条。", occurred_at=stamp)

    before_second = await rt.repos.utterance.context_before_event(
        second.conversation_id, second.event_id, limit=10
    )
    assert [item.event_id for item in before_second] == [first.event_id]
    before_first = await rt.repos.utterance.context_before_event(
        first.conversation_id, first.event_id, limit=10
    )
    assert before_first == []


async def test_context_before_event_stays_inside_one_conversation(rt):
    person = await _person(rt)
    other = await _event(rt, person, "另一段对话。", conversation="c2")
    target = await _event(rt, person, "这段对话。", conversation="c1")
    context = await rt.repos.utterance.context_before_event(
        target.conversation_id, target.event_id, limit=10
    )
    assert other.event_id not in {item.event_id for item in context}
    assert await rt.repos.utterance.context_before_event(
        "missing", target.event_id, limit=10
    ) == []


async def test_later_correction_does_not_rewrite_the_earlier_decision(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我正在做机器狗。", occurred_at=_at(10, 0))
    await _event(rt, person, "我不做了。", occurred_at=_at(10, 1))
    await _event(rt, person, "算了，我继续做。", occurred_at=_at(10, 2))

    engine = _engine(
        rt,
        _proposal(
            "projects.active",
            "机器狗",
            events=[first.event_id],
            quotes=["我正在做机器狗"],
        ),
    )
    result = await engine.review(person.person_id, first.event_id)
    assert result.status == "ACCEPTED"
    assert result.changes and result.changes[0].accepted
    assert result.rejected == []


# -- evidence validity -------------------------------------------------------


async def test_a_persons_own_quoted_message_is_valid_evidence(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我最近在学嵌入式开发。")
    engine = _engine(
        rt,
        _proposal(
            "knowledge.skills",
            "嵌入式开发",
            events=[event.event_id],
            quotes=["在学嵌入式开发"],
        ),
    )
    result = await engine.review(person.person_id, event.event_id)
    applied = await engine.apply_proposals(result)
    assert applied.applied
    assert applied.applied[0]["field_key"] == "knowledge.skills"


async def test_ava_reply_is_context_not_evidence(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天随便聊聊。")

    # The bundle structurally separates the person's messages from context.
    bundle = ProfileEvidenceBundle(
        person_id=person.person_id,
        source_event_id=event.event_id,
        current_turn=event.text,
        recent_context=[
            {"role": "human", "text": event.text},
            {"role": "assistant", "text": "你好像对摄影很感兴趣。"},
        ],
        eligible_user_evidence=[
            {
                "event_id": event.event_id,
                "speaker_person_id": person.person_id,
                "text": event.text,
                "role": "user",
            }
        ],
        allowed_event_ids=[event.event_id],
    )
    assert bundle.quote_texts() == [event.text]
    assert "摄影" not in " ".join(bundle.quote_texts())

    engine = _engine(
        rt,
        _proposal(
            "interests.topics",
            "摄影",
            events=[event.event_id],
            quotes=["你好像对摄影很感兴趣"],
        ),
    )
    result = await engine.review(person.person_id, event.event_id)
    assert result.rejected[0]["reason"] == "evidence_quote_not_found"

    # Citing an assistant-authored utterance as proof is not authorized either.
    reply = await _event(
        rt,
        person,
        "你好像对摄影很感兴趣。",
        speaker_person_id=None,
    )
    cited = await _engine(
        rt,
        _proposal(
            "interests.topics", "摄影", events=[reply.event_id]
        ),
    ).review(person.person_id, event.event_id)
    assert cited.rejected[0]["reason"] == "evidence_event_not_authorized"


async def test_wrong_nonexistent_and_foreign_events_are_refused(rt):
    person = await _person(rt)
    other = await _person(rt, "Other")
    event = await _event(rt, person, "我喜欢机器人。")
    unrelated = await _event(rt, person, "今天天气不错。")
    foreign = await _event(rt, other, "我喜欢摄影。", conversation="c9")

    wrong = await _engine(
        rt,
        _proposal(
            "interests.topics",
            "机器人",
            events=[unrelated.event_id],
            quotes=["我喜欢机器人"],
        ),
    ).review(person.person_id, event.event_id)
    assert wrong.rejected[0]["reason"] == "evidence_event_not_authorized"

    missing = await _engine(
        rt, _proposal("interests.topics", "机器人", events=["evt_does_not_exist"])
    ).review(person.person_id, event.event_id)
    assert missing.rejected[0]["reason"] == "evidence_event_not_authorized"

    other_person = await _engine(
        rt,
        _proposal(
            "interests.topics", "摄影", events=[foreign.event_id]
        ),
    ).review(person.person_id, event.event_id)
    assert other_person.rejected[0]["reason"] == "evidence_event_not_authorized"


async def test_no_evidence_or_confidence_alone_is_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我可能喜欢赛车吧。")
    result = await _engine(
        rt,
        ProfileChangeProposal(
            operation="ADD",
            field_key="interests.topics",
            proposed_content="赛车",
            confidence=1.0,
        ),
    ).review(person.person_id, event.event_id)
    assert result.status == "REJECTED"
    assert result.accepted_changes() == []
    assert result.rejected[0]["reason"] == "no_evidence"


async def test_hypothetical_and_third_party_claims_are_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "如果我有时间，我也想学冲浪。")
    result = await _engine(
        rt,
        _proposal(
            "interests.topics",
            "冲浪",
            events=[event.event_id],
            quotes=["如果我有时间"],
        ),
    ).review(person.person_id, event.event_id)
    assert result.status == "REJECTED"
    assert result.rejected[0]["reason"] == "hypothetical_or_reported_evidence"

    third_party = await _event(rt, person, "他说我喜欢潜水。")
    reported = await _engine(
        rt,
        _proposal(
            "interests.topics",
            "潜水",
            events=[third_party.event_id],
            quotes=["他说我喜欢潜水"],
        ),
    ).review(person.person_id, third_party.event_id)
    assert reported.status == "REJECTED"
    assert reported.rejected[0]["reason"] == "hypothetical_or_reported_evidence"


async def test_a_quote_must_be_located_in_the_cited_event(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢机器人。")
    # The quote is not in the cited event's text.
    result = await _engine(
        rt,
        _proposal(
            "interests.topics",
            "机器人",
            events=[event.event_id],
            quotes=["我喜欢摄影"],
        ),
    ).review(person.person_id, event.event_id)
    assert result.rejected[0]["reason"] == "evidence_quote_not_found"

    # A quote without any event cannot be verified and may not auto-apply.
    unanchored = await _engine(
        rt,
        _proposal("interests.topics", "机器人", quotes=["我喜欢机器人"]),
    ).review(person.person_id, event.event_id)
    assert unanchored.status == "REJECTED"
    assert unanchored.rejected[0]["reason"] == "evidence_quote_without_event"


async def test_service_write_boundary_rechecks_evidence(rt, monkeypatch):
    """A hand-built service decision cannot cite another person's message."""
    from types import SimpleNamespace

    from companion.people.profile_refinement import ValidatedChange

    person = await _person(rt)
    other = await _person(rt, "Other")
    event = await _event(rt, person, "随便聊聊。")
    foreign = await _event(rt, other, "我喜欢摄影。", conversation="c9")

    async def passthrough(person_id, changes):
        return list(changes), []

    monkeypatch.setattr(
        rt.profile_service, "_precheck_changes", passthrough, raising=False
    )
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision
    decision = SimpleNamespace(
        status="PROPOSE_CHANGES",
        base_revision=revision,
        reason="hand built",
        changes=[
            ValidatedChange(
                proposal=ProfileChangeProposal(
                    operation="ADD",
                    field_key="interests.topics",
                    proposed_content="摄影",
                    evidence_event_ids=[foreign.event_id],
                    confidence=1.0,
                )
            )
        ],
    )
    with pytest.raises(ValueError, match="evidence_event_not_accessible"):
        await rt.profile_service.apply_refinement(
            person.person_id, decision, source_event_id=event.event_id
        )
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision
    assert await rt.repos.profile.list_entries(person.person_id) == []
