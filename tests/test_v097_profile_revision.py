"""V0.9.7: optimistic revisions, idempotency and atomic multi-change commits."""

from __future__ import annotations

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider


async def _person(rt):
    person = await rt.registry.register(
        "Revision",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.profile_service.ensure_profile(person.person_id, actor="test")
    return person


async def _event(rt, person, text, conversation="c1"):
    await rt.conversations.get_or_create(conversation)
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    return event


def _engine(rt, *decisions):
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(list(decisions)),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )


def _change(field_key, content, evidence, operation="ADD"):
    return ProfileChangeProposal(
        operation=operation,
        field_key=field_key,
        proposed_content=content,
        evidence_event_ids=[evidence],
        confidence=0.95,
    )


def _propose(person_id, revision, *changes):
    return ProfileReviewDecision(
        person_id=person_id,
        base_revision=revision,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=list(changes),
    )


async def test_one_review_commits_one_revision_with_every_change(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器人，也喜欢爵士。")
    engine = _engine(
        rt,
        _propose(
            person.person_id,
            1,
            _change("projects.active", "机器人", event.event_id),
            _change("interests.topics", "爵士乐", event.event_id),
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    applied = await engine.apply_proposals(review)
    assert applied.status == "APPLIED"
    assert applied.new_revision == 2
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",)
    )
    assert len(entries) == 2
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert len(revisions) == 1
    assert revisions[0]["revision"] == 2
    assert len(revisions[0]["changes"]["changes"]) == 2
    assert revisions[0]["source_event_id"] == event.event_id


async def test_no_change_does_not_bump_revision(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天天气不错。")
    engine = _engine(
        rt,
        ProfileReviewDecision(
            person_id=person.person_id,
            base_revision=1,
            decision="NO_CHANGE",
            changes=[],
        ),
    )
    applied = await engine.apply_proposals(
        await engine.review(person.person_id, event.event_id)
    )
    assert applied.status == "NO_CHANGE"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1
    assert await rt.repos.profile.list_revisions(person.person_id) == []


async def test_stale_review_reports_conflict_without_writing(rt):
    person = await _person(rt)
    first_event = await _event(rt, person, "我在做机器人。")
    second_event = await _event(rt, person, "顺便说下，我也在学日语。", "c2")
    first_engine = _engine(
        rt,
        _propose(
            person.person_id,
            1,
            _change("projects.active", "机器人", first_event.event_id),
        ),
    )
    second_engine = _engine(
        rt,
        _propose(
            person.person_id,
            1,
            _change("knowledge.skills", "日语", second_event.event_id),
        ),
    )
    first_review = await first_engine.review(person.person_id, first_event.event_id)
    second_review = await second_engine.review(person.person_id, second_event.event_id)
    assert first_review.base_revision == second_review.base_revision == 1

    assert (await first_engine.apply_proposals(first_review)).status == "APPLIED"
    conflict = await second_engine.apply_proposals(second_review)
    assert conflict.status == "REVISION_CONFLICT"
    assert conflict.base_revision == 1
    assert conflict.new_revision is None
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",)
    )
    assert [item["content"] for item in entries] == ["机器人"]
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert [item["revision"] for item in revisions] == [2]


async def test_replaying_the_same_event_is_idempotent(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机械臂。")
    engine = _engine(
        rt,
        _propose(
            person.person_id,
            1,
            _change("projects.active", "机械臂", event.event_id),
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    assert (await engine.apply_proposals(review)).status == "APPLIED"
    replay = await engine.apply_proposals(review)
    assert replay.status == "ALREADY_APPLIED"
    assert replay.new_revision == 2
    assert len(await rt.repos.profile.list_entries(person.person_id)) == 1
    assert len(await rt.repos.profile.list_revisions(person.person_id)) == 1


async def test_strict_identity_from_model_is_rejected_without_candidate(rt):
    """A model proposal can never write strict identity, not even as pending."""
    person = await _person(rt)
    event = await _event(rt, person, "我叫阿明。")
    engine = _engine(
        rt,
        _propose(
            person.person_id,
            1,
            _change("identity.name", "阿明", event.event_id),
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    applied = await engine.apply_proposals(review)
    assert applied.status == "REJECTED"
    assert applied.applied == []
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1
    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert await rt.repos.profile.list_revisions(person.person_id) == []
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []

    # The person's own explicit statement still writes directly.
    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id=event.event_id,
    )
    assert result.status == "APPLIED"
    assert result.revision == 2
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",)
    )
    assert [item["content"] for item in entries] == ["阿明"]
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []


async def test_multi_change_review_is_all_or_nothing(rt, monkeypatch):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器人，喜欢爵士。")
    engine = _engine(
        rt,
        _propose(
            person.person_id,
            1,
            _change("projects.active", "机器人", event.event_id),
            _change("interests.topics", "爵士乐", event.event_id),
        ),
    )
    review = await engine.review(person.person_id, event.event_id)

    calls = {"n": 0}
    original = rt.repos.profile.insert_entry

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second insert failed")
        return await original(*args, **kwargs)

    monkeypatch.setattr(rt.repos.profile, "insert_entry", flaky)
    with pytest.raises(RuntimeError):
        await engine.apply_proposals(review)
    monkeypatch.undo()
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1
    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert await rt.repos.profile.list_revisions(person.person_id) == []
