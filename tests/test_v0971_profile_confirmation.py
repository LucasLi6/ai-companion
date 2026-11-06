"""V0.9.7.1 Section B: stale and concurrent proposal confirmation.

A candidate is bound to the profile revision it was proposed against. A later
profile change makes confirmation explicitly stale: nothing is overwritten,
bumped or confirmed, and only a fresh re-review may be confirmed.
"""

from __future__ import annotations

import asyncio

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
from companion.people.profile_service import (
    ProfileProposalStale,
    ProfileUnauthorized,
    TRUSTED_CONFIRMATION,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider


async def _person(rt, name="Confirm"):
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


def _decision(changes):
    return ProfileReviewDecision(
        person_id="ignored",
        base_revision=0,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=changes,
    )


def _engine(rt, decision):
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(decision),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )


def _proposal(operation, field_key, content, evidence, targets=None):
    return ProfileChangeProposal(
        operation=operation,
        field_key=field_key,
        target_entry_ids=targets or [],
        proposed_content=content,
        evidence_event_ids=[evidence],
        confidence=0.95,
    )


async def _identity_candidate(rt, person):
    event = await _event(rt, person, "我叫阿明。")
    engine = _engine(
        rt,
        _decision(
            [
                _proposal(
                    "ADD", "identity.name", "阿明", event.event_id
                )
            ]
        ),
    )
    applied = await engine.apply_proposals(
        await engine.review(person.person_id, event.event_id)
    )
    assert applied.status == "PENDING_CONFIRMATION"
    pending = await rt.repos.profile.find_pending(person.person_id, "identity.name")
    assert len(pending) == 1
    return pending[0]


async def test_a_bare_actor_string_is_not_confirmation_authority(rt):
    person = await _person(rt)
    candidate = await _identity_candidate(rt, person)

    with pytest.raises(ProfileUnauthorized):
        await rt.profile_service.confirm_profile_proposal(
            candidate["candidate_id"], actor="user"
        )
    stored = await rt.repos.profile.get_candidate(candidate["candidate_id"])
    assert stored["status"] == "PENDING"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1
    assert await rt.repos.profile.list_entries(person.person_id) == []


async def test_confirmation_of_a_stale_proposal_changes_nothing(rt):
    person = await _person(rt)
    candidate = await _identity_candidate(rt, person)
    assert candidate["expected_revision"] == 1

    # An unrelated, legitimate change moves the profile to revision 2.
    event = await _event(rt, person, "我喜欢爵士。")
    engine = _engine(
        rt, _decision([_proposal("ADD", "interests.topics", "爵士乐", event.event_id)])
    )
    applied = await engine.apply_proposals(
        await engine.review(person.person_id, event.event_id)
    )
    assert applied.new_revision == 2

    with pytest.raises(ProfileProposalStale) as raised:
        await rt.profile_service.confirm_profile_proposal(
            candidate["candidate_id"], authority=TRUSTED_CONFIRMATION
        )
    assert raised.value.status == "STALE_PROPOSAL"
    assert raised.value.expected_revision == 1
    assert raised.value.current_revision == 2

    # Not confirmed, not bumped, no identity row written and no target touched.
    stored = await rt.repos.profile.get_candidate(candidate["candidate_id"])
    assert stored["status"] == "NEEDS_REVIEW"
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 2
    assert await rt.repos.profile.list_entries(
        person.person_id, field_key="identity.name"
    ) == []
    # The candidate is kept for re-review, with its evidence intact.
    assert stored["evidence_event_ids"] == candidate["evidence_event_ids"]


async def test_a_stale_proposal_becomes_confirmable_after_re_review(rt):
    person = await _person(rt)
    candidate = await _identity_candidate(rt, person)
    event = await _event(rt, person, "我喜欢爵士。")
    engine = _engine(
        rt, _decision([_proposal("ADD", "interests.topics", "爵士乐", event.event_id)])
    )
    await engine.apply_proposals(await engine.review(person.person_id, event.event_id))

    with pytest.raises(ProfileProposalStale):
        await rt.profile_service.confirm_profile_proposal(
            candidate["candidate_id"], authority=TRUSTED_CONFIRMATION
        )

    # Re-read the turn against the current revision and review again.
    turn = await rt.repos.utterance.get(candidate["evidence_event_ids"][0])
    fresh_engine = _engine(
        rt,
        _decision(
            [_proposal("ADD", "identity.name", "阿明", turn.event_id)]
        ),
    )
    reapplied = await fresh_engine.apply_proposals(
        await fresh_engine.review(person.person_id, turn.event_id)
    )
    assert reapplied.status == "PENDING_CONFIRMATION"
    fresh = [
        item
        for item in await rt.repos.profile.find_pending(
            person.person_id, "identity.name"
        )
        if item["status"] == "PENDING" and item["expected_revision"] == 2
    ]
    assert len(fresh) == 1
    assert (
        await rt.profile_service.confirm_profile_proposal(
            fresh[0]["candidate_id"], authority=TRUSTED_CONFIRMATION
        )
        == 3
    )
    active = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    )
    assert [item["content"] for item in active] == ["阿明"]


async def test_two_concurrent_confirmations_commit_at_most_once(rt):
    person = await _person(rt)
    candidate = await _identity_candidate(rt, person)

    results = await asyncio.gather(
        rt.profile_service.confirm_profile_proposal(
            candidate["candidate_id"], authority=TRUSTED_CONFIRMATION
        ),
        rt.profile_service.confirm_profile_proposal(
            candidate["candidate_id"], authority=TRUSTED_CONFIRMATION
        ),
        return_exceptions=True,
    )
    successes = [item for item in results if isinstance(item, int)]
    failures = [item for item in results if isinstance(item, Exception)]
    assert successes == [2]
    assert len(failures) == 1
    assert isinstance(failures[0], (ProfileProposalStale, ValueError))

    # Exactly one revision and one identity row: the loser left no partial
    # write behind.
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 2
    entries = await rt.repos.profile.list_entries(
        person.person_id, field_key="identity.name"
    )
    assert len(entries) == 1
    assert entries[0]["content"] == "阿明"
    stored = await rt.repos.profile.get_candidate(candidate["candidate_id"])
    assert stored["status"] == "CONFIRMED"
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert [item["revision"] for item in revisions] == [2]


async def test_confirmation_rechecks_that_target_entries_are_still_effective(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的项目叫机器狗。")
    engine = _engine(
        rt,
        _decision(
            [_proposal("ADD", "projects.active", "机器狗", event.event_id)]
        ),
    )
    applied = await engine.apply_proposals(
        await engine.review(person.person_id, event.event_id)
    )
    target_id = applied.applied[0]["entry_id"]
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision

    candidate_id = await rt.repos.profile.insert_candidate(
        person_id=person.person_id,
        field_key="projects.active",
        value="机器狗",
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_memory_ids=[],
        evidence_event_ids=[event.event_id],
        evidence_candidate_ids=[],
        reason="test archive proposal",
        operation="ARCHIVE",
        target_entry_ids=[target_id],
        expected_revision=revision,
        requires_confirmation=True,
    )

    # The target stops being effective without a revision bump (for example a
    # concurrent archive). The revision alone would look current; the target
    # recheck must still refuse.
    from companion.domain.models import utcnow

    await rt.repos.profile.update_entry(
        target_id, status="ARCHIVED", valid_until=utcnow()
    )

    with pytest.raises(ProfileProposalStale) as raised:
        await rt.profile_service.confirm_profile_proposal(
            candidate_id, authority=TRUSTED_CONFIRMATION
        )
    assert raised.value.reason == "target_entries_changed"
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision
    stored = await rt.repos.profile.get_candidate(candidate_id)
    assert stored["status"] == "NEEDS_REVIEW"
