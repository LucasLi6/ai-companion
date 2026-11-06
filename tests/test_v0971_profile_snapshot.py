"""V0.9.7.1 Section G: the review snapshot and pending-candidate handling.

The review model sees one consistent, budgeted view of the profile. A pending
proposal is visibly *pending* and never presented as confirmed understanding.
Repeated identical proposals merge their evidence instead of piling up rows,
and a proposal built on an older revision is parked for re-review.
"""

from __future__ import annotations

import json

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_refinement import (
    MAX_SNAPSHOT_LEGACY_FIELDS,
    MAX_SNAPSHOT_PENDING,
    MAX_SNAPSHOT_TEXT_CHARS,
    FullProfileSnapshot,
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider


async def _person(rt, name="Snapshot"):
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


def _engine(rt, *decisions):
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(list(decisions)),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )


def _decision(*proposals):
    return ProfileReviewDecision(
        person_id="ignored",
        base_revision=0,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=list(proposals),
    )


def _proposal(
    operation,
    field_key,
    content,
    event_id,
    *,
    targets=(),
    confidence=0.95,
    pending=False,
):
    return ProfileChangeProposal(
        operation=operation,
        field_key=field_key,
        target_entry_ids=list(targets),
        proposed_content=content,
        evidence_event_ids=[event_id] if event_id else [],
        confidence=confidence,
        requires_confirmation=pending,
    )


async def _review(rt, person, event, *proposals):
    engine = _engine(rt, _decision(*proposals))
    return await engine.apply_proposals(
        await engine.review(person.person_id, event.event_id)
    )


async def test_snapshot_carries_identity_relationship_entries_and_revision(rt):
    person = await _person(rt)
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="identity.name",
        content="小林",
        status="ACTIVE",
        source="USER_CONFIRMED",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )
    event = await _event(rt, person, "我在做机器狗。")
    applied = await _review(
        rt, person, event, _proposal("ADD", "projects.active", "机器狗", event.event_id)
    )
    assert applied.applied
    entry_id = applied.applied[0]["entry_id"]

    snapshot = await rt.profile_service.get_full_snapshot(person.person_id)
    assert snapshot.revision == 2
    assert snapshot.identity["name"] == "小林"
    assert snapshot.identity["preferred_name"] == "UNKNOWN"
    assert snapshot.relationship["familiarity"] == "STRANGER"
    assert entry_id in snapshot.entry_ids()
    assert snapshot.entries["projects.active"][0].content == "机器狗"

    payload = snapshot.model_input()
    assert payload["profile_revision"] == 2
    assert payload["existing_profile"]["identity"]["name"] == "小林"
    assert payload["existing_profile"]["relationship"]["familiarity"] == "STRANGER"
    # The snapshot's display name is a registration label, never a learned fact.
    assert payload["existing_profile"]["display_name_is_registration_label"] is True
    assert payload["entry_ids"]["projects.active"] == [entry_id]
    assert payload["existing_profile"]["projects"]["active_projects"][0][
        "entry_id"
    ] == entry_id


async def test_a_pending_proposal_is_never_shown_as_confirmed_fact(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我明年可能想跑马拉松。")
    applied = await _review(
        rt,
        person,
        event,
        _proposal(
            "ADD", "goals.general", "跑马拉松", event.event_id, pending=True
        ),
    )
    assert applied.applied == []
    assert applied.pending

    snapshot = await rt.profile_service.get_full_snapshot(person.person_id)
    assert snapshot.revision == 1
    assert snapshot.goals == {}
    assert snapshot.entries == {}
    assert len(snapshot.pending_candidates) == 1
    candidate = snapshot.pending_candidates[0]
    assert candidate["proposed_content"] == "跑马拉松"
    assert candidate["status"] == "PENDING"

    payload = snapshot.model_input()
    confirmed = json.dumps(payload["existing_profile"], ensure_ascii=False)
    assert "跑马拉松" not in confirmed
    assert payload["pending_proposals"][0]["proposed_content"] == "跑马拉松"
    assert payload["pending_proposals"][0]["status"] == "PENDING"
    assert payload["entry_ids"] == {}


async def test_identical_pending_proposals_merge_but_add_and_update_do_not(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我想跑马拉松。")
    second = await _event(rt, person, "我真的想跑马拉松。")
    await _review(
        rt, person, first,
        _proposal("ADD", "goals.general", "跑马拉松", first.event_id, pending=True),
    )
    await _review(
        rt, person, second,
        _proposal("ADD", "goals.general", "跑马拉松", second.event_id, pending=True),
    )
    candidates = await rt.repos.profile.list_candidates(person.person_id)
    assert len(candidates) == 1
    assert set(candidates[0]["evidence_event_ids"]) == {
        first.event_id,
        second.event_id,
    }

    # The same words with a different operation or target is a different
    # proposal and must not be folded into the pending ADD.
    base = await _event(rt, person, "我平时种花。")
    await _review(
        rt, person, base,
        _proposal("ADD", "interests.topics", "园艺", base.event_id),
    )
    entry = (
        await rt.repos.profile.list_entries(person.person_id, field_key="interests.topics")
    )[0]
    third = await _event(rt, person, "园艺还在坚持。")
    await _review(
        rt, person, third,
        _proposal(
            "UPDATE",
            "interests.topics",
            "园艺",
            third.event_id,
            targets=[entry["entry_id"]],
            pending=True,
        ),
    )
    candidates = await rt.repos.profile.list_candidates(person.person_id)
    assert len(candidates) == 2
    assert {item["operation"] for item in candidates} == {"ADD", "UPDATE"}


async def test_a_proposal_based_on_an_old_revision_is_parked_and_reissued(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我想跑马拉松。")
    await _review(
        rt, person, first,
        _proposal("ADD", "goals.general", "跑马拉松", first.event_id, pending=True),
    )
    old = (await rt.repos.profile.list_candidates(person.person_id))[0]
    assert old["expected_revision"] == 1

    moved = await _event(rt, person, "我喜欢摄影。")
    applied = await _review(
        rt, person, moved,
        _proposal("ADD", "interests.topics", "摄影", moved.event_id),
    )
    assert applied.applied
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision
    assert revision == 2

    again = await _event(rt, person, "我还是想跑马拉松。")
    await _review(
        rt, person, again,
        _proposal("ADD", "goals.general", "跑马拉松", again.event_id, pending=True),
    )
    candidates = await rt.repos.profile.list_candidates(
        person.person_id, status=None
    )
    assert len(candidates) == 2
    stale = [item for item in candidates if item["candidate_id"] == old["candidate_id"]]
    assert len(stale) == 1
    assert stale[0]["status"] == "NEEDS_REVIEW"
    assert stale[0]["reason"] == "stale_base_revision"
    fresh = [item for item in candidates if item["candidate_id"] != old["candidate_id"]]
    assert len(fresh) == 1
    assert fresh[0]["status"] == "PENDING"
    assert fresh[0]["expected_revision"] == revision

    # The snapshot exposes both and keeps the revision current.
    snapshot = await rt.profile_service.get_full_snapshot(person.person_id)
    assert snapshot.revision == revision
    statuses = {
        item["candidate_id"]: item["status"]
        for item in snapshot.pending_candidates
    }
    assert statuses[old["candidate_id"]] == "NEEDS_REVIEW"


def test_snapshot_model_input_has_explicit_budgets():
    pending = [
        {
            "candidate_id": f"cand_{index:03d}",
            "field_key": "goals.general",
            "operation": "ADD",
            "proposed_content": "x" * 500,
            "status": "PENDING",
            "review_status": "PENDING",
        }
        for index in range(MAX_SNAPSHOT_PENDING + 5)
    ]
    legacy = {
        f"legacy.field_{index:03d}": "y" * 50
        for index in range(MAX_SNAPSHOT_LEGACY_FIELDS + 5)
    }
    snapshot = FullProfileSnapshot(
        person_id="person_budget",
        revision=7,
        pending_candidates=pending,
        legacy_fields=legacy,
        compact_summary="z" * (MAX_SNAPSHOT_TEXT_CHARS + 100),
    )

    payload = snapshot.model_input()
    assert len(payload["pending_proposals"]) == MAX_SNAPSHOT_PENDING
    assert len(payload["legacy_unconverted"]) == MAX_SNAPSHOT_LEGACY_FIELDS
    assert len(payload["compact_summary"]) == MAX_SNAPSHOT_TEXT_CHARS
    assert all(
        len(item["proposed_content"]) <= 240
        for item in payload["pending_proposals"]
    )
    assert payload["snapshot_limits"] == {
        "pending_total": MAX_SNAPSHOT_PENDING + 5,
        "pending_included": MAX_SNAPSHOT_PENDING,
        "legacy_total": MAX_SNAPSHOT_LEGACY_FIELDS + 5,
        "legacy_included": MAX_SNAPSHOT_LEGACY_FIELDS,
        "truncated": True,
    }
    assert payload["profile_revision"] == 7
