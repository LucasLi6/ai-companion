"""V0.9.7.1 Section G: the review snapshot and candidate-free decisions.

The review model sees one consistent, budgeted view of the profile. Only
effective entries and legacy fields are visible; nothing is presented as a
"pending proposal" because V0.9.9.6 removed that queue entirely.
"""

from __future__ import annotations

import json

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_refinement import (
    MAX_SNAPSHOT_LEGACY_FIELDS,
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


async def test_requires_confirmation_flag_does_not_block_a_legal_write(rt):
    """The legacy flag is accepted but ignored: local evidence decides."""
    person = await _person(rt)
    event = await _event(rt, person, "我想跑马拉松。")
    applied = await _review(
        rt,
        person,
        event,
        _proposal(
            "ADD", "goals.general", "跑马拉松", event.event_id, pending=True
        ),
    )
    assert applied.status == "APPLIED"
    assert len(applied.applied) == 1

    snapshot = await rt.profile_service.get_full_snapshot(person.person_id)
    assert snapshot.revision == 2
    assert [item.content for item in snapshot.entries["goals.general"]] == [
        "跑马拉松"
    ]
    assert not hasattr(snapshot, "pending_candidates")

    payload = snapshot.model_input()
    assert "pending_proposals" not in payload
    assert "跑马拉松" in json.dumps(
        payload["existing_profile"], ensure_ascii=False
    )


async def test_an_unsupported_proposal_is_rejected_and_never_queued(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天天气不错。")
    applied = await _review(
        rt,
        person,
        event,
        _proposal("ADD", "goals.general", "跑马拉松", event.event_id),
    )
    assert applied.status == "REJECTED"
    assert applied.applied == []
    assert applied.rejected
    assert applied.rejected[0]["reason"] in {
        "content_not_supported_by_evidence",
        "content_not_in_evidence",
    }

    snapshot = await rt.profile_service.get_full_snapshot(person.person_id)
    assert snapshot.revision == 1
    assert snapshot.goals == {}
    assert snapshot.entries == {}
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []


async def test_snapshot_does_not_expose_legacy_candidate_rows(rt):
    """A stale archive row never leaks into the model's context."""
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器人。")
    now = "2026-01-01T00:00:00.000Z"
    await rt.repos.profile.db.execute(
        "INSERT INTO person_profile_candidates_legacy("
        " candidate_id, person_id, field_key, candidate_value_json, source,"
        " confidence, status, evidence_event_ids_json, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "cand_legacy_1",
            person.person_id,
            "identity.name",
            json.dumps("阿明", ensure_ascii=False),
            "INTERACTION_INFERENCE",
            0.9,
            "PENDING",
            json.dumps([event.event_id]),
            now,
            now,
        ),
    )
    snapshot = await rt.profile_service.get_full_snapshot(person.person_id)
    payload = snapshot.model_input()
    assert "pending_proposals" not in payload
    assert "阿明" not in json.dumps(payload, ensure_ascii=False)
    assert snapshot.entries == {}


def test_snapshot_model_input_has_explicit_budgets():
    legacy = {
        f"legacy.field_{index:03d}": "y" * 50
        for index in range(MAX_SNAPSHOT_LEGACY_FIELDS + 5)
    }
    snapshot = FullProfileSnapshot(
        person_id="person_budget",
        revision=7,
        legacy_fields=legacy,
        compact_summary="z" * (MAX_SNAPSHOT_TEXT_CHARS + 100),
    )

    payload = snapshot.model_input()
    assert "pending_proposals" not in payload
    assert len(payload["legacy_unconverted"]) == MAX_SNAPSHOT_LEGACY_FIELDS
    assert len(payload["compact_summary"]) == MAX_SNAPSHOT_TEXT_CHARS
    assert payload["snapshot_limits"] == {
        "legacy_total": MAX_SNAPSHOT_LEGACY_FIELDS + 5,
        "legacy_included": MAX_SNAPSHOT_LEGACY_FIELDS,
        "truncated": True,
    }
    assert payload["profile_revision"] == 7
