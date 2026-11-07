"""V0.9.7.1 Section E: deletion propagates into derived profile data.

Retracting one source must not silently keep content it alone supported, and a
permanent privacy delete must not leave the private body recoverable in
entries, candidates, revisions, receipts or legacy fields.
"""

from __future__ import annotations

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource, JobType
from companion.domain.models import make_utterance_event, utcnow
from companion.jobs.profile_review import (
    ProfileReviewScheduler,
    build_profile_review_handler,
)
from companion.memory.models import AccessContext
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider
from companion.routing.models import DataControlRequest


async def _person(rt, name="Deletion"):
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


async def _entry(
    rt,
    person,
    *,
    content="机器人",
    field_key="interests.topics",
    events=(),
    memories=(),
    merged_from=None,
    status="ACTIVE",
):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key=field_key,
        content=content,
        status=status,
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_event_ids=list(events),
        evidence_memory_ids=list(memories),
        merged_from=list(merged_from or []),
    )


async def _trace(rt, person_id) -> str:
    """Every stored string that could still expose profile-derived content."""
    parts: list[str] = []
    for sql in (
        "SELECT content FROM person_profile_entries WHERE person_id = ?",
        "SELECT candidate_value_json FROM person_profile_candidates_legacy WHERE person_id = ?",
        "SELECT changes_json FROM person_profile_revisions WHERE person_id = ?",
        "SELECT changes_json, reason, last_error FROM profile_review_receipts"
        " WHERE person_id = ?",
        "SELECT value_json FROM person_profile_fields WHERE person_id = ?",
        "SELECT compact_summary FROM person_interaction_profiles WHERE person_id = ?",
    ):
        cur = await rt.db.execute(sql, (person_id,))
        for row in await cur.fetchall():
            parts.extend(str(value or "") for value in tuple(row))
    return "\n".join(parts)


async def test_losing_the_only_source_archives_the_entry(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢机器人。")
    entry_id = await _entry(rt, person, events=[event.event_id])

    result = await rt.profile_service.invalidate_profile_source(
        person.person_id, event_id=event.event_id
    )
    assert result.entries_archived == 1
    assert result.revision == 2
    entry = await rt.repos.profile.get_entry(entry_id)
    assert entry["status"] == "ARCHIVED"
    assert entry["evidence_event_ids"] == []
    effective = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE", "TENTATIVE")
    )
    assert effective == []


async def test_an_entry_with_another_independent_source_keeps_its_content(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢机器人。")
    second = await _event(rt, person, "机器人真的很有意思。")
    entry_id = await _entry(
        rt, person, content="机器人", events=[first.event_id, second.event_id]
    )

    result = await rt.profile_service.invalidate_profile_source(
        person.person_id, event_id=first.event_id
    )
    assert result.entries_updated == 1
    entry = await rt.repos.profile.get_entry(entry_id)
    assert entry["status"] == "ACTIVE"
    assert entry["content"] == "机器人"
    assert entry["evidence_event_ids"] == [second.event_id]


async def test_a_merged_entry_stops_being_effective_when_one_source_goes(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我会 STM32。")
    second = await _event(rt, person, "我也会嵌入式开发。")
    entry_id = await _entry(
        rt,
        person,
        content="STM32 嵌入式开发",
        field_key="knowledge.skills",
        events=[first.event_id, second.event_id],
        merged_from=["pfe_a", "pfe_b"],
    )

    result = await rt.profile_service.invalidate_profile_source(
        person.person_id, event_id=first.event_id
    )
    assert result.entries_archived == 1
    entry = await rt.repos.profile.get_entry(entry_id)
    # The survivors cannot be proven to support every detail of the merged
    # text, so it is suspended for review rather than silently trimmed.
    assert entry["status"] == "PENDING_REVIEW"


async def test_memory_retraction_follows_the_same_rules(rt):
    person = await _person(rt)
    entry_id = await _entry(
        rt, person, content="住在城东", events=[], memories=["mem_one"]
    )
    result = await rt.profile_service.invalidate_profile_source(
        person.person_id, memory_id="mem_one", deletion_mode="RETRACTION"
    )
    assert result.entries_archived == 1
    assert (await rt.repos.profile.get_entry(entry_id))["status"] == "ARCHIVED"


async def _secret_profile(rt, person):
    """Two applied entries and one review receipt for one event."""
    event = await _event(rt, person, "我的秘密代号是蓝鲸。")
    proposal = ProfileChangeProposal(
        operation="ADD",
        field_key="interests.topics",
        proposed_content="蓝鲸",
        evidence_event_ids=[event.event_id],
        confidence=0.95,
    )
    pending = ProfileChangeProposal(
        operation="ADD",
        field_key="goals.general",
        proposed_content="蓝鲸计划",
        evidence_event_ids=[event.event_id],
        confidence=0.6,
        requires_confirmation=True,
    )
    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(
            ProfileReviewDecision(
                person_id="ignored",
                base_revision=0,
                decision="PROPOSE_CHANGES",
                reason="test",
                changes=[proposal, pending],
            )
        ),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    applied = await engine.apply_proposals(
        await engine.review(person.person_id, event.event_id)
    )
    assert applied.applied

    scheduler = ProfileReviewScheduler(rt.repos.profile, rt.repos.job)
    review_id = await scheduler.request(person.person_id, event.event_id)
    await rt.repos.profile.update_review_receipt(
        review_id,
        status="COMPLETED",
        decision="PROPOSE_CHANGES",
        changes=[{"content": "蓝鲸", "note": "蓝鲸计划"}],
        completed=True,
    )
    # A legacy field derived from the same event must also be scrubbed.
    await rt.repos.profile.upsert_field(
        person.person_id,
        "background.occupation",
        value="蓝鲸研究员",
        source="INTERACTION_INFERENCE",
        confidence=0.8,
        evidence_memory_ids=[],
        evidence_event_ids=[event.event_id],
        evidence_candidate_ids=[],
        revision=1,
    )
    return event, review_id


async def test_privacy_delete_leaves_no_recoverable_private_body(rt):
    person = await _person(rt)
    event, review_id = await _secret_profile(rt, person)
    entry_ids = [
        item["entry_id"]
        for item in await rt.repos.profile.list_entries(person.person_id)
    ]
    assert entry_ids

    result = await rt.profile_service.invalidate_profile_source(
        person.person_id, event_id=event.event_id, deletion_mode="PRIVACY_DELETE"
    )
    assert result.entries_deleted >= 2

    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert (
        await rt.repos.profile.list_legacy_candidates(person.person_id)
    ) == []
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["changes"] == []
    field = await rt.repos.profile.get_field(
        person.person_id, "background.occupation"
    )
    assert field["status"] == "RETIRED"
    assert field["evidence_event_ids"] == []

    stored = await _trace(rt, person.person_id)
    assert "蓝鲸" not in stored
    # Revision numbers stay continuous even though their bodies are redacted.
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert [item["revision"] for item in revisions] == [2, 3]


async def test_data_control_event_delete_propagates_to_the_profile(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的秘密代号是蓝鲸。")
    entry_id = await _entry(rt, person, content="蓝鲸", events=[event.event_id])
    scheduler = ProfileReviewScheduler(rt.repos.profile, rt.repos.job)
    review_id = await scheduler.request(person.person_id, event.event_id)
    await rt.repos.profile.update_review_receipt(
        review_id,
        status="PENDING",
        changes=[{"content": "蓝鲸"}],
    )

    pending = await rt.data_control.resolve_request(
        DataControlRequest(
            operation="DELETE",
            target_type="EVENT",
            target_id=event.event_id,
            scope="SELF",
        ),
        event,
    )
    result = await rt.data_control.execute_confirmed(pending.operation_id)
    assert result.status == "COMPLETED"
    assert await rt.repos.utterance.get(event.event_id) is None
    assert await rt.repos.profile.get_entry(entry_id) is None
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "CANCELLED"
    assert receipt["changes"] == []
    assert "蓝鲸" not in await _trace(rt, person.person_id)


async def test_data_control_memory_delete_propagates_to_the_profile(rt):
    person = await _person(rt)
    event = await _event(rt, person, "随便聊聊。")
    memory = await rt.memory_service.create(
        kind="EPISODE",
        content="蓝鲸的秘密",
        access=AccessContext("admin", is_admin=True),
        visibility="PRIVATE",
        subjects=[("PERSON", person.person_id)],
    )
    entry_id = await _entry(
        rt, person, content="蓝鲸", events=[], memories=[memory.memory_id]
    )

    pending = await rt.data_control.resolve_request(
        DataControlRequest(
            operation="DELETE",
            target_type="MEMORY",
            target_id=memory.memory_id,
            scope="SELF",
        ),
        event,
    )
    result = await rt.data_control.execute_confirmed(pending.operation_id)
    assert result.status == "COMPLETED"
    assert await rt.memory_service.repo.get(memory.memory_id) is None
    assert await rt.repos.profile.get_entry(entry_id) is None
    assert "蓝鲸" not in await _trace(rt, person.person_id)


async def test_a_mid_delete_failure_rolls_back_and_is_not_reported_complete(
    rt, monkeypatch
):
    person = await _person(rt)
    event = await _event(rt, person, "我的秘密代号是蓝鲸。")
    entry_id = await _entry(rt, person, content="蓝鲸", events=[event.event_id])
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision

    async def explode(entry_id):
        raise RuntimeError("simulated mid-delete failure")

    monkeypatch.setattr(rt.repos.profile, "delete_entry", explode)

    pending = await rt.data_control.resolve_request(
        DataControlRequest(
            operation="DELETE",
            target_type="EVENT",
            target_id=event.event_id,
            scope="SELF",
        ),
        event,
    )
    result = await rt.data_control.execute_confirmed(pending.operation_id)
    assert result.status == "FAILED"

    # The raw event and the profile row were not lost while invalidation
    # failed: one transaction means all or nothing.
    assert await rt.repos.utterance.get(event.event_id) is not None
    entry = await rt.repos.profile.get_entry(entry_id)
    assert entry is not None and entry["status"] == "ACTIVE"
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision


async def test_a_deleted_event_is_never_relearned(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的秘密代号是蓝鲸。")
    scheduler = ProfileReviewScheduler(
        rt.repos.profile, rt.repos.job, persons=rt.repos.person
    )
    review_id = await scheduler.request(person.person_id, event.event_id)

    pending = await rt.data_control.resolve_request(
        DataControlRequest(
            operation="DELETE",
            target_type="EVENT",
            target_id=event.event_id,
            scope="SELF",
        ),
        event,
    )
    await rt.data_control.execute_confirmed(pending.operation_id)
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "CANCELLED"

    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(
            ProfileReviewDecision(
                person_id="ignored",
                base_revision=0,
                decision="PROPOSE_CHANGES",
                reason="test",
                changes=[
                    ProfileChangeProposal(
                        operation="ADD",
                        field_key="interests.topics",
                        proposed_content="蓝鲸",
                        evidence_event_ids=[event.event_id],
                        confidence=0.99,
                    )
                ],
            )
        ),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        engine
    )
    await rt.worker.run_until_idle()

    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "CANCELLED"
    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert "蓝鲸" not in await _trace(rt, person.person_id)
