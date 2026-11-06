"""V0.9.7.2 Section A: Event -> Memory -> Profile source graph.

A profile entry may cite only a *derived memory* and never the utterance that
produced it. Deleting the utterance must therefore collect the complete source
set (event + derived memories) before anything is removed, invalidate every
affected profile row in one transaction, and leave no private body behind in
candidates, revisions, receipts or the compiled summary.
"""

from __future__ import annotations

import logging

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.jobs.profile_review import (
    ProfileReviewScheduler,
    build_profile_review_handler,
    reconcile_profile_review_jobs,
)
from companion.memory.models import AccessContext, MemoryVisibility
from companion.people.profile_compiler import compile_profile_context
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
    ValidatedChange,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider
from companion.routing.models import DataControlRequest

LOG = logging.getLogger("test.v0972_source_graph")


async def _person(rt, name="Graph"):
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


async def _memory(rt, person, content, source_event_id):
    item = await rt.memory_service.create(
        kind="PROJECT_STATE",
        content=content,
        access=AccessContext("admin", is_admin=True),
        visibility=MemoryVisibility.PRIVATE,
        subjects=[("PERSON", person.person_id)],
    )
    async with rt.repos.db.transaction():
        await rt.memory_service.repo.add_provenance(
            item.memory_id,
            source_event_id=source_event_id,
            candidate_id=None,
            speaker_person_id=person.person_id,
            subject_person_id=person.person_id,
        )
    return item.memory_id


async def _entry(rt, person, *, content, events=(), memories=()):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="interests.topics",
        content=content,
        status="ACTIVE",
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_event_ids=list(events),
        evidence_memory_ids=list(memories),
    )


async def _trace(rt, person_id) -> str:
    """Every stored string that could still expose deleted content."""
    parts: list[str] = []
    for sql in (
        "SELECT content FROM person_profile_entries WHERE person_id = ?",
        "SELECT candidate_value_json FROM person_profile_candidates WHERE person_id = ?",
        "SELECT changes_json, reason, last_error FROM profile_review_receipts"
        " WHERE person_id = ?",
        "SELECT changes_json FROM person_profile_revisions WHERE person_id = ?",
        "SELECT value_json FROM person_profile_fields WHERE person_id = ?",
        "SELECT compact_summary FROM person_interaction_profiles WHERE person_id = ?",
    ):
        cur = await rt.db.execute(sql, (person_id,))
        for row in await cur.fetchall():
            parts.extend(str(value or "") for value in tuple(row))
    cur = await rt.db.execute("SELECT payload FROM processing_jobs")
    for row in await cur.fetchall():
        parts.append(str(row["payload"] or ""))
    return "\n".join(parts)


async def _delete_event(rt, event):
    pending = await rt.data_control.resolve_request(
        DataControlRequest(
            operation="DELETE",
            target_type="EVENT",
            target_id=event.event_id,
            scope="SELF",
        ),
        event,
    )
    return await rt.data_control.execute_confirmed(pending.operation_id)


def _handler(rt, provider):
    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        provider,
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    return engine, build_profile_review_handler(engine)


async def test_event_delete_removes_a_memory_only_entry_and_all_its_bodies(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的秘密代号是蓝鲸。")
    memory_id = await _memory(rt, person, "代号蓝鲸", event.event_id)
    # The entry and candidate cite only the derived memory, never the event.
    entry_id = await _entry(rt, person, content="蓝鲸项目", memories=[memory_id])
    candidate_id = await rt.repos.profile.insert_candidate(
        person_id=person.person_id,
        field_key="interests.topics",
        value="蓝鲸",
        source="INTERACTION_INFERENCE",
        confidence=0.7,
        evidence_memory_ids=[memory_id],
        evidence_event_ids=[],
        evidence_candidate_ids=[],
        reason="derived_memory_only",
    )
    scheduler = ProfileReviewScheduler(rt.repos.profile, rt.repos.job)
    review_id = await scheduler.request(person.person_id, event.event_id)
    await rt.repos.profile.update_review_receipt(
        review_id,
        status="PENDING",
        changes=[{"content": "蓝鲸项目", "note": "代号蓝鲸"}],
    )

    result = await _delete_event(rt, event)
    assert result.status == "COMPLETED"

    assert await rt.repos.utterance.get(event.event_id) is None
    assert await rt.memory_service.repo.get(memory_id) is None
    assert await rt.repos.profile.get_entry(entry_id) is None
    assert await rt.repos.profile.get_candidate(candidate_id) is None
    assert await rt.repos.profile.list_candidates(person.person_id) == []

    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "CANCELLED"
    assert receipt["changes"] == []
    assert "蓝鲸" not in await _trace(rt, person.person_id)

    # The summary was recomputed, not left stale: the deleted entry's content
    # is gone even though the block always opens with the identity header.
    profile = await rt.profile_service.get(person.person_id)
    assert profile.entries == {}
    assert "蓝鲸" not in profile.compact_summary
    assert profile.compact_summary == compile_profile_context(profile)
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert revisions  # revision numbers stay continuous for audit
    assert [item["revision"] for item in revisions] == sorted(
        item["revision"] for item in revisions
    )


async def test_a_deleted_event_and_memory_are_never_valid_evidence_again(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的秘密代号是蓝鲸。")
    memory_id = await _memory(rt, person, "代号蓝鲸", event.event_id)

    await _delete_event(rt, event)

    revision = (await rt.profile_service.get(person.person_id)).metadata.revision
    decision = type(
        "Decision",
        (),
        {
            "status": "PROPOSE_CHANGES",
            "base_revision": revision,
            "reason": "hand built",
            "changes": [
                ValidatedChange(
                    proposal=ProfileChangeProposal(
                        operation="ADD",
                        field_key="interests.topics",
                        proposed_content="蓝鲸",
                        evidence_event_ids=[event.event_id],
                        confidence=1.0,
                    ),
                    auto_apply=True,
                )
            ],
        },
    )()
    applied = await rt.profile_service.apply_refinement(
        person.person_id,
        decision,
        source_event_id=event.event_id,
    )
    assert applied.status == "NO_CHANGE"
    assert applied.rejected[0]["reason"] == "evidence_event_not_accessible"
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision
    assert await rt.repos.profile.list_entries(person.person_id) == []
    # A rejected proposal is not stored, so it cannot re-introduce the secret.
    assert await rt.repos.profile.list_candidates(person.person_id) == []
    assert await rt.memory_service.repo.get(memory_id) is None


async def test_the_worker_cannot_relearn_a_deleted_source(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的秘密代号是蓝鲸。")
    memory_id = await _memory(rt, person, "代号蓝鲸", event.event_id)
    await _entry(rt, person, content="蓝鲸项目", memories=[memory_id])
    scheduler = ProfileReviewScheduler(rt.repos.profile, rt.repos.job)
    review_id = await scheduler.request(person.person_id, event.event_id)

    await _delete_event(rt, event)
    jobs_before = await rt.repos.job.list_jobs()

    # Reconcile must cancel, never re-enqueue.
    fixed = await reconcile_profile_review_jobs(rt.repos, person_id=person.person_id)
    assert fixed["enqueued"] == 0
    assert fixed["cancelled"] == 0  # the deletion already cancelled it

    # Requesting the same deleted event again must not create a new job.
    again = await scheduler.request(person.person_id, event.event_id)
    assert again == review_id
    jobs_after = await rt.repos.job.list_jobs()
    assert {job.job_key for job in jobs_after} == {
        job.job_key for job in jobs_before
    }

    # A handler that would happily re-add the secret never runs the model.
    provider = ScriptedProfileRefinementProvider(
        [
            ProfileReviewDecision(
                person_id="ignored",
                base_revision=0,
                decision="PROPOSE_CHANGES",
                reason="must never run",
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
        ]
    )
    _engine, handler = _handler(rt, provider)
    await handler(
        {"person_id": person.person_id, "source_event_id": event.event_id},
        rt.repos,
        LOG,
    )
    assert provider.payloads == []
    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert "蓝鲸" not in await _trace(rt, person.person_id)


async def test_aggregate_invalidation_bumps_the_revision_once(rt):
    person = await _person(rt)
    one = await _entry(rt, person, content="机器人", memories=["mem-a"])
    two = await _entry(rt, person, content="摄影", memories=["mem-b"])
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision

    result = await rt.profile_service.invalidate_profile_sources(
        person.person_id,
        memory_ids=("mem-a", "mem-b"),
        deletion_mode="PRIVACY_DELETE",
    )
    assert result.entries_deleted == 2
    assert result.revision == revision + 1
    assert await rt.repos.profile.get_entry(one) is None
    assert await rt.repos.profile.get_entry(two) is None
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert [item["revision"] for item in revisions] == [revision + 1]
