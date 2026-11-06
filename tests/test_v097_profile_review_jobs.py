"""V0.9.7: durable profile-review jobs, ordering, retry and recovery."""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource, JobType
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.jobs.profile_review import (
    JobError,
    ProfileReviewScheduler,
    build_profile_review_handler,
    profile_review_job_key,
)
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileRefinementProviderError,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider

LOG = logging.getLogger("test.profile_review")


async def _person(rt, *, external_ai=True):
    person = await rt.registry.register(
        "Jobs",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
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


def _scheduler(rt):
    return ProfileReviewScheduler(
        rt.repos.profile, rt.repos.job, persons=rt.repos.person, delay_seconds=0
    )


def _engine(rt, provider):
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        provider,
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )


def _add(field_key, content, event_id):
    return ProfileReviewDecision(
        person_id="person_placeholder",
        base_revision=0,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=[
            ProfileChangeProposal(
                operation="ADD",
                field_key=field_key,
                proposed_content=content,
                evidence_event_ids=[event_id],
                confidence=0.95,
            )
        ],
    )


async def test_local_only_person_never_gets_a_review_job(rt):
    person = await _person(rt, external_ai=False)
    event = await _event(rt, person, "你好。")
    scheduler = _scheduler(rt)
    assert await scheduler.request(person.person_id, event.event_id) is None
    assert await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    ) is None
    assert await rt.repos.profile.list_review_receipts(person.person_id) == []
    assert [job for job in await rt.repos.job.pending() if job.job_type == JobType.PROFILE_REVIEW] == []


async def test_repeated_requests_share_one_receipt_and_job(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器人。")
    scheduler = _scheduler(rt)
    first = await scheduler.request(person.person_id, event.event_id)
    second = await scheduler.request(person.person_id, event.event_id)
    assert first and first == second
    receipts = await rt.repos.profile.list_review_receipts(person.person_id)
    assert len(receipts) == 1
    jobs = [
        job
        for job in await rt.repos.job.pending()
        if job.job_type == JobType.PROFILE_REVIEW
    ]
    assert len(jobs) == 1
    assert jobs[0].job_key == profile_review_job_key(
        person.person_id, event.event_id
    )
    assert jobs[0].payload == {
        "person_id": person.person_id,
        "source_event_id": event.event_id,
    }


async def test_worker_processes_oldest_event_first_and_reads_latest_profile(rt):
    person = await _person(rt)
    first_event = await _event(rt, person, "我在做机器人。")
    second_event = await _event(rt, person, "最近又开始听爵士。", "c2")
    scheduler = _scheduler(rt)
    await scheduler.request(person.person_id, first_event.event_id)
    await scheduler.request(person.person_id, second_event.event_id)

    provider = ScriptedProfileRefinementProvider(
        [
            _add("projects.active", "机器人", first_event.event_id),
            _add("interests.topics", "爵士乐", second_event.event_id),
        ]
    )
    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(rt, provider)
    )
    await rt.worker.run_until_idle()

    receipts = await rt.repos.profile.list_review_receipts(person.person_id)
    assert [item["status"] for item in receipts] == ["COMPLETED", "COMPLETED"]
    assert [item["source_event_id"] for item in receipts] == [
        first_event.event_id,
        second_event.event_id,
    ]
    assert [item["result_revision"] for item in receipts] == [2, 3]
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert [item["revision"] for item in revisions] == [2, 3]

    # The second review already saw the first turn's committed understanding.
    assert len(provider.payloads) == 2
    second_profile = provider.payloads[1]["existing_profile"]
    assert [item["content"] for item in second_profile["projects"]["active_projects"]] == [
        "机器人"
    ]


async def test_retry_after_provider_failure_applies_exactly_once(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机械臂。")
    scheduler = _scheduler(rt)
    await scheduler.request(person.person_id, event.event_id)

    provider = ScriptedProfileRefinementProvider(
        [_add("projects.active", "机械臂", event.event_id)]
    )
    provider.error = ProfileRefinementProviderError("api down")
    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(rt, provider)
    )
    await rt.worker.run_once()

    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "PENDING"
    assert "api down" in (receipt["last_error"] or "")
    assert receipt["attempt_count"] == 1
    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1

    provider.error = None
    await rt.worker.run_until_idle()
    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "COMPLETED"
    assert receipt["attempt_count"] == 1
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2
    assert [
        item["content"]
        for item in await rt.repos.profile.list_entries(
            person.person_id, statuses=("ACTIVE",)
        )
    ] == ["机械臂"]

    # A replayed job sees a terminal receipt and does no second write.
    handler = rt.worker.handlers[JobType.PROFILE_REVIEW.value]
    await handler(
        {"person_id": person.person_id, "source_event_id": event.event_id},
        rt.repos,
        LOG,
    )
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2
    assert len(await rt.repos.profile.list_revisions(person.person_id)) == 1


async def test_missing_payload_is_rejected_without_side_effects(rt):
    handler = build_profile_review_handler(_engine(rt, ScriptedProfileRefinementProvider()))
    with pytest.raises(JobError):
        await handler({"person_id": "x"}, rt.repos, LOG)
    with pytest.raises(JobError):
        await handler({"person_id": "x", "source_event_id": "evt"}, rt.repos, LOG)
    assert await rt.repos.job.pending() == []


async def test_crashed_running_receipt_is_recovered(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器人。")
    scheduler = _scheduler(rt)
    review_id = await scheduler.request(person.person_id, event.event_id)
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    await rt.db.execute(
        "UPDATE profile_review_receipts SET status = 'RUNNING', updated_at = ?"
        " WHERE review_id = ?",
        ((utcnow() - timedelta(seconds=1200)).isoformat(), review_id),
    )
    assert receipt is not None

    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(
            rt, ScriptedProfileRefinementProvider(
                [_add("projects.active", "机器人", event.event_id)]
            )
        )
    )
    await rt.worker.run_until_idle()
    completed = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert completed["status"] == "COMPLETED"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2


async def test_withdrawn_external_ai_marks_receipt_skipped(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器人。")
    scheduler = _scheduler(rt)
    review_id = await scheduler.request(person.person_id, event.event_id)

    consent = await rt.registry.get_consent(person.person_id)
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=consent.allow_processing,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    handler = build_profile_review_handler(
        _engine(
            rt, ScriptedProfileRefinementProvider(
                [_add("projects.active", "机器人", event.event_id)]
            )
        )
    )
    await handler(
        {"person_id": person.person_id, "source_event_id": event.event_id},
        rt.repos,
        LOG,
    )
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "SKIPPED"
    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1


async def test_a_committed_turn_records_its_review_request(rt, say):
    person = await _person(rt)
    report = await say(person.person_id, "agent", "我在做机器人。", conversation="turn")
    assert report is not None
    receipts = await rt.repos.profile.list_review_receipts(person.person_id)
    assert len(receipts) == 1
    assert receipts[0]["source_event_id"] == report.event_id
    assert receipts[0]["status"] == "PENDING"

    # The chat committed; the delayed job is durable but invisible to a turn.
    job = await rt.repos.job.get_by_key(
        profile_review_job_key(person.person_id, report.event_id)
    )
    assert job is not None
    assert job.job_type == JobType.PROFILE_REVIEW
    assert job.next_run_at is not None
    assert job.next_run_at > utcnow()
