"""V0.9.7.1 Section F: per-person serial review, leases and recovery.

One person is reviewed by at most one execution at a time, across workers. A
lost lease can never commit. A job that exhausts its retries leaves a terminal
receipt, and reconciliation repairs orphaned work without replaying a change.
"""

from __future__ import annotations

import logging

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource, JobStatus, JobType
from companion.domain.models import make_utterance_event
from companion.jobs.profile_review import (
    JobError,
    ProfileReviewScheduler,
    build_profile_review_handler,
    profile_review_job_key,
    reconcile_profile_review_jobs,
)
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileRefinementProviderError,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider
from companion.tasks.models import TaskPaused

LOG = logging.getLogger("test.v0971_jobs")


async def _person(rt, name="Jobs"):
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


def _scheduler(rt):
    return ProfileReviewScheduler(rt.repos.profile, rt.repos.job)


def _engine(rt, provider):
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        provider,
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )


def _add(field_key, content, event_id, *, confidence=0.95):
    return ProfileReviewDecision(
        person_id="ignored",
        base_revision=0,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=[
            ProfileChangeProposal(
                operation="ADD",
                field_key=field_key,
                proposed_content=content,
                evidence_event_ids=[event_id],
                confidence=confidence,
            )
        ],
    )


async def _payload(person, event):
    return {"person_id": person.person_id, "source_event_id": event.event_id}


async def test_a_live_person_lease_blocks_a_second_review(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    await _scheduler(rt).request(person.person_id, event.event_id)

    handler = build_profile_review_handler(
        _engine(rt, ScriptedProfileRefinementProvider(
            [_add("projects.active", "机器狗", event.event_id)]
        ))
    )
    assert await rt.repos.profile.acquire_review_lease(
        person.person_id, "another-worker"
    )
    with pytest.raises(TaskPaused):
        await handler(await _payload(person, event), rt.repos, LOG)

    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "PENDING"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1
    assert await rt.repos.profile.list_entries(person.person_id) == []

    await rt.repos.profile.release_review_lease(person.person_id, "another-worker")
    await handler(await _payload(person, event), rt.repos, LOG)
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2


async def test_different_people_do_not_block_each_other(rt):
    blocked = await _person(rt, "Blocked")
    free = await _person(rt, "Free")
    blocked_event = await _event(rt, blocked, "我在做机器狗。", "c1")
    free_event = await _event(rt, free, "我喜欢爵士。", "c2")
    await _scheduler(rt).request(blocked.person_id, blocked_event.event_id)
    await _scheduler(rt).request(free.person_id, free_event.event_id)

    assert await rt.repos.profile.acquire_review_lease(
        blocked.person_id, "other-worker"
    )
    handler = build_profile_review_handler(
        _engine(
            rt,
            ScriptedProfileRefinementProvider(
                [_add("interests.topics", "爵士乐", free_event.event_id)]
            ),
        )
    )
    await handler(await _payload(free, free_event), rt.repos, LOG)

    receipt = await rt.repos.profile.get_review_receipt(
        free.person_id, free_event.event_id
    )
    assert receipt["status"] == "COMPLETED"
    assert (await rt.profile_service.get(free.person_id)).metadata.revision == 2


async def test_an_expired_lease_can_be_taken_over_and_is_owner_scoped(rt):
    person = await _person(rt)
    repository = rt.repos.profile
    assert await repository.acquire_review_lease(person.person_id, "worker-a")
    await rt.db.execute(
        "UPDATE profile_review_leases SET expires_at = ? WHERE person_id = ?",
        ("2000-01-01T00:00:00+00:00", person.person_id),
    )

    assert await repository.acquire_review_lease(person.person_id, "worker-b")
    # The old execution can neither renew nor release the new owner's lease.
    assert not await repository.renew_review_lease(person.person_id, "worker-a")
    assert not await repository.release_review_lease(person.person_id, "worker-a")
    lease = await repository.get_review_lease(person.person_id)
    assert lease["owner_id"] == "worker-b"
    assert await repository.renew_review_lease(person.person_id, "worker-b")
    assert await repository.release_review_lease(person.person_id, "worker-b")
    assert await repository.get_review_lease(person.person_id) is None


async def test_a_worker_that_lost_its_lease_cannot_write(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    await _scheduler(rt).request(person.person_id, event.event_id)
    # The first attempt consumes one decision, then loses the lease; the retry
    # needs its own review decision because the abandoned one was never applied.
    handler = build_profile_review_handler(
        _engine(rt, ScriptedProfileRefinementProvider(
            [
                _add("projects.active", "机器狗", event.event_id),
                _add("projects.active", "机器狗", event.event_id),
            ]
        ))
    )

    original = rt.repos.profile.renew_review_lease

    async def lost(*args, **kwargs):
        return False

    rt.repos.profile.renew_review_lease = lost
    try:
        with pytest.raises(JobError, match="lease_lost"):
            await handler(await _payload(person, event), rt.repos, LOG)
    finally:
        rt.repos.profile.renew_review_lease = original

    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "PENDING"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1
    assert await rt.repos.profile.list_entries(person.person_id) == []

    # With the lease held normally, the retry succeeds.
    await handler(await _payload(person, event), rt.repos, LOG)
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2


async def test_exhausted_retries_leave_a_terminal_receipt(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    await _scheduler(rt).request(person.person_id, event.event_id)

    provider = ScriptedProfileRefinementProvider(
        [_add("projects.active", "机器狗", event.event_id)]
    )
    provider.error = ProfileRefinementProviderError("api down")
    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(rt, provider)
    )
    await rt.worker.run_until_idle()

    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "FAILED"
    assert receipt["decision"] == "FAILED"
    assert receipt["attempt_count"] == 3
    assert "api down" in (receipt["last_error"] or "")
    job = await rt.repos.job.get_by_key(
        profile_review_job_key(person.person_id, event.event_id)
    )
    assert job.status == JobStatus.FAILED
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1


async def test_a_successful_review_is_not_applied_twice(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    await _scheduler(rt).request(person.person_id, event.event_id)
    handler = build_profile_review_handler(
        _engine(rt, ScriptedProfileRefinementProvider(
            [_add("projects.active", "机器狗", event.event_id)]
        ))
    )
    await handler(await _payload(person, event), rt.repos, LOG)
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2

    await handler(await _payload(person, event), rt.repos, LOG)
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2
    assert len(await rt.repos.profile.list_revisions(person.person_id)) == 1
    assert len(
        await rt.repos.profile.list_entries(person.person_id, field_key="projects.active")
    ) == 1


async def test_manual_retry_uses_a_new_execution_record(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    scheduler = _scheduler(rt)
    await scheduler.request(person.person_id, event.event_id)
    old_key = profile_review_job_key(person.person_id, event.event_id)
    await rt.db.execute(
        "UPDATE processing_jobs SET max_attempts = 1 WHERE job_key = ?", (old_key,)
    )

    provider = ScriptedProfileRefinementProvider(
        [_add("projects.active", "机器狗", event.event_id)]
    )
    provider.error = ProfileRefinementProviderError("api down")
    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(rt, provider)
    )
    await rt.worker.run_until_idle()
    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "FAILED"
    assert (await rt.repos.job.get_by_key(old_key)).status == JobStatus.FAILED

    review_id = await scheduler.retry(person.person_id, event.event_id)
    assert review_id == receipt["review_id"]
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "PENDING"
    assert receipt["attempt_count"] == 0
    assert receipt["last_error"] is None
    assert receipt["job_key"] != old_key
    new_job = await rt.repos.job.get_by_key(receipt["job_key"])
    assert new_job is not None and new_job.status == JobStatus.PENDING

    provider.error = None
    await rt.worker.run_until_idle()
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "COMPLETED"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2
    assert len(await rt.repos.profile.list_revisions(person.person_id)) == 1


async def test_reconcile_reenqueues_an_orphaned_receipt(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    await _scheduler(rt).request(person.person_id, event.event_id)
    job_key = profile_review_job_key(person.person_id, event.event_id)
    await rt.db.execute("DELETE FROM processing_jobs WHERE job_key = ?", (job_key,))
    assert await rt.repos.job.get_by_key(job_key) is None

    fixed = await reconcile_profile_review_jobs(rt.repos)
    assert fixed["enqueued"] == 1
    assert await rt.repos.job.get_by_key(job_key) is not None

    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(rt, ScriptedProfileRefinementProvider(
            [_add("projects.active", "机器狗", event.event_id)]
        ))
    )
    await rt.worker.run_until_idle()
    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "COMPLETED"


async def test_reconcile_turns_a_failed_job_into_a_failed_receipt(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    await _scheduler(rt).request(person.person_id, event.event_id)
    job_key = profile_review_job_key(person.person_id, event.event_id)
    await rt.db.execute(
        "UPDATE processing_jobs SET status = 'FAILED', last_error = 'boom'"
        " WHERE job_key = ?",
        (job_key,),
    )

    fixed = await reconcile_profile_review_jobs(rt.repos)
    assert fixed["failed"] == 1
    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "job_failed"
    assert "boom" in (receipt["last_error"] or "")

    # No permanent PENDING receipt, and a second pass is a no-op.
    again = await reconcile_profile_review_jobs(rt.repos)
    assert again["checked"] == 0


async def test_reconcile_resets_a_dead_running_receipt(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    review_id = await _scheduler(rt).request(person.person_id, event.event_id)
    job_key = profile_review_job_key(person.person_id, event.event_id)
    await rt.db.execute(
        "UPDATE profile_review_receipts SET status = 'RUNNING' WHERE review_id = ?",
        (review_id,),
    )
    await rt.db.execute("DELETE FROM processing_jobs WHERE job_key = ?", (job_key,))

    fixed = await reconcile_profile_review_jobs(rt.repos, person_id=person.person_id)
    assert fixed["reset_running"] == 1
    assert fixed["enqueued"] == 1
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "PENDING"


async def test_a_later_job_still_processes_the_older_receipt_first(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我在做机器狗。", "c1")
    second = await _event(rt, person, "我喜欢爵士。", "c2")
    scheduler = _scheduler(rt)
    await scheduler.request(person.person_id, first.event_id)
    await scheduler.request(person.person_id, second.event_id)

    provider = ScriptedProfileRefinementProvider(
        [
            _add("projects.active", "机器狗", first.event_id),
            _add("interests.topics", "爵士乐", second.event_id),
        ]
    )
    handler = build_profile_review_handler(_engine(rt, provider))
    # Run the *later* event's job only; it must drain the older receipt first.
    await handler(await _payload(person, second), rt.repos, LOG)

    receipts = await rt.repos.profile.list_review_receipts(person.person_id)
    assert [item["source_event_id"] for item in receipts] == [
        first.event_id,
        second.event_id,
    ]
    assert [item["status"] for item in receipts] == ["COMPLETED", "COMPLETED"]
    assert [item["result_revision"] for item in receipts] == [2, 3]
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert [item["revision"] for item in revisions] == [2, 3]


async def test_reconcile_never_replays_a_completed_receipt(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    await _scheduler(rt).request(person.person_id, event.event_id)
    handler = build_profile_review_handler(
        _engine(rt, ScriptedProfileRefinementProvider(
            [_add("projects.active", "机器狗", event.event_id)]
        ))
    )
    await handler(await _payload(person, event), rt.repos, LOG)
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision

    fixed = await reconcile_profile_review_jobs(rt.repos)
    assert fixed["checked"] == 0
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision
