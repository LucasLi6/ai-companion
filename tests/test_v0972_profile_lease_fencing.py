"""V0.9.7.2 Section C: the person lease fences the final profile commit.

These tests use two independent SQLite connections to the same database file,
so the guarantee is the database's conditional-UPDATE serialisation, not an
in-process asyncio lock. A worker that finished its model call but lost the
lease may not commit, may not overwrite the newer worker's receipt and may not
release the newer worker's lease.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event, utcnow
from companion.jobs.profile_review import ProfileReviewScheduler
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileReviewExecution,
    ValidatedChange,
)
from companion.people.profile_service import (
    PersonProfileService,
    ReviewExecutionLost,
)
from companion.runtime import build_runtime
from companion.storage.database import Database
from companion.storage.repositories import make_repositories


async def _seed(path):
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db)
    person = await rt.registry.register(
        "Fence",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.profile_service.ensure_profile(person.person_id, actor="test")
    await rt.conversations.get_or_create("c1")
    event = make_utterance_event(
        conversation_id="c1",
        text="我在做机器狗。",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    review_id = await ProfileReviewScheduler(
        rt.repos.profile, rt.repos.job
    ).request(person.person_id, event.event_id)
    return db, rt, person, event, review_id


def _decision(person_id, event_id, base_revision, content):
    return SimpleNamespace(
        status="PROPOSE_CHANGES",
        base_revision=base_revision,
        reason="fencing test",
        changes=[
            ValidatedChange(
                proposal=ProfileChangeProposal(
                    operation="ADD",
                    field_key="projects.active",
                    proposed_content=content,
                    evidence_event_ids=[event_id],
                    confidence=0.95,
                ),
                auto_apply=True,
            )
        ],
    )


async def _expire_everything(db, person_id, review_id):
    """Make A's lease and its claimed receipt look abandoned."""
    past = (utcnow() - timedelta(hours=1)).isoformat()
    await db.execute(
        "UPDATE profile_review_leases SET expires_at = ? WHERE person_id = ?",
        (past, person_id),
    )
    await db.execute(
        "UPDATE profile_review_receipts SET updated_at = ? WHERE review_id = ?",
        (past, review_id),
    )


@pytest.mark.asyncio
async def test_a_worker_that_lost_the_lease_cannot_commit_or_touch_b(tmp_path):
    db_a, rt_a, person, event, review_id = await _seed(tmp_path / "fence.db")
    person_id = person.person_id
    revision = (await rt_a.profile_service.get(person_id)).metadata.revision
    decision_a = _decision(person_id, event.event_id, revision, "A 的机器狗")

    # A owns the person lease and has claimed the receipt.
    assert await rt_a.repos.profile.acquire_review_lease(
        person_id, "worker-a", lease_seconds=30
    )
    assert await rt_a.repos.profile.claim_review_receipt(review_id, "worker-a")

    # A has finished its model review and is paused before the commit.
    await _expire_everything(db_a, person_id, review_id)

    db_b = await Database(tmp_path / "fence.db").connect()
    repos_b = make_repositories(db_b)
    service_b = PersonProfileService(
        repos_b.profile, repos_b.person, audit=repos_b.audit
    )

    # B takes over the dead lease and the stale receipt.
    assert await repos_b.profile.acquire_review_lease(
        person_id, "worker-b", lease_seconds=60
    )
    assert not await repos_b.profile.acquire_review_lease(
        person_id, "worker-c", lease_seconds=60
    )
    assert (
        await repos_b.profile.recover_stale_review_receipts(
            person_id, older_than_seconds=60
        )
        == 1
    )
    assert await repos_b.profile.claim_review_receipt(review_id, "worker-b")

    # A resumes. Its execution identity is stale: the lease belongs to B and
    # the receipt is owned by B.
    execution_a = ProfileReviewExecution(
        person_id=person_id,
        review_id=review_id,
        lease_owner_id="worker-a",
        source_event_id=event.event_id,
    )
    with pytest.raises(ReviewExecutionLost):
        await rt_a.profile_service.apply_refinement(
            person_id,
            decision_a,
            source_event_id=event.event_id,
            execution=execution_a,
        )
    assert (
        await rt_a.profile_service.get(person_id)
    ).metadata.revision == revision
    assert await rt_a.repos.profile.list_entries(person_id) == []

    # A cannot overwrite B's receipt, reopen it, or release B's lease.
    assert not await rt_a.repos.profile.update_review_receipt(
        review_id,
        status="COMPLETED",
        decision="PROPOSE_CHANGES",
        completed=True,
        execution_owner_id="worker-a",
    )
    receipt = await repos_b.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "RUNNING"
    assert receipt["execution_owner_id"] == "worker-b"
    assert not await rt_a.repos.profile.release_review_receipt(
        review_id, reason="stale", execution_owner_id="worker-a"
    )
    assert not await rt_a.repos.profile.release_review_lease(person_id, "worker-a")
    lease = await repos_b.profile.get_review_lease(person_id)
    assert lease["owner_id"] == "worker-b"

    # B commits through the same fence and completes its own receipt.
    execution_b = ProfileReviewExecution(
        person_id=person_id,
        review_id=review_id,
        lease_owner_id="worker-b",
        source_event_id=event.event_id,
    )
    decision_b = _decision(person_id, event.event_id, revision, "B 的机器狗")
    applied = await service_b.apply_refinement(
        person_id,
        decision_b,
        source_event_id=event.event_id,
        execution=execution_b,
    )
    assert applied.status == "APPLIED"
    assert applied.new_revision == revision + 1
    assert await repos_b.profile.update_review_receipt(
        review_id,
        status="COMPLETED",
        decision="PROPOSE_CHANGES",
        changes=applied.applied,
        result_revision=applied.new_revision,
        completed=True,
        execution_owner_id="worker-b",
    )
    assert await repos_b.profile.release_review_lease(person_id, "worker-b")

    entries = await repos_b.profile.list_entries(
        person_id, statuses=("ACTIVE", "TENTATIVE")
    )
    assert [entry["content"] for entry in entries] == ["B 的机器狗"]
    revisions = await repos_b.profile.list_revisions(person_id)
    assert [item["revision"] for item in revisions] == [revision + 1]

    await db_b.close()
    await db_a.close()


@pytest.mark.asyncio
async def test_an_expired_lease_blocks_commit_even_for_its_owner(tmp_path):
    db, rt, person, event, review_id = await _seed(tmp_path / "expiry.db")
    person_id = person.person_id
    revision = (await rt.profile_service.get(person_id)).metadata.revision

    assert await rt.repos.profile.acquire_review_lease(
        person_id, "worker-a", lease_seconds=30
    )
    assert await rt.repos.profile.claim_review_receipt(review_id, "worker-a")
    await rt.db.execute(
        "UPDATE profile_review_leases SET expires_at = ? WHERE person_id = ?",
        ((utcnow() - timedelta(seconds=5)).isoformat(), person_id),
    )

    execution = ProfileReviewExecution(
        person_id=person_id,
        review_id=review_id,
        lease_owner_id="worker-a",
        source_event_id=event.event_id,
    )
    with pytest.raises(ReviewExecutionLost, match="review_lease_lost"):
        await rt.profile_service.apply_refinement(
            person_id,
            _decision(person_id, event.event_id, revision, "过期提交"),
            source_event_id=event.event_id,
            execution=execution,
        )
    assert (
        await rt.profile_service.get(person_id)
    ).metadata.revision == revision
    assert await rt.repos.profile.list_entries(person_id) == []
    await db.close()


@pytest.mark.asyncio
async def test_an_execution_for_another_receipt_cannot_commit(tmp_path):
    db, rt, person, event, review_id = await _seed(tmp_path / "scope.db")
    person_id = person.person_id
    revision = (await rt.profile_service.get(person_id)).metadata.revision
    assert await rt.repos.profile.acquire_review_lease(
        person_id, "worker-a", lease_seconds=60
    )
    assert await rt.repos.profile.claim_review_receipt(review_id, "worker-a")

    # Right person and lease, wrong review identity.
    execution = ProfileReviewExecution(
        person_id=person_id,
        review_id="pfrv_someone_else",
        lease_owner_id="worker-a",
        source_event_id=event.event_id,
    )
    with pytest.raises(ReviewExecutionLost, match="review_receipt_not_owned"):
        await rt.profile_service.apply_refinement(
            person_id,
            _decision(person_id, event.event_id, revision, "错误回执"),
            source_event_id=event.event_id,
            execution=execution,
        )
    assert await rt.repos.profile.list_entries(person_id) == []
    await db.close()


@pytest.mark.asyncio
async def test_a_receipt_can_only_be_claimed_by_one_execution(tmp_path):
    db_a, rt_a, person, event, review_id = await _seed(tmp_path / "claim.db")
    db_b = await Database(tmp_path / "claim.db").connect()
    repos_b = make_repositories(db_b)

    assert await rt_a.repos.profile.claim_review_receipt(review_id, "worker-a")
    assert not await repos_b.profile.claim_review_receipt(review_id, "worker-b")
    receipt = await repos_b.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "RUNNING"
    assert receipt["execution_owner_id"] == "worker-a"
    assert receipt["lease_epoch"] == 1

    await db_b.close()
    await db_a.close()
