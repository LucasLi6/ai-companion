"""A3 — job ownership and leases.

A live worker's in-flight job is protected from ``recover_running``; an expired
(or lease-less) job is reclaimed; manual retry is restricted to ``FAILED`` jobs;
and a heartbeat extends a lease for the owning worker only.
"""

from __future__ import annotations

from companion.domain.enums import JobStatus, JobType
from companion.storage.database import Database
from companion.storage.repositories import make_repositories


async def _enqueue(repos, key="k") -> str:
    return await repos.job.enqueue(JobType.MEMORY_CONSOLIDATION, key, {})


async def test_claim_sets_owner_and_lease_and_is_exclusive(tmp_path):
    db = await Database(tmp_path / "a.db").connect()
    await db.initialize()
    repos = make_repositories(db)
    job_id = await _enqueue(repos)

    assert await repos.job.claim(job_id, "w1", lease_seconds=300) is True
    job = await repos.job.get(job_id)
    assert job.status == JobStatus.RUNNING
    assert job.owner_id == "w1"
    assert job.lease_expires_at is not None
    assert job.heartbeat_at is not None

    # A second worker cannot claim a job already owned by a live worker.
    assert await repos.job.claim(job_id, "w2", lease_seconds=300) is False
    job = await repos.job.get(job_id)
    assert job.owner_id == "w1"
    await db.close()


async def test_recover_running_protects_live_worker(tmp_path):
    db = await Database(tmp_path / "b.db").connect()
    await db.initialize()
    repos = make_repositories(db)
    job_id = await _enqueue(repos)
    await repos.job.claim(job_id, "w1", lease_seconds=300)

    # A live worker holds an unexpired lease: recovery must not touch it.
    assert await repos.job.recover_running() == 0
    job = await repos.job.get(job_id)
    assert job.status == JobStatus.RUNNING
    assert job.owner_id == "w1"
    await db.close()


async def test_recover_running_reclaims_expired_lease(tmp_path):
    db = await Database(tmp_path / "c.db").connect()
    await db.initialize()
    repos = make_repositories(db)
    job_id = await _enqueue(repos)
    await repos.job.claim(job_id, "w1", lease_seconds=300)

    # Simulate the owning process dying and its lease lapsing.
    await db.execute(
        "UPDATE processing_jobs SET lease_expires_at = '2000-01-01T00:00:00+00:00'"
        " WHERE job_id = ?",
        (job_id,),
    )
    assert await repos.job.recover_running() == 1
    job = await repos.job.get(job_id)
    assert job.status == JobStatus.PENDING
    assert job.owner_id is None
    await db.close()


async def test_retry_restricted_to_failed_jobs(tmp_path):
    db = await Database(tmp_path / "d.db").connect()
    await db.initialize()
    repos = make_repositories(db)

    failed = await _enqueue(repos, "k-failed")
    await repos.job.fail(failed, "boom")

    running = await _enqueue(repos, "k-running")
    await repos.job.claim(running, "w1", lease_seconds=300)

    completed = await _enqueue(repos, "k-completed")
    await repos.job.claim(completed, "w1", lease_seconds=300)
    await repos.job.complete(completed)

    # Only the FAILED job is retryable.
    assert await repos.job.retry(failed) is True
    assert await repos.job.retry(running) is False
    assert await repos.job.retry(completed) is False

    reset = await repos.job.get(failed)
    assert reset.status == JobStatus.PENDING
    assert reset.attempts == 0
    assert reset.last_error is None

    # RUNNING and COMPLETED are untouched.
    assert (await repos.job.get(running)).status == JobStatus.RUNNING
    assert (await repos.job.get(completed)).status == JobStatus.COMPLETED
    await db.close()


async def test_heartbeat_extends_lease_for_owner_only(tmp_path):
    db = await Database(tmp_path / "e.db").connect()
    await db.initialize()
    repos = make_repositories(db)
    job_id = await _enqueue(repos)
    await repos.job.claim(job_id, "w1", lease_seconds=300)
    before = (await repos.job.get(job_id)).lease_expires_at

    # A non-owner cannot extend the lease.
    assert await repos.job.heartbeat(job_id, "w2", lease_seconds=300) is False
    # The owner can.
    assert await repos.job.heartbeat(job_id, "w1", lease_seconds=300) is True
    after = (await repos.job.get(job_id)).lease_expires_at
    assert after >= before
    await db.close()
