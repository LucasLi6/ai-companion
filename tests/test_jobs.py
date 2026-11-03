"""Test F: durable job recovery; retry and dedup semantics."""

from __future__ import annotations

from companion.domain.enums import JobStatus, JobType
from companion.jobs.worker import JobWorker
from companion.runtime import build_runtime
from companion.storage.database import Database
from companion.storage.repositories import make_repositories


async def test_job_dedup_by_key(rt):
    first = await rt.worker.enqueue(
        JobType.MEMORY_CONSOLIDATION, "memory:evt_x", {"event_id": "evt_x"}
    )
    second = await rt.worker.enqueue(
        JobType.MEMORY_CONSOLIDATION, "memory:evt_x", {"event_id": "evt_x"}
    )
    assert first == second
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS c FROM processing_jobs WHERE job_key = 'memory:evt_x'"
    )
    assert (await cur.fetchone())["c"] == 1


async def test_job_recovers_after_restart(tmp_path, register, say, rt):
    p = await register("Speaker")
    report = await say(p.person_id, "agent", "hello")  # enqueues a PENDING job
    job = await rt.repos.job.get_by_key(f"memory:{report.event_id}")
    assert job.status == JobStatus.PENDING

    path = rt.db.path
    await rt.db.close()  # simulate shutdown before the worker ran

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        await rt2.worker.recover()
        await rt2.worker.run_once()
        job2 = await rt2.repos.job.get(job.job_id)
        assert job2.status == JobStatus.COMPLETED
        assert job2.completed_at is not None
    finally:
        await db2.close()


async def test_running_job_recovered_after_crash(tmp_path, register, say, rt):
    p = await register("Speaker")
    report = await say(p.person_id, "agent", "hello")
    job = await rt.repos.job.get_by_key(f"memory:{report.event_id}")
    # Worker "crashed" mid-processing with no lease (an older build), or its
    # lease has since lapsed — either way it is reclaimable.
    await rt.repos.job.claim(job.job_id, "dead-worker", lease_seconds=0)

    path = rt.db.path
    await rt.db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        recovered = await rt2.worker.recover()
        assert recovered >= 1
        await rt2.worker.run_once()
        job2 = await rt2.repos.job.get(job.job_id)
        assert job2.status == JobStatus.COMPLETED
    finally:
        await db2.close()


async def test_job_fails_after_max_attempts(tmp_path):
    db = await Database(tmp_path / "f.db").connect()
    await db.initialize()
    repos = make_repositories(db)

    async def boom(payload, repos, logger):
        raise RuntimeError("boom")

    worker = JobWorker(repos, {"MEMORY_CONSOLIDATION": boom})
    job_id = await worker.enqueue(JobType.MEMORY_CONSOLIDATION, "k", {})
    await worker.run_until_idle()

    job = await repos.job.get(job_id)
    assert job.status == JobStatus.FAILED
    assert job.attempts == job.max_attempts
    assert "boom" in (job.last_error or "")
    await db.close()


async def test_automatic_retry_succeeds(tmp_path):
    db = await Database(tmp_path / "r.db").connect()
    await db.initialize()
    repos = make_repositories(db)
    calls = {"n": 0}

    async def flaky(payload, repos, logger):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")

    worker = JobWorker(repos, {"MEMORY_CONSOLIDATION": flaky})
    job_id = await worker.enqueue(JobType.MEMORY_CONSOLIDATION, "k", {})
    await worker.run_until_idle()

    job = await repos.job.get(job_id)
    assert job.status == JobStatus.COMPLETED
    assert calls["n"] == 2  # failed once, retried, succeeded
    await db.close()


async def test_manual_retry_resets_failed_job(tmp_path):
    db = await Database(tmp_path / "m.db").connect()
    await db.initialize()
    repos = make_repositories(db)

    async def boom(payload, repos, logger):
        raise RuntimeError("boom")

    worker = JobWorker(repos, {"MEMORY_CONSOLIDATION": boom})
    job_id = await worker.enqueue(JobType.MEMORY_CONSOLIDATION, "k", {})
    await worker.run_until_idle()
    assert (await repos.job.get(job_id)).status == JobStatus.FAILED

    ok = await repos.job.retry(job_id)
    assert ok is True
    reset = await repos.job.get(job_id)
    assert reset.status == JobStatus.PENDING
    assert reset.attempts == 0
    assert reset.last_error is None
    await db.close()
