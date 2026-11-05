"""V0.9 unified off-task execution tests.

These exercise the task substrate directly: durable enqueue, dedupe, priority,
foreground preemption, cooperative checkpoints, cancellation, permission
revocation and truthful results. Research integration is covered with the
offline mock providers only.
"""

from __future__ import annotations

import asyncio

import pytest

from companion.domain.enums import JobStatus, JobType, TaskMode, TaskPriority, TaskSource
from companion.idle import build_idle_scheduler
from companion.jobs.worker import JobWorker
from companion.storage.repositories import make_repositories
from companion.tasks.handlers import build_research_task_handler
from companion.tasks.models import (
    TaskOutcome,
    TaskPaused,
    TaskRejected,
    TaskRequest,
    TaskStatus,
)
from companion.tasks.service import TaskService


def make_service(db, handlers, **kwargs):
    repos = make_repositories(db)
    worker = JobWorker(repos, handlers, poll_interval=0.01)
    service = TaskService(repos, handlers=handlers, worker=worker, **kwargs)
    return repos, worker, service


@pytest.mark.asyncio
async def test_migration_adds_task_columns_without_touching_old_rows(db):
    repos = make_repositories(db)
    await db.execute(
        "INSERT INTO processing_jobs(job_id, job_key, job_type, payload, status,"
        " attempts, max_attempts, created_at)"
        " VALUES ('old', 'old-key', 'MEMORY_CONSOLIDATION', '{}', 'PENDING',"
        " 0, 3, '2026-01-01T00:00:00+00:00')"
    )
    job = await repos.job.get("old")
    assert job.priority == "NORMAL"
    assert job.source_kind == "SYSTEM"
    assert job.execution_mode == "BACKGROUND"
    assert job.checkpoint == {}
    assert job.cancel_requested is False
    assert job.result is None
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] >= 23


@pytest.mark.asyncio
async def test_execute_now_returns_and_persists_real_result(db):
    async def handler(payload, repos, logger):
        return {"echo": payload["value"]}

    repos, worker, service = make_service(db, {"ECHO": handler})
    request = TaskRequest(
        type="ECHO",
        payload={"value": 7},
        source=TaskSource.USER,
        source_event_id="evt-1",
    )
    result = await service.execute_now(request)

    assert result.ok
    assert str(result.status) == "COMPLETED"
    assert result.result == {"echo": 7}
    stored = await repos.job.get(result.task_id)
    assert stored.status == JobStatus.COMPLETED
    assert stored.result == {"echo": 7}
    assert stored.source_kind == "USER"
    assert stored.source_ref == "evt-1"
    assert stored.execution_mode == "FOREGROUND"
    assert stored.priority == "HIGH"


@pytest.mark.asyncio
async def test_execute_now_failure_is_truthful_and_not_retried(db):
    calls = 0

    async def handler(payload, repos, logger):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider down")

    repos, worker, service = make_service(db, {"FAIL": handler})
    result = await service.execute_now(TaskRequest(type="FAIL", source=TaskSource.USER))

    assert str(result.status) == "FAILED"
    assert result.error == "provider down"
    assert result.result is None
    stored = await repos.job.get(result.task_id)
    assert stored.status == JobStatus.FAILED
    assert calls == 1


@pytest.mark.asyncio
async def test_concurrent_duplicate_enqueue_creates_one_task(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = make_service(db, {"NOOP": handler})
    request = TaskRequest(type="NOOP", job_key="stable-job", source=TaskSource.SYSTEM)
    ids = await asyncio.gather(service.enqueue(request), service.enqueue(request))

    assert ids[0] == ids[1]
    cur = await db.execute("SELECT COUNT(*) AS n FROM processing_jobs")
    assert (await cur.fetchone())["n"] == 1


@pytest.mark.asyncio
async def test_pending_orders_by_priority_then_created_at(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = make_service(db, {"NOOP": handler})
    low = await service.enqueue(
        TaskRequest(type="NOOP", job_key="low", priority=TaskPriority.LOW)
    )
    high = await service.enqueue(
        TaskRequest(type="NOOP", job_key="high", priority=TaskPriority.HIGH)
    )
    normal = await service.enqueue(
        TaskRequest(type="NOOP", job_key="normal", priority=TaskPriority.NORMAL)
    )

    pending = await repos.job.pending()
    assert [job.job_id for job in pending] == [high, normal, low]


@pytest.mark.asyncio
async def test_foreground_activity_defers_background_without_consuming_attempts(db):
    calls = 0

    async def handler(payload, repos, logger):
        nonlocal calls
        calls += 1
        return {"ok": True}

    repos, worker, service = make_service(db, {"WORK": handler}, resume_idle_seconds=0)
    task_id = await service.enqueue(TaskRequest(type="WORK", job_key="w1"))

    with service.foreground.scope():
        await worker.run_once()
        assert calls == 0
        job = await repos.job.get(task_id)
        assert job.status == JobStatus.PENDING
        assert job.attempts == 0

    await worker.run_once()
    job = await repos.job.get(task_id)
    assert calls == 1
    assert job.status == JobStatus.COMPLETED
    assert job.attempts == 1


@pytest.mark.asyncio
async def test_cooperative_checkpoint_pause_does_not_count_as_retry(db):
    seen = []

    async def handler(payload, repos, logger):
        from companion.tasks.context import current_task_context

        context = current_task_context()
        seen.append(dict(context.checkpoint))
        if context.checkpoint.get("stage") != "ONE":
            return TaskOutcome(
                status=TaskStatus.PENDING,
                checkpoint={"stage": "ONE", "partial_ref": "ref-1"},
            )
        return TaskOutcome(
            status=TaskStatus.COMPLETED,
            result={"resumed": True},
            checkpoint={"stage": "DONE"},
        )

    repos, worker, service = make_service(db, {"STAGED": handler}, resume_idle_seconds=0)
    task_id = await service.enqueue(TaskRequest(type="STAGED", job_key="staged"))

    await worker.run_once()
    first = await repos.job.get(task_id)
    assert first.status == JobStatus.PENDING
    assert first.attempts == 0
    assert first.checkpoint["partial_ref"] == "ref-1"

    await worker.run_once()
    final = await repos.job.get(task_id)
    assert final.status == JobStatus.COMPLETED
    assert final.result == {"resumed": True}
    assert final.attempts == 1
    assert seen[0] == {}
    assert seen[1]["stage"] == "ONE"


@pytest.mark.asyncio
async def test_task_paused_exception_is_a_pause_not_a_failure(db):
    async def handler(payload, repos, logger):
        raise TaskPaused({"stage": "SEARCHED"})

    repos, worker, service = make_service(db, {"PAUSE": handler}, resume_idle_seconds=0)
    task_id = await service.enqueue(TaskRequest(type="PAUSE", job_key="pause"))
    await worker.run_once()
    job = await repos.job.get(task_id)
    assert job.status == JobStatus.PENDING
    assert job.attempts == 0
    assert job.checkpoint["stage"] == "SEARCHED"


@pytest.mark.asyncio
async def test_cancel_before_execution_is_terminal_and_skips_handler(db):
    calls = 0

    async def handler(payload, repos, logger):
        nonlocal calls
        calls += 1

    repos, worker, service = make_service(db, {"NOOP": handler})
    task_id = await service.enqueue(TaskRequest(type="NOOP", job_key="cancel"))
    assert await service.cancel(task_id)
    await worker.run_once()

    job = await repos.job.get(task_id)
    assert job.status == JobStatus.CANCELLED
    assert calls == 0
    assert await service.cancel(task_id)


@pytest.mark.asyncio
async def test_cancellation_during_execution_is_not_reported_as_success(db):
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(payload, repos, logger):
        started.set()
        await release.wait()
        return {"side_effect": "done"}

    repos, worker, service = make_service(db, {"SLOW": handler})
    runner = asyncio.create_task(
        service.execute_now(TaskRequest(type="SLOW", source=TaskSource.USER))
    )
    await started.wait()
    task_id = (await repos.job.get_by_key(
        TaskRequest(type="SLOW", source=TaskSource.USER).stable_key
    )).job_id
    assert await service.cancel(task_id)
    release.set()
    result = await runner

    assert str(result.status) == "CANCELLED"
    stored = await repos.job.get(task_id)
    assert stored.status == JobStatus.CANCELLED
    assert stored.result is None


@pytest.mark.asyncio
async def test_permission_revocation_blocks_pre_claim_execution(db):
    permitted = True
    calls = 0

    def permission(request):
        return permitted

    async def handler(payload, repos, logger):
        nonlocal calls
        calls += 1

    repos, worker, service = make_service(
        db, {"NOOP": handler}, permission_checker=permission
    )
    task_id = await service.enqueue(
        TaskRequest(type="NOOP", job_key="perm", source=TaskSource.SYSTEM)
    )
    permitted = False
    await worker.run_once()

    job = await repos.job.get(task_id)
    assert job.status == JobStatus.CANCELLED
    assert calls == 0


@pytest.mark.asyncio
async def test_revoke_source_cancels_pending_queue_and_blocks_new_nodes(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = make_service(db, {"NOOP": handler})
    task_id = await service.enqueue(
        TaskRequest(
            type="NOOP", job_key="rev", source=TaskSource.ROUTER, source_ref="person-1"
        )
    )
    cancelled = await service.revoke_source(TaskSource.ROUTER, "person-1")
    assert cancelled == 1
    assert (await repos.job.get(task_id)).status == JobStatus.CANCELLED
    with pytest.raises(TaskRejected) as exc:
        await service.enqueue(
            TaskRequest(
                type="NOOP",
                job_key="rev-2",
                source=TaskSource.ROUTER,
                source_ref="person-1",
            )
        )
    assert exc.value.reason == "PERMISSION_REVOKED"


@pytest.mark.asyncio
async def test_autonomous_queue_limit_never_drops_user_work(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = make_service(
        db, {"NOOP": handler}, autonomous_pending_limit=1
    )
    await service.enqueue(TaskRequest(type="NOOP", job_key="a1"))
    with pytest.raises(TaskRejected) as exc:
        await service.enqueue(TaskRequest(type="NOOP", job_key="a2"))
    assert exc.value.reason == "AUTONOMOUS_QUEUE_FULL"
    user_id = await service.enqueue(
        TaskRequest(type="NOOP", job_key="u1", source=TaskSource.USER)
    )
    assert (await repos.job.get(user_id)).status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_unsupported_task_type_is_rejected_before_a_row_is_written(db):
    repos, worker, service = make_service(db, {})
    with pytest.raises(TaskRejected) as exc:
        await service.enqueue(TaskRequest(type="UNKNOWN"))
    assert exc.value.reason == "UNSUPPORTED_TASK_TYPE"
    cur = await db.execute("SELECT COUNT(*) AS n FROM processing_jobs")
    assert (await cur.fetchone())["n"] == 0


@pytest.mark.asyncio
async def test_background_failure_retries_then_fails_truthfully(db):
    calls = 0

    async def handler(payload, repos, logger):
        nonlocal calls
        calls += 1
        raise RuntimeError("network down")

    repos, worker, service = make_service(db, {"NET": handler})
    task_id = await service.enqueue(
        TaskRequest(type="NET", job_key="net", max_attempts=2)
    )
    await worker.run_once()
    await worker.run_once()
    await worker.run_once()

    job = await repos.job.get(task_id)
    assert job.status == JobStatus.FAILED
    assert "network down" in (job.last_error or "")
    assert calls == 2
    result = await service.get_result(task_id)
    assert str(result.status) == "FAILED"
    assert "network down" in (result.error or "")


@pytest.mark.asyncio
async def test_paused_task_resumes_after_a_simulated_restart(db):
    seen = []

    async def handler(payload, repos, logger):
        from companion.tasks.context import current_task_context

        stage = current_task_context().checkpoint.get("stage")
        seen.append(stage)
        if stage != "SEARCHED":
            return TaskOutcome(status=TaskStatus.PENDING, checkpoint={"stage": "SEARCHED"})
        return {"resumed": True}

    repos = make_repositories(db)
    first_worker = JobWorker(repos, {"RESTART": handler})
    first = TaskService(repos, handlers={"RESTART": handler}, worker=first_worker,
                        resume_idle_seconds=0)
    task_id = await first.enqueue(TaskRequest(type="RESTART", job_key="restart"))
    await first_worker.run_once()
    assert (await repos.job.get(task_id)).status == JobStatus.PENDING

    # A new process/service sees the durable row and its checkpoint.
    second_worker = JobWorker(repos, {"RESTART": handler})
    second = TaskService(repos, handlers={"RESTART": handler}, worker=second_worker,
                         resume_idle_seconds=0)
    await second_worker.run_once()
    job = await repos.job.get(task_id)
    assert job.status == JobStatus.COMPLETED
    assert job.result == {"resumed": True}
    assert seen == [None, "SEARCHED"]


@pytest.mark.asyncio
async def test_lost_lease_cannot_be_reported_as_completed(db):
    async def handler(payload, repos, logger):
        return {"ok": True}

    repos, worker, service = make_service(db, {"LEASE": handler})
    task_id = await service.enqueue(TaskRequest(type="LEASE", job_key="lease"))
    job = await repos.job.get(task_id)
    assert await repos.job.claim(task_id, "other-worker", 300)
    # This worker no longer owns the job.
    assert not await repos.job.complete_owned(task_id, worker.owner_id, {"ok": True})
    stored = await repos.job.get(task_id)
    assert stored.status == JobStatus.RUNNING
    assert stored.owner_id == "other-worker"


@pytest.mark.asyncio
async def test_research_resume_reuses_search_and_synthesis_without_duplicate_memory(
    rt, db
):
    scheduler = build_idle_scheduler(rt)
    research = scheduler.research
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("可恢复的研究", source_kind="test")

    searches = 0
    syntheses = 0
    original_search = research.web_search.search
    original_synthesize = research.deepseek.synthesize

    async def counted_search(query, limit=5):
        nonlocal searches
        searches += 1
        return await original_search(query, limit=limit)

    async def counted_synthesize(question, results):
        nonlocal syntheses
        syntheses += 1
        return await original_synthesize(question, results)

    research.web_search.search = counted_search
    research.deepseek.synthesize = counted_synthesize

    saved: dict = {}

    async def save(update):
        saved.update(update)

    checks = {"n": 0}

    def pause_after_search():
        checks["n"] += 1
        return checks["n"] >= 2

    with pytest.raises(TaskPaused) as paused:
        await research.run_resumable(
            task_id, checkpoint={}, pause_check=pause_after_search, save_checkpoint=save
        )
    checkpoint = paused.value.checkpoint
    assert checkpoint["stage"] == "SEARCHED"
    assert searches == 1
    assert syntheses == 0
    # The search output is persisted by reference, not re-run on resume.
    resumed = await research.run_resumable(
        task_id, checkpoint=checkpoint, pause_check=lambda: False,
        save_checkpoint=save,
    )
    assert resumed["status"] == "COMPLETED"
    assert searches == 1
    assert syntheses == 1
    assert saved["stage"] == "DONE"

    await research.run_resumable(task_id, checkpoint=checkpoint)
    cur = await db.execute(
        "SELECT COUNT(*) AS n FROM memory_items WHERE memory_id = ?",
        (f"mem:{task_id}",),
    )
    assert (await cur.fetchone())["n"] == 1
    assert searches == 1
    assert syntheses == 1


@pytest.mark.asyncio
async def test_research_task_pauses_for_foreground_then_resumes_without_duplicate_memory(
    rt, db
):
    scheduler = build_idle_scheduler(rt)
    research = scheduler.research
    await rt.repos.settings.set("research.enabled", "true")
    task_id = await rt.repos.research.create("什么是模拟研究？", source_kind="test")

    rt.task_service.register_handler(
        JobType.RESEARCH_TASK, build_research_task_handler(research)
    )
    rt.task_service.resume_idle_seconds = 0
    job_id = await rt.task_service.enqueue(
        TaskRequest(
            type=JobType.RESEARCH_TASK.value,
            payload={"research_task_id": task_id},
            source=TaskSource.SYSTEM,
            priority=TaskPriority.LOW,
            job_key=f"research:{task_id}",
        )
    )

    with rt.task_service.foreground.scope():
        await rt.worker.run_once()
    paused = await rt.repos.job.get(job_id)
    assert paused.status == JobStatus.PENDING
    assert paused.attempts == 0

    await rt.worker.run_once()
    done = await rt.repos.job.get(job_id)
    assert done.status == JobStatus.COMPLETED
    assert done.result["memory_id"] == f"mem:{task_id}"
    assert (await rt.repos.research.get(task_id))["status"] == "COMPLETED"

    # Re-running the same task must reuse the deterministic memory id.
    await research.run_resumable(task_id)
    cur = await db.execute(
        "SELECT COUNT(*) AS n FROM memory_items WHERE memory_id = ?",
        (f"mem:{task_id}",),
    )
    assert (await cur.fetchone())["n"] == 1
