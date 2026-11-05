"""V0.9.1 regression tests for the off-task execution layer.

These exercise the real call chain (Runtime -> TaskService -> SQLite queue ->
JobWorker -> handler -> engine), not a fake handler's return value. They cover

- unified research scheduling (curiosity/scheduler enqueue, no bypass),
- capacity that counts only autonomous research,
- cancellation that wins before any memory write,
- explicit-memory permission re-read from the database,
- unified research ownership/lease.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest

from companion.domain.enums import (
    EventSource,
    JobStatus,
    JobType,
    MemoryKind,
    MemoryVisibility,
    ReplyNecessity,
    TaskMode,
    TaskPriority,
    TaskSource,
)
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.idle import build_idle_scheduler
from companion.jobs.worker import JobWorker
from companion.memory.models import AccessContext
from companion.routing.models import (
    ProcessingIntent,
    RouteAction,
    RouteAddressee,
    ValidatedRouteDecision,
)
from companion.storage.repositories import make_repositories
from companion.tasks.handlers import build_explicit_memory_task_handler
from companion.tasks.models import TaskLeaseLost, TaskRejected, TaskRequest, TaskStatus
from companion.tasks.service import TaskService

ADMIN = AccessContext("admin", is_admin=True)
LOG = logging.getLogger("test.v091")


def _research_request(task_id: str, **kwargs) -> TaskRequest:
    return TaskRequest(
        type=JobType.RESEARCH_TASK.value,
        payload={"research_task_id": task_id},
        mode=TaskMode.BACKGROUND,
        source=TaskSource.SYSTEM,
        priority=TaskPriority.LOW,
        source_ref=task_id,
        job_key=f"research:{task_id}",
        **kwargs,
    )


async def _memory_count(rt, task_id: str) -> int:
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS n FROM memory_items WHERE memory_id = ?",
        (f"mem:{task_id}",),
    )
    return int((await cur.fetchone())["n"])


async def _enable_research(rt) -> None:
    await rt.repos.settings.set("research.enabled", "true")


# -- unified scheduling ---------------------------------------------------


@pytest.mark.asyncio
async def test_normal_runtime_registers_the_research_handler(rt):
    assert JobType.RESEARCH_TASK.value in rt.task_service.supported_types
    assert JobType.RESEARCH_TASK.value in rt.worker.handlers
    assert rt.research_engine is not None
    # The idle pipeline reuses the runtime's engine and handler.
    scheduler = build_idle_scheduler(rt)
    assert scheduler.research is rt.research_engine
    assert scheduler.task_service is rt.task_service


@pytest.mark.asyncio
async def test_curiosity_enqueues_research_through_the_task_service(rt):
    await rt.memory_service.create(
        kind=MemoryKind.PROJECT_STATE,
        content="机械设计进行中",
        access=ADMIN,
        visibility=MemoryVisibility.PUBLIC,
    )
    scheduler = build_idle_scheduler(rt)
    task_ids = await scheduler.curiosity.run()

    assert len(task_ids) == 1
    job = await rt.repos.job.get_by_key(f"research:{task_ids[0]}")
    assert job is not None
    assert job.status == JobStatus.PENDING
    assert job.source_kind == TaskSource.SYSTEM.value
    assert job.source_ref == task_ids[0]
    # Creating the need did not execute it.
    assert await _memory_count(rt, task_ids[0]) == 0


@pytest.mark.asyncio
async def test_scheduler_enqueues_instead_of_running_research(rt):
    await _enable_research(rt)
    await rt.repos.settings.set("idle.enabled", "true")
    task_id = await rt.repos.research.create("调度器不应自己执行", source_kind="test")
    scheduler = build_idle_scheduler(rt)

    result = await scheduler.run_cycle()

    assert result["skipped"] is False
    assert result["research"] == 1
    job = await rt.repos.job.get_by_key(f"research:{task_id}")
    assert job is not None and job.status == JobStatus.PENDING
    # No search and no memory happened in the dialogue process.
    assert await _memory_count(rt, task_id) == 0
    assert (await rt.repos.research.get(task_id))["status"] == "PENDING"

    await rt.worker.run_once()
    assert (await rt.repos.job.get(job.job_id)).status == JobStatus.COMPLETED
    assert (await rt.repos.research.get(task_id))["status"] == "COMPLETED"
    assert await _memory_count(rt, task_id) == 1


@pytest.mark.asyncio
async def test_existing_pending_research_is_taken_over_without_a_duplicate(rt):
    await _enable_research(rt)
    await rt.repos.settings.set("idle.enabled", "true")
    task_id = await rt.repos.research.create("历史遗留的 PENDING", source_kind="manual")
    # A second scheduling pass must reuse the same durable job row.
    scheduler = build_idle_scheduler(rt)
    await scheduler.run_cycle()
    await scheduler.run_cycle()

    cur = await rt.db.execute(
        "SELECT COUNT(*) AS n FROM processing_jobs WHERE job_key = ?",
        (f"research:{task_id}",),
    )
    assert (await cur.fetchone())["n"] == 1

    await rt.worker.run_once()
    await rt.worker.run_once()
    assert await _memory_count(rt, task_id) == 1


@pytest.mark.asyncio
async def test_a_second_research_executor_is_rejected_while_its_lease_is_live(rt):
    await _enable_research(rt)
    task_id = await rt.repos.research.create("只允许一个执行者", source_kind="test")
    # A legacy direct engine already owns this task with a live lease.
    assert await rt.repos.research.claim(task_id, "legacy-owner", 300)
    searches = 0
    original = rt.research_engine.web_search.search

    async def counted(query, limit=5):
        nonlocal searches
        searches += 1
        return await original(query, limit=limit)

    rt.research_engine.web_search.search = counted
    job_id = await rt.task_service.enqueue(_research_request(task_id))
    await rt.worker.run_once()

    job = await rt.repos.job.get(job_id)
    assert job.status == JobStatus.PENDING
    task = await rt.repos.research.get(task_id)
    assert task["status"] == "RUNNING"
    assert task["owner_id"] == "legacy-owner"
    assert searches == 0
    assert await _memory_count(rt, task_id) == 0


# -- capacity -------------------------------------------------------------


def _make_service(db, handlers, **kwargs):
    repos = make_repositories(db)
    worker = JobWorker(repos, handlers, poll_interval=0.01)
    service = TaskService(repos, handlers=handlers, worker=worker, **kwargs)
    return repos, worker, service


@pytest.mark.asyncio
async def test_turn_analysis_is_never_dropped_when_research_queue_is_full(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = _make_service(
        db,
        {
            JobType.RESEARCH_TASK.value: handler,
            JobType.TURN_ANALYSIS.value: handler,
        },
        autonomous_pending_limit=1,
    )
    await service.enqueue(
        TaskRequest(type=JobType.RESEARCH_TASK.value, job_key="research-1")
    )
    with pytest.raises(TaskRejected) as exc:
        await service.enqueue(
            TaskRequest(type=JobType.RESEARCH_TASK.value, job_key="research-2")
        )
    assert exc.value.reason == "AUTONOMOUS_QUEUE_FULL"
    # The rejection carries a concrete reason, never a silent drop.
    assert "queue is full" in str(exc.value)
    assert "1/1" in str(exc.value)

    analysis_id = await service.enqueue(
        TaskRequest(type=JobType.TURN_ANALYSIS.value, job_key="analysis-1")
    )
    assert (await repos.job.get(analysis_id)).status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_concurrent_submits_cannot_exceed_the_autonomous_capacity(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = _make_service(
        db, {JobType.RESEARCH_TASK.value: handler}, autonomous_pending_limit=1
    )
    results = await asyncio.gather(
        *[
            service.enqueue(
                TaskRequest(
                    type=JobType.RESEARCH_TASK.value, job_key=f"burst-{i}"
                )
            )
            for i in range(6)
        ],
        return_exceptions=True,
    )
    accepted = [r for r in results if isinstance(r, str)]
    rejected = [r for r in results if isinstance(r, TaskRejected)]
    assert len(accepted) == 1
    assert len(rejected) == 5
    cur = await db.execute(
        "SELECT COUNT(*) AS n FROM processing_jobs WHERE status = 'PENDING'"
        " AND job_type = ?",
        (JobType.RESEARCH_TASK.value,),
    )
    assert (await cur.fetchone())["n"] == 1


@pytest.mark.asyncio
async def test_heavy_tasks_queue_instead_of_being_rejected_on_enqueue(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = _make_service(
        db,
        {JobType.RESEARCH_TASK.value: handler},
        autonomous_pending_limit=5,
        heavy_task_limit=1,
    )
    first = await service.enqueue(
        TaskRequest(type=JobType.RESEARCH_TASK.value, job_key="heavy-1")
    )
    second = await service.enqueue(
        TaskRequest(type=JobType.RESEARCH_TASK.value, job_key="heavy-2")
    )
    assert (await repos.job.get(first)).status == JobStatus.PENDING
    assert (await repos.job.get(second)).status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_pending_order_keeps_maintenance_ahead_of_research(db):
    async def handler(payload, repos, logger):
        return None

    repos, worker, service = _make_service(
        db,
        {
            JobType.RESEARCH_TASK.value: handler,
            JobType.TURN_ANALYSIS.value: handler,
        },
    )
    await service.enqueue(
        TaskRequest(type=JobType.RESEARCH_TASK.value, job_key="r-low")
    )
    await service.enqueue(
        TaskRequest(
            type=JobType.TURN_ANALYSIS.value,
            job_key="analysis-normal",
            priority=TaskPriority.NORMAL,
        )
    )
    pending = await repos.job.pending()
    assert [job.job_key for job in pending] == ["analysis-normal", "r-low"]


# -- cancellation ---------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_after_search_prevents_the_memory_write(rt):
    await _enable_research(rt)
    task_id = await rt.repos.research.create("取消后不得写记忆", source_kind="test")
    job_id = await rt.task_service.enqueue(_research_request(task_id))

    original = rt.research_engine.web_search.search

    async def cancel_during_search(query, limit=5):
        await rt.task_service.cancel(job_id)
        return await original(query, limit=limit)

    rt.research_engine.web_search.search = cancel_during_search
    await rt.worker.run_once()

    assert (await rt.repos.job.get(job_id)).status == JobStatus.CANCELLED
    assert (await rt.repos.research.get(task_id))["status"] == "CANCELLED"
    assert await _memory_count(rt, task_id) == 0


@pytest.mark.asyncio
async def test_cancel_before_the_write_transaction_prevents_commit(rt):
    await _enable_research(rt)
    task_id = await rt.repos.research.create("分析后取消", source_kind="test")
    job_id = await rt.task_service.enqueue(_research_request(task_id))

    original = rt.research_engine.deepseek.synthesize

    async def cancel_during_synthesis(question, results):
        await rt.task_service.cancel(job_id)
        return await original(question, results)

    rt.research_engine.deepseek.synthesize = cancel_during_synthesis
    await rt.worker.run_once()

    job = await rt.repos.job.get(job_id)
    assert job.status == JobStatus.CANCELLED
    assert job.result is None
    assert (await rt.repos.research.get(task_id))["status"] == "CANCELLED"
    assert await _memory_count(rt, task_id) == 0


@pytest.mark.asyncio
async def test_cancel_after_the_atomic_commit_does_not_undo_completion(rt):
    await _enable_research(rt)
    task_id = await rt.repos.research.create("提交后取消", source_kind="test")
    job_id = await rt.task_service.enqueue(_research_request(task_id))
    await rt.worker.run_once()

    assert (await rt.repos.job.get(job_id)).status == JobStatus.COMPLETED
    assert (await rt.repos.research.get(task_id))["status"] == "COMPLETED"
    assert await _memory_count(rt, task_id) == 1

    # A later cancellation request cannot demote committed work.
    assert await rt.task_service.cancel(job_id) is False
    assert (await rt.repos.job.get(job_id)).status == JobStatus.COMPLETED
    assert (await rt.repos.research.get(task_id))["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_stale_worker_cannot_checkpoint_or_commit(rt):
    await _enable_research(rt)
    task_id = await rt.repos.research.create("过期执行者不得提交", source_kind="test")
    job_id = await rt.task_service.enqueue(_research_request(task_id))
    original_owner = rt.worker.owner_id
    assert await rt.repos.job.claim(job_id, original_owner, 300)

    # Another worker takes the row while the old owner still believes it runs.
    await rt.repos.db.execute(
        "UPDATE processing_jobs SET owner_id = ?, lease_expires_at = ?"
        " WHERE job_id = ?",
        ("other-worker", (utcnow() + timedelta(days=1)).isoformat(), job_id),
    )
    assert not await rt.repos.job.save_checkpoint(
        job_id, {"stage": "SEARCHED"}, owner_id=original_owner
    )
    action_ran = False

    async def action():
        nonlocal action_ran
        action_ran = True

    with pytest.raises(TaskLeaseLost):
        await rt.task_service.commit_atomic(job_id, original_owner, action)
    assert action_ran is False
    assert await _memory_count(rt, task_id) == 0


@pytest.mark.asyncio
async def test_pause_and_resume_run_search_and_memory_once(rt):
    await _enable_research(rt)
    task_id = await rt.repos.research.create("暂停后恢复只做一次", source_kind="test")
    searches = 0
    syntheses = 0
    original_search = rt.research_engine.web_search.search
    original_synthesize = rt.research_engine.deepseek.synthesize

    async def counted_search(query, limit=5):
        nonlocal searches
        searches += 1
        return await original_search(query, limit=limit)

    async def counted_synthesize(question, results):
        nonlocal syntheses
        syntheses += 1
        return await original_synthesize(question, results)

    rt.research_engine.web_search.search = counted_search
    rt.research_engine.deepseek.synthesize = counted_synthesize
    rt.task_service.resume_idle_seconds = 0

    job_id = await rt.task_service.enqueue(_research_request(task_id))
    with rt.task_service.foreground.scope():
        await rt.worker.run_once()
    paused = await rt.repos.job.get(job_id)
    assert paused.status == JobStatus.PENDING
    assert paused.attempts == 0

    await rt.worker.run_once()
    done = await rt.repos.job.get(job_id)
    assert done.status == JobStatus.COMPLETED
    assert searches == 1
    assert syntheses == 1
    assert await _memory_count(rt, task_id) == 1


@pytest.mark.asyncio
async def test_cancellation_survives_a_restart_without_resurrecting_research(rt):
    await _enable_research(rt)
    task_id = await rt.repos.research.create("重启后仍取消", source_kind="test")
    job_id = await rt.task_service.enqueue(_research_request(task_id))
    await rt.task_service.cancel(job_id)

    assert (await rt.repos.job.get(job_id)).status == JobStatus.CANCELLED
    assert (await rt.repos.research.get(task_id))["status"] == "CANCELLED"

    # Recovery only resets *expired* RUNNING work; it cannot revive a cancel.
    await rt.repos.job.recover_running()
    await rt.repos.research.recover_running()
    await rt.worker.run_once()
    assert (await rt.repos.job.get(job_id)).status == JobStatus.CANCELLED
    assert (await rt.repos.research.get(task_id))["status"] == "CANCELLED"
    assert await _memory_count(rt, task_id) == 0


# -- explicit memory permission -------------------------------------------


def _route(intent=ProcessingIntent.MEMORY_SAVE) -> ValidatedRouteDecision:
    return ValidatedRouteDecision(
        action=RouteAction.RESPOND,
        reply_necessity=ReplyNecessity.REQUIRED,
        addressee_assessment=RouteAddressee.AGENT,
        processing_intents=[intent],
    )


async def _persisted_event(rt, person_id: str, text: str = "我叫 Alice。"):
    event = make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )
    await rt.repos.conversation.get_or_create(event.conversation_id)
    await rt.repos.utterance.save(event)
    return event


def _explicit_payload(event, route, consent) -> dict:
    return {
        "event_id": event.event_id,
        "event": event.model_dump(mode="json"),
        "route": route.model_dump(mode="json"),
        "consent": consent.model_dump(mode="json") if consent is not None else None,
    }


@pytest.mark.asyncio
async def test_explicit_memory_resolves_the_event_from_its_trusted_reference(rt, register):
    person = await register("Alice")
    event = await _persisted_event(rt, person.person_id)
    route = _route()
    consent = await rt.repos.person.get_consent(person.person_id)

    job_id = await rt.task_service.enqueue(
        TaskRequest(
            type=JobType.MEMORY_SAVE.value,
            payload=_explicit_payload(event, route, consent),
            source=TaskSource.ROUTER,
            source_event_id=event.event_id,
            job_key=f"explicit_memory:{event.event_id}",
        )
    )
    job = await rt.repos.job.get(job_id)
    assert job.source_ref == event.event_id


@pytest.mark.asyncio
async def test_revoked_memory_permission_blocks_execution_after_creation(rt, register):
    person = await register("Alice")
    event = await _persisted_event(rt, person.person_id)
    route = _route()
    consent = await rt.repos.person.get_consent(person.person_id)
    job_id = await rt.task_service.enqueue(
        TaskRequest(
            type=JobType.MEMORY_SAVE.value,
            payload=_explicit_payload(event, route, consent),
            source=TaskSource.ROUTER,
            source_event_id=event.event_id,
            job_key=f"explicit_memory:{event.event_id}",
        )
    )

    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            updated_at=utcnow(),
        )
    )
    await rt.worker.run_once()

    assert (await rt.repos.job.get(job_id)).status == JobStatus.CANCELLED
    assert await rt.repos.analysis.get_candidates(event.event_id) == []
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0


@pytest.mark.asyncio
async def test_forged_payload_consent_cannot_grant_permission(rt, register):
    person = await register("Alice")
    event = await _persisted_event(rt, person.person_id)
    crafted = Consent(
        person_id=person.person_id,
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        updated_at=utcnow(),
    )
    payload = _explicit_payload(event, _route(), crafted)
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            updated_at=utcnow(),
        )
    )

    handler = build_explicit_memory_task_handler(rt.explicit_memory_operations)
    outcome = await handler(payload, rt.repos, LOG)

    assert str(outcome.status) == TaskStatus.FAILED
    assert outcome.error == "permission_denied"
    assert await rt.repos.analysis.get_candidates(event.event_id) == []


@pytest.mark.asyncio
async def test_unverifiable_source_event_is_rejected_not_default_allowed(rt, register):
    person = await register("Alice")
    route = _route()
    consent = await rt.repos.person.get_consent(person.person_id)

    # The job points at an event that does not exist.
    with pytest.raises(TaskRejected) as exc:
        await rt.task_service.enqueue(
            TaskRequest(
                type=JobType.MEMORY_SAVE.value,
                payload={
                    "event_id": "missing-event",
                    "route": route.model_dump(mode="json"),
                    "consent": consent.model_dump(mode="json"),
                },
                source=TaskSource.ROUTER,
                source_event_id="missing-event",
                job_key="explicit_memory:missing",
            )
        )
    assert exc.value.reason == "PERMISSION_DENIED"

    # No reference at all is also unverifiable.
    with pytest.raises(TaskRejected):
        await rt.task_service.enqueue(
            TaskRequest(
                type=JobType.MEMORY_SAVE.value,
                payload={"route": route.model_dump(mode="json")},
                source=TaskSource.ROUTER,
                job_key="explicit_memory:none",
            )
        )


@pytest.mark.asyncio
async def test_permission_revoked_between_claim_and_commit_blocks_the_result(db):
    permitted = True

    def permission(request):
        return permitted

    async def handler(payload, repos, logger):
        nonlocal permitted
        permitted = False  # revoked while the task is running
        return {"side_effect": "would_have_been_written"}

    repos, worker, service = _make_service(
        db, {"WORK": handler}, permission_checker=permission
    )
    result = await service.execute_now(
        TaskRequest(type="WORK", job_key="commit-perm", source=TaskSource.USER)
    )

    assert result.status == TaskStatus.CANCELLED
    stored = await repos.job.get(result.task_id)
    assert stored.status == JobStatus.CANCELLED
    assert stored.result is None


@pytest.mark.asyncio
async def test_handler_uses_the_database_event_and_live_consent(rt, register):
    person = await register("Alice")
    event = await _persisted_event(rt, person.person_id, "真实事件内容")
    forged = event.model_copy(update={"text": "伪造的事件内容"})
    live_consent = await rt.repos.person.get_consent(person.person_id)
    payload = _explicit_payload(forged, _route(), live_consent)

    captured: dict = {}

    class CapturingOps:
        async def execute(self, event, route, consent, routing_context):
            captured["event"] = event
            captured["consent"] = consent

            class Result:
                completed = True
                status = "SAVED"
                error_code = None

                def as_dict(self):
                    return {
                        "event_id": event.event_id,
                        "intent": "MEMORY_SAVE",
                        "status": "SAVED",
                        "completed": True,
                    }

            return Result()

    handler = build_explicit_memory_task_handler(CapturingOps())
    outcome = await handler(payload, rt.repos, LOG)

    # A successful handler returns the serialisable result payload.
    assert outcome["status"] == "SAVED"
    # The DB row wins over the payload snapshot, and consent is live.
    assert captured["event"].text == "真实事件内容"
    assert captured["consent"].allow_memory_learning is True
