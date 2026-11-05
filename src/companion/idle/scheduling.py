"""Durable research scheduling through the unified off-task task service.

A research detail row and its ``RESEARCH_TASK`` job are written inside one
``BEGIN IMMEDIATE`` transaction: a crash between the two inserts can never
leave a scheduled question without an executor, or an executor without its
detail row. Dedupe happens in the same transaction, so two curiosity cycles
that propose the same question create exactly one record.

There is no second scheduler here. This module only decides *whether* a
research need deserves a durable row and then hands the execution authority to
the existing :class:`~companion.tasks.service.TaskService` / ``JobWorker``.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.domain.enums import JobType, TaskMode, TaskPriority, TaskSource
from companion.storage.repositories import ResearchCapacityError
from companion.tasks.models import TaskRejected, TaskRequest

RESEARCH_JOB_TYPE = JobType.RESEARCH_TASK.value

logger = logging.getLogger("companion.idle.scheduling")


def research_job_request(
    task_id: str,
    *,
    source: TaskSource = TaskSource.SYSTEM,
    priority: TaskPriority = TaskPriority.LOW,
) -> TaskRequest:
    return TaskRequest(
        type=RESEARCH_JOB_TYPE,
        payload={"research_task_id": task_id},
        mode=TaskMode.BACKGROUND,
        source=source,
        priority=priority,
        source_ref=task_id,
        job_key=f"research:{task_id}",
    )


async def enqueue_existing_research(
    task_service, task_id: str, *, source: TaskSource = TaskSource.SYSTEM
) -> Optional[str]:
    """Ensure one PENDING research row has exactly one durable job."""
    if task_service is None:
        return None
    return await task_service.enqueue(
        research_job_request(task_id, source=source)
    )


async def schedule_research(
    db,
    research_repo,
    task_service,
    question: str,
    *,
    origin: str,
    source_kind: str = "curiosity",
    source_memory_ids: list[str] | tuple[str, ...] = (),
    priority: int = 0,
    max_active: Optional[int] = None,
    recheck_seconds: Optional[int] = None,
    source: Optional[TaskSource] = None,
    task_priority: Optional[TaskPriority] = None,
    log: Optional[logging.Logger] = None,
) -> dict:
    """Create (deduped) research + job atomically.

    Returns a truthful report: ``created`` says whether a new detail row was
    inserted; ``reason`` explains dedupe/capacity; ``job_id`` is the durable
    executor (``None`` only when no task service is wired). A rejected enqueue
    leaves the detail row PENDING for a later cycle and is reported instead of
    silently swallowed.
    """
    log = log or logger
    if source is None:
        source = TaskSource.USER if str(origin) == "user" else TaskSource.SYSTEM
    if task_priority is None:
        task_priority = (
            TaskPriority.HIGH if source is TaskSource.USER else TaskPriority.LOW
        )

    try:
        async with db.transaction(immediate=True):
            task_id, created, reason = await research_repo.create_unique(
                question,
                origin=origin,
                source_kind=source_kind,
                source_memory_ids=source_memory_ids,
                priority=priority,
                max_active=max_active,
                recheck_seconds=recheck_seconds,
            )
            if not created:
                log.info(
                    "research_schedule_deduped research_id=%s reason=%s",
                    task_id,
                    reason,
                )
                return {
                    "task_id": task_id,
                    "created": False,
                    "reason": reason,
                    "job_id": None,
                }

            job_id = None
            if task_service is not None:
                try:
                    job_id = await task_service.enqueue(
                        research_job_request(
                            task_id, source=source, priority=task_priority
                        )
                    )
                except TaskRejected as exc:
                    # Keep the detail row: the idle scheduler can schedule it
                    # in a later cycle. The reason is reported, never hidden.
                    log.info(
                        "research_schedule_not_enqueued research_id=%s reason=%s",
                        task_id,
                        exc.reason,
                    )
                    return {
                        "task_id": task_id,
                        "created": True,
                        "reason": exc.reason,
                        "job_id": None,
                    }
            return {
                "task_id": task_id,
                "created": True,
                "reason": reason,
                "job_id": job_id,
            }
    except ResearchCapacityError as exc:
        log.info(
            "research_schedule_capacity_reached active=%s limit=%s",
            exc.active,
            exc.limit,
        )
        return {
            "task_id": None,
            "created": False,
            "reason": "AUTONOMOUS_RESEARCH_FULL",
            "job_id": None,
            "active": exc.active,
            "limit": exc.limit,
        }


__all__ = [
    "schedule_research",
    "enqueue_existing_research",
    "research_job_request",
    "RESEARCH_JOB_TYPE",
]
