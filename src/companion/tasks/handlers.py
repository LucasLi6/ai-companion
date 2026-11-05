"""Task handlers for the unified off-task layer.

The existing job handlers keep their signatures. This module adds only the
adapter needed to run an existing research task as a durable, resumable
background job, and a tiny helper for building handler maps without a second
registry. There is no second research scheduler here: the task handler executes
one explicitly enqueued research task through the same ``ResearchEngine`` the
idle pipeline uses.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from companion.domain.enums import JobType
from companion.storage.repositories import Repositories
from companion.tasks.context import current_task_context
from companion.tasks.models import TaskOutcome, TaskPaused, TaskStatus

Handler = Callable[[dict, Repositories, logging.Logger], Awaitable[Any]]


def build_explicit_memory_task_handler(explicit_service) -> Handler:
    """Adapt ``ExplicitMemoryOperationService`` to the task boundary.

    A foreground explicit memory request is a real task: the row is durable,
    deduplicated by event/intent, and its result is the *same* auditable
    ``MemoryOperationResult`` object the speaker already uses. Serializing the
    checked event/route/consent into the payload keeps a retry local and
    permission-revalidating; the handler never trusts a model-declared grant.
    """
    from companion.domain.models import Consent, UtteranceEvent
    from companion.routing.models import RoutingContext, ValidatedRouteDecision

    async def handle(payload: dict, repos: Repositories, logger: logging.Logger):
        event = UtteranceEvent.model_validate(payload["event"])
        route = ValidatedRouteDecision.model_validate(payload["route"])
        consent = (
            Consent.model_validate(payload["consent"])
            if payload.get("consent") is not None
            else None
        )
        routing_context = (
            RoutingContext.model_validate(payload["routing_context"])
            if payload.get("routing_context") is not None
            else None
        )
        result = await explicit_service.execute(
            event, route, consent, routing_context
        )
        data = result.as_dict()
        if not result.completed or str(result.status).upper() == "FAILED":
            return TaskOutcome(
                status=TaskStatus.FAILED,
                error=result.error_code or result.status,
                result=data,
            )
        return data

    return handle


def build_research_task_handler(research_engine) -> Handler:
    """Adapt ``ResearchEngine`` to the durable task handler signature.

    The checkpoint/ownership state lives on the *job* row; the research task
    itself remains the detail store linked by ``task_id``. A cooperative pause
    propagates as :class:`TaskPaused` only at a stage boundary inside the
    engine, after outputs already produced have been checkpointed by reference.
    """

    async def handle(payload: dict, repos: Repositories, logger: logging.Logger):
        task_id = payload.get("research_task_id") or payload.get("task_id")
        if not task_id:
            return TaskOutcome(
                status=TaskStatus.FAILED, error="missing research_task_id"
            )
        context = current_task_context()
        checkpoint = dict(context.checkpoint) if context is not None else {}
        try:
            result = await research_engine.run_resumable(
                str(task_id),
                checkpoint=checkpoint,
                pause_check=context.should_pause if context is not None else None,
                save_checkpoint=context.save_checkpoint if context is not None else None,
            )
        except TaskPaused:
            raise
        if not isinstance(result, dict):
            return TaskOutcome(status=TaskStatus.COMPLETED, result=result)
        status = str(result.get("status") or "").upper()
        if status == "COMPLETED":
            return TaskOutcome(
                status=TaskStatus.COMPLETED,
                result={
                    "research_task_id": task_id,
                    "memory_id": result.get("memory_id"),
                    "result": result.get("result"),
                },
                checkpoint={"stage": "DONE"},
            )
        if status == "CANCELLED":
            return TaskOutcome(status=TaskStatus.CANCELLED)
        if status == "PENDING":
            return TaskOutcome(
                status=TaskStatus.PENDING, checkpoint=checkpoint
            )
        return TaskOutcome(
            status=TaskStatus.FAILED,
            error=result.get("error") or f"research task ended {status or 'UNKNOWN'}",
        )

    return handle


def build_default_task_handlers(
    *,
    memory_consolidation: Optional[Handler] = None,
    turn_analysis: Optional[Handler] = None,
    embedding: Optional[Handler] = None,
    research_engine=None,
) -> dict[str, Handler]:
    """Build a handler map without introducing a second dispatcher.

    Only handlers that are actually available are registered; ``TaskService``
    rejects every other type as unsupported before writing a row.
    """
    handlers: dict[str, Handler] = {}
    if memory_consolidation is not None:
        handlers[JobType.MEMORY_CONSOLIDATION.value] = memory_consolidation
    if turn_analysis is not None:
        handlers[JobType.TURN_ANALYSIS.value] = turn_analysis
    if embedding is not None:
        handlers[JobType.EMBEDDING.value] = embedding
    if research_engine is not None:
        handlers[JobType.RESEARCH_TASK.value] = build_research_task_handler(
            research_engine
        )
    return handlers
