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
    from companion.routing.models import RoutingContext, ValidatedRouteDecision

    async def handle(payload: dict, repos: Repositories, logger: logging.Logger):
        route = ValidatedRouteDecision.model_validate(payload["route"])
        routing_context = (
            RoutingContext.model_validate(payload["routing_context"])
            if payload.get("routing_context") is not None
            else None
        )
        # Resolve the source event from a trusted reference, then re-read the
        # real row. The event/consent snapshots in the payload are only a
        # historical record of what was proposed; they can never grant
        # permission or substitute a different event.
        snapshot_event = payload.get("event") or {}
        event_id = payload.get("event_id") or snapshot_event.get("event_id")
        if not event_id and routing_context is not None:
            event_id = routing_context.current_event_id
        if not event_id:
            return TaskOutcome(
                status=TaskStatus.FAILED, error="missing_source_event"
            )
        if (
            snapshot_event.get("event_id")
            and snapshot_event.get("event_id") != event_id
        ):
            return TaskOutcome(
                status=TaskStatus.FAILED, error="event_identity_mismatch"
            )
        event = await repos.utterance.get(str(event_id))
        if event is None:
            return TaskOutcome(
                status=TaskStatus.FAILED, error="source_event_unavailable"
            )
        speaker = event.speaker_person_id
        if not speaker:
            return TaskOutcome(
                status=TaskStatus.FAILED, error="no_registered_speaker"
            )
        consent = await repos.person.get_consent(speaker)
        if consent is None or not (
            consent.allow_processing
            and consent.allow_persistence
            and consent.allow_memory_learning
        ):
            return TaskOutcome(
                status=TaskStatus.FAILED, error="permission_denied"
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
        task_id = str(task_id)
        context = current_task_context()
        checkpoint = dict(context.checkpoint) if context is not None else {}
        # The job row is the sole execution authority. Its stable id names the
        # research detail owner, and its lease/owner is re-checked at every
        # stage and at the atomic commit.
        owner = f"job:{context.task_id}" if context is not None else None

        async def authorize() -> bool:
            return await context.can_continue()

        async def commit(action, *, result=None):
            return await context.commit_atomic(action, result=result)

        try:
            result = await research_engine.run_resumable(
                task_id,
                checkpoint=checkpoint,
                pause_check=context.should_pause if context is not None else None,
                save_checkpoint=context.save_checkpoint if context is not None else None,
                owner_id=owner,
                authorize=authorize if context is not None else None,
                commit=commit if context is not None else None,
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
            cancel = getattr(research_engine, "cancel_detail", None)
            if cancel is not None:
                await cancel(task_id, owner)
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
