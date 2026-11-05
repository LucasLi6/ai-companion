"""Minimal unified task service (V0.9).

This is the *only* entry point for off-task work:

- :meth:`TaskService.execute_now` runs a checked task synchronously and returns
  its real result; the speaker waits for it.
- :meth:`TaskService.enqueue` persists an idempotent background task for the
  existing :class:`~companion.jobs.worker.JobWorker`.
- :meth:`TaskService.cancel` / :meth:`get_result` are the local control and
  observation API.

It intentionally contains no planner, no workflow DAG and no model call. All
checks are local Python: handler support, permission, stable-key dedupe,
autonomous queue budget and an optional necessity predicate. A model-declared
proposal is only data; it cannot choose its own source, permission or priority.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Mapping, Optional

from companion.domain.enums import JobStatus, JobType, TaskMode, TaskPriority, TaskSource
from companion.domain.models import ProcessingJob, utcnow
from companion.storage.repositories import Repositories
from companion.tasks.context import TaskContext, reset_task_context, set_task_context
from companion.tasks.foreground import ForegroundActivity
from companion.tasks.models import (
    TaskOutcome,
    TaskPaused,
    TaskRejected,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

__all__ = ["TaskService", "TaskHandler", "TaskResult"]

TaskHandler = Callable[[dict, Repositories, logging.Logger], Awaitable[Any]]

_DEFAULT_HEAVY_TYPES = (
    JobType.RESEARCH_TASK.value,
    JobType.TURN_ANALYSIS.value,
)


class TaskService:
    def __init__(
        self,
        repos: Repositories,
        handlers: Optional[Mapping[str, TaskHandler]] = None,
        *,
        worker=None,
        logger: Optional[logging.Logger] = None,
        foreground_activity: Optional[ForegroundActivity] = None,
        autonomous_pending_limit: int = 10,
        heavy_task_limit: int = 1,
        heavy_types: tuple[str, ...] = _DEFAULT_HEAVY_TYPES,
        resume_idle_seconds: float = 0.5,
        permission_checker: Optional[Callable[[TaskRequest], Any]] = None,
        necessity_checker: Optional[Callable[[TaskRequest], Any]] = None,
    ) -> None:
        self.repos = repos
        self.logger = logger or logging.getLogger("companion.tasks")
        self.foreground = foreground_activity or ForegroundActivity()
        self.autonomous_pending_limit = max(0, int(autonomous_pending_limit))
        self.heavy_task_limit = max(1, int(heavy_task_limit))
        self.heavy_types = tuple(dict.fromkeys(heavy_types))
        self.resume_idle_seconds = max(0.0, float(resume_idle_seconds))
        self.permission_checker = permission_checker
        self.necessity_checker = necessity_checker
        self._handlers: dict[str, TaskHandler] = {}
        self._revoked_sources: set[tuple[str, str]] = set()
        self.worker = worker
        if handlers:
            for key, handler in handlers.items():
                self.register_handler(key, handler)
        if worker is not None:
            # The worker consults this service before claiming so permission
            # revocation and foreground activity are not bypassed.
            worker.task_service = self
            if not worker.handlers:
                worker.handlers = self._handlers

    # -- registration ------------------------------------------------------

    def register_handler(self, task_type: str | JobType, handler: TaskHandler) -> None:
        name = task_type.value if isinstance(task_type, JobType) else str(task_type)
        if not name:
            raise ValueError("task type must be non-empty")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers[name] = handler
        if self.worker is not None and self.worker.handlers is not self._handlers:
            self.worker.handlers[name] = handler

    def unregister_handler(self, task_type: str | JobType) -> None:
        name = task_type.value if isinstance(task_type, JobType) else str(task_type)
        self._handlers.pop(name, None)

    @property
    def handlers(self) -> dict[str, TaskHandler]:
        return self._handlers

    @property
    def supported_types(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def foreground_scope(self):
        return self.foreground.scope()

    # -- checks ------------------------------------------------------------

    async def _check_supported(self, request: TaskRequest) -> None:
        if request.type not in self._handlers:
            raise TaskRejected(
                "UNSUPPORTED_TASK_TYPE",
                detail=f"no handler registered for {request.type}",
            )

    async def _check_permission(self, request: TaskRequest) -> None:
        if self._source_revoked(request):
            raise TaskRejected("PERMISSION_REVOKED", detail="source was revoked")
        if self.permission_checker is None:
            return
        allowed = await self._call_checker(self.permission_checker, request)
        if not allowed:
            raise TaskRejected("PERMISSION_DENIED", detail="local permission check")

    async def _check_necessity(self, request: TaskRequest) -> None:
        if self.necessity_checker is None:
            return
        if request.resolved_source is TaskSource.USER or request.resolved_mode is TaskMode.FOREGROUND:
            # A direct user/foreground request is authoritative; an autonomous
            # heuristic may not silently discard it.
            return
        necessary = await self._call_checker(self.necessity_checker, request)
        if not necessary:
            raise TaskRejected("NOT_NECESSARY", detail="autonomous necessity check")

    async def _call_checker(self, checker, request: TaskRequest) -> Any:
        """Call a local predicate that may be sync/async and take one or two args."""
        try:
            signature = inspect.signature(checker)
            accepts_request_only = len(signature.parameters) <= 1
        except (TypeError, ValueError):
            accepts_request_only = True
        value = checker(request) if accepts_request_only else checker(request, self.repos)
        if inspect.isawaitable(value):
            value = await value
        return value

    def _source_revoked(self, request: TaskRequest) -> bool:
        if request.source_ref is None:
            return False
        return (
            str(request.resolved_source.value),
            str(request.source_ref),
        ) in self._revoked_sources

    @staticmethod
    def _source_ref(request: TaskRequest) -> Optional[str]:
        if request.source_ref or request.source_event_id:
            return request.source_ref or request.source_event_id
        if request.type == JobType.RESEARCH_TASK.value:
            # Stable link to the research detail row without a second owner.
            return request.payload.get("research_task_id") or request.payload.get(
                "task_id"
            )
        return None

    async def _check_budget(self, request: TaskRequest) -> None:
        if request.resolved_source is TaskSource.USER:
            # User-requested work is never dropped because the autonomous queue
            # is full.
            return
        queued = await self.repos.job.count_pending(autonomous_only=True)
        if queued >= self.autonomous_pending_limit:
            raise TaskRejected(
                "AUTONOMOUS_QUEUE_FULL",
                detail=f"{queued} autonomous tasks already pending",
            )
        if request.type in self.heavy_types:
            running = await self.repos.job.count_running(self.heavy_types)
            if running >= self.heavy_task_limit:
                raise TaskRejected(
                    "RESOURCE_BUSY",
                    detail="heavy background task already running",
                )

    # -- enqueue -----------------------------------------------------------

    async def enqueue(self, request: TaskRequest) -> str:
        """Persist a durable background task, returning its stable id.

        The request is deduplicated on its local stable key *inside* the
        enqueue transaction. All checks happen before the row is written.
        """
        if not isinstance(request, TaskRequest):
            raise TypeError("enqueue expects a TaskRequest")
        # ``enqueue`` is by definition the background path. A caller that needs
        # an immediate result must use ``execute_now``; silently running here
        # would block the speaker.
        bg = request if request.resolved_mode is TaskMode.BACKGROUND else request.with_mode(
            TaskMode.BACKGROUND
        )
        await self._check_supported(bg)
        await self._check_permission(bg)
        await self._check_necessity(bg)
        existing = await self.repos.job.get_by_key(bg.stable_key)
        if existing is not None:
            return existing.job_id
        await self._check_budget(bg)
        job_id = await self.repos.job.enqueue(
            bg.type,
            bg.stable_key,
            bg.payload,
            bg.max_attempts,
            bg.next_run_at,
            priority=bg.resolved_priority.value,
            source_kind=bg.resolved_source.value,
            source_ref=self._source_ref(bg),
            execution_mode=TaskMode.BACKGROUND.value,
            checkpoint=bg.checkpoint,
        )
        self.logger.info(
            "task_enqueued task_id=%s type=%s priority=%s source=%s",
            job_id,
            bg.type,
            bg.resolved_priority.value,
            bg.resolved_source.value,
        )
        return job_id

    # -- foreground --------------------------------------------------------

    async def execute_now(self, request: TaskRequest) -> TaskResult:
        """Execute a task synchronously and return its truthful result.

        The row is durable before execution begins. Concurrent submission of
        the same stable key does not run the handler twice: one caller claims
        it, the other observes the current state. A foreground request may
        promote an existing PENDING background row instead of duplicating it.
        """
        if not isinstance(request, TaskRequest):
            raise TypeError("execute_now expects a TaskRequest")
        fg = request.with_mode(TaskMode.FOREGROUND)
        # Foreground work is HIGH by construction: the current turn is waiting,
        # so it must not inherit an autonomous LOW/NORMAL hint.
        fg = TaskRequest(
            type=fg.type,
            payload=fg.payload,
            mode=fg.mode,
            source=fg.source,
            priority=TaskPriority.HIGH,
            source_ref=fg.source_ref,
            source_event_id=fg.source_event_id,
            job_key=fg.job_key,
            max_attempts=fg.max_attempts,
            next_run_at=None,
            checkpoint=fg.checkpoint,
        )
        await self._check_supported(fg)
        await self._check_permission(fg)
        await self._check_necessity(fg)

        with self.foreground.scope():
            job_id = await self.repos.job.enqueue(
                fg.type,
                fg.stable_key,
                fg.payload,
                fg.max_attempts,
                None,
                priority=fg.resolved_priority.value,
                source_kind=fg.resolved_source.value,
                source_ref=self._source_ref(fg),
                execution_mode=TaskMode.FOREGROUND.value,
                checkpoint=fg.checkpoint,
            )
            job = await self.repos.job.get(job_id)
            if job is None:
                return TaskResult(job_id, TaskStatus.FAILED, error="task_not_found")
            if str(job.status.value if isinstance(job.status, JobStatus) else job.status) in {
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            }:
                return self._result_from_job(job)
            if job.cancel_requested:
                await self.repos.job.mark_cancelled(job_id)
                job = await self.repos.job.get(job_id)
                return self._result_from_job(job)

            await self.repos.job.promote_foreground(job_id)
            job = await self.repos.job.get(job_id)
            if job is None:
                return TaskResult(job_id, TaskStatus.FAILED, error="task_not_found")
            if str(job.status.value if isinstance(job.status, JobStatus) else job.status) != "PENDING":
                # Another executor already owns it; never run the side effect
                # twice. Report the real observed state.
                return self._result_from_job(job)

            claimed = await self.repos.job.claim(
                job_id, f"foreground:{fg.stable_key}", 300
            )
            if not claimed:
                latest = await self.repos.job.get(job_id)
                return self._result_from_job(latest) if latest else TaskResult(
                    job_id, TaskStatus.FAILED, error="claim_failed"
                )
            job = await self.repos.job.get(job_id)
            if job is None:
                return TaskResult(job_id, TaskStatus.FAILED, error="task_not_found")
            return await self._run_claimed(job)

    async def _run_claimed(self, job: ProcessingJob) -> TaskResult:
        task_id = job.job_id
        job_type = job.job_type.value if isinstance(job.job_type, JobType) else str(job.job_type)
        handler = self._handlers.get(job_type)
        if handler is None:
            await self.repos.job.fail(task_id, f"no handler for {job_type}")
            return TaskResult(task_id, TaskStatus.FAILED, error=f"no handler for {job_type}")
        context = TaskContext(
            task_id=task_id,
            mode=TaskMode(job.execution_mode),
            source=TaskSource(job.source_kind),
            checkpoint=dict(job.checkpoint or {}),
            service=self,
        )
        token = set_task_context(context)
        try:
            try:
                outcome = await handler(job.payload, self.repos, self.logger)
            except TaskPaused as paused:
                return await self._pause(task_id, paused.checkpoint or context.checkpoint)
            except Exception as exc:  # noqa: BLE001 — truthful failure.
                # A cancellation that arrived while the handler was running is
                # terminal, even when the handler then failed. Never report a
                # retryable failure for work the user already cancelled.
                if await self._cancelled(task_id):
                    await self.repos.job.mark_cancelled(task_id)
                    return TaskResult(
                        task_id,
                        TaskStatus.CANCELLED,
                        error="cancelled",
                        checkpoint=context.checkpoint,
                    )
                message = str(exc) or type(exc).__name__
                await self.repos.job.fail(task_id, message)
                self.logger.error(
                    "task_failed task_id=%s type=%s error=%s",
                    task_id,
                    job_type,
                    type(exc).__name__,
                )
                return TaskResult(task_id, TaskStatus.FAILED, error=message)

            if isinstance(outcome, TaskResult):
                return await self._apply_result(job, outcome, context.checkpoint)
            if isinstance(outcome, TaskOutcome):
                return await self._apply_outcome(job, outcome, context.checkpoint)
            return await self._commit(job, outcome, context.checkpoint)
        finally:
            reset_task_context(token)

    async def _apply_result(
        self, job: ProcessingJob, result: TaskResult, checkpoint: dict
    ) -> TaskResult:
        status = str(result.status)
        if status == TaskStatus.PENDING:
            return await self._pause(job.job_id, result.checkpoint or checkpoint)
        if status == TaskStatus.CANCELLED:
            await self.repos.job.mark_cancelled(job.job_id)
            return result
        if status == TaskStatus.FAILED:
            if await self._cancelled(job.job_id):
                await self.repos.job.mark_cancelled(job.job_id)
                return TaskResult(
                    job.job_id,
                    TaskStatus.CANCELLED,
                    error="cancelled",
                    checkpoint=checkpoint,
                )
            message = result.error or "task failed"
            await self.repos.job.fail(job.job_id, message, result.result)
            return TaskResult(
                job.job_id,
                TaskStatus.FAILED,
                result=result.result,
                error=message,
                checkpoint=result.checkpoint or checkpoint,
            )
        return await self._commit(job, result.result, result.checkpoint or checkpoint)

    async def _apply_outcome(
        self, job: ProcessingJob, outcome: TaskOutcome, checkpoint: dict
    ) -> TaskResult:
        status = str(outcome.status)
        if status == TaskStatus.PENDING:
            return await self._pause(
                job.job_id, outcome.checkpoint or checkpoint
            )
        if status == TaskStatus.CANCELLED:
            await self.repos.job.mark_cancelled(job.job_id)
            return TaskResult(
                job.job_id,
                TaskStatus.CANCELLED,
                error=outcome.error,
                checkpoint=outcome.checkpoint,
            )
        if status == TaskStatus.FAILED:
            if await self._cancelled(job.job_id):
                await self.repos.job.mark_cancelled(job.job_id)
                return TaskResult(
                    job.job_id,
                    TaskStatus.CANCELLED,
                    error="cancelled",
                    checkpoint=checkpoint,
                )
            message = outcome.error or "task failed"
            await self.repos.job.fail(job.job_id, message, outcome.result)
            return TaskResult(
                job.job_id,
                TaskStatus.FAILED,
                result=outcome.result,
                error=message,
                checkpoint=outcome.checkpoint or checkpoint,
            )
        return await self._commit(
            job, outcome.result, outcome.checkpoint or checkpoint
        )

    async def _commit(
        self, job: ProcessingJob, result: Any, checkpoint: dict
    ) -> TaskResult:
        """Run pre-commit checks and commit only while this owner holds the lease."""
        allowed, reason = await self.can_commit(job)
        if not allowed:
            await self.repos.job.mark_cancelled(job.job_id)
            return TaskResult(
                job.job_id,
                TaskStatus.CANCELLED,
                error=reason.lower(),
                checkpoint=checkpoint,
            )
        committed = await self.repos.job.complete_owned(
            job.job_id, job.owner_id or "", result
        )
        if not committed:
            return TaskResult(
                job.job_id,
                TaskStatus.FAILED,
                error="lease_lost",
                checkpoint=checkpoint,
            )
        return TaskResult(
            job.job_id,
            TaskStatus.COMPLETED,
            result=result,
            checkpoint=checkpoint or None,
        )

    async def _cancelled(self, task_id: str) -> bool:
        job = await self.repos.job.get(task_id)
        return bool(job is not None and job.cancel_requested)

    async def _pause(self, task_id: str, checkpoint: dict) -> TaskResult:
        resume_at = utcnow() + timedelta(seconds=self.resume_idle_seconds)
        await self.repos.job.yield_for_foreground(
            task_id, checkpoint, resume_after=resume_at
        )
        self.logger.info("task_paused task_id=%s reason=foreground", task_id)
        return TaskResult(
            task_id, TaskStatus.PENDING, checkpoint=dict(checkpoint or {})
        )

    # -- observation / control --------------------------------------------

    async def get_result(self, task_id: str) -> Optional[TaskResult]:
        job = await self.repos.job.get(task_id)
        if job is None:
            return None
        return self._result_from_job(job)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a task locally. Running work yields at its next checkpoint."""
        return await self.repos.job.cancel(task_id)

    async def revoke_source(self, source_kind: str | TaskSource, source_ref: str) -> int:
        """Trusted revocation, independent of queue order.

        Pending rows are cancelled immediately and running rows get the
        cooperative cancel flag; later pre-claim checks also reject the source.
        The method returns the number of PENDING rows cancelled.
        """
        source = (
            source_kind.value
            if isinstance(source_kind, TaskSource)
            else str(source_kind).upper()
        )
        self._revoked_sources.add((source, str(source_ref)))
        cancelled = await self.repos.job.cancel_pending_for_source(source, str(source_ref))
        cur = await self.repos.db.execute(
            "UPDATE processing_jobs SET cancel_requested = 1"
            " WHERE status = 'RUNNING' AND source_kind = ? AND source_ref = ?",
            (source, str(source_ref)),
        )
        self.logger.info(
            "task_source_revoked source=%s ref=%s cancelled=%d running_flagged=%d",
            source,
            source_ref,
            cancelled,
            cur.rowcount if hasattr(cur, "rowcount") else 0,
        )
        return cancelled

    async def can_commit(self, job: ProcessingJob) -> tuple[bool, str]:
        """Local pre-commit check: cancellation and permission only.

        Foreground activity is not a reason to discard a *finished* result; a
        checkpoint pause happens before the work, not at commit time.
        """
        if job.cancel_requested or await self.repos.job.cancellation_requested(
            job.job_id
        ):
            return False, "CANCELLED"
        request = self._request_from_job(job)
        if self._source_revoked(request):
            return False, "PERMISSION_REVOKED"
        if self.permission_checker is not None:
            try:
                await self._check_permission(request)
            except TaskRejected:
                return False, "PERMISSION_REVOKED"
        return True, "OK"

    async def can_claim(self, job: ProcessingJob) -> tuple[bool, str]:
        """Worker pre-claim check. All decisions are local.

        Returns ``(False, reason)`` where ``reason`` is one of ``CANCELLED``,
        ``PERMISSION_REVOKED``, ``FOREGROUND_ACTIVE`` or ``RESOURCE_BUSY``.
        """
        if job.cancel_requested:
            return False, "CANCELLED"
        if job.execution_mode == TaskMode.BACKGROUND.value and self.foreground.active:
            return False, "FOREGROUND_ACTIVE"
        request = self._request_from_job(job)
        if self._source_revoked(request):
            return False, "PERMISSION_REVOKED"
        if self.permission_checker is not None:
            try:
                await self._check_permission(request)
            except TaskRejected:
                return False, "PERMISSION_REVOKED"
        job_type = job.job_type.value if isinstance(job.job_type, JobType) else str(job.job_type)
        if job_type in self.heavy_types:
            running = await self.repos.job.count_running(self.heavy_types)
            if running >= self.heavy_task_limit:
                return False, "RESOURCE_BUSY"
        return True, "OK"

    def request_from_job(self, job: ProcessingJob) -> TaskRequest:
        return self._request_from_job(job)

    @staticmethod
    def _request_from_job(job: ProcessingJob) -> TaskRequest:
        return TaskRequest(
            type=job.job_type.value
            if isinstance(job.job_type, JobType)
            else str(job.job_type),
            payload=dict(job.payload or {}),
            mode=job.execution_mode,
            source=job.source_kind,
            priority=job.priority,
            source_ref=job.source_ref,
            job_key=job.job_key,
            max_attempts=job.max_attempts,
            checkpoint=dict(job.checkpoint or {}),
        )

    @staticmethod
    def _result_from_job(job: ProcessingJob) -> TaskResult:
        status_value = (
            job.status.value if isinstance(job.status, JobStatus) else str(job.status)
        )
        return TaskResult(
            task_id=job.job_id,
            status=TaskStatus(status_value),
            result=job.result,
            error=job.last_error,
            checkpoint=dict(job.checkpoint or {}) or None,
        )
