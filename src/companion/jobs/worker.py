"""A durable background job processor.

Jobs live in SQLite (``processing_jobs``), not in an ``asyncio.Queue``, so a
restart can recover them. On startup, leftover ``RUNNING`` jobs are reset to
``PENDING``; idempotent handlers ensure this never duplicates side effects.
Claiming is atomic and conditional (``UPDATE … WHERE status='PENDING'`` with a
``rowcount`` check), so multiple workers — even on separate connections — can
compete for jobs safely and at most one runs each handler. V0.0 deliberately
runs one worker by default and no complex concurrency pool.

V0.9 adds cooperative scheduling only:

- the poller never claims a ``FOREGROUND`` task;
- while foreground activity is active, background work is deferred, not killed;
- a handler may return ``TaskOutcome(PENDING, checkpoint=...)`` or raise
  ``TaskPaused`` at a safe stage boundary. The job returns to PENDING without
  consuming a retry, and resumes from its checkpoint later.

No database transaction or external request is interrupted mid-flight.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import timedelta
from typing import Any, Awaitable, Callable, Optional

from companion.domain.enums import JobStatus, JobType
from companion.domain.models import ProcessingJob, utcnow
from companion.storage.repositories import JobRepository, Repositories
from companion.tasks.context import TaskContext, reset_task_context, set_task_context
from companion.tasks.models import TaskMode, TaskOutcome, TaskPaused, TaskSource, TaskStatus

Handler = Callable[[dict, Repositories, logging.Logger], Awaitable[Any]]


class JobWorker:
    def __init__(
        self,
        repos: Repositories,
        handlers: dict[str, Handler],
        logger: Optional[logging.Logger] = None,
        poll_interval: float = 0.5,
        lease_seconds: int = 300,
        owner_id: Optional[str] = None,
    ) -> None:
        self.repos = repos
        self.job_repo: JobRepository = repos.job
        self.handlers = handlers
        self.logger = logger or logging.getLogger("companion.jobs")
        self.poll_interval = poll_interval
        self.lease_seconds = lease_seconds
        # Each worker instance has a stable owner id. A claim records it, and a
        # lease guards against another worker (or a restarted process) stealing
        # an in-flight job while this one is still alive.
        self.owner_id = owner_id or f"worker-{uuid.uuid4().hex[:12]}"
        # Set by TaskService after construction. When absent, the worker still
        # drains background jobs exactly as before V0.9.
        self.task_service = None
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def enqueue(
        self, job_type: JobType, job_key: str, payload: dict, max_attempts: int = 3
    ) -> str:
        return await self.job_repo.enqueue(job_type, job_key, payload, max_attempts)

    async def recover(self) -> int:
        """Reset leftover RUNNING jobs so they are retried after a restart."""
        return await self.job_repo.recover_running()

    async def run_once(self) -> int:
        """Process every currently-PENDING background job once."""
        processed = 0
        for job in await self.job_repo.pending():
            await self._process(job)
            processed += 1
        return processed

    async def run_until_idle(self, max_iterations: int = 20) -> int:
        """Drain the pending queue, honouring automatic retries."""
        total = 0
        for _ in range(max_iterations):
            pending = await self.job_repo.pending()
            if not pending:
                break
            total += await self.run_once()
        return total

    async def _process(self, job: ProcessingJob) -> None:
        if self.task_service is not None:
            allowed, reason = await self.task_service.can_claim(job)
            if not allowed:
                if reason in ("CANCELLED", "PERMISSION_REVOKED"):
                    await self.job_repo.mark_cancelled(job.job_id)
                    self.logger.info(
                        "job_cancelled job_id=%s reason=%s", job.job_id, reason
                    )
                elif reason == "FOREGROUND_ACTIVE":
                    # Do not even claim: claiming increments attempts. Delay a
                    # short idle window so a busy turn does not spin the poller.
                    resume = utcnow() + timedelta(
                        seconds=self.task_service.resume_idle_seconds
                    )
                    await self.job_repo.postpone(job.job_id, resume)
                # RESOURCE_BUSY: leave PENDING; the next pass tries again.
                return
        claimed = await self.job_repo.claim(
            job.job_id, self.owner_id, self.lease_seconds
        )
        if not claimed:
            # Another worker claimed it first (or attempts exhausted).
            return
        job_type_value = (
            job.job_type.value
            if isinstance(job.job_type, JobType)
            else str(job.job_type)
        )
        handler = self.handlers.get(job_type_value)
        t0 = time.monotonic()
        if handler is None:
            await self.job_repo.fail(job.job_id, f"no handler for {job_type_value}")
            self.logger.error(
                "job_no_handler job_id=%s job_type=%s", job.job_id, job_type_value
            )
            return
        claimed_job = await self.job_repo.get(job.job_id) or job
        context = TaskContext(
            task_id=job.job_id,
            mode=TaskMode(claimed_job.execution_mode),
            source=TaskSource(claimed_job.source_kind),
            checkpoint=dict(claimed_job.checkpoint or {}),
            service=self.task_service,
        )
        token = set_task_context(context)
        try:
            try:
                outcome = await handler(job.payload, self.repos, self.logger)
            except TaskPaused as paused:
                await self._pause(job.job_id, paused.checkpoint or context.checkpoint)
                return
            except Exception as exc:  # noqa: BLE001 — report and retry/fail.
                await self._handle_failure(job, exc, t0)
                return
            await self._finish(job, outcome, context, t0)
        finally:
            reset_task_context(token)

    async def _finish(
        self,
        job: ProcessingJob,
        outcome: Any,
        context: TaskContext,
        t0: float,
    ) -> None:
        if isinstance(outcome, TaskOutcome):
            status = str(outcome.status)
            if status == TaskStatus.PENDING:
                await self._pause(job.job_id, outcome.checkpoint or context.checkpoint)
                return
            if status == TaskStatus.CANCELLED:
                await self.job_repo.mark_cancelled(job.job_id)
                return
            if status == TaskStatus.FAILED:
                await self._handle_failure(
                    job, RuntimeError(outcome.error or "task failed"), t0
                )
                return
            if outcome.checkpoint:
                await self.job_repo.save_checkpoint(job.job_id, outcome.checkpoint)
            result = outcome.result
        else:
            result = outcome

        # Pre-commit checks: cancellation/permission may have changed while
        # the handler ran. Never mark revoked or cancelled work as completed.
        if self.task_service is not None:
            allowed, reason = await self.task_service.can_commit(job)
            if not allowed:
                await self.job_repo.mark_cancelled(job.job_id)
                self.logger.info(
                    "job_cancelled job_id=%s phase=pre_commit reason=%s",
                    job.job_id,
                    reason,
                )
                return
        elif await self.job_repo.cancellation_requested(job.job_id):
            await self.job_repo.mark_cancelled(job.job_id)
            self.logger.info("job_cancelled job_id=%s phase=pre_commit", job.job_id)
            return
        committed = await self.job_repo.complete_owned(
            job.job_id, self.owner_id, result
        )
        if not committed:
            self.logger.error(
                "job_commit_rejected job_id=%s reason=lease_or_cancel",
                job.job_id,
            )
            return
        self.logger.info(
            "job_completed job_id=%s job_type=%s attempts=%d duration_ms=%.2f",
            job.job_id,
            job.job_type.value
            if isinstance(job.job_type, JobType)
            else job.job_type,
            job.attempts + 1,
            (time.monotonic() - t0) * 1000,
        )

    async def _pause(self, job_id: str, checkpoint: dict) -> None:
        resume_at = (
            utcnow()
            + timedelta(
                seconds=self.task_service.resume_idle_seconds
                if self.task_service is not None
                else 0.5
            )
        )
        await self.job_repo.yield_for_foreground(
            job_id, checkpoint, resume_after=resume_at
        )
        self.logger.info(
            "job_paused job_id=%s reason=foreground attempts_not_consumed=true",
            job_id,
        )

    async def _handle_failure(
        self, job: ProcessingJob, exc: Exception, t0: float
    ) -> None:
        # Cancellation is terminal. A handler that raises after the user
        # cancelled must not be retried or reported as a plain failure.
        if await self.job_repo.cancellation_requested(job.job_id):
            await self.job_repo.mark_cancelled(job.job_id)
            self.logger.info("job_cancelled job_id=%s phase=failure", job.job_id)
            return
        attempts = job.attempts + 1
        if attempts >= job.max_attempts:
            await self.job_repo.fail(job.job_id, str(exc))
            self.logger.error(
                "job_failed job_id=%s job_type=%s attempts=%d error=%s",
                job.job_id,
                job.job_type.value
                if isinstance(job.job_type, JobType)
                else job.job_type,
                attempts,
                exc,
            )
        else:
            await self.job_repo.requeue(job.job_id, str(exc))
            self.logger.warning(
                "job_requeued job_id=%s job_type=%s attempts=%d error=%s",
                job.job_id,
                job.job_type.value
                if isinstance(job.job_type, JobType)
                else job.job_type,
                attempts,
                exc,
            )

    async def start(self) -> None:
        """Start a background loop (used by long-running sessions/demo)."""
        await self.recover()
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
