"""A durable background job processor.

Jobs live in SQLite (``processing_jobs``), not in an ``asyncio.Queue``, so a
restart can recover them. On startup, leftover ``RUNNING`` jobs are reset to
``PENDING``; idempotent handlers ensure this never duplicates side effects.
Claiming is atomic and conditional (``UPDATE … WHERE status='PENDING'`` with a
``rowcount`` check), so multiple workers — even on separate connections — can
compete for jobs safely and at most one runs each handler. V0.0 deliberately
runs one worker by default and no complex concurrency pool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from companion.domain.enums import JobType
from companion.storage.repositories import JobRepository, Repositories

Handler = Callable[[dict, Repositories, logging.Logger], Awaitable[None]]


class JobWorker:
    def __init__(
        self,
        repos: Repositories,
        handlers: dict[str, Handler],
        logger: Optional[logging.Logger] = None,
        poll_interval: float = 0.5,
    ) -> None:
        self.repos = repos
        self.job_repo: JobRepository = repos.job
        self.handlers = handlers
        self.logger = logger or logging.getLogger("companion.jobs")
        self.poll_interval = poll_interval
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
        """Process every currently-PENDING job once."""
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

    async def _process(self, job) -> None:
        claimed = await self.job_repo.claim(job.job_id)
        if not claimed:
            # Another worker claimed it first (or attempts exhausted).
            return
        handler = self.handlers.get(job.job_type.value)
        t0 = time.monotonic()
        if handler is None:
            await self.job_repo.fail(job.job_id, f"no handler for {job.job_type.value}")
            self.logger.error(
                "job_no_handler job_id=%s job_type=%s", job.job_id, job.job_type.value
            )
            return
        try:
            await handler(job.payload, self.repos, self.logger)
            await self.job_repo.complete(job.job_id)
            self.logger.info(
                "job_completed job_id=%s job_type=%s attempts=%d duration_ms=%.2f",
                job.job_id,
                job.job_type.value,
                job.attempts + 1,
                (time.monotonic() - t0) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 — report and retry/fail.
            attempts = job.attempts + 1
            if attempts >= job.max_attempts:
                await self.job_repo.fail(job.job_id, str(exc))
                self.logger.error(
                    "job_failed job_id=%s job_type=%s attempts=%d error=%s",
                    job.job_id,
                    job.job_type.value,
                    attempts,
                    exc,
                )
            else:
                await self.job_repo.requeue(job.job_id, str(exc))
                self.logger.warning(
                    "job_requeued job_id=%s job_type=%s attempts=%d error=%s",
                    job.job_id,
                    job.job_type.value,
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
