"""Per-execution task context for cooperative checkpoints.

The existing job handlers keep their ``(payload, repos, logger)`` signature.
Rather than changing every handler, a task execution publishes a context var;
a multi-stage handler may call ``current_task_context()`` to

- see its persisted checkpoint,
- save a new checkpoint immediately (before doing the next expensive step),
- ask whether a background task should yield to foreground activity, and
- observe a cancellation request.

Outside a task execution the context is ``None``, so old handlers and direct
engine tests are unaffected.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from companion.domain.enums import TaskMode, TaskSource
from companion.tasks.models import TaskLeaseLost, TaskPaused

if TYPE_CHECKING:  # pragma: no cover - typing only
    from companion.tasks.service import TaskService


_current: ContextVar[Optional["TaskContext"]] = ContextVar(
    "companion_task_context", default=None
)


@dataclass
class TaskContext:
    task_id: str
    mode: TaskMode
    source: TaskSource
    checkpoint: dict = field(default_factory=dict)
    service: Optional["TaskService"] = None
    # Durable owner token of the current execution. When present, every
    # checkpoint and commit is conditional on this execution still holding a
    # live lease on ``task_id``.
    owner_id: Optional[str] = None

    async def save_checkpoint(self, update: dict) -> None:
        self.checkpoint.update(update or {})
        if self.service is None:
            return
        saved = await self.service.repos.job.save_checkpoint(
            self.task_id, update, owner_id=self.owner_id
        )
        if self.owner_id is not None and not saved:
            # A stale execution must not keep staging outputs as if it still
            # owned the task. The caller aborts without further side effects.
            raise TaskLeaseLost("checkpoint_rejected")

    async def can_continue(self) -> bool:
        """Whether this execution still owns a live, permitted lease.

        Also renews the lease, so a long multi-stage task does not expire
        while it is making progress. Legacy direct executions (no service or
        no owner token) remain allowed, preserving the pre-V0.9 engine path.
        """
        if self.service is None or self.owner_id is None:
            return True
        return await self.service.authorize_execution(
            self.task_id, self.owner_id, renew=True
        )

    async def commit_atomic(self, action, *, result: Any = None) -> Any:
        """Run ``action`` and the job completion in one SQLite transaction.

        ``action`` must perform only the persistent side effects that belong
        to this task; the service re-validates cancellation, permission and
        lease before running it and rolls everything back if completion loses
        the lease.
        """
        if self.service is None or self.owner_id is None:
            raise TaskLeaseLost("no_owner")
        return await self.service.commit_atomic(
            self.task_id, self.owner_id, action, result=result
        )

    async def should_pause(self) -> bool:
        if self.mode is not TaskMode.BACKGROUND or self.service is None:
            return False
        return self.service.foreground.active

    async def cancellation_requested(self) -> bool:
        if self.service is None:
            return False
        return await self.service.repos.job.cancellation_requested(self.task_id)

    async def pause_if_needed(self) -> None:
        """Raise :class:`TaskPaused` at a safe stage boundary."""
        if await self.should_pause():
            raise TaskPaused(dict(self.checkpoint))

    def publish(self) -> Token:
        return _current.set(self)

    def reset(self, token: Token) -> None:
        _current.reset(token)


def current_task_context() -> Optional[TaskContext]:
    return _current.get()


def set_task_context(context: Optional[TaskContext]) -> Token:
    return _current.set(context)


def reset_task_context(token: Token) -> None:
    _current.reset(token)
