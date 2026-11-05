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
from companion.tasks.models import TaskPaused

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

    async def save_checkpoint(self, update: dict) -> None:
        self.checkpoint.update(update or {})
        if self.service is not None:
            await self.service.repos.job.save_checkpoint(self.task_id, update)

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
