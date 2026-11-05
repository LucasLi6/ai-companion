"""Minimal models for the V0.9 off-task execution layer.

The task layer is deliberately small: a request, a result, and a cooperative
checkpoint signal. It is not a planner or a workflow engine. A task is either

- ``FOREGROUND`` — the current turn waits for it (explicit memory write, a
  required web search), executed through ``TaskService.execute_now``; or
- ``BACKGROUND`` — a durable SQLite row processed later by the existing
  ``JobWorker`` (never blocking the speaker).

Source identity is local metadata. A model may *propose* a task, but only local
code chooses ``source`` and ``mode``; no field in this module is model-granted
permission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

from companion.domain.enums import (
    JobStatus,
    JobType,
    TaskMode,
    TaskPriority,
    TaskSource,
)

__all__ = [
    "TaskMode",
    "TaskPriority",
    "TaskSource",
    "TaskStatus",
    "TaskRequest",
    "TaskResult",
    "TaskOutcome",
    "TaskPaused",
    "TaskRejected",
    "TaskLeaseLost",
    "TaskCommitRejected",
    "TaskStatus",
]


class TaskStatus(str):
    """String status compatible with both ``JobStatus`` and ``TaskStatus``.

    Kept as a tiny ``str`` subclass rather than a second enum so persisted
    statuses and API results compare cleanly (``result.status == "COMPLETED"``)
    without a conversion table. Use :class:`companion.domain.enums.JobStatus`
    where an enum is required.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def __new__(cls, value: str) -> "TaskStatus":
        return str.__new__(cls, value)

    @classmethod
    def from_job(cls, status: JobStatus | str) -> "TaskStatus":
        value = status.value if isinstance(status, JobStatus) else str(status)
        return cls(value)


@dataclass(frozen=True)
class TaskRequest:
    """One request to execute work off the speaker path.

    ``type`` is the string name of a registered task handler. ``task_type`` is
    accepted as an alias for callers that prefer an explicit field name.
    """

    type: str = ""
    payload: dict = field(default_factory=dict)
    mode: TaskMode | str = TaskMode.BACKGROUND
    source: TaskSource | str = TaskSource.SYSTEM
    priority: TaskPriority | str | None = None
    source_ref: Optional[str] = None
    source_event_id: Optional[str] = None
    job_key: Optional[str] = None
    max_attempts: int = 3
    next_run_at: Optional[datetime] = None
    checkpoint: dict = field(default_factory=dict)
    # Alias accepted by older call sites / proposals.
    task_type: Optional[str] = None

    def __post_init__(self) -> None:
        raw_type = self.type or self.task_type or ""
        if isinstance(raw_type, JobType):
            raw_type = raw_type.value
        resolved_type = str(raw_type).strip()
        if not resolved_type:
            raise ValueError("TaskRequest requires a non-empty type")
        raw_alias = self.task_type
        if isinstance(raw_alias, JobType):
            raw_alias = raw_alias.value
        if raw_alias and self.type and str(raw_alias) != resolved_type:
            raise ValueError("TaskRequest.type and task_type disagree")
        object.__setattr__(self, "type", resolved_type)
        object.__setattr__(self, "mode", _mode(self.mode))
        object.__setattr__(self, "source", _source(self.source))
        if self.priority is not None:
            object.__setattr__(self, "priority", _priority(self.priority))
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(self, "checkpoint", dict(self.checkpoint or {}))

    @property
    def task_type_value(self) -> str:
        return self.type

    @property
    def resolved_mode(self) -> TaskMode:
        return _mode(self.mode)

    @property
    def resolved_source(self) -> TaskSource:
        return _source(self.source)

    @property
    def resolved_priority(self) -> TaskPriority:
        if self.priority is not None:
            return _priority(self.priority)
        if self.resolved_mode is TaskMode.FOREGROUND:
            return TaskPriority.HIGH
        if self.resolved_source is TaskSource.USER:
            return TaskPriority.HIGH
        if self.resolved_source is TaskSource.ROUTER:
            return TaskPriority.NORMAL
        return TaskPriority.LOW

    @property
    def stable_key(self) -> str:
        """Idempotency key chosen locally, never model-declared.

        A caller-supplied key wins. Otherwise identity is the source event /
        source ref when present, falling back to a canonical payload hash.
        """
        if self.job_key:
            return self.job_key
        identity = self.source_event_id or self.source_ref
        if not identity:
            canonical = json.dumps(
                self.payload, ensure_ascii=False, sort_keys=True, default=str
            )
            identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
        return f"task:{self.type}:{identity}"

    def with_mode(self, mode: TaskMode | str) -> "TaskRequest":
        return TaskRequest(
            type=self.type,
            payload=self.payload,
            mode=mode,
            source=self.source,
            priority=self.priority,
            source_ref=self.source_ref,
            source_event_id=self.source_event_id,
            job_key=self.job_key,
            max_attempts=self.max_attempts,
            next_run_at=self.next_run_at,
            checkpoint=self.checkpoint,
        )


@dataclass
class TaskResult:
    """Truthful result of a foreground or background task.

    ``result`` is present only when the work actually completed. A failure has
    ``status=FAILED`` and an error; a cooperative pause is reported as
    ``PENDING`` with a checkpoint and is never presented as success.
    """

    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    checkpoint: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return str(self.status) == TaskStatus.COMPLETED

    @property
    def terminal(self) -> bool:
        return str(self.status) in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": str(self.status),
            "result": self.result,
            "error": self.error,
            "checkpoint": self.checkpoint,
        }


@dataclass
class TaskOutcome:
    """What a handler may return when it wants more than plain success.

    - ``COMPLETED`` (default): persist ``result`` and mark the job complete.
    - ``PENDING`` (with checkpoint): cooperative pause; no retry consumed.
    - ``CANCELLED``: cancellation observed at a safe checkpoint.
    - ``FAILED``: business failure with a truthful error.
    """

    status: TaskStatus | str = TaskStatus.COMPLETED
    result: Any = None
    error: Optional[str] = None
    checkpoint: Optional[dict] = None

    def __post_init__(self) -> None:
        self.status = TaskStatus(str(self.status))


class TaskPaused(Exception):
    """Raised by a handler to yield to foreground activity at a safe point."""

    def __init__(self, checkpoint: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__("task paused for foreground activity")
        self.checkpoint = dict(checkpoint or {})


class TaskRejected(Exception):
    """A local, pre-execution rejection (unsupported/permission/budget)."""

    def __init__(self, reason: str, *, detail: Optional[str] = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason


class TaskLeaseLost(Exception):
    """This execution no longer owns a live lease, so it must not write.

    Raised at a commit boundary when the durable row was reclaimed, completed
    or cancelled by someone else. The caller reports the loss; it never
    converts it into a retryable failure or a success.
    """

    def __init__(self, reason: str = "lease_lost") -> None:
        super().__init__(reason)
        self.reason = reason


class TaskCommitRejected(Exception):
    """A pre-commit local check (cancel/permission) refused the write."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail

    def as_dict(self) -> dict:
        return {"reason": self.reason, "detail": self.detail}


def _mode(value: TaskMode | str) -> TaskMode:
    if isinstance(value, TaskMode):
        return value
    try:
        return TaskMode(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"unknown task mode {value!r}") from exc


def _source(value: TaskSource | str) -> TaskSource:
    if isinstance(value, TaskSource):
        return value
    try:
        return TaskSource(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"unknown task source {value!r}") from exc


def _priority(value: TaskPriority | str) -> TaskPriority:
    if isinstance(value, TaskPriority):
        return value
    try:
        return TaskPriority(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"unknown task priority {value!r}") from exc
