"""V0.9 unified off-task execution package."""

from companion.domain.enums import TaskMode, TaskPriority, TaskSource
from companion.tasks.context import TaskContext, current_task_context
from companion.tasks.foreground import ForegroundActivity
from companion.tasks.models import (
    TaskOutcome,
    TaskPaused,
    TaskRejected,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from companion.tasks.service import TaskHandler, TaskService

__all__ = [
    "ForegroundActivity",
    "TaskContext",
    "TaskHandler",
    "TaskMode",
    "TaskOutcome",
    "TaskPaused",
    "TaskPriority",
    "TaskRejected",
    "TaskRequest",
    "TaskResult",
    "TaskService",
    "TaskSource",
    "TaskStatus",
    "current_task_context",
]
