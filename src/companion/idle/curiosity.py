"""Curiosity engine: turn memory gaps into queued research questions.

Questions are derived only from externally-authorised memories (via the
context builder), so the text of a queued question can never leak a private
memory of a person who has not granted ``allow_external_ai``.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.domain.enums import JobType, TaskMode, TaskPriority, TaskSource
from companion.memory.service import MemoryService
from companion.learning.state import LearningRepository
from companion.tasks.models import TaskRejected, TaskRequest

from . import IDLE_AGENT


class CuriosityEngine:
    def __init__(
        self,
        memory_service: MemoryService,
        provider,
        research_repo,
        context_builder,
        logger: Optional[logging.Logger] = None,
        *,
        learning: LearningRepository | None = None,
        task_service=None,
    ) -> None:
        self.memory_service = memory_service
        self.provider = provider
        self.research_repo = research_repo
        self.context_builder = context_builder
        self.logger = logger or logging.getLogger("companion.idle.curiosity")
        self.learning = learning or LearningRepository(research_repo.db)
        # When the unified task service is wired, a research need is made
        # durable immediately. The detail row remains the research record;
        # the job is the execution authority.
        self.task_service = task_service

    async def run(self, limit: int = 50) -> list[str]:
        """Queue research questions derived from externally-authorised memories."""
        memories = await self.context_builder.external_context(limit=limit)
        questions = await self.provider.ask_questions(memories)
        ids = []
        for question in questions:
            task_id = await self.research_repo.create(question, source_kind="curiosity")
            await self._enqueue_research(task_id)
            item_id = await self.learning.enqueue_curiosity(
                question, source_kind="curiosity", source_ref=task_id
            )
            # The task is already created, so the queue item is no longer
            # pending; ``QUEUED`` truthfully describes its state.
            await self.learning.mark_curiosity(item_id, "QUEUED")
            ids.append(task_id)
            self.logger.info(
                "curiosity_question_queued research_id=%s curiosity_id=%s",
                task_id,
                item_id,
            )
        return ids

    async def _enqueue_research(self, task_id: str) -> None:
        """Hand the research need to the unified task service.

        A rejected enqueue (for example, a full autonomous queue) is recorded
        rather than silently swallowed: the detail row stays PENDING, so the
        idle scheduler can schedule it in a later cycle.
        """
        if self.task_service is None:
            return
        try:
            await self.task_service.enqueue(
                TaskRequest(
                    type=JobType.RESEARCH_TASK.value,
                    payload={"research_task_id": task_id},
                    mode=TaskMode.BACKGROUND,
                    source=TaskSource.SYSTEM,
                    priority=TaskPriority.LOW,
                    source_ref=task_id,
                    job_key=f"research:{task_id}",
                )
            )
        except TaskRejected as exc:
            self.logger.info(
                "curiosity_research_not_queued research_id=%s reason=%s",
                task_id,
                exc.reason,
            )
