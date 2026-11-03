"""Curiosity engine: turn memory gaps into queued research questions."""

from __future__ import annotations

import logging
from typing import Optional

from companion.memory.service import MemoryService

from . import SYSTEM_ADMIN


class CuriosityEngine:
    def __init__(self, memory_service: MemoryService, provider, research_repo, logger: Optional[logging.Logger] = None) -> None:
        self.memory_service = memory_service
        self.provider = provider
        self.research_repo = research_repo
        self.logger = logger or logging.getLogger("companion.idle.curiosity")

    async def run(self, limit: int = 50) -> list[str]:
        """Queue research questions derived from active memories."""
        memories = await self.memory_service.search(
            access=SYSTEM_ADMIN, statuses=["ACTIVE"], limit=limit
        )
        questions = await self.provider.ask_questions(memories)
        ids = []
        for question in questions:
            task_id = await self.research_repo.create(question, source_kind="curiosity")
            ids.append(task_id)
            self.logger.info("curiosity_question_queued research_id=%s", task_id)
        return ids
