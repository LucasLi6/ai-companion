"""Curiosity engine: turn memory gaps into queued research questions.

Questions are derived only from externally-authorised memories (via the
context builder), so the text of a queued question can never leak a private
memory of a person who has not granted ``allow_external_ai``.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.memory.service import MemoryService

from . import IDLE_AGENT


class CuriosityEngine:
    def __init__(
        self,
        memory_service: MemoryService,
        provider,
        research_repo,
        context_builder,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.memory_service = memory_service
        self.provider = provider
        self.research_repo = research_repo
        self.context_builder = context_builder
        self.logger = logger or logging.getLogger("companion.idle.curiosity")

    async def run(self, limit: int = 50) -> list[str]:
        """Queue research questions derived from externally-authorised memories."""
        memories = await self.context_builder.external_context(limit=limit)
        questions = await self.provider.ask_questions(memories)
        ids = []
        for question in questions:
            task_id = await self.research_repo.create(question, source_kind="curiosity")
            ids.append(task_id)
            self.logger.info("curiosity_question_queued research_id=%s", task_id)
        return ids
