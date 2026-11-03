"""Reflection engine: propose reflections over recent memories.

Reflections are always *proposals* — they enter the unified store as
``CANDIDATE`` (or ``NEEDS_REVIEW`` if they conflict with a human-confirmed
fact) and never overwrite anything. Before any memory reaches the provider it
passes through :class:`~companion.idle.context.IdleMemoryContextBuilder`, so a
future real provider sees only externally-authorised context; the mock provider
sees exactly the same filtered input.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.memory.service import MemoryService
from companion.learning.state import LearningRepository

from . import IDLE_AGENT, SYSTEM_ADMIN


class ReflectionEngine:
    def __init__(
        self,
        memory_service: MemoryService,
        provider,
        context_builder,
        logger: Optional[logging.Logger] = None,
        *,
        learning: LearningRepository | None = None,
    ) -> None:
        self.memory_service = memory_service
        self.provider = provider
        self.context_builder = context_builder
        self.logger = logger or logging.getLogger("companion.idle.reflection")
        self.learning = learning or LearningRepository(memory_service.repo.db)

    async def run(self, limit: int = 50) -> list:
        """Reflect over externally-authorised memories and record proposals."""
        memories = await self.context_builder.external_context(limit=limit)
        proposals = await self.provider.reflect(memories)
        results = []
        for p in proposals:
            item = await self.memory_service.propose(
                access=IDLE_AGENT,
                content=p["content"],
                kind=p.get("kind", "REFLECTION"),
                conflicts_with=p.get("conflicts_with"),
                source_ids=[m["memory_id"] for m in memories],
                source_access=SYSTEM_ADMIN,
            )
            await self.learning.record_reflection(
                p["content"],
                subject_memory_id=memories[0]["memory_id"] if memories else None,
                status="PROPOSED",
                source_access="SYSTEM_ADMIN",
            )
            results.append(item)
            self.logger.info("reflection_proposed memory_id=%s", item.memory_id)
        return results
