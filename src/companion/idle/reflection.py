"""Reflection engine: propose reflections over recent memories.

Reflections are always *proposals* — they enter the unified store as
``CANDIDATE`` (or ``NEEDS_REVIEW`` if they conflict with a human-confirmed
fact) and never overwrite anything.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.memory.models import AccessContext
from companion.memory.service import MemoryService

from . import IDLE_AGENT, SYSTEM_ADMIN


class ReflectionEngine:
    def __init__(self, memory_service: MemoryService, provider, logger: Optional[logging.Logger] = None) -> None:
        self.memory_service = memory_service
        self.provider = provider
        self.logger = logger or logging.getLogger("companion.idle.reflection")

    async def run(self, limit: int = 50) -> list:
        """Reflect over active memories and record any resulting proposals."""
        memories = await self.memory_service.search(
            access=SYSTEM_ADMIN, statuses=["ACTIVE"], limit=limit
        )
        proposals = await self.provider.reflect(memories)
        results = []
        for p in proposals:
            item = await self.memory_service.propose(
                access=IDLE_AGENT,
                content=p["content"],
                kind=p.get("kind", "REFLECTION"),
                conflicts_with=p.get("conflicts_with"),
            )
            results.append(item)
            self.logger.info("reflection_proposed memory_id=%s", item.memory_id)
        return results
