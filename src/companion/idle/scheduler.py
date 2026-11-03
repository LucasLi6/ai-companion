"""Idle scheduler: one full idle-learning cycle.

A cycle runs reflection → curiosity → research, in that order, and is gated by
the ``idle.enabled`` setting (off by default). The cycle is intentionally
small and deterministic so a single explicit ``idle run`` is cheap and safe.
"""

from __future__ import annotations

import logging
from typing import Optional


class IdleScheduler:
    def __init__(
        self,
        settings,
        reflection,
        curiosity,
        research,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.settings = settings
        self.reflection = reflection
        self.curiosity = curiosity
        self.research = research
        self.logger = logger or logging.getLogger("companion.idle")

    async def run_cycle(self) -> dict:
        """Run one cycle, or report a skip when idle learning is disabled."""
        if not await self.settings.get_bool("idle.enabled", False):
            self.logger.info("idle_cycle_skipped reason=disabled")
            return {"skipped": True, "reflections": 0, "questions": 0, "research": 0}

        reflections = await self.reflection.run()
        questions = await self.curiosity.run()
        research = await self.research.run_all(limit=3)
        return {
            "skipped": False,
            "reflections": len(reflections),
            "questions": len(questions),
            "research": len(research),
        }
