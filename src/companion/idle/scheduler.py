"""Idle scheduler: one full idle-learning cycle.

A cycle runs reflection → curiosity → research, in that order, and is gated by
the ``idle.enabled`` setting (off by default). Limits are read from settings so
an operator can bound how much a single cycle does; a learning run records the
budget it was given and a checkpoint after each stage, so an interrupted cycle
can be inspected instead of disappearing.

The run record is bookkeeping only: resuming still requires the same gates
(``idle.enabled``, ``research.enabled``, ``web_search.enabled``) as a fresh
cycle, and no checkpoint grants permission.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.learning.state import LearningRepository


class IdleScheduler:
    def __init__(
        self,
        settings,
        reflection,
        curiosity,
        research,
        logger: Optional[logging.Logger] = None,
        *,
        learning: LearningRepository | None = None,
    ) -> None:
        self.settings = settings
        self.reflection = reflection
        self.curiosity = curiosity
        self.research = research
        self.logger = logger or logging.getLogger("companion.idle")
        self.learning = learning

    async def _int_setting(self, key: str, default: int) -> int:
        try:
            raw = await self.settings.get(key)
            return max(0, int(raw)) if raw is not None else default
        except (TypeError, ValueError):
            return default

    async def run_cycle(self) -> dict:
        """Run one cycle, or report a skip when idle learning is disabled."""
        if not await self.settings.get_bool("idle.enabled", False):
            self.logger.info("idle_cycle_skipped reason=disabled")
            return {"skipped": True, "reflections": 0, "questions": 0, "research": 0}

        budget = {
            "reflection_limit": await self._int_setting("idle.reflection_limit", 50),
            "curiosity_limit": await self._int_setting("idle.curiosity_limit", 50),
            "research_limit": await self._int_setting("idle.research_limit", 3),
        }
        provider_name = getattr(
            getattr(self.reflection, "provider", None), "name", None
        )
        run_id = None
        if self.learning is not None:
            run_id = await self.learning.start_run(
                "idle_cycle", provider=provider_name, budget=budget
            )
        try:
            reflections = await self.reflection.run(
                limit=budget["reflection_limit"]
            )
            if run_id is not None:
                await self.learning.checkpoint(
                    run_id, "reflection", {"count": len(reflections)}
                )
            questions = await self.curiosity.run(limit=budget["curiosity_limit"])
            if run_id is not None:
                await self.learning.checkpoint(
                    run_id, "curiosity", {"count": len(questions)}
                )
            research = await self.research.run_all(limit=budget["research_limit"])
            if run_id is not None:
                await self.learning.checkpoint(
                    run_id, "research", {"count": len(research)}
                )
                await self.learning.finish_run(run_id, "COMPLETED")
        except Exception as exc:
            if run_id is not None:
                await self.learning.finish_run(run_id, "FAILED", error=str(exc)[:1000])
            raise
        return {
            "skipped": False,
            "reflections": len(reflections),
            "questions": len(questions),
            "research": len(research),
        }
