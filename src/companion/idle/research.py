"""Research planner & worker, plus knowledge consolidation.

A research task flows: PENDING → RUNNING → (search → synthesize → consolidate)
→ COMPLETED, or → FAILED with the error preserved. Consolidation writes the
synthesised answer into the unified store as a ``WORLD_KNOWLEDGE`` memory with
its web sources attached as evidence — ``model_inferred`` and never
human-confirmed, so a human still reviews it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from companion.domain.enums import MemoryKind, MemoryVisibility, VerificationLevel
from companion.memory.models import MemoryEvidence, MemorySource
from companion.memory.service import MemoryService

from . import SYSTEM_ADMIN


class ResearchEngine:
    def __init__(
        self,
        memory_service: MemoryService,
        research_repo,
        web_search,
        deepseek,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.memory_service = memory_service
        self.research_repo = research_repo
        self.web_search = web_search
        self.deepseek = deepseek
        self.logger = logger or logging.getLogger("companion.idle.research")

    async def run_one(self) -> dict | None:
        """Execute the highest-priority pending task; return its final state."""
        pending = await self.research_repo.list(status="PENDING", limit=1)
        if not pending:
            return None
        task = pending[0]
        task_id = task["task_id"]
        await self.research_repo.set_status(task_id, "RUNNING")
        try:
            results = await self.web_search.search(task["question"])
            summary = await self.deepseek.synthesize(
                task["question"], [asdict(r) for r in results]
            )
            await self._consolidate(task, summary, results)
            await self.research_repo.complete(task_id, summary)
            self.logger.info("research_completed research_id=%s", task_id)
        except Exception as exc:
            await self.research_repo.set_status(task_id, "FAILED", error=str(exc))
            self.logger.error("research_failed research_id=%s error=%s", task_id, exc)
        return await self.research_repo.get(task_id)

    async def run_all(self, limit: int = 3) -> list[dict]:
        """Run up to ``limit`` pending tasks, stopping when the queue is empty."""
        done: list[dict] = []
        for _ in range(limit):
            task = await self.run_one()
            if task is None:
                break
            done.append(task)
        return done

    async def _consolidate(self, task: dict, summary: str, results) -> None:
        sources = []
        evidence = []
        for i, r in enumerate(results):
            source_id = f"web:{task['task_id']}:{i}"
            sources.append(
                MemorySource(source_id=source_id, source_type="web", title=r.title, url=r.url)
            )
            evidence.append(
                MemoryEvidence(
                    source_id=source_id, verification_level=VerificationLevel.SINGLE_SOURCE
                )
            )
        item = await self.memory_service.create(
            kind=MemoryKind.WORLD_KNOWLEDGE,
            content=summary,
            access=SYSTEM_ADMIN,
            visibility=MemoryVisibility.PUBLIC,
            source_supported=True,
            model_inferred=True,
            sources=sources,
            evidence=evidence,
        )
        self.logger.info("research_consolidated memory_id=%s", item.memory_id)
