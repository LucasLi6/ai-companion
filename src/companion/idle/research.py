"""Research planner & worker, plus knowledge consolidation.

A research task flows: ``PENDING -> RUNNING -> (search -> synthesize ->
consolidate) -> COMPLETED``, or ``RUNNING -> FAILED`` on a permanent error.
Claiming is atomic and lease-gated, so any number of workers may compete for a
task and at most one runs it. Consolidation is idempotent: the consolidated
memory id is derived deterministically from the task id, and the memory and the
task's ``COMPLETED`` state are committed in one transaction, so a crash can
never produce a duplicate memory.

Mock output never becomes trusted knowledge: consolidations enter as
``CANDIDATE`` with ``source_supported=False`` and ``UNVERIFIED`` evidence,
labelled with the search provider's name, and only a human can confirm them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
    VerificationLevel,
)
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
        *,
        settings=None,
        owner_id: str = "idle",
        lease_seconds: int = 300,
        search_limit: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.memory_service = memory_service
        self.research_repo = research_repo
        self.web_search = web_search
        self.deepseek = deepseek
        self.settings = settings
        self.owner_id = owner_id
        self.lease_seconds = lease_seconds
        self.search_limit = search_limit
        self.logger = logger or logging.getLogger("companion.idle.research")

    async def _enabled(self) -> bool:
        if self.settings is None:
            return True
        return await self.settings.get_bool("research.enabled", False)

    async def run_one(self, owner_id: str | None = None) -> dict | None:
        """Execute (claim) the highest-priority pending task; return final state.

        Returns ``None`` when research is disabled, the queue is empty, or the
        top task was claimed by another worker first.
        """
        owner = owner_id or self.owner_id
        if not await self._enabled():
            self.logger.info("research_skipped reason=disabled")
            return None
        pending = await self.research_repo.list(status="PENDING", limit=1)
        if not pending:
            return None
        task_id = pending[0]["task_id"]
        if not await self.research_repo.claim(task_id, owner, self.lease_seconds):
            task = await self.research_repo.get(task_id)
            if task and task["status"] == "PENDING" and task["attempts"] >= task["max_attempts"]:
                await self.research_repo.fail(task_id, "max attempts exceeded")
            return None
        task = await self.research_repo.get(task_id)
        try:
            memory_id = task["memory_id"]
            if memory_id:
                # A prior attempt already consolidated a memory but the task
                # was not marked COMPLETED (crash window). Reuse it, never
                # create a duplicate.
                existing = await self.memory_service.get_item(memory_id, SYSTEM_ADMIN)
                summary = existing.content if existing is not None else (task["result"] or "")
                await self.research_repo.complete(task_id, summary, memory_id=memory_id)
                self.logger.info(
                    "research_resumed research_id=%s memory_id=%s", task_id, memory_id
                )
            else:
                await self._run_new(task)
                self.logger.info("research_completed research_id=%s", task_id)
        except Exception as exc:  # noqa: BLE001 — retryable vs permanent.
            task = await self.research_repo.get(task_id)
            if task and task["attempts"] >= task["max_attempts"]:
                await self.research_repo.fail(task_id, str(exc))
                self.logger.error("research_failed research_id=%s error=%s", task_id, exc)
            else:
                await self.research_repo.requeue(task_id, str(exc))
                self.logger.warning("research_requeued research_id=%s error=%s", task_id, exc)
        return await self.research_repo.get(task_id)

    async def _run_new(self, task: dict) -> None:
        # Real network search must be both currently enabled and authorised.
        # A mock provider never hits the network; a real one refuses otherwise.
        if getattr(self.web_search, "name", "") == "http-web-search":
            if self.settings is None or not await self.settings.get_bool(
                "web_search.enabled", False
            ):
                raise RuntimeError(
                    "real web search is not authorised (web_search.enabled=false)"
                )
        results = await self.web_search.search(task["question"], limit=self.search_limit)
        summary = await self.deepseek.synthesize(
            task["question"], [asdict(r) for r in results]
        )
        await self._consolidate(task, summary, results)

    async def run_all(self, limit: int = 3, owner_id: str | None = None) -> list[dict]:
        """Run up to ``limit`` pending tasks, recovering expired leases first."""
        if not await self._enabled():
            return []
        await self.research_repo.recover_running()
        done: list[dict] = []
        for _ in range(limit):
            if not await self.research_repo.list(status="PENDING", limit=1):
                break
            result = await self.run_one(owner_id)
            if result is not None:
                done.append(result)
        return done

    async def _consolidate(self, task: dict, summary: str, results) -> None:
        memory_id = f"mem:{task['task_id']}"
        provider_name = getattr(self.web_search, "name", None)
        sources = []
        evidence = []
        for i, r in enumerate(results):
            source_id = f"web:{task['task_id']}:{i}"
            sources.append(
                MemorySource(
                    source_id=source_id,
                    source_type="web",
                    title=r.title,
                    url=r.url,
                    provider=provider_name,
                )
            )
            evidence.append(
                MemoryEvidence(source_id=source_id, verification_level=VerificationLevel.UNVERIFIED)
            )
        # One transaction: the memory and the task's COMPLETED state commit
        # together, so a crash cannot leave a memory without a completed task
        # (and vice versa).
        async with self.memory_service.repo.db.transaction():
            existing = await self.memory_service.get_item(memory_id, SYSTEM_ADMIN)
            if existing is None:
                item = await self.memory_service.create(
                    kind=MemoryKind.WORLD_KNOWLEDGE,
                    content=summary,
                    access=SYSTEM_ADMIN,
                    visibility=MemoryVisibility.PUBLIC,
                    status=MemoryStatus.CANDIDATE,
                    source_supported=False,
                    model_inferred=True,
                    sources=sources,
                    evidence=evidence,
                    memory_id=memory_id,
                )
            else:
                item = existing
            await self.research_repo.complete(task["task_id"], summary, memory_id=memory_id)
        self.logger.info(
            "research_consolidated memory_id=%s provider=%s",
            item.memory_id,
            provider_name or "unknown",
        )
