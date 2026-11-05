"""Curiosity engine: turn memory gaps into queued research questions.

Questions are derived only from externally-authorised memories (via the
context builder), and the same consent filter is re-applied immediately before
the provider call, so the text of a queued question can never leak a private
memory of a person who has not granted — or has just revoked — ``allow_external_ai``.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.domain.enums import TaskPriority
from companion.memory.service import MemoryService
from companion.learning.state import LearningRepository
from companion.storage.repositories import normalize_research_question

from .scheduling import schedule_research


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
        source_guard=None,
        settings=None,
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
        self.source_guard = source_guard
        self.settings = settings

    async def _setting_int(self, key: str, default: int) -> int:
        if self.settings is None:
            return default
        try:
            raw = await self.settings.get(key)
            return max(0, int(raw)) if raw is not None else default
        except (TypeError, ValueError):
            return default

    async def _proposals(self, memories: list[dict]) -> list[dict]:
        method = getattr(self.provider, "ask_question_proposals", None)
        if method is not None:
            raw = await method(memories)
        else:
            raw = await self.provider.ask_questions(memories)
        proposals: list[dict] = []
        for item in raw or []:
            if isinstance(item, str):
                proposals.append({"question": item, "source_memory_ids": []})
            elif isinstance(item, dict):
                proposals.append(item)
        return proposals

    async def run(self, limit: int = 50) -> list[str]:
        """Queue research questions derived from externally-authorised memories.

        Each question carries the exact memory ids it was derived from. Ids
        that are not in this cycle's authorised external-context set are
        dropped; if none remain the question is not persisted, because an
        autonomous question without verifiable sources must never reach a
        real network search.
        """
        memories = await self.context_builder.external_context(limit=limit)
        if self.source_guard is not None:
            # Second pass immediately before the provider call: a person may
            # have revoked ``allow_external_ai`` after the context was read.
            memories = await self.source_guard.filter_external_safe(memories)
        if not memories:
            # Nothing may leave the machine, so do not call the model at all.
            self.logger.info("curiosity_skipped reason=no_external_context")
            return []
        moment_ids = {
            str(m.get("memory_id")) for m in memories if m.get("memory_id")
        }
        if self.source_guard is not None:
            allowed_ids = await self.source_guard.allowed_question_sources(memories)
        else:
            allowed_ids = set(moment_ids)

        proposals = await self._proposals(memories)
        max_active = await self._setting_int("research.max_active", 10)
        recheck = await self._setting_int("research.recheck_seconds", 604800)
        ids: list[str] = []
        for proposal in proposals:
            question = str(proposal.get("question") or "").strip()
            if not question:
                continue
            raw_ids = proposal.get("source_memory_ids")
            proposed_ids = (
                [str(value) for value in raw_ids if value]
                if isinstance(raw_ids, (list, tuple))
                else []
            )
            source_ids = [
                item
                for item in dict.fromkeys(proposed_ids)
                if item in moment_ids and item in allowed_ids
            ]
            if not source_ids:
                self.logger.info(
                    "curiosity_question_skipped reason=no_verified_source"
                    " question_chars=%d",
                    len(question),
                )
                continue
            report = await schedule_research(
                self.research_repo.db,
                self.research_repo,
                self.task_service,
                question,
                origin="curiosity",
                source_kind="curiosity",
                source_memory_ids=source_ids,
                priority=0,
                max_active=max_active,
                recheck_seconds=recheck,
                task_priority=TaskPriority.LOW,
                log=self.logger,
            )
            task_id = report.get("task_id")
            if not task_id or not report.get("created"):
                continue
            item_id = await self.learning.enqueue_curiosity(
                question, source_kind="curiosity", source_ref=task_id
            )
            await self.learning.mark_curiosity(item_id, "QUEUED")
            ids.append(task_id)
            self.logger.info(
                "curiosity_question_queued research_id=%s curiosity_id=%s"
                " normalized_chars=%d sources=%d",
                task_id,
                item_id,
                len(normalize_research_question(question)),
                len(source_ids),
            )
        return ids

