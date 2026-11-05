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

from companion.domain.enums import JobType
from companion.learning.state import LearningRepository
from companion.tasks.models import TaskRejected


async def _provider_label(provider) -> str | None:
    """Name the mode this provider would actually use right now.

    A settings-gated adapter keeps its class name (``gated-deepseek-idle``)
    regardless of mode, so the run record must ask ``describe()`` for the
    effective provider name instead of reading ``.name`` — otherwise a mock
    cycle would be recorded under a real-looking name.
    """
    describe = getattr(provider, "describe", None)
    if callable(describe):
        try:
            _, name = await describe()
            return str(name)
        except Exception:
            pass
    return getattr(provider, "name", None)


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
        task_service=None,
    ) -> None:
        self.settings = settings
        self.reflection = reflection
        self.curiosity = curiosity
        self.research = research
        self.logger = logger or logging.getLogger("companion.idle")
        self.learning = learning
        # With the unified task service wired, this scheduler *proposes*
        # research work by enqueueing it. It never executes research in the
        # dialogue process, so it cannot race the JobWorker.
        self.task_service = task_service

    async def _int_setting(self, key: str, default: int) -> int:
        try:
            raw = await self.settings.get(key)
            return max(0, int(raw)) if raw is not None else default
        except (TypeError, ValueError):
            return default

    async def _research_phase(self, limit: int) -> list:
        """Schedule research; execute it only on the legacy direct-engine path.

        With a task service present, the scheduler's job is to make every
        PENDING research detail row durable as a ``RESEARCH_TASK`` job. The
        JobWorker is the only executor. Existing PENDING rows (including ones
        created before this change) are taken over idempotently through the
        stable ``research:<task_id>`` job key.

        When the active search provider would make a real network request,
        each row's provenance is checked before a job is scheduled. A legacy
        row without source ids, or one whose source consent was revoked, is
        never handed to the network: it is failed/cancelled with a truthful
        reason instead of being silently retried forever.
        """
        task_service = self.task_service
        if task_service is None or (
            JobType.RESEARCH_TASK.value not in task_service.supported_types
        ):
            return await self.research.run_all(limit=limit)
        if not await self.settings.get_bool("research.enabled", False):
            return []
        pending = await self.research.research_repo.list(
            status="PENDING", limit=limit
        )
        network_required = False
        guard = getattr(self.research, "source_guard", None)
        report = getattr(self.research, "provider_report", None)
        if callable(report):
            try:
                network_required = (
                    (await report()).get("search_mode") == "real"
                )
            except Exception:
                network_required = True
        scheduled: list[str] = []
        for task in pending:
            task_id = str(task["task_id"])
            if network_required and guard is not None:
                decision = await guard.check(task, network_required=True)
                if not decision.allowed:
                    if decision.reason == "permission_revoked":
                        await self.research.research_repo.cancel_for_job(
                            task_id, ""
                        )
                    else:
                        await self.research.research_repo.fail(
                            task_id, f"source_blocked:{decision.reason}"
                        )
                    self.logger.info(
                        "research_not_scheduled research_id=%s reason=%s",
                        task_id,
                        decision.reason,
                    )
                    continue
            try:
                from .scheduling import enqueue_existing_research

                await enqueue_existing_research(task_service, task_id)
            except TaskRejected as exc:
                # No silent drop: the row stays PENDING and the reason is
                # recorded for the cycle report.
                self.logger.info(
                    "research_not_scheduled research_id=%s reason=%s",
                    task_id,
                    exc.reason,
                )
                break
            scheduled.append(task_id)
        return scheduled

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
        provider_name = await _provider_label(
            getattr(self.reflection, "provider", None)
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
            research = await self._research_phase(budget["research_limit"])
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
