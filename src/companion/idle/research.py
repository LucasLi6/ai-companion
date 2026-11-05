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
from companion.research.state import ResearchStateRepository
from companion.tasks.models import (
    TaskCommitRejected,
    TaskLeaseLost,
    TaskPaused,
)

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
        state: ResearchStateRepository | None = None,
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
        self.state = state or ResearchStateRepository(research_repo.db)
        self.logger = logger or logging.getLogger("companion.idle.research")

    async def _setting_int(self, key: str, default: int) -> int:
        if self.settings is None:
            return default
        try:
            raw = await self.settings.get(key)
            return int(raw) if raw is not None else default
        except (TypeError, ValueError):
            return default

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
                # Repair the durable run record if the crash happened after
                # consolidation but before the run was marked completed.
                run_id = task.get("run_id")
                if run_id:
                    run = await self.state.get_run(run_id)
                    if run is not None and run["status"] == "RUNNING":
                        await self.state.finish_run(
                            run_id,
                            "COMPLETED",
                            budget_used={"resumed": True, "memory_reused": True},
                        )
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

        search_limit = max(1, await self._setting_int("research.search_limit", self.search_limit))
        excerpt_chars = max(80, await self._setting_int("research.excerpt_chars", 500))
        refresh_seconds = max(60, await self._setting_int(
            "research.source_refresh_seconds", 86400
        ))
        web_name = getattr(self.web_search, "name", None) or "unknown-search"
        synth_name = getattr(self.deepseek, "name", None) or "unknown-synthesizer"
        provider_name = f"{web_name}+{synth_name}"

        # A prior attempt that crashed mid-run leaves its run RUNNING. Record
        # it as abandoned so the new attempt is the only live run.
        await self.state.supersede_running(task["task_id"])
        project_id = await self.state.ensure_project(
            task["question"], task_id=task["task_id"]
        )
        run_id = await self.state.start_run(
            project_id,
            task_id=task["task_id"],
            provider=provider_name,
            budget={"search_limit": search_limit, "excerpt_chars": excerpt_chars},
        )
        try:
            results = await self.web_search.search(
                task["question"], limit=search_limit
            )
            await self.state.record_query(
                run_id,
                task["question"],
                provider=web_name,
                result_count=len(results),
            )
            sources, excerpts = await self._persist_sources(
                task, results, run_id, excerpt_chars, refresh_seconds
            )
            summary = await self.deepseek.synthesize(
                task["question"], [asdict(r) for r in results]
            )
            claim_id = f"claim:{task['task_id']}"
            await self.state.record_claim(
                summary,
                project_id=project_id,
                status="PROPOSED",
                confidence=0.0,
                verification_level="UNVERIFIED",
                claim_id=claim_id,
            )
            for source, excerpt_id in zip(sources, excerpts):
                await self.state.link_evidence(
                    claim_id, source.source_id, excerpt_id=excerpt_id or ""
                )
            await self.state.record_finding(
                project_id,
                summary,
                run_id=run_id,
                status="DRAFT",
                finding_id=f"find:{task['task_id']}",
            )
            await self._consolidate(task, summary, results, sources=sources)
            await self.state.finish_run(
                run_id,
                "COMPLETED",
                budget_used={
                    "search_limit": search_limit,
                    "results": len(results),
                    "sources": len(sources),
                },
            )
        except Exception as exc:  # noqa: BLE001 — recorded, then retried/failed.
            await self.state.finish_run(run_id, "FAILED", error=str(exc)[:1000])
            raise

    async def _persist_sources(
        self, task: dict, results, run_id: str, excerpt_chars: int, refresh_seconds: int
    ) -> tuple[list[MemorySource], list[str]]:
        """Record durable sources + bounded excerpts, returning memory sources.

        A search hit is stored as an UNVERIFIED source candidate. Its snippet
        is quoted (bounded) material, never an instruction.
        """
        provider_name = getattr(self.web_search, "name", None)
        sources: list[MemorySource] = []
        excerpt_ids: list[str] = []
        for i, r in enumerate(results):
            source_id = f"web:{task['task_id']}:{i}"
            await self.state.record_source(
                source_id=source_id,
                url=r.url,
                title=r.title,
                publisher=None,
                source_kind="web",
                trust_level="UNVERIFIED",
                extra={
                    "task_id": task["task_id"],
                    "run_id": run_id,
                    "provider": provider_name,
                },
            )
            excerpt_id = await self.state.record_excerpt(
                source_id,
                r.snippet or "",
                run_id=run_id,
                max_chars=excerpt_chars,
                excerpt_id=f"{source_id}:excerpt",
            )
            await self.state.create_refresh_rule(
                source_id,
                interval_seconds=refresh_seconds,
                rule_id=f"refresh:{source_id}",
            )
            excerpt_ids.append(excerpt_id)
            sources.append(
                MemorySource(
                    source_id=source_id,
                    source_type="web",
                    title=r.title,
                    url=r.url,
                    provider=provider_name,
                )
            )
        return sources, excerpt_ids

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

    def _owner_token(self, task_id: str, owner_id: str | None) -> str:
        return owner_id or f"{self.owner_id}:{task_id}"

    async def run_resumable(
        self,
        task_id: str,
        *,
        checkpoint: dict | None = None,
        pause_check=None,
        save_checkpoint=None,
        owner_id: str | None = None,
        authorize=None,
        commit=None,
    ) -> dict:
        """Execute one research task with durable stages.

        Stages are ``SEARCH -> ANALYZE -> SAVE -> DONE``. A checkpoint stores
        search results by value (bounded snippets) and the durable run/source
        references needed to resume, never a claim that permission or a lease
        is still valid.

        When invoked through ``TaskService`` the caller supplies

        - ``owner_id``: a stable token such as ``job:<job_id>`` so the research
          detail row and the job name the same execution;
        - ``authorize``: an async predicate over the *job* lease/cancel/
          permission state, checked before each stage;
        - ``commit``: an atomic-commit callback that runs the memory write,
          the research detail update and the job completion in one SQLite
          transaction.

        A task is never run merely because its detail row says RUNNING: a live
        lease owned by a different execution rejects this one, and an expired
        lease is reclaimed through the normal claim path.
        """
        import inspect

        state = dict(checkpoint or {})
        if not await self._enabled():
            return {"status": "CANCELLED", "reason": "research_disabled"}
        if authorize is not None and not await authorize():
            return {"status": "CANCELLED", "reason": "authorization_lost"}
        task = await self.research_repo.get(task_id)
        if task is None:
            return {"status": "FAILED", "error": "research task not found"}
        owner = self._owner_token(task_id, owner_id)

        if task["status"] == "RUNNING":
            same_owner = task.get("owner_id") == owner
            live = await self.research_repo.lease_live(task_id)
            if not same_owner and live:
                # A different execution still holds a valid lease. Do not
                # race it, and do not treat its RUNNING row as our own.
                return {"status": "PENDING", "reason": "claimed_elsewhere"}
            if not live:
                await self.research_repo.recover_task(task_id)
            task = await self.research_repo.get(task_id)
            if task is None:
                return {"status": "FAILED", "error": "research task not found"}

        if task["status"] == "COMPLETED":
            await self._repair_completed_run(task)
            return self._completed_payload(task)
        if task["status"] == "FAILED":
            return {
                "status": "FAILED",
                "error": task.get("error") or "research task failed",
                "memory_id": task.get("memory_id"),
            }

        if task["status"] == "PENDING":
            if not await self.research_repo.claim(
                task_id, owner, self.lease_seconds
            ):
                current = await self.research_repo.get(task_id)
                if current and current["status"] == "RUNNING":
                    return {"status": "PENDING", "reason": "claimed_elsewhere"}
                if current and current["status"] == "FAILED":
                    return {
                        "status": "FAILED",
                        "error": current.get("error") or "research task failed",
                    }
                return {"status": "PENDING", "reason": "claim_failed"}
            task = await self.research_repo.get(task_id)
        if task is None or task["status"] != "RUNNING":
            return {"status": "PENDING", "reason": "not_owned"}
        if task.get("owner_id") != owner:
            # Lost a claim race after recovery; never run another's task.
            return {"status": "PENDING", "reason": "claimed_elsewhere"}

        async def guard() -> None:
            """Abort without writes when this execution has no right to run."""
            if authorize is not None and not await authorize():
                raise TaskCommitRejected("CANCELLED")
            if not await self.research_repo.renew(
                task_id, owner, self.lease_seconds
            ):
                raise TaskLeaseLost("research_lease_lost")

        async def save(update: dict, *, final: bool = False) -> None:
            # ``final`` is used only by the legacy direct-engine path after a
            # successful consolidation: the detail row is already COMPLETED,
            # so a lease renewal there would fail even though the work ended
            # correctly. With an atomic commit the job is COMPLETED and no
            # final checkpoint is written at all.
            if not final:
                await guard()
            state.update(update)
            if save_checkpoint is not None:
                result = save_checkpoint(dict(state))
                if inspect.isawaitable(result):
                    await result

        async def pause_if_needed() -> None:
            await guard()
            if pause_check is None:
                return
            paused = pause_check()
            if inspect.isawaitable(paused):
                paused = await paused
            if paused:
                await save({"stage": state.get("stage", "START")})
                raise TaskPaused(dict(state))

        # A prior attempt may have consolidated the memory. Finalize the
        # detail row through the same atomic commit path so cancellation can
        # never leave a job COMPLETED with an abandoned research row.
        if task.get("memory_id"):
            try:
                await self._finalize_existing(task, owner, commit)
            except TaskCommitRejected as rejected:
                await self.research_repo.cancel_owned(task_id, owner)
                return {"status": "CANCELLED", "reason": rejected.reason}
            except TaskLeaseLost as lost:
                current = await self.research_repo.get(task_id)
                if current is not None and current["status"] == "COMPLETED":
                    return self._completed_payload(current)
                return {"status": "PENDING", "reason": lost.reason}
            return self._completed_payload(task)

        try:
            await self._resume_stages(
                task,
                state,
                save=save,
                pause_if_needed=pause_if_needed,
                owner_id=owner,
                commit=commit,
            )
        except TaskPaused:
            raise
        except TaskCommitRejected as rejected:
            await self.research_repo.cancel_owned(task_id, owner)
            return {"status": "CANCELLED", "reason": rejected.reason}
        except TaskLeaseLost as lost:
            current = await self.research_repo.get(task_id)
            if current is not None and current["status"] == "COMPLETED":
                return self._completed_payload(current)
            return {"status": "PENDING", "reason": lost.reason}
        except Exception as exc:  # noqa: BLE001 — retried by the job worker.
            current = await self.research_repo.get(task_id)
            if current is not None and current["status"] == "RUNNING":
                if current.get("owner_id") == owner:
                    if current["attempts"] >= current["max_attempts"]:
                        await self.research_repo.fail_owned(
                            task_id, owner, str(exc)
                        )
                    else:
                        await self.research_repo.requeue_owned(
                            task_id, owner, str(exc)
                        )
            raise
        task = await self.research_repo.get(task_id)
        if task is None:
            return {"status": "FAILED", "error": "research task not found"}
        if task["status"] == "COMPLETED":
            return self._completed_payload(task)
        if task["status"] == "CANCELLED":
            return {"status": "CANCELLED"}
        return {"status": "PENDING", "reason": "not finished"}

    async def cancel_detail(self, task_id: str, owner_id: str | None = None) -> bool:
        """Cancel a research detail row on behalf of its cancelled job.

        The owner token, when supplied, confines the RUNNING branch to this
        execution; a PENDING row is cancelled because the job that would have
        claimed it is already terminal.
        """
        if owner_id is not None and await self.research_repo.cancel_owned(
            task_id, owner_id
        ):
            return True
        return await self.research_repo.cancel_for_job(task_id, owner_id or "")

    async def _finalize_existing(self, task: dict, owner: str, commit) -> None:
        """Complete a task whose memory was already written by a prior attempt."""
        task_id = task["task_id"]
        memory_id = task.get("memory_id")
        summary = task.get("result") or ""
        payload = {
            "status": "COMPLETED",
            "research_task_id": task_id,
            "memory_id": memory_id,
            "result": summary,
        }

        async def action() -> None:
            if commit is not None:
                ok = await self.research_repo.complete_owned(
                    task_id, owner, summary, memory_id=memory_id
                )
                if not ok:
                    raise TaskLeaseLost("research_owner_lost")
            else:
                await self.research_repo.complete(
                    task_id, summary, memory_id=memory_id
                )

        if commit is not None:
            await commit(action, result=payload)
        else:
            await action()
        await self._repair_completed_run(task)

    async def _resume_stages(
        self,
        task: dict,
        state: dict,
        *,
        save,
        pause_if_needed,
        owner_id: str | None = None,
        commit=None,
    ) -> None:
        started_from_checkpoint = bool(state.get("stage"))
        if getattr(self.web_search, "name", "") == "http-web-search":
            if self.settings is None or not await self.settings.get_bool(
                "web_search.enabled", False
            ):
                raise RuntimeError(
                    "real web search is not authorised (web_search.enabled=false)"
                )

        search_limit = max(
            1, await self._setting_int("research.search_limit", self.search_limit)
        )
        excerpt_chars = max(
            80, await self._setting_int("research.excerpt_chars", 500)
        )
        refresh_seconds = max(
            60, await self._setting_int("research.source_refresh_seconds", 86400)
        )
        web_name = getattr(self.web_search, "name", None) or "unknown-search"
        synth_name = getattr(self.deepseek, "name", None) or "unknown-synthesizer"
        provider_name = f"{web_name}+{synth_name}"

        project_id = state.get("project_id") or task.get("project_id")
        run_id = state.get("run_id") or task.get("run_id")
        if not project_id or not run_id:
            await self.state.supersede_running(task["task_id"])
            project_id = project_id or await self.state.ensure_project(
                task["question"], task_id=task["task_id"]
            )
            # Reuse a durable RUNNING run when this is a fresh attempt after a
            # pause; otherwise start one.
            existing_run = (
                await self.state.get_run(run_id) if run_id else None
            )
            if existing_run is None or existing_run.get("status") != "RUNNING":
                run_id = await self.state.start_run(
                    project_id,
                    task_id=task["task_id"],
                    provider=provider_name,
                    budget={
                        "search_limit": search_limit,
                        "excerpt_chars": excerpt_chars,
                    },
                )
            await save({"stage": "START", "project_id": project_id, "run_id": run_id})

        stage = str(state.get("stage") or "START")
        if stage not in {"SEARCHED", "ANALYZED", "SAVING", "SAVED", "DONE"}:
            await pause_if_needed()
            # -- SEARCH ----------------------------------------------------
            results = await self.web_search.search(
                task["question"], limit=search_limit
            )
            await self.state.record_query(
                run_id,
                task["question"],
                provider=web_name,
                result_count=len(results),
            )
            sources, excerpts = await self._persist_sources(
                task, results, run_id, excerpt_chars, refresh_seconds
            )
            state["search_results"] = [asdict(r) for r in results]
            state["source_ids"] = [source.source_id for source in sources]
            state["excerpt_ids"] = excerpt_ids = list(excerpts)
            state["sources"] = [
                {
                    "source_id": source.source_id,
                    "source_type": source.source_type,
                    "title": source.title,
                    "url": source.url,
                    "provider": source.provider,
                }
                for source in sources
            ]
            state["excerpt_ids"] = excerpt_ids
            state["stage"] = "SEARCHED"
            await save(dict(state))
            stage = "SEARCHED"

        if stage == "SEARCHED":
            await pause_if_needed()
            # -- ANALYZE ---------------------------------------------------
            summary = state.get("summary")
            if not summary:
                summary = await self.deepseek.synthesize(
                    task["question"], list(state.get("search_results") or [])
                )
                claim_id = f"claim:{task['task_id']}"
                await self.state.record_claim(
                    summary,
                    project_id=project_id,
                    status="PROPOSED",
                    confidence=0.0,
                    verification_level="UNVERIFIED",
                    claim_id=claim_id,
                )
                source_ids = list(state.get("source_ids") or [])
                excerpt_ids = list(state.get("excerpt_ids") or [])
                for index, source_id in enumerate(source_ids):
                    await self.state.link_evidence(
                        claim_id,
                        source_id,
                        excerpt_id=(
                            excerpt_ids[index]
                            if index < len(excerpt_ids)
                            else ""
                        ),
                    )
                await self.state.record_finding(
                    project_id,
                    summary,
                    run_id=run_id,
                    status="DRAFT",
                    finding_id=f"find:{task['task_id']}",
                )
                state["summary"] = summary
                state["stage"] = "ANALYZED"
                await save(dict(state))
            stage = "ANALYZED"

        if stage in {"ANALYZED", "SAVING", "SAVED"}:
            await pause_if_needed()
            # -- SAVE ------------------------------------------------------
            summary = state.get("summary") or ""
            sources = self._sources_from_state(task, state)
            state["stage"] = "SAVING"
            await save(dict(state))
            await self._consolidate(
                task,
                summary,
                [],
                sources=sources,
                owner_id=owner_id,
                commit=commit,
            )
            try:
                await self.state.finish_run(
                    run_id,
                    "COMPLETED",
                    budget_used={
                        "search_limit": search_limit,
                        "results": len(state.get("search_results") or []),
                        "sources": len(sources),
                        "resumed": started_from_checkpoint,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - repair can finish it later.
                self.logger.warning(
                    "research_run_finish_failed research_id=%s error=%s",
                    task["task_id"],
                    type(exc).__name__,
                )
            state["stage"] = "DONE"
            if commit is None:
                # With an atomic commit the job is already COMPLETED; writing
                # a checkpoint afterwards would be rejected (correctly) and
                # must not turn a committed result into an error.
                await save({"stage": "DONE"}, final=True)

    @staticmethod
    def _sources_from_state(task: dict, state: dict) -> list[MemorySource]:
        sources = state.get("sources")
        if sources:
            return [
                MemorySource(
                    source_id=str(item["source_id"]),
                    source_type=str(item.get("source_type") or "web"),
                    title=item.get("title"),
                    url=item.get("url"),
                    provider=item.get("provider"),
                )
                for item in sources
            ]
        return [
            MemorySource(
                source_id=f"web:{task['task_id']}:{i}",
                source_type="web",
                title=item.get("title"),
                url=item.get("url"),
                provider=None,
            )
            for i, item in enumerate(state.get("search_results") or [])
        ]

    async def _repair_completed_run(self, task: dict) -> None:
        run_id = task.get("run_id")
        if not run_id:
            return
        run = await self.state.get_run(run_id)
        if run is not None and run["status"] == "RUNNING":
            await self.state.finish_run(
                run_id,
                "COMPLETED",
                budget_used={"resumed": True, "memory_reused": True},
            )

    @staticmethod
    def _completed_payload(task: dict) -> dict:
        return {
            "status": "COMPLETED",
            "research_task_id": task["task_id"],
            "memory_id": task.get("memory_id"),
            "result": task.get("result"),
        }


    async def _consolidate(
        self,
        task: dict,
        summary: str,
        results,
        *,
        sources: list[MemorySource] | None = None,
        owner_id: str | None = None,
        commit=None,
    ) -> None:
        memory_id = f"mem:{task['task_id']}"
        provider_name = getattr(self.web_search, "name", None)
        if sources is None:
            sources = []
            for i, r in enumerate(results):
                sources.append(
                    MemorySource(
                        source_id=f"web:{task['task_id']}:{i}",
                        source_type="web",
                        title=r.title,
                        url=r.url,
                        provider=provider_name,
                    )
                )
        evidence = [
            MemoryEvidence(
                source_id=s.source_id,
                verification_level=VerificationLevel.UNVERIFIED,
            )
            for s in sources
        ]

        async def write() -> None:
            existing = await self.memory_service.get_item(memory_id, SYSTEM_ADMIN)
            if existing is None:
                await self.memory_service.create(
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
            if commit is not None:
                # Detail-row update is conditional on this exact execution
                # still owning the task. If the token changed, raising rolls
                # the memory write back in the same transaction.
                if owner_id is None:
                    raise TaskLeaseLost("no_research_owner")
                completed = await self.research_repo.complete_owned(
                    task["task_id"], owner_id, summary, memory_id=memory_id
                )
                if not completed:
                    raise TaskLeaseLost("research_owner_lost")
            else:
                await self.research_repo.complete(
                    task["task_id"], summary, memory_id=memory_id
                )

        if commit is not None:
            # The outer task transaction runs the memory write, the detail-row
            # COMPLETED update and the job COMPLETED update together. A
            # cancellation before this point writes nothing; one after it
            # cannot undo the committed result.
            payload = {
                "status": "COMPLETED",
                "research_task_id": task["task_id"],
                "memory_id": memory_id,
                "result": summary,
            }
            await commit(write, result=payload)
        else:
            # Legacy direct-engine path: the memory and the task's COMPLETED
            # state still commit together.
            async with self.memory_service.repo.db.transaction():
                await write()
        self.logger.info(
            "research_consolidated memory_id=%s provider=%s",
            memory_id,
            provider_name or "unknown",
        )
