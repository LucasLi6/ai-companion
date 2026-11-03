"""Durable research state: projects, runs, queries, sources, claims.

The distinctions this layer preserves:

- a search hit is a *source candidate*, not verified evidence;
- an excerpt is quoted material, never an instruction;
- a synthesized summary is a *proposed claim*, never a trusted fact;
- retrieval time and publication time are separate columns;
- a mock provider is recorded by name so it can never be presented as real;
- expiration/refresh rules describe staleness, not falsity.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

from companion.domain.models import utcnow
from companion.storage.database import Database


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _iso(value: datetime) -> str:
    return value.isoformat()


class ResearchStateRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def ensure_project(
        self, question: str, *, title: str | None = None, task_id: str | None = None
    ) -> str:
        """Return the project for a task, creating it once."""
        if task_id:
            cur = await self.db.execute(
                "SELECT project_id FROM research_tasks WHERE task_id = ?", (task_id,)
            )
            row = await cur.fetchone()
            if row is not None and row["project_id"]:
                return row["project_id"]
        now = _iso(utcnow())
        project_id = _new("proj")
        short_title = (title or question).strip()[:200]
        await self.db.execute(
            "INSERT INTO research_projects(project_id, title, question, status,"
            " created_at, updated_at) VALUES (?, ?, ?, 'OPEN', ?, ?)",
            (project_id, short_title, question, now, now),
        )
        if task_id:
            await self.db.execute(
                "UPDATE research_tasks SET project_id = ? WHERE task_id = ?",
                (project_id, task_id),
            )
        return project_id

    async def start_run(
        self,
        project_id: str,
        *,
        task_id: str | None = None,
        provider: str | None = None,
        budget: dict | None = None,
    ) -> str:
        run_id = _new("run")
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO research_runs(run_id, project_id, task_id, status,"
            " provider, started_at, budget_used, extra) VALUES (?, ?, ?, 'RUNNING',"
            " ?, ?, '{}', ?)",
            (
                run_id,
                project_id,
                task_id,
                provider,
                now,
                json.dumps(
                    {"budget": budget or {}}, ensure_ascii=False, sort_keys=True
                ),
            ),
        )
        if task_id:
            await self.db.execute(
                "UPDATE research_tasks SET run_id = ? WHERE task_id = ?",
                (run_id, task_id),
            )
        return run_id

    async def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        budget_used: dict | None = None,
    ) -> None:
        await self.db.execute(
            "UPDATE research_runs SET status = ?, completed_at = ?, error = ?,"
            " budget_used = COALESCE(?, budget_used) WHERE run_id = ?",
            (
                status,
                _iso(utcnow()),
                error,
                json.dumps(budget_used, ensure_ascii=False, sort_keys=True)
                if budget_used is not None
                else None,
                run_id,
            ),
        )

    async def get_run(self, run_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def supersede_running(self, task_id: str) -> None:
        """Mark a crashed attempt's run abandoned before a new attempt starts."""
        await self.db.execute(
            "UPDATE research_runs SET status = 'ABANDONED', completed_at = ?,"
            " error = COALESCE(error, 'attempt did not finish')"
            " WHERE task_id = ? AND status = 'RUNNING'",
            (_iso(utcnow()), task_id),
        )

    async def record_query(
        self,
        run_id: str,
        query_text: str,
        *,
        provider: str | None = None,
        result_count: int = 0,
    ) -> str:
        query_id = _new("query")
        await self.db.execute(
            "INSERT INTO search_queries(query_id, run_id, query_text, provider,"
            " executed_at, result_count) VALUES (?, ?, ?, ?, ?, ?)",
            (query_id, run_id, query_text, provider, _iso(utcnow()), result_count),
        )
        return query_id

    async def record_source(
        self,
        *,
        url: str | None,
        title: str | None,
        run_id: str | None = None,
        publisher: str | None = None,
        published_at: datetime | None = None,
        source_kind: str = "web",
        trust_level: str = "UNVERIFIED",
        extra: dict | None = None,
        source_id: str | None = None,
    ) -> str:
        source_id = source_id or _new("src")
        await self.db.execute(
            "INSERT OR IGNORE INTO web_sources(source_id, url, title, publisher,"
            " published_at, retrieved_at, source_kind, trust_level, extra)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                url,
                title,
                publisher,
                _iso(published_at) if published_at is not None else None,
                _iso(utcnow()),
                source_kind,
                trust_level,
                json.dumps(extra or {}, ensure_ascii=False, sort_keys=True),
            ),
        )
        return source_id

    async def record_excerpt(
        self,
        source_id: str,
        excerpt: str,
        *,
        run_id: str | None = None,
        max_chars: int = 500,
        excerpt_id: str | None = None,
    ) -> str:
        """Store a bounded excerpt. Long content is truncated, not trusted."""
        text = " ".join(str(excerpt).split())[:max_chars]
        excerpt_id = excerpt_id or _new("exc")
        await self.db.execute(
            "INSERT OR IGNORE INTO source_excerpts(excerpt_id, source_id, run_id,"
            " excerpt, char_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (excerpt_id, source_id, run_id, text, len(text), _iso(utcnow())),
        )
        return excerpt_id

    async def record_claim(
        self,
        claim_text: str,
        *,
        project_id: str | None = None,
        status: str = "PROPOSED",
        confidence: float = 0.0,
        verification_level: str = "UNVERIFIED",
        claim_id: str | None = None,
    ) -> str:
        claim_id = claim_id or _new("claim")
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT OR IGNORE INTO knowledge_claims(claim_id, project_id, claim_text,"
            " status, confidence, verification_level, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                claim_id,
                project_id,
                claim_text,
                status,
                confidence,
                verification_level,
                now,
                now,
            ),
        )
        return claim_id

    async def link_evidence(
        self,
        claim_id: str,
        source_id: str,
        *,
        excerpt_id: str = "",
        relation: str = "supports",
    ) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO claim_evidence(claim_id, source_id, excerpt_id,"
            " relation, created_at) VALUES (?, ?, ?, ?, ?)",
            (claim_id, source_id, excerpt_id, relation, _iso(utcnow())),
        )

    async def record_finding(
        self,
        project_id: str,
        summary: str,
        *,
        run_id: str | None = None,
        status: str = "DRAFT",
        finding_id: str | None = None,
    ) -> str:
        finding_id = finding_id or _new("find")
        await self.db.execute(
            "INSERT OR IGNORE INTO research_findings(finding_id, project_id, run_id,"
            " summary, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (finding_id, project_id, run_id, summary, status, _iso(utcnow())),
        )
        return finding_id

    async def create_refresh_rule(
        self,
        source_id: str,
        *,
        interval_seconds: int = 86400,
        active: bool = True,
        rule_id: str | None = None,
    ) -> str:
        rule_id = rule_id or _new("refresh")
        next_check = utcnow() + timedelta(seconds=interval_seconds)
        await self.db.execute(
            "INSERT OR IGNORE INTO source_refresh_rules(rule_id, source_id,"
            " next_check_at, interval_seconds, active) VALUES (?, ?, ?, ?, ?)",
            (rule_id, source_id, _iso(next_check), interval_seconds, int(active)),
        )
        return rule_id

    # -- read helpers ------------------------------------------------------

    async def get_project(self, project_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM research_projects WHERE project_id = ?", (project_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_runs(self, project_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM research_runs WHERE project_id = ? ORDER BY started_at",
            (project_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_queries(self, run_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM search_queries WHERE run_id = ? ORDER BY executed_at",
            (run_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_claims(self, project_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM knowledge_claims WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_claim_evidence(self, claim_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM claim_evidence WHERE claim_id = ?", (claim_id,)
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_sources(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM web_sources ORDER BY retrieved_at"
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_excerpts(self, source_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM source_excerpts WHERE source_id = ? ORDER BY created_at",
            (source_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_findings(self, project_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM research_findings WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_refresh_rules(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM source_refresh_rules ORDER BY next_check_at"
        )
        return [dict(r) for r in await cur.fetchall()]
