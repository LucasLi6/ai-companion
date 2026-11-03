"""Durable learning state: runs, checkpoints, curiosity, reflection.

Learning runs are bookkeeping, not authority. A checkpoint records what an idle
cycle had reached so a crash can resume from a known point; it never grants
permission, and a restored run must re-check the same gates as a fresh one.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from companion.domain.models import utcnow
from companion.storage.database import Database


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _iso(value: datetime) -> str:
    return value.isoformat()


class LearningRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def start_run(
        self,
        kind: str,
        *,
        job_id: str | None = None,
        provider: str | None = None,
        budget: dict | None = None,
    ) -> str:
        run_id = _new("learn")
        await self.db.execute(
            "INSERT INTO learning_runs(run_id, job_id, kind, status, started_at,"
            " budget_json, provider) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?)",
            (
                run_id,
                job_id,
                kind,
                _iso(utcnow()),
                json.dumps(budget or {}, ensure_ascii=False, sort_keys=True),
                provider,
            ),
        )
        return run_id

    async def finish_run(
        self, run_id: str, status: str, *, error: str | None = None
    ) -> None:
        await self.db.execute(
            "UPDATE learning_runs SET status = ?, completed_at = ?, error = ?"
            " WHERE run_id = ?",
            (status, _iso(utcnow()), error, run_id),
        )

    async def checkpoint(self, run_id: str, stage: str, payload: dict | None = None) -> str:
        checkpoint_id = _new("ckpt")
        await self.db.execute(
            "INSERT INTO learning_checkpoints(checkpoint_id, run_id, stage, payload,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (
                checkpoint_id,
                run_id,
                stage,
                json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                _iso(utcnow()),
            ),
        )
        return checkpoint_id

    async def get_run(self, run_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM learning_runs WHERE run_id = ?", (run_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_checkpoints(self, run_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM learning_checkpoints WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def last_checkpoint(self, kind: str) -> dict | None:
        """Most recent checkpoint for the newest run of ``kind``."""
        cur = await self.db.execute(
            "SELECT c.* FROM learning_checkpoints c"
            " JOIN learning_runs r ON r.run_id = c.run_id"
            " WHERE r.kind = ? ORDER BY c.created_at DESC LIMIT 1",
            (kind,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    # -- curiosity queue ---------------------------------------------------

    async def enqueue_curiosity(
        self,
        question: str,
        *,
        source_kind: str = "curiosity",
        source_ref: str | None = None,
        priority: int = 0,
        scheduled_at: str | None = None,
    ) -> str:
        item_id = _new("curio")
        await self.db.execute(
            "INSERT INTO curiosity_queue(item_id, question, source_kind, source_ref,"
            " priority, status, created_at, scheduled_at) VALUES (?, ?, ?, ?, ?,"
            " 'PENDING', ?, ?)",
            (
                item_id,
                question,
                source_kind,
                source_ref,
                priority,
                _iso(utcnow()),
                scheduled_at,
            ),
        )
        return item_id

    async def mark_curiosity(self, item_id: str, status: str) -> None:
        await self.db.execute(
            "UPDATE curiosity_queue SET status = ? WHERE item_id = ?",
            (status, item_id),
        )

    async def list_curiosity(self, status: str | None = None) -> list[dict]:
        if status is None:
            cur = await self.db.execute(
                "SELECT * FROM curiosity_queue ORDER BY priority DESC, created_at"
            )
        else:
            cur = await self.db.execute(
                "SELECT * FROM curiosity_queue WHERE status = ?"
                " ORDER BY priority DESC, created_at",
                (status,),
            )
        return [dict(r) for r in await cur.fetchall()]

    # -- reflection records ------------------------------------------------

    async def record_reflection(
        self,
        reflection_text: str,
        *,
        subject_memory_id: str | None = None,
        status: str = "PROPOSED",
        source_access: str | None = None,
    ) -> str:
        reflection_id = _new("refl")
        await self.db.execute(
            "INSERT INTO reflection_records(reflection_id, subject_memory_id,"
            " reflection_text, status, source_access, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                reflection_id,
                subject_memory_id,
                reflection_text,
                status,
                source_access,
                _iso(utcnow()),
            ),
        )
        return reflection_id

    async def list_reflections(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM reflection_records ORDER BY created_at"
        )
        return [dict(r) for r in await cur.fetchall()]

    # -- knowledge refresh jobs --------------------------------------------

    async def enqueue_refresh(
        self,
        *,
        source_id: str | None = None,
        claim_id: str | None = None,
        scheduled_at: str | None = None,
    ) -> str:
        job_id = _new("refreshjob")
        await self.db.execute(
            "INSERT INTO knowledge_refresh_jobs(job_id, source_id, claim_id, status,"
            " scheduled_at, attempts) VALUES (?, ?, ?, 'PENDING', ?, 0)",
            (job_id, source_id, claim_id, scheduled_at or _iso(utcnow())),
        )
        return job_id

    async def list_refresh_jobs(self, status: str | None = None) -> list[dict]:
        if status is None:
            cur = await self.db.execute(
                "SELECT * FROM knowledge_refresh_jobs ORDER BY scheduled_at"
            )
        else:
            cur = await self.db.execute(
                "SELECT * FROM knowledge_refresh_jobs WHERE status = ?"
                " ORDER BY scheduled_at",
                (status,),
            )
        return [dict(r) for r in await cur.fetchall()]
