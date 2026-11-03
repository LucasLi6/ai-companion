"""Data access for ``embedding_models`` and ``memory_embeddings``.

Thin, parameterised SQL over :class:`Database`. No business rules live here —
validation, idempotency and stale-write protection belong to the embedding
service. The vector BLOB is written exactly as packed float32; nothing here
interprets it.
"""

from __future__ import annotations

import json
from typing import Optional

from companion.domain.models import utcnow
from companion.memory.embeddings import EmbeddingModelSpec
from companion.storage.database import Database


def _iso(dt) -> str:
    return dt.isoformat()


class EmbeddingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -- model registry -----------------------------------------------------

    async def upsert_model(self, spec: EmbeddingModelSpec) -> str:
        """Insert (or refresh) one model row; return its deterministic id."""
        model_id = spec.model_id()
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO embedding_models(model_id, provider, model_name, dimensions,"
            " model_revision, normalization, query_prefix, document_prefix,"
            " distance_metric, max_input_tokens, configuration_hash, created_at, extra)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')"
            " ON CONFLICT(model_id) DO UPDATE SET"
            " provider = excluded.provider, model_name = excluded.model_name,"
            " dimensions = excluded.dimensions,"
            " model_revision = excluded.model_revision,"
            " normalization = excluded.normalization,"
            " query_prefix = excluded.query_prefix,"
            " document_prefix = excluded.document_prefix,"
            " distance_metric = excluded.distance_metric,"
            " max_input_tokens = excluded.max_input_tokens,"
            " configuration_hash = excluded.configuration_hash",
            (
                model_id,
                spec.provider,
                spec.model_name,
                spec.dimensions,
                spec.model_revision,
                spec.normalization,
                spec.query_prefix,
                spec.document_prefix,
                spec.distance_metric,
                spec.max_input_tokens,
                spec.configuration_hash(),
                now,
            ),
        )
        return model_id

    async def get_model(self, model_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM embedding_models WHERE model_id = ?", (model_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_models(self) -> list[dict]:
        cur = await self.db.execute("SELECT * FROM embedding_models ORDER BY created_at ASC")
        return [dict(r) for r in await cur.fetchall()]

    # -- embeddings ---------------------------------------------------------

    async def get_embedding(self, memory_id: str, model_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT memory_id, model_id, vector, memory_version, content_hash,"
            " embedding_status, updated_at, last_error FROM memory_embeddings"
            " WHERE memory_id = ? AND model_id = ?",
            (memory_id, model_id),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def upsert_embedding(
        self,
        memory_id: str,
        model_id: str,
        vector: bytes,
        *,
        memory_version: int,
        content_hash: str,
        embedding_status: str = "READY",
        last_error: Optional[str] = None,
    ) -> None:
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO memory_embeddings(memory_id, model_id, vector, memory_version,"
            " content_hash, embedding_status, updated_at, last_error, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(memory_id, model_id) DO UPDATE SET"
            " vector = excluded.vector, memory_version = excluded.memory_version,"
            " content_hash = excluded.content_hash,"
            " embedding_status = excluded.embedding_status,"
            " updated_at = excluded.updated_at, last_error = excluded.last_error",
            (
                memory_id,
                model_id,
                vector,
                memory_version,
                content_hash,
                embedding_status,
                now,
                last_error,
                now,
            ),
        )

    async def record_failure(
        self,
        memory_id: str,
        model_id: str,
        error: str,
        *,
        memory_version: int = 0,
        content_hash: str | None = None,
    ) -> None:
        """Record a FAILED embedding without touching the memory row itself.

        An embedding failure must never destroy or mutate a memory: it only
        marks the derived vector as failed so the backfill can retry it.
        """
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO memory_embeddings(memory_id, model_id, vector, memory_version,"
            " content_hash, embedding_status, updated_at, last_error, created_at)"
            " VALUES (?, ?, ?, ?, ?, 'FAILED', ?, ?, ?)"
            " ON CONFLICT(memory_id, model_id) DO UPDATE SET"
            " memory_version = excluded.memory_version,"
            " content_hash = excluded.content_hash,"
            " embedding_status = 'FAILED', updated_at = excluded.updated_at,"
            " last_error = excluded.last_error",
            (memory_id, model_id, b"", memory_version, content_hash, now, error, now),
        )

    async def set_status(self, memory_id: str, model_id: str, status: str) -> None:
        await self.db.execute(
            "UPDATE memory_embeddings SET embedding_status = ?, updated_at = ?"
            " WHERE memory_id = ? AND model_id = ?",
            (status, _iso(utcnow()), memory_id, model_id),
        )

    async def list_ready(self, model_id: str) -> list[dict]:
        """Return every READY vector row for a model (id + unpacked vector)."""
        cur = await self.db.execute(
            "SELECT memory_id, vector, memory_version, content_hash FROM memory_embeddings"
            " WHERE model_id = ? AND embedding_status = 'READY'",
            (model_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_all(self, model_id: str | None = None) -> list[dict]:
        sql = (
            "SELECT e.memory_id, e.model_id, e.memory_version, e.content_hash,"
            " e.embedding_status, e.updated_at, e.last_error, m.content, m.status"
            " FROM memory_embeddings e JOIN memory_items m ON m.memory_id = e.memory_id"
        )
        params: tuple = ()
        if model_id is not None:
            sql += " WHERE e.model_id = ?"
            params = (model_id,)
        sql += " ORDER BY e.updated_at ASC"
        cur = await self.db.execute(sql, params)
        return [dict(r) for r in await cur.fetchall()]

    async def counts(self, model_id: str | None = None) -> dict:
        sql = "SELECT embedding_status, COUNT(*) AS c FROM memory_embeddings"
        params: tuple = ()
        if model_id is not None:
            sql += " WHERE model_id = ?"
            params = (model_id,)
        sql += " GROUP BY embedding_status"
        cur = await self.db.execute(sql, params)
        return {r["embedding_status"]: r["c"] for r in await cur.fetchall()}

    async def delete_model_embeddings(self, model_id: str) -> None:
        await self.db.execute(
            "DELETE FROM memory_embeddings WHERE model_id = ?", (model_id,)
        )

    async def mark_all_stale(self, model_id: str) -> None:
        """Mark every vector for a model stale (used by ``rebuild``)."""
        await self.db.execute(
            "UPDATE memory_embeddings SET embedding_status = 'STALE'"
            " WHERE model_id = ?",
            (model_id,),
        )
