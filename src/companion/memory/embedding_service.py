"""Embedding lifecycle: model registry, idempotent vector writes, backfill.

This is the trusted layer that turns a configured :class:`EmbeddingProvider`
into durable vectors in ``memory_embeddings``. It owns the invariants:

- **Idempotency.** A write is keyed by ``embed:{memory_id}:{model_id}:
  {memory_version}`` and content hash; re-running the same write is a no-op.
- **Crash safety.** Writing a vector never touches the memory row itself, so a
  failed or interrupted embedding cannot destroy or corrupt a memory.
- **Stale-write protection.** A job whose ``memory_version`` is older than the
  currently stored vector is dropped — a stale job never overwrites a newer
  vector for the same memory.
- **Failure isolation.** A provider error for one memory records a ``FAILED``
  status on that vector row and leaves the memory retrievable by keyword.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from companion.domain.enums import MemoryStatus
from companion.memory.embeddings import (
    EmbeddingModelSpec,
    EmbeddingProvider,
    VectorValidationError,
    normalize_l2,
    pack_vector,
    validate_vector,
)
from companion.memory.embedding_repository import EmbeddingRepository
from companion.memory.vector_index import VectorIndex

READY = "READY"
FAILED = "FAILED"
STALE = "STALE"
UNVERIFIED = "UNVERIFIED"
PENDING = "PENDING"


def content_hash(kind: str, content: str) -> str:
    """Deterministic hash of the fields a memory vector must reflect.

    A change to kind or content yields a different hash, so a backfill detects
    that the stored vector is out of date and rebuilds it.
    """
    digest = hashlib.sha256(f"{kind}\0{content}".encode("utf-8")).hexdigest()
    return digest


class EmbeddingService:
    def __init__(
        self,
        repo: EmbeddingRepository,
        provider: EmbeddingProvider,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repo = repo
        self.provider = provider
        self.index = VectorIndex(repo, logger=logger)
        self.logger = logger or logging.getLogger("companion.memory.embedding")

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    @property
    def spec(self) -> EmbeddingModelSpec:
        return self.provider.spec

    async def ensure_model(self) -> str:
        """Register the provider's model configuration and return its id."""
        return await self.repo.upsert_model(self.spec)

    # -- single-memory embedding -------------------------------------------

    async def embed_memory(
        self,
        memory_id: str,
        *,
        kind: str,
        content: str,
        memory_version: int,
        force: bool = False,
    ) -> dict:
        """Embed one memory idempotently, guarding against stale writes.

        Returns a small status dict. ``force`` re-embeds even when the stored
        content hash already matches (used by ``rebuild``).
        """
        model_id = await self.ensure_model()
        expected_hash = content_hash(kind, content)

        existing = await self.repo.get_embedding(memory_id, model_id)
        if existing is not None and not force:
            # Idempotency: the same content is already embedded.
            if (
                existing.get("content_hash") == expected_hash
                and existing.get("embedding_status") == READY
            ):
                return {"memory_id": memory_id, "model_id": model_id, "status": "UNCHANGED"}
            # Stale-write guard: a job carrying an older version must never
            # overwrite a vector already written for a newer version.
            if (existing.get("memory_version") or 0) > memory_version:
                return {"memory_id": memory_id, "model_id": model_id, "status": "STALE_SKIPPED"}

        try:
            vector = validate_vector(
                self.provider.embed_documents([content])[0], self.provider.dimensions
            )
            if self.spec.normalization == "l2":
                vector = normalize_l2(vector)
            blob = pack_vector(vector)
        except VectorValidationError as exc:
            self.logger.warning(
                "embedding_invalid memory_id=%s error=%s", memory_id, exc
            )
            await self.repo.record_failure(
                memory_id,
                model_id,
                str(exc),
                memory_version=memory_version,
                content_hash=expected_hash,
            )
            return {"memory_id": memory_id, "model_id": model_id, "status": FAILED}
        except Exception as exc:
            # A provider crash is transient: record it and re-raise so the job
            # can retry without losing the memory.
            self.logger.error("embedding_error memory_id=%s error=%s", memory_id, exc)
            await self.repo.record_failure(
                memory_id,
                model_id,
                str(exc),
                memory_version=memory_version,
                content_hash=expected_hash,
            )
            raise

        await self.repo.upsert_embedding(
            memory_id,
            model_id,
            blob,
            memory_version=memory_version,
            content_hash=expected_hash,
            embedding_status=READY,
        )
        self.logger.info(
            "embedding_written memory_id=%s model_id=%s version=%d",
            memory_id,
            model_id,
            memory_version,
        )
        return {"memory_id": memory_id, "model_id": model_id, "status": READY}

    # -- query --------------------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        vector = validate_vector(self.provider.embed_query(text), self.provider.dimensions)
        if self.spec.normalization == "l2":
            vector = normalize_l2(vector)
        return vector

    # -- backfill / rebuild / status ----------------------------------------

    async def active_memory_rows(self) -> list[dict]:
        cur = await self.repo.db.execute(
            "SELECT memory_id, kind, content, version FROM memory_items"
            " WHERE status = ? ORDER BY updated_at ASC",
            (MemoryStatus.ACTIVE.value,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def backfill(self, *, force: bool = False) -> list[dict]:
        """Embed every ACTIVE memory that lacks an up-to-date READY vector.

        Returns one status dict per memory. This is the synchronous, foreground
        form of the idempotent backfill command; the job queue wraps it in
        resumable units when run through the worker.
        """
        await self.ensure_model()
        results: list[dict] = []
        for row in await self.active_memory_rows():
            result = await self.embed_memory(
                row["memory_id"],
                kind=row["kind"],
                content=row["content"],
                memory_version=row["version"],
                force=force,
            )
            results.append(result)
        return results

    async def rebuild(self) -> dict:
        """Mark every vector stale and re-embed all ACTIVE memories."""
        model_id = await self.ensure_model()
        await self.repo.mark_all_stale(model_id)
        results = await self.backfill(force=True)
        return {
            "model_id": model_id,
            "stale_marked": True,
            "results": results,
            "counts": await self.repo.counts(model_id),
        }

    async def status(self) -> dict:
        model_id = await self.ensure_model()
        models = await self.repo.list_models()
        return {
            "model_id": model_id,
            "models": models,
            "counts": await self.repo.counts(model_id),
            "using_sqlite_vec": self.index.using_sqlite_vec,
            "sqlite_vec_available": self.index.using_sqlite_vec,
        }

    def job_key(self, memory_id: str, model_id: str, memory_version: int) -> str:
        return f"embed:{memory_id}:{model_id}:{memory_version}"
