"""Embedding lifecycle: model registry, idempotent vector writes, backfill.

This is the trusted layer that turns a configured :class:`EmbeddingProvider`
into durable vectors in ``memory_embeddings``. It owns the invariants:

- **The database is authoritative.** :meth:`EmbeddingService.embed_memory`
  re-reads the current ``memory_items`` row; the caller's ``kind``/``content``/
  ``version`` are treated as a job hint. A job whose hint disagrees with the
  current ACTIVE row is stale and is dropped.
- **Only ACTIVE memories are embedded.** A retracted/deleted memory is never
  (re)indexed, and an existing vector stops being searchable as soon as the
  memory version or content hash changes.
- **Compute outside, commit inside.** The (potentially slow) provider runs in a
  worker thread before the transaction; inside ``BEGIN IMMEDIATE`` the current
  row, version, hash and stored vector are re-checked before anything is
  written, so a job that started earlier can never overwrite a newer vector.
- **Failure isolation.** A provider error records ``FAILED`` only when that
  record still describes the current memory and does not replace a newer
  READY vector; the memory itself is never touched.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from companion.domain.enums import MemoryStatus
from companion.memory.embeddings import (
    EmbeddingModelSpec,
    EmbeddingProvider,
    VectorValidationError,
    content_hash,
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
MISSING = "MISSING"
SKIPPED_INACTIVE = "SKIPPED_INACTIVE"
STALE_SKIPPED = "STALE_SKIPPED"


class EmbeddingService:
    def __init__(
        self,
        repo: EmbeddingRepository,
        provider: EmbeddingProvider,
        *,
        logger: Optional[logging.Logger] = None,
        max_concurrent: int = 1,
    ) -> None:
        self.repo = repo
        self.provider = provider
        self.index = VectorIndex(repo, logger=logger)
        self.logger = logger or logging.getLogger("companion.memory.embedding")
        # A local model is not thread-safe and can saturate a CPU; one in-flight
        # embedding is the safe default. The semaphore is held only around the
        # off-loop provider call, never around a database transaction.
        self._gate = asyncio.Semaphore(max(1, int(max_concurrent)))

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    @property
    def spec(self) -> EmbeddingModelSpec:
        return self.provider.spec

    async def ensure_model(self) -> str:
        """Register the provider's model configuration and return its id."""
        return await self.repo.upsert_model(self.spec)

    # -- off-loop provider calls --------------------------------------------

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        async with self._gate:
            return await asyncio.to_thread(self.provider.embed_documents, texts)

    async def _embed_query(self, text: str) -> list[float]:
        async with self._gate:
            return await asyncio.to_thread(self.provider.embed_query, text)

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

        The current ACTIVE row is read first; a job hint that no longer matches
        it is ``STALE_SKIPPED``. The vector is computed off the event loop, then
        the row is re-read inside ``BEGIN IMMEDIATE`` before the write.
        """
        model_id = await self.ensure_model()

        current = await self.repo.get_memory(memory_id)
        if current is None:
            return self._result(memory_id, model_id, MISSING)
        if current["status"] != MemoryStatus.ACTIVE.value:
            return self._result(memory_id, model_id, SKIPPED_INACTIVE)
        if (
            current["kind"] != kind
            or current["content"] != content
            or int(current["version"]) != int(memory_version)
        ):
            # This job was produced for an older (or never-current) revision.
            return self._result(memory_id, model_id, STALE_SKIPPED)
        current_version = int(current["version"])
        expected_hash = content_hash(current["kind"], current["content"])

        existing = await self.repo.get_embedding(memory_id, model_id)
        if existing is not None and not force:
            if (
                existing.get("content_hash") == expected_hash
                and existing.get("embedding_status") == READY
                and int(existing.get("memory_version") or 0) == current_version
            ):
                return self._result(memory_id, model_id, "UNCHANGED")
            if int(existing.get("memory_version") or 0) > current_version:
                return self._result(memory_id, model_id, STALE_SKIPPED)

        try:
            vector = validate_vector(
                (await self._embed_documents([current["content"]]))[0],
                self.provider.dimensions,
            )
            if self.spec.normalization == "l2":
                vector = normalize_l2(vector)
            blob = pack_vector(vector)
        except VectorValidationError as exc:
            self.logger.warning(
                "embedding_invalid memory_id=%s error=%s", memory_id, exc
            )
            await self._record_failure_if_current(
                memory_id,
                model_id,
                str(exc),
                memory_version=current_version,
                expected_hash=expected_hash,
            )
            return self._result(memory_id, model_id, FAILED)
        except Exception as exc:
            # A provider crash is transient: record it and re-raise so the job
            # can retry without losing the memory.
            self.logger.error("embedding_error memory_id=%s error=%s", memory_id, exc)
            await self._record_failure_if_current(
                memory_id,
                model_id,
                str(exc),
                memory_version=current_version,
                expected_hash=expected_hash,
            )
            raise

        async with self.repo.db.transaction(immediate=True):
            fresh = await self.repo.get_memory(memory_id)
            if (
                fresh is None
                or fresh["status"] != MemoryStatus.ACTIVE.value
                or int(fresh["version"]) != current_version
                or content_hash(fresh["kind"], fresh["content"]) != expected_hash
            ):
                # The memory changed while the model was running: drop this
                # vector rather than publish one that describes old content.
                return self._result(memory_id, model_id, STALE_SKIPPED)

            stored = await self.repo.get_embedding(memory_id, model_id)
            if (
                stored is not None
                and not force
                and int(stored.get("memory_version") or 0) > current_version
            ):
                return self._result(memory_id, model_id, STALE_SKIPPED)

            await self.repo.upsert_embedding(
                memory_id,
                model_id,
                blob,
                memory_version=current_version,
                content_hash=expected_hash,
                embedding_status=READY,
            )
        self.logger.info(
            "embedding_written memory_id=%s model_id=%s version=%d",
            memory_id,
            model_id,
            current_version,
        )
        return self._result(memory_id, model_id, READY)

    async def _record_failure_if_current(
        self,
        memory_id: str,
        model_id: str,
        error: str,
        *,
        memory_version: int,
        expected_hash: str,
    ) -> None:
        """Record FAILED only for the current revision and never over READY."""
        async with self.repo.db.transaction(immediate=True):
            fresh = await self.repo.get_memory(memory_id)
            if (
                fresh is None
                or fresh["status"] != MemoryStatus.ACTIVE.value
                or int(fresh["version"]) != memory_version
                or content_hash(fresh["kind"], fresh["content"]) != expected_hash
            ):
                return
            stored = await self.repo.get_embedding(memory_id, model_id)
            if stored is not None:
                stored_version = int(stored.get("memory_version") or 0)
                if stored_version > memory_version:
                    return
                if (
                    stored_version == memory_version
                    and stored.get("embedding_status") == READY
                ):
                    return
            await self.repo.record_failure(
                memory_id,
                model_id,
                error,
                memory_version=memory_version,
                content_hash=expected_hash,
            )

    @staticmethod
    def _result(memory_id: str, model_id: str, status: str) -> dict:
        return {"memory_id": memory_id, "model_id": model_id, "status": status}

    # -- query --------------------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        vector = validate_vector(
            await self._embed_query(text), self.provider.dimensions
        )
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
        backend = self.index.backend_status()
        return {
            "model_id": model_id,
            "models": models,
            "counts": await self.repo.counts(model_id),
            **backend,
            "sqlite_vec_available": backend["sqlite_vec_loadable"],
        }

    def job_key(self, memory_id: str, model_id: str, memory_version: int) -> str:
        return f"embed:{memory_id}:{model_id}:{memory_version}"
