"""``EMBEDDING`` job handler: embed one memory idempotently.

Each job payload carries ``memory_id`` and ``memory_version`` (and, when
available, ``kind``/``content``). The handler re-reads the *current* memory row
and embeds it through :class:`EmbeddingService.embed_memory`, which is
idempotent (keyed by content hash) and refuses to overwrite a newer vector. A
stale job therefore never destroys a newer embedding, and a provider failure
records a ``FAILED`` status on the vector row while leaving the memory intact.
"""

from __future__ import annotations

import logging

from companion.memory.embedding_service import EmbeddingService
from companion.storage.repositories import Repositories


class EmbeddingJobError(Exception):
    """Raised by the embedding handler to record a business-level failure."""


def build_embedding_handler(service: EmbeddingService):
    """Return an ``EMBEDDING`` handler bound to a configured service."""

    async def embedding_job(
        payload: dict, repos: Repositories, logger: logging.Logger
    ) -> None:
        memory_id = payload.get("memory_id")
        if not memory_id:
            raise EmbeddingJobError("embedding payload missing 'memory_id'")
        memory_version = int(payload.get("memory_version") or 0)

        row = await repos.db.execute(
            "SELECT memory_id, kind, content, version FROM memory_items"
            " WHERE memory_id = ?",
            (memory_id,),
        )
        cur = await row.fetchone()
        if cur is None:
            # The memory was deleted or retracted since enqueue: nothing to do.
            return
        row = dict(cur)

        await service.embed_memory(
            memory_id,
            kind=row["kind"],
            content=row["content"],
            memory_version=row["version"],
        )
        logger.info(
            "embedding_job_done memory_id=%s version=%d", memory_id, row["version"]
        )

    return embedding_job
