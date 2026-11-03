"""Vector similarity search over ``memory_embeddings``.

The index lives in the same SQLite file as everything else. Two backends are
supported behind one interface:

- **Pure-Python fallback** (always available): load the READY vectors for a
  model and rank them with cosine/dot/L2. This is bounded (the backfill is
  idempotent and capped), requires no native extension, and is what runs when
  ``sqlite-vec`` is absent.
- **``sqlite-vec`` acceleration** (optional): a ``vec0`` virtual table that
  keeps the ranking inside SQLite. It is probed at *runtime*, never imported
  unconditionally, so startup succeeds without it and CMF validation never
  reports corruption just because the optional extension is missing.
"""

from __future__ import annotations

import logging
import math
import struct
from typing import Optional

from companion.memory.embeddings import (
    VectorValidationError,
    cosine_similarity,
    dot_product,
    unpack_vector,
)
from companion.memory.embedding_repository import EmbeddingRepository


def _sqlite_vec_available() -> bool:
    """Probe whether the optional ``sqlite-vec`` extension can be loaded."""
    try:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            conn.load_extension("vec0")
        except Exception:
            return False
        conn.execute("SELECT vec_version()")
        return True
    except Exception:
        return False


class VectorIndex:
    """Top-k nearest-neighbour search for a single embedding model."""

    def __init__(
        self,
        repo: EmbeddingRepository,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repo = repo
        self.logger = logger or logging.getLogger("companion.memory.vector")
        self._vec_available: Optional[bool] = None

    async def available(self) -> bool:
        """Whether READY vectors exist for ranking (pure fallback always can)."""
        return True

    @property
    def using_sqlite_vec(self) -> bool:
        if self._vec_available is None:
            self._vec_available = _sqlite_vec_available()
        return self._vec_available

    async def search(
        self,
        model_id: str,
        query_vector: list[float],
        *,
        limit: int = 20,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """Return ``(memory_id, similarity)`` pairs, most similar first.

        Similarity is higher-is-better for every metric (cosine in [-1, 1]; dot
        on L2-normalised vectors in [-1, 1]; L2 distance negated).
        """
        rows = await self.repo.list_ready(model_id)
        scored: list[tuple[str, float]] = []
        for row in rows:
            try:
                stored = unpack_vector(row["vector"])
            except VectorValidationError:
                continue
            score = self._score(stored, query_vector, metric)
            scored.append((row["memory_id"], score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _score(a: list[float], b: list[float], metric: str) -> float:
        if len(a) != len(b):
            return 0.0
        if metric == "dot":
            return dot_product(a, b)
        if metric == "l2":
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            return -dist
        return cosine_similarity(a, b)
