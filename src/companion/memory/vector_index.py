"""Vector similarity search over ``memory_embeddings``.

Ranking currently runs in pure Python over the READY vectors for one model.
That is deliberate and honest: ``sqlite-vec`` may be installed, but this
project does **not** maintain a ``vec0`` shadow table, so it cannot claim that
KNN executes inside SQLite. :meth:`VectorIndex.backend_status` therefore
reports installation/loadability separately from the *active* backend.

Only vectors that still match their current ACTIVE memory are ranked — the
repository's ``list_ready`` re-checks memory status, version and content hash,
so a stale vector can never surface content the keyword layer would not show.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from typing import Optional

from companion.memory.embeddings import (
    VectorValidationError,
    cosine_similarity,
    dot_product,
    unpack_vector,
)
from companion.memory.embedding_repository import EmbeddingRepository


def _probe_sqlite_vec() -> dict[str, bool]:
    """Probe the optional extension without importing it unconditionally.

    ``installed`` means the Python package or a loadable extension was found;
    ``loadable`` additionally means a fresh connection could call
    ``vec_version()``. Neither implies this project uses it for search.
    """
    installed = False
    try:  # pragma: no cover - depends on the optional native package
        import sqlite_vec  # noqa: F401

        installed = True
    except Exception:
        sqlite_vec = None

    loadable = False
    try:
        conn = sqlite3.connect(":memory:")
        try:
            try:
                conn.enable_load_extension(True)
            except Exception:
                pass
            loaded = False
            if installed:
                try:  # pragma: no cover - optional native package
                    sqlite_vec.load(conn)
                    loaded = True
                except Exception:
                    loaded = False
            if not loaded:
                try:
                    conn.load_extension("vec0")
                    loaded = True
                except Exception:
                    loaded = False
            if loaded:
                try:
                    conn.execute("SELECT vec_version()")
                    loadable = True
                except Exception:
                    loadable = False
        finally:
            conn.close()
    except Exception:
        loadable = False
    return {"installed": installed or loadable, "loadable": loadable}


def _sqlite_vec_available() -> bool:
    """Backwards-compatible probe: whether vec0 is loadable (not in use)."""
    return _probe_sqlite_vec()["loadable"]


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
        self._probe: Optional[dict[str, bool]] = None

    def backend_status(self, requested: str = "python") -> dict:
        """Honest backend report: available vs. actually used.

        ``sqlite_vec_enabled`` is always ``False`` because no ``vec0`` index is
        maintained; the active backend is the Python fallback. If a future
        change implements real KNN, it must flip this based on a successful
        in-database query, not merely on the extension being loadable.
        """
        if self._probe is None:
            self._probe = _probe_sqlite_vec()
        return {
            "sqlite_vec_installed": self._probe["installed"],
            "sqlite_vec_loadable": self._probe["loadable"],
            "sqlite_vec_enabled": False,
            "active_backend": "python",
            "requested_backend": requested,
        }

    @property
    def using_sqlite_vec(self) -> bool:
        """Whether search runs in sqlite-vec (never, currently)."""
        return bool(self.backend_status()["sqlite_vec_enabled"])

    async def available(self) -> bool:
        """Whether READY vectors exist for ranking (pure fallback always can)."""
        return True

    async def search(
        self,
        model_id: str,
        query_vector: list[float],
        *,
        limit: int = 20,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """Return ``(memory_id, similarity)`` pairs, most similar first.

        Only current ACTIVE vectors are considered (see ``list_ready``).
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
