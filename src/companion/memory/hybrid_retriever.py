"""Hybrid retrieval: fuse keyword (FTS/CJK) and vector rankings with RRF.

The design contract:

- **Authorised scope first.** The keyword candidate list is produced by
  :class:`MemoryService` (the policy gateway) and is therefore already visible
  to the requester. Vector search returns *ids only*; every id is re-checked
  through the same access-controlled load callback before it may enter the
  ranking. A vector can never reveal a memory the keyword layer would not show.
- **Identity is preserved.** Exact identity matches are ranked ahead of any
  fused result — a question about the person's own name is answered by the
  deterministic subject+topic lookup, not by similarity.
- **Diagnostics without leaking content.** Fused results carry ``keyword_rank``,
  ``vector_rank``, ``fusion_score``, ``matched_memory_id``, ``index_model`` and
  ``index_status`` — but only on the returned dictionaries, never the private
  *content* of a memory the requester could not already see.

Reciprocal Rank Fusion: ``score = sum(1 / (k + rank))`` across the lists a
memory appears in. ``k`` defaults to 60 (a common choice) and is stable so the
ranking is reproducible.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from companion.memory.embedding_service import EmbeddingService
from companion.memory.models import AccessContext

RRF_K = 60
_VECTOR_CANDIDATES = 50


# ``load_memory(memory_id, access)`` returns the access-checked result dict for
# one memory, or ``None`` if the requester may not see it. Visibility is always
# enforced through the caller's access context — never assumed from the index.
LoadMemory = Callable[[str, AccessContext], Awaitable[dict | None]]


class HybridRetriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        *,
        load_memory: LoadMemory,
        logger: Optional[logging.Logger] = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self.embedding = embedding_service
        self.load_memory = load_memory
        self.logger = logger or logging.getLogger("companion.memory.hybrid")
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        *,
        keyword_results: list[dict],
        access: AccessContext,
        identity_ids: set[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Fuse keyword and vector rankings into one ordered, diagnosed list.

        ``keyword_results`` are already-authorised dictionaries (most relevant
        first) and ``access`` is the requester's context. Vector search returns
        *ids only*; each id is re-checked through ``load_memory(id, access)``
        before it may enter the ranking. ``identity_ids`` are memory ids from
        deterministic identity lookup, which always sort first.
        """
        identity_ids = identity_ids or set()
        keyword_results = [m for m in keyword_results if m.get("memory_id")]

        model_id = None
        index_status = "NO_MODEL"
        vector_ranked: list[dict] = []

        try:
            model_id = self.embedding.model_id
            index_status = "READY"
            query_vector = await self.embedding.embed_query(query)
            candidates = await self.embedding.index.search(
                model_id,
                query_vector,
                limit=_VECTOR_CANDIDATES,
                metric=self.embedding.spec.distance_metric,
            )
            for rank, (memory_id, _score) in enumerate(candidates):
                if memory_id in identity_ids:
                    continue
                # Vector results are ids only: enforce visibility before use.
                loaded = await self.load_memory(memory_id, access)
                if loaded is None or loaded.get("status") != "ACTIVE":
                    continue
                vector_ranked.append({**loaded, "vector_rank": rank + 1})
        except Exception as exc:  # a vector failure degrades to keyword-only
            self.logger.warning("hybrid_vector_degraded error=%s", exc)
            index_status = "UNAVAILABLE"

        # RRF across the two ranked lists.
        keyword_rank = {
            m["memory_id"]: i + 1 for i, m in enumerate(keyword_results)
        }
        vector_rank = {
            m["memory_id"]: m["vector_rank"] for m in vector_ranked
        }

        fused: dict[str, dict] = {}
        for i, memory in enumerate(keyword_results):
            mid = memory["memory_id"]
            score = 1.0 / (self.rrf_k + (i + 1))
            if mid in vector_rank:
                score += 1.0 / (self.rrf_k + vector_rank[mid])
            fused[mid] = {
                **memory,
                "keyword_rank": i + 1,
                "vector_rank": vector_rank.get(mid),
                "fusion_score": round(score, 6),
                "matched_memory_id": mid,
                "index_model": model_id,
                "index_status": index_status,
            }
        for memory in vector_ranked:
            mid = memory["memory_id"]
            if mid in fused:
                continue
            rank = vector_rank[mid]
            score = 1.0 / (self.rrf_k + rank)
            fused[mid] = {
                **memory,
                "keyword_rank": None,
                "vector_rank": rank,
                "fusion_score": round(score, 6),
                "matched_memory_id": mid,
                "index_model": model_id,
                "index_status": index_status,
            }

        ordered = sorted(
            fused.values(),
            key=lambda m: (
                0 if m["memory_id"] in identity_ids else 1,
                -(m["fusion_score"] or 0.0),
                m.get("updated_at") or "",
                m.get("memory_id") or "",
            ),
        )
        # Identity matches always lead; drop their fusion fields (they were not
        # fused) and keep them pristine at the head.
        result: list[dict] = []
        seen: set[str] = set()
        for memory in ordered:
            mid = memory["memory_id"]
            if mid in seen:
                continue
            seen.add(mid)
            if mid in identity_ids:
                result.append({k: v for k, v in memory.items()
                               if k not in ("keyword_rank", "vector_rank",
                                            "fusion_score", "matched_memory_id",
                                            "index_model", "index_status")})
            else:
                result.append(memory)
        return result[:limit]
