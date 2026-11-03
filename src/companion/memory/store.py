"""MemoryStore adapter over the unified :class:`MemoryService`.

The foreground cognitive flow reads memory through this adapter. It runs with
a *non-admin* :class:`AccessContext` for the speaking person, so it sees only
that person's own subjects plus public world knowledge — never another
person's private facts.

Retrieval (Phase 4) is keyword-based rather than a raw-text substring match:
the query is split into keywords (CJK/ASCII, on punctuation and whitespace)
and a memory matches if it matches *any* keyword, so a sentence like
"你好，今天怎么样？" retrieves on both "你好" and "今天怎么样" instead of
requiring the whole sentence to appear verbatim. Only ``ACTIVE`` memories are
ever returned.
"""

from __future__ import annotations

import re
from typing import Optional

from companion.memory.models import AccessContext
from companion.memory.service import MemoryService
from companion.memory.topics import detect_identity_query

RETRIEVAL_KINDS = [
    "PERSONAL_FACT",
    "EPISODE",
    "RELATIONSHIP",
    "PROJECT_STATE",
    "WORLD_KNOWLEDGE",
]

_MAX_KEYWORDS = 6
_MIN_KEYWORD_LEN = 2
_MAX_KEYWORD_LEN = 64
_PER_KEYWORD_LIMIT = 20
_RETRIEVAL_LIMIT = 50
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def extract_keywords(text: str | None, max_keywords: int = _MAX_KEYWORDS) -> list[str]:
    """Split free text into a small, ordered, de-duplicated keyword list.

    Full whitespace/punctuation-delimited tokens come first. For CJK runs of
    three or more characters, overlapping two-character windows are appended as
    a segmenter-free fallback, so a clause like "机器学习是什么" can still
    retrieve a memory containing "机器学习". No vector search, no segmenter —
    both are explicitly out of scope.
    """
    if not text:
        return []
    tokens = [
        t.strip()
        for t in re.split(r"[^\w]+", text, flags=re.UNICODE)
        if t.strip()
    ]
    seen: set[str] = set()
    out: list[str] = []

    def _add(token: str) -> bool:
        token = token.strip()
        if (
            len(token) < _MIN_KEYWORD_LEN
            or len(token) > _MAX_KEYWORD_LEN
            or token in seen
        ):
            return False
        seen.add(token)
        out.append(token)
        return len(out) >= max_keywords

    for token in tokens:
        if _add(token):
            return out
    for token in tokens:
        if _has_cjk(token) and 3 <= len(token) <= 32:
            for i in range(len(token) - 1):
                if _add(token[i : i + 2]):
                    return out
    return out


def validate_referenced_memory_ids(
    referenced_ids: list[str], retrieved_memories: list[dict]
) -> list[str]:
    """Return the subset of ``referenced_ids`` present in retrieved memories.

    ``referenced_memory_ids`` come from the *model*, so they are untrusted: an
    id the model fabricated (not in the retrieved set) is dropped. Order is
    preserved and duplicates removed. This is the trust boundary that stops a
    hallucinated id from being written as a real citation.
    """
    valid = {m["memory_id"] for m in retrieved_memories}
    out: list[str] = []
    for mid in referenced_ids:
        if mid in valid and mid not in out:
            out.append(mid)
    return out


class SqliteMemoryStore:
    def __init__(self, service: MemoryService) -> None:
        self.service = service

    async def retrieve(self, query: str, person_id: Optional[str]) -> list[dict]:
        """Retrieve the person's ``ACTIVE`` memories matching the query.

        Identity questions ("我叫什么名字？") are resolved *deterministically*
        by subject + canonical topic before keyword search, so the answer does
        not depend on the question sharing wording with the stored fact. All
        other queries use keyword matching (OR semantics), de-duplicated,
        most recently updated first, and capped.
        """
        if person_id is None:
            return []
        access = AccessContext(actor_id=person_id, is_admin=False, role="cognitive")

        identity: list[dict] = []
        identity_topics = detect_identity_query(query)
        if identity_topics:
            rows = await self.service.repo.list_subject_topic(
                person_id,
                identity_topics,
                statuses=["ACTIVE"],
                limit=_PER_KEYWORD_LIMIT,
            )
            for row in rows:
                identity.append(
                    {
                        "memory_id": row["memory_id"],
                        "kind": row["kind"],
                        "content": row["content"],
                        "status": row["status"],
                        "visibility": row["visibility"],
                        "created_by": row["created_by"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "version": row["version"],
                        "human_confirmed": bool(row["human_confirmed"]),
                        "source_supported": bool(row["source_supported"]),
                        "self_reported": bool(row["self_reported"]),
                        "third_party_reported": bool(row["third_party_reported"]),
                        "model_inferred": bool(row["model_inferred"]),
                        "subjects": [
                            {"subject_type": "PERSON", "subject_id": person_id}
                        ],
                        "sources": [],
                    }
                )

        keywords = extract_keywords(query)

        by_id: dict[str, dict] = {
            memory["memory_id"]: memory for memory in identity
        }
        if keywords:
            for keyword in keywords:
                hits = await self.service.search(
                    access=access,
                    query=keyword,
                    statuses=["ACTIVE"],
                    kinds=RETRIEVAL_KINDS,
                    limit=_PER_KEYWORD_LIMIT,
                )
                for memory in hits:
                    by_id.setdefault(memory["memory_id"], memory)
        elif not identity:
            for memory in await self.service.search(
                access=access,
                query=None,
                statuses=["ACTIVE"],
                kinds=RETRIEVAL_KINDS,
                limit=_RETRIEVAL_LIMIT,
            ):
                by_id.setdefault(memory["memory_id"], memory)

        identity_ids = {m["memory_id"] for m in identity}
        merged = list(by_id.values())
        merged.sort(
            key=lambda m: (
                1 if m.get("memory_id") in identity_ids else 0,
                m.get("updated_at") or "",
                m.get("memory_id") or "",
            ),
            reverse=True,
        )
        return merged[:_RETRIEVAL_LIMIT]
