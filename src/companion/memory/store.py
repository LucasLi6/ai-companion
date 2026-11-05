"""MemoryStore adapter over the unified :class:`MemoryService`.

The foreground cognitive flow reads memory through this adapter. It runs with
a *non-admin* :class:`AccessContext` for the speaking person, so it sees only
that person's own subjects plus public world knowledge — never another
person's private facts.

Retrieval combines three permitted sources:

- deterministic identity lookup (subject + canonical topic) for questions such
  as "what is my name?";
- keyword search (CJK/ASCII, OR semantics);
- memories *assigned* to the speaker.

Assignment is an association, not access: every assigned memory is re-checked
through :meth:`MemoryService.get_item` with the same non-admin access context.
An assignment therefore helps recall a memory the speaker could already see;
it can never reveal a private memory about someone else. Identity results now
carry their *real* subject associations (loaded from ``memory_subjects``)
instead of a fabricated ``PERSON:speaker`` entry, so the privacy gateway can
correctly drop a memory whose other subject has not authorised external AI.

Only ``ACTIVE`` memories are ever returned.
"""

from __future__ import annotations

import re
from typing import Optional

from companion.memory.models import AccessContext, MemoryItem
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
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# An unbroken CJK run longer than this is a whole clause, not a term. Storing
# it would consume one of the few keyword slots without ever matching a real
# memory (a memory is not a substring of the question).
_MAX_FULL_CJK_KEYWORD = 8

# High-frequency grammatical characters. A two-character window containing one
# ("我想", "一下", "的项") is a poor search key, so it is tried only after the
# content windows. Ordering only: none of these windows are discarded, so a
# term that happens to contain one is still searchable when the budget allows.
_CJK_FUNCTION_CHARS = frozenset(
    "我你他她它您咱们的了着过是在有无和与及跟就都也还很太更最要会能可"
    "以把被给对从到向于说想问请帮让使为而但如因所之其这那个一下上中里"
    "吗呢吧啊呀嘛呗么什怎谁哪"
)


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _cjk_keyword_candidates(token: str) -> list[str]:
    """Order a CJK run's bigrams so content terms survive the keyword budget.

    Both parities are used (start 0 and start 1), which together cover every
    adjacent two-character window exactly once; content windows are emitted
    first in text order, function-word windows afterwards. This reaches terms
    at the end of a long clause instead of spending the budget on the leading
    "我想 / 想问 / 问一" sequence.
    """
    windows: list[str] = []
    for offset in (0, 1):
        for i in range(offset, len(token) - 1, 2):
            pair = token[i : i + 2]
            if len(pair) == 2:
                windows.append(pair)

    def _rank(pair: str) -> tuple[bool, bool]:
        # Pure-CJK content windows first, then mixed/ASCII content windows,
        # then function-word windows. Ties keep text order (stable sort).
        has_function = any(ch in _CJK_FUNCTION_CHARS for ch in pair)
        pure_cjk = all(_CJK_RE.fullmatch(ch) for ch in pair)
        return (has_function, not pure_cjk)

    return sorted(windows, key=_rank)


def extract_keywords(text: str | None, max_keywords: int = _MAX_KEYWORDS) -> list[str]:
    """Split free text into a small, ordered, de-duplicated keyword list.

    Full whitespace/punctuation-delimited tokens come first, except that a long
    unbroken CJK clause is never used whole (it cannot be a stored substring).
    For CJK runs of three or more characters, two-character windows are added
    as a segmenter-free fallback, content windows before function-word ones, so
    a clause like "我想问一下关于机器学习的项目进度" still searches "机器",
    "项目" and "进度". No vector search, no segmenter — both are explicitly out
    of scope.
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
        if _has_cjk(token) and len(token) > _MAX_FULL_CJK_KEYWORD:
            # Do not lose an embedded ASCII term ("Python") just because it is
            # glued to a long Chinese run.
            for ascii_token in re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]*", token):
                if _add(ascii_token):
                    return out
            continue
        if _add(token):
            return out

    # Interleave the per-token candidate lists so one long clause cannot take
    # the whole remaining budget before another token is considered.
    candidate_lists = [
        _cjk_keyword_candidates(token)
        for token in tokens
        if _has_cjk(token) and 3 <= len(token) <= 32
    ]
    for position in range(max((len(c) for c in candidate_lists), default=0)):
        for candidates in candidate_lists:
            if position < len(candidates) and _add(candidates[position]):
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
    def __init__(self, service: MemoryService, hybrid=None) -> None:
        self.service = service
        # Optional hybrid (vector + keyword) retriever. ``None`` preserves the
        # legacy keyword/identity-only behaviour exactly.
        self.hybrid = hybrid

    async def retrieve(self, query: str, person_id: Optional[str]) -> list[dict]:
        """Retrieve the person's permitted ``ACTIVE`` memories.

        Identity questions ("我叫什么名字？") are resolved *deterministically*
        by subject + canonical topic before keyword search, so the answer does
        not depend on the question sharing wording with the stored fact. All
        other queries use keyword matching (OR semantics). Assigned memories
        are included only after the same visibility check. When a hybrid
        retriever is wired in, the keyword results are fused with vector
        similarity (RRF) *after* the same access checks; identity matches still
        lead. Results are de-duplicated, most recently updated first, and
        capped.
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
                result = await self._visible_result(row["memory_id"], access)
                if result is not None:
                    identity.append(result)

        keywords = extract_keywords(query)

        by_id: dict[str, dict] = {
            memory["memory_id"]: memory for memory in identity
        }
        # How many distinct query keywords each memory matched. Used only for
        # ordering: a memory matching "机器/学习/项目" is more relevant than one
        # that merely shares "今天", and the caller only sees the first few.
        match_counts: dict[str, int] = {}
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
                    memory_id = memory["memory_id"]
                    match_counts[memory_id] = match_counts.get(memory_id, 0) + 1
                    by_id.setdefault(memory_id, memory)
        elif not identity:
            for memory in await self.service.search(
                access=access,
                query=None,
                statuses=["ACTIVE"],
                kinds=RETRIEVAL_KINDS,
                limit=_RETRIEVAL_LIMIT,
            ):
                by_id.setdefault(memory["memory_id"], memory)

        # Associations help recall, never grant access. Each id is re-checked
        # through the access-controlled service before being added.
        for memory_id in await self.service.repo.list_assigned_memory_ids(person_id):
            if memory_id in by_id:
                continue
            result = await self._visible_result(memory_id, access)
            if result is not None and result["status"] == "ACTIVE":
                by_id[memory_id] = result

        identity_ids = {m["memory_id"] for m in identity}
        merged = list(by_id.values())
        merged.sort(
            key=lambda m: (
                1 if m.get("memory_id") in identity_ids else 0,
                match_counts.get(m.get("memory_id"), 0),
                m.get("updated_at") or "",
                m.get("memory_id") or "",
            ),
            reverse=True,
        )
        merged = merged[:_RETRIEVAL_LIMIT]

        if self.hybrid is not None:
            return await self.hybrid.retrieve(
                query,
                keyword_results=merged,
                access=access,
                identity_ids=identity_ids,
                limit=_RETRIEVAL_LIMIT,
            )
        return merged

    async def browse(self, person_id: Optional[str], *, limit: int = 3) -> list[dict]:
        """Any visible ``ACTIVE`` memories, newest first, with no keyword.

        Used only when the speaker explicitly asks to see a memory sample. The
        same access-checked service path as retrieval applies, so a sample can
        never reveal a memory this person may not see.
        """
        if person_id is None:
            return []
        access = AccessContext(actor_id=person_id, is_admin=False, role="cognitive")
        return await self.service.search(
            access=access,
            query=None,
            statuses=["ACTIVE"],
            kinds=RETRIEVAL_KINDS,
            limit=max(1, limit),
        )

    async def _visible_result(
        self, memory_id: str, access: AccessContext
    ) -> dict | None:
        """Load one memory's real associations, applying access control."""
        item = await self.service.get_item(memory_id, access)
        if item is None:
            return None
        subjects = await self.service.repo.get_subjects(memory_id)
        sources = await self.service.repo.get_sources(memory_id)
        return self._result(item, subjects, sources)

    @staticmethod
    def _result(
        item: MemoryItem, subjects: list[tuple[str, str]], sources: list[dict]
    ) -> dict:
        return {
            "memory_id": item.memory_id,
            "kind": item.kind.value,
            "content": item.content,
            "status": item.status.value,
            "visibility": item.visibility.value,
            "created_by": item.created_by,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "version": item.version,
            "human_confirmed": item.human_confirmed,
            "source_supported": item.source_supported,
            "self_reported": item.self_reported,
            "third_party_reported": item.third_party_reported,
            "model_inferred": item.model_inferred,
            "subjects": [
                {"subject_type": subject_type, "subject_id": subject_id}
                for subject_type, subject_id in subjects
            ],
            "sources": sources,
        }
