"""MemoryStore abstraction and its V0.0 SQLite implementation.

V0.0 does not implement semantic vector search. :class:`SqliteMemoryStore`
returns hand-authored facts (from ``memory_facts``) via exact/substring match,
which is enough to prove the retrieval interface without pretending to
"understand" anything.
"""

from __future__ import annotations

from typing import Optional

from companion.storage.repositories import MemoryFactRepository


class SqliteMemoryStore:
    def __init__(self, repo: MemoryFactRepository) -> None:
        self.repo = repo

    async def retrieve(self, query: str, person_id: Optional[str]) -> list:
        # A blank query retrieves every fact for the person; a non-blank query
        # performs a literal substring match over fact content.
        return await self.repo.retrieve(query or None, person_id)
