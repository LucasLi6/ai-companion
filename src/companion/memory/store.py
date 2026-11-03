"""MemoryStore adapter over the unified :class:`MemoryService`.

The foreground cognitive flow reads memory through this adapter. It runs with
a *non-admin* :class:`AccessContext` for the speaking person, so it sees only
that person's own subjects plus public world knowledge — never another
person's private facts. Retrieval uses the unified store (which includes
hand-authored facts, project state, relationships, episodes and world
knowledge), not just the legacy ``memory_facts`` table.
"""

from __future__ import annotations

from typing import Optional

from companion.memory.models import AccessContext
from companion.memory.service import MemoryService


class SqliteMemoryStore:
    def __init__(self, service: MemoryService) -> None:
        self.service = service

    async def retrieve(self, query: str, person_id: Optional[str]) -> list:
        if person_id is None:
            return []
        access = AccessContext(actor_id=person_id, is_admin=False, role="cognitive")
        return await self.service.search(
            access=access,
            query=query or None,
            statuses=["ACTIVE"],
            kinds=[
                "PERSONAL_FACT",
                "EPISODE",
                "RELATIONSHIP",
                "PROJECT_STATE",
                "WORLD_KNOWLEDGE",
            ],
            limit=50,
        )
