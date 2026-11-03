"""Idle external-context builder: the privacy chokepoint before an AI provider.

Local *read* access (as ``SYSTEM_ADMIN``) is distinct from *external send*
access. A memory may be handed to a remote model only when it is ``PUBLIC``,
or when **every** ``PERSON`` subject of the memory currently holds the
independent ``allow_external_ai`` grant. ``allow_memory_learning`` alone never
qualifies a memory for upload: storing a fact locally is not permission to send
it to an external model.

This builder is the single place a future real DeepSeek provider will go
through, so the mock and real paths are filtered identically. It also records
which memory ids were released and why, for audit.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from companion.memory.models import AccessContext
from companion.memory.service import MemoryService

from . import IDLE_AGENT, SYSTEM_ADMIN


class IdleMemoryContextBuilder:
    def __init__(
        self,
        memory_service: MemoryService,
        registry,
        *,
        audit=None,
        admin: AccessContext = SYSTEM_ADMIN,
        agent: AccessContext = IDLE_AGENT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.memory_service = memory_service
        self.registry = registry
        self.audit = audit
        self.admin = admin
        self.agent = agent
        self.logger = logger or logging.getLogger("companion.idle.context")

    async def external_context(
        self,
        *,
        kinds: Optional[list[str]] = None,
        statuses: tuple[str, ...] = ("ACTIVE",),
        limit: int = 50,
    ) -> list[dict]:
        """Return the memories safe to send to an external provider.

        Reads the store locally as admin, then keeps only PUBLIC memories and
        PRIVATE/SHARED memories whose every PERSON subject currently authorises
        external AI (active + ``allow_external_ai``). A private memory with no
        PERSON subject is excluded conservatively. Multi-subject memories
        require *all* subjects to authorise.
        """
        memories = await self.memory_service.search(
            access=self.admin, statuses=list(statuses), kinds=kinds, limit=limit
        )
        included: list[dict] = []
        reasons: dict[str, str] = {}
        for m in memories:
            if m["visibility"] == "PUBLIC":
                included.append(m)
                reasons[m["memory_id"]] = "public"
                continue
            person_ids = [
                s["subject_id"]
                for s in m.get("subjects", [])
                if s["subject_type"] == "PERSON"
            ]
            if not person_ids:
                # Non-person-scoped private content is never auto-exported.
                continue
            authorized = True
            for pid in person_ids:
                if not await self.registry.is_external_ai_authorized(pid):
                    authorized = False
                    break
            if authorized:
                included.append(m)
                reasons[m["memory_id"]] = "external_ai_consent"

        if self.audit is not None:
            await self.audit.record(
                actor=self.agent.actor_id,
                action="idle_external_context",
                entity_type="memory",
                detail=json.dumps(reasons, ensure_ascii=False),
            )
        self.logger.info(
            "idle_external_context released=%d considered=%d",
            len(included),
            len(memories),
        )
        return included
