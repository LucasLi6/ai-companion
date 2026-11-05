"""Source and permission checks for autonomous research.

A curiosity question is derived from one or more memory records. That origin
is recorded with the task and re-validated before every real network search:

- every source id must still exist and be ``ACTIVE``;
- a ``PUBLIC`` memory is safe to send;
- for any other memory, **every** ``PERSON`` subject must currently hold the
  independent ``allow_external_ai`` grant (``allow_memory_learning`` is not
  enough);
- a private memory with no ``PERSON`` subject is never exported;
- legacy tasks without source ids are refused conservatively when a real
  network provider is active.

User-submitted research (``origin='user'``) uses the query the user typed; it
is a separate source mechanism and does not read private memories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from companion.memory.models import AccessContext

SYSTEM_ADMIN = AccessContext("admin", is_admin=True, role="idle")


@dataclass(frozen=True)
class SourceDecision:
    allowed: bool
    reason: str
    mode: str


def source_ids_from_task(task: dict) -> list[str]:
    raw = task.get("source_memory_ids")
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item)]
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item)]


class ResearchSourceGuard:
    def __init__(self, memory_service, registry, *, logger=None) -> None:
        self.memory_service = memory_service
        self.registry = registry
        self.logger = logger

    async def check(
        self, task: dict, *, network_required: bool
    ) -> SourceDecision:
        origin = str(task.get("origin") or "legacy")
        if origin == "user":
            # The query is the user's own public submission, not extracted
            # private memory content.
            return SourceDecision(True, "user_submitted", "user")
        ids = source_ids_from_task(task)
        if not ids:
            if not network_required:
                # Mock/offline research may run; nothing leaves the machine.
                return SourceDecision(True, "mock_no_sources", "mock")
            if origin == "legacy":
                return SourceDecision(False, "legacy_source_unknown", "blocked")
            return SourceDecision(False, "source_unverified", "blocked")

        for memory_id in ids:
            view = await self.memory_service.get_view(memory_id, SYSTEM_ADMIN)
            if view is None:
                return SourceDecision(False, "source_unavailable", "blocked")
            item = view.get("item") or {}
            status = str(item.get("status") or "").upper()
            if status != "ACTIVE":
                return SourceDecision(False, "source_inactive", "blocked")
            visibility = str(item.get("visibility") or "").upper()
            if visibility == "PUBLIC":
                continue
            person_ids = [
                str(subject.get("subject_id"))
                for subject in (view.get("subjects") or [])
                if str(subject.get("subject_type") or "").upper() == "PERSON"
                and subject.get("subject_id")
            ]
            if not person_ids:
                return SourceDecision(
                    False, "source_private_no_subject", "blocked"
                )
            for person_id in person_ids:
                if not await self.registry.is_external_ai_authorized(person_id):
                    return SourceDecision(False, "permission_revoked", "blocked")
        return SourceDecision(True, "sources_authorized", "validated")

    async def allowed_question_sources(self, memories: list[dict]) -> set[str]:
        """The authorized source ids a curiosity proposal may cite."""
        allowed: set[str] = set()
        for memory in memories:
            memory_id = memory.get("memory_id")
            if memory_id and await self._memory_is_external_safe(memory_id):
                allowed.add(str(memory_id))
        return allowed

    async def _memory_is_external_safe(self, memory_id: str) -> bool:
        view = await self.memory_service.get_view(memory_id, SYSTEM_ADMIN)
        if view is None:
            return False
        item = view.get("item") or {}
        if str(item.get("status") or "").upper() != "ACTIVE":
            return False
        if str(item.get("visibility") or "").upper() == "PUBLIC":
            return True
        person_ids = [
            str(subject.get("subject_id"))
            for subject in (view.get("subjects") or [])
            if str(subject.get("subject_type") or "").upper() == "PERSON"
            and subject.get("subject_id")
        ]
        if not person_ids:
            return False
        for person_id in person_ids:
            if not await self.registry.is_external_ai_authorized(person_id):
                return False
        return True


__all__ = [
    "ResearchSourceGuard",
    "SourceDecision",
    "source_ids_from_task",
    "SYSTEM_ADMIN",
]
