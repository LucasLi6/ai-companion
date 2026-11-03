"""Durable scene snapshots with optimistic concurrency and idempotent receipts.

Two invariants live here, not in the orchestrator:

1. **CAS on revision.** A writer may only advance the snapshot it actually
   read. If another process advanced it first, the update affects zero rows and
   :class:`SceneRevisionConflict` is raised. Newer state is never overwritten.
2. **One receipt per source event.** ``scene_transition_receipts`` is keyed by
   ``source_event_id``. A retry or crash-recovery pass can therefore tell that
   an event was already applied, even when the caller lost its in-memory
   candidate.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

from companion.cognition.scene_models import SceneState

from companion.storage.database import Database
from companion.storage.repositories import _iso, _dt


class SceneRevisionConflict(Exception):
    """Another writer advanced the scene; the caller must reload and retry."""

    def __init__(self, conversation_id: str, person_id: str, expected: int):
        super().__init__(
            "scene revision conflict for "
            f"({conversation_id}, {person_id}): expected revision {expected}"
        )
        self.conversation_id = conversation_id
        self.person_id = person_id
        self.expected = expected


@dataclass(frozen=True)
class SceneSnapshotRecord:
    conversation_id: str
    person_id: str
    revision: int
    state: SceneState
    source_event_ids: list[str]
    last_committed_event_id: Optional[str]
    updated_at: object


class SceneRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def load(
        self, conversation_id: str, person_id: str
    ) -> Optional[SceneSnapshotRecord]:
        cur = await self.db.execute(
            "SELECT conversation_id, person_id, revision, state_json,"
            " source_event_ids_json, last_committed_event_id, updated_at"
            " FROM scene_snapshots WHERE conversation_id = ? AND person_id = ?",
            (conversation_id, person_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            state = SceneState.model_validate_json(row["state_json"])
        except (ValueError, TypeError):
            return None
        try:
            source_ids = json.loads(row["source_event_ids_json"] or "[]")
        except (TypeError, ValueError):
            source_ids = []
        if not isinstance(source_ids, list):
            source_ids = []
        return SceneSnapshotRecord(
            conversation_id=row["conversation_id"],
            person_id=row["person_id"],
            revision=int(row["revision"]),
            state=state,
            source_event_ids=[str(v) for v in source_ids],
            last_committed_event_id=row["last_committed_event_id"],
            updated_at=_dt(row["updated_at"]),
        )

    async def get_receipt(self, source_event_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT source_event_id, conversation_id, person_id, result_revision,"
            " transition_json, applied_at FROM scene_transition_receipts"
            " WHERE source_event_id = ?",
            (source_event_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def apply_transition(
        self,
        *,
        source_event_id: str,
        conversation_id: str,
        person_id: str,
        expected_revision: int,
        state: SceneState,
        transition: dict,
    ) -> dict:
        """Apply one scene transition exactly once.

        Returns the receipt mapping. If a receipt already exists it is returned
        unchanged and no state is written. Raises :class:`SceneRevisionConflict`
        if the stored revision is not ``expected_revision``.
        """
        async with self.db.transaction(immediate=True):
            existing = await self.get_receipt(source_event_id)
            if existing is not None:
                return existing

            current = await self.load(conversation_id, person_id)
            now = _iso(state.updated_at)
            payload = json.dumps(state.model_dump(mode="json"), ensure_ascii=False)
            source_json = json.dumps(state.source_event_ids, ensure_ascii=False)

            if current is None:
                if expected_revision != 0:
                    raise SceneRevisionConflict(
                        conversation_id, person_id, expected_revision
                    )
                cur = await self.db.execute(
                    "INSERT OR IGNORE INTO scene_snapshots("
                    " conversation_id, person_id, revision, state_json,"
                    " source_event_ids_json, last_committed_event_id, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        conversation_id,
                        person_id,
                        state.revision,
                        payload,
                        source_json,
                        state.last_committed_event_id,
                        now,
                    ),
                )
                if cur.rowcount != 1:
                    raise SceneRevisionConflict(
                        conversation_id, person_id, expected_revision
                    )
            else:
                if current.revision != expected_revision:
                    raise SceneRevisionConflict(
                        conversation_id, person_id, expected_revision
                    )
                expected_next = expected_revision + 1
                if state.revision != expected_next:
                    raise SceneRevisionConflict(
                        conversation_id, person_id, expected_revision
                    )
                cur = await self.db.execute(
                    "UPDATE scene_snapshots SET revision = ?, state_json = ?,"
                    " source_event_ids_json = ?, last_committed_event_id = ?,"
                    " updated_at = ?"
                    " WHERE conversation_id = ? AND person_id = ? AND revision = ?",
                    (
                        state.revision,
                        payload,
                        source_json,
                        state.last_committed_event_id,
                        now,
                        conversation_id,
                        person_id,
                        expected_revision,
                    ),
                )
                if cur.rowcount != 1:
                    raise SceneRevisionConflict(
                        conversation_id, person_id, expected_revision
                    )

            return await self._insert_receipt(
                source_event_id=source_event_id,
                conversation_id=conversation_id,
                person_id=person_id,
                result_revision=state.revision,
                transition=transition,
                applied_at=now,
            )

    async def save_receipt_only(
        self,
        *,
        source_event_id: str,
        conversation_id: str,
        person_id: str,
        result_revision: int,
        transition: dict,
    ) -> dict:
        """Record a receipt for a transition already reflected in the snapshot.

        Used by crash/legacy recovery when the snapshot's
        ``last_committed_event_id`` is already this event but the receipt row is
        missing. It never modifies the scene state.
        """
        async with self.db.transaction(immediate=True):
            existing = await self.get_receipt(source_event_id)
            if existing is not None:
                return existing
            return await self._insert_receipt(
                source_event_id=source_event_id,
                conversation_id=conversation_id,
                person_id=person_id,
                result_revision=result_revision,
                transition={**transition, "recovered": True},
                applied_at=_iso(_now()),
            )

    async def _insert_receipt(
        self,
        *,
        source_event_id: str,
        conversation_id: str,
        person_id: str,
        result_revision: int,
        transition: dict,
        applied_at: str,
    ) -> dict:
        try:
            await self.db.execute(
                "INSERT INTO scene_transition_receipts("
                " source_event_id, conversation_id, person_id, result_revision,"
                " transition_json, applied_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_event_id,
                    conversation_id,
                    person_id,
                    result_revision,
                    json.dumps(transition, ensure_ascii=False),
                    applied_at,
                ),
            )
        except sqlite3.IntegrityError:
            existing = await self.get_receipt(source_event_id)
            if existing is None:
                raise
            return existing
        return {
            "source_event_id": source_event_id,
            "conversation_id": conversation_id,
            "person_id": person_id,
            "result_revision": result_revision,
            "transition_json": json.dumps(transition, ensure_ascii=False),
            "applied_at": applied_at,
        }

    async def delete_scope(self, conversation_id: str, person_id: str) -> None:
        """Remove derived scene state and receipts for a scope.

        Receipts contain the transition summary (topic/tone/phase), so they are
        derived scene data too. A privacy invalidation must remove both, not
        leave an old summary behind in the idempotency table.
        """

        async with self.db.transaction(immediate=True):
            await self.db.execute(
                "DELETE FROM scene_snapshots WHERE conversation_id = ? AND person_id = ?",
                (conversation_id, person_id),
            )
            await self.db.execute(
                "DELETE FROM scene_transition_receipts"
                " WHERE conversation_id = ? AND person_id = ?",
                (conversation_id, person_id),
            )


def _now():
    from companion.domain.models import utcnow

    return utcnow()
