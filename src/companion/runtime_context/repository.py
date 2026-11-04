"""Persistence for :class:`RuntimeSnapshot` (one row per input event)."""

from __future__ import annotations

import json
from datetime import datetime

from companion.domain.models import utcnow

from .snapshot import RuntimeSnapshot


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class RuntimeSnapshotRepository:
    def __init__(self, db) -> None:
        self.db = db

    async def get(self, event_id: str) -> RuntimeSnapshot | None:
        cur = await self.db.execute(
            "SELECT snapshot_json FROM runtime_snapshots WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["snapshot_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        try:
            return RuntimeSnapshot.model_validate(data)
        except ValueError:
            return None

    async def save_if_absent(self, snapshot: RuntimeSnapshot) -> bool:
        """Insert once; a concurrent/retried writer keeps the original row."""
        cur = await self.db.execute(
            "INSERT OR IGNORE INTO runtime_snapshots("
            "event_id, conversation_id, speaker_person_id, received_at_utc,"
            " occurred_at_utc, timezone_name, timezone_source, local_time,"
            " snapshot_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.event_id,
                snapshot.conversation_id,
                snapshot.speaker_person_id,
                _iso(snapshot.received_at_utc),
                _iso(snapshot.occurred_at_utc),
                snapshot.timezone,
                snapshot.timezone_source,
                snapshot.local_time,
                snapshot.model_dump_json(),
                _iso(utcnow()),
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def set_emotion(self, event_id: str, emotion: str | None) -> None:
        """Attach the companion's own state to the stored snapshot.

        Only the prompt-safe snapshot JSON is rewritten; the fixed timestamp
        columns (and therefore retry stability) are untouched.
        """
        if not emotion:
            return
        cur = await self.db.execute(
            "SELECT snapshot_json FROM runtime_snapshots WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return
        try:
            data = json.loads(row["snapshot_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        data["ava_emotion"] = str(emotion)
        await self.db.execute(
            "UPDATE runtime_snapshots SET snapshot_json = ? WHERE event_id = ?",
            (json.dumps(data, ensure_ascii=False), event_id),
        )


__all__ = ["RuntimeSnapshotRepository"]
