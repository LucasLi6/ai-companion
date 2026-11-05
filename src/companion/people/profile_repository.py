"""Persistence for the per-person understanding profile (V0.7).

Plain parameterised SQL over the single CMF database. Business rules
(authorisation, revisioning, candidate confirmation) live in
:mod:`companion.people.profile_service`; this module only reads and writes the
tables added by schema migration 20.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from companion.domain.models import new_id, utcnow
from companion.storage.database import Database


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, default):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class PersonProfileRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -- main record -------------------------------------------------------

    async def ensure(
        self,
        person_id: str,
        *,
        relationship_tier: str = "STRANGER",
        actor: str = "system",
    ) -> bool:
        """Create the one row per person. Returns True when newly created."""
        now = _iso(utcnow())
        cur = await self.db.execute(
            "INSERT OR IGNORE INTO person_interaction_profiles("
            " person_id, relationship_tier, style_overrides, updated_at,"
            " compact_summary, revision, created_at, updated_by)"
            " VALUES (?, ?, '{}', ?, '', 1, ?, ?)",
            (person_id, relationship_tier, now, now, actor),
        )
        created = bool(cur.rowcount)
        # Rows created by the pre-V0.7 relationship setters have no created_at.
        await self.db.execute(
            "UPDATE person_interaction_profiles"
            " SET created_at = COALESCE(created_at, updated_at)"
            " WHERE person_id = ?",
            (person_id,),
        )
        return created

    async def get(self, person_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT person_id, relationship_tier, compact_summary, revision,"
            " created_at, updated_at, last_reviewed_at, updated_by"
            " FROM person_interaction_profiles WHERE person_id = ?",
            (person_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def touch(
        self,
        person_id: str,
        *,
        updated_by: str,
        summary: Optional[str] = None,
    ) -> int:
        """Increment the profile revision and return the new value."""
        row = await self.get(person_id)
        if row is None:
            return 0
        revision = int(row["revision"]) + 1
        now = _iso(utcnow())
        if summary is None:
            await self.db.execute(
                "UPDATE person_interaction_profiles SET revision = ?,"
                " updated_at = ?, updated_by = ? WHERE person_id = ?",
                (revision, now, updated_by, person_id),
            )
        else:
            await self.db.execute(
                "UPDATE person_interaction_profiles SET revision = ?,"
                " updated_at = ?, updated_by = ?, compact_summary = ?"
                " WHERE person_id = ?",
                (revision, now, updated_by, summary[:2000], person_id),
            )
        return revision

    async def set_summary(
        self, person_id: str, summary: str, *, updated_by: str
    ) -> None:
        await self.db.execute(
            "UPDATE person_interaction_profiles SET compact_summary = ?,"
            " updated_at = ?, updated_by = ? WHERE person_id = ?",
            (summary[:2000], _iso(utcnow()), updated_by, person_id),
        )

    async def set_reviewed(self, person_id: str) -> None:
        await self.db.execute(
            "UPDATE person_interaction_profiles SET last_reviewed_at = ?"
            " WHERE person_id = ?",
            (_iso(utcnow()), person_id),
        )

    # -- effective fields --------------------------------------------------

    async def get_field(self, person_id: str, field_key: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT person_id, field_key, value_json, source, confidence,"
            " status, evidence_memory_ids_json, evidence_event_ids_json,"
            " evidence_candidate_ids_json, revision, created_at, updated_at"
            " FROM person_profile_fields WHERE person_id = ? AND field_key = ?",
            (person_id, field_key),
        )
        row = await cur.fetchone()
        return self._field_row(row) if row is not None else None

    async def list_fields(
        self, person_id: str, *, status: str = "ACTIVE"
    ) -> list[dict]:
        cur = await self.db.execute(
            "SELECT person_id, field_key, value_json, source, confidence,"
            " status, evidence_memory_ids_json, evidence_event_ids_json,"
            " evidence_candidate_ids_json, revision, created_at, updated_at"
            " FROM person_profile_fields WHERE person_id = ? AND status = ?"
            " ORDER BY field_key",
            (person_id, status),
        )
        rows = await cur.fetchall()
        return [self._field_row(row) for row in rows]

    async def upsert_field(
        self,
        person_id: str,
        field_key: str,
        *,
        value: Any,
        source: str,
        confidence: float,
        evidence_memory_ids: list[str],
        evidence_event_ids: list[str],
        evidence_candidate_ids: list[str],
        revision: int,
    ) -> None:
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO person_profile_fields("
            " person_id, field_key, value_json, source, confidence, status,"
            " evidence_memory_ids_json, evidence_event_ids_json,"
            " evidence_candidate_ids_json, revision, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(person_id, field_key) DO UPDATE SET"
            " value_json=excluded.value_json,"
            " source=excluded.source,"
            " confidence=excluded.confidence,"
            " status='ACTIVE',"
            " evidence_memory_ids_json=excluded.evidence_memory_ids_json,"
            " evidence_event_ids_json=excluded.evidence_event_ids_json,"
            " evidence_candidate_ids_json=excluded.evidence_candidate_ids_json,"
            " revision=excluded.revision,"
            " updated_at=excluded.updated_at",
            (
                person_id,
                field_key,
                _json(value),
                source,
                float(confidence),
                _json(sorted(set(evidence_memory_ids))),
                _json(sorted(set(evidence_event_ids))),
                _json(sorted(set(evidence_candidate_ids))),
                int(revision),
                now,
                now,
            ),
        )

    async def retire_field(
        self, person_id: str, field_key: str, *, revision: int
    ) -> None:
        await self.db.execute(
            "UPDATE person_profile_fields SET status = 'RETIRED', revision = ?,"
            " updated_at = ? WHERE person_id = ? AND field_key = ?",
            (int(revision), _iso(utcnow()), person_id, field_key),
        )

    @staticmethod
    def _field_row(row) -> dict:
        return {
            "person_id": row["person_id"],
            "field_key": row["field_key"],
            "value": _loads(row["value_json"], None),
            "source": row["source"],
            "confidence": float(row["confidence"]),
            "status": row["status"],
            "evidence_memory_ids": _loads(
                row["evidence_memory_ids_json"], []
            ),
            "evidence_event_ids": _loads(row["evidence_event_ids_json"], []),
            "evidence_candidate_ids": _loads(
                row["evidence_candidate_ids_json"], []
            ),
            "revision": int(row["revision"]),
            "created_at": _dt(row["created_at"]),
            "updated_at": _dt(row["updated_at"]),
        }

    # -- candidates --------------------------------------------------------

    async def insert_candidate(
        self,
        *,
        person_id: str,
        field_key: str,
        value: Any,
        source: str,
        confidence: float,
        evidence_memory_ids: list[str],
        evidence_event_ids: list[str],
        evidence_candidate_ids: list[str],
        reason: Optional[str] = None,
    ) -> str:
        candidate_id = new_id("pfc")
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO person_profile_candidates("
            " candidate_id, person_id, field_key, candidate_value_json, source,"
            " confidence, status, evidence_memory_ids_json,"
            " evidence_event_ids_json, evidence_candidate_ids_json, reason,"
            " created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?)",
            (
                candidate_id,
                person_id,
                field_key,
                _json(value),
                source,
                float(confidence),
                _json(sorted(set(evidence_memory_ids))),
                _json(sorted(set(evidence_event_ids))),
                _json(sorted(set(evidence_candidate_ids))),
                reason,
                now,
                now,
            ),
        )
        return candidate_id

    async def get_candidate(self, candidate_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM person_profile_candidates WHERE candidate_id = ?",
            (candidate_id,),
        )
        row = await cur.fetchone()
        return self._candidate_row(row) if row is not None else None

    async def list_candidates(
        self, person_id: str, *, status: str = "PENDING"
    ) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM person_profile_candidates"
            " WHERE person_id = ? AND status = ? ORDER BY created_at ASC",
            (person_id, status),
        )
        rows = await cur.fetchall()
        return [self._candidate_row(row) for row in rows]

    async def find_pending(
        self, person_id: str, field_key: str
    ) -> list[dict]:
        return [
            candidate
            for candidate in await self.list_candidates(person_id)
            if candidate["field_key"] == field_key
        ]

    async def set_candidate_status(
        self, candidate_id: str, status: str, *, reason: Optional[str] = None
    ) -> None:
        await self.db.execute(
            "UPDATE person_profile_candidates SET status = ?, updated_at = ?,"
            " reason = COALESCE(?, reason) WHERE candidate_id = ?",
            (status, _iso(utcnow()), reason, candidate_id),
        )

    async def add_candidate_evidence(
        self,
        candidate_id: str,
        *,
        memory_ids: list[str],
        event_ids: list[str],
        candidate_ids: list[str],
    ) -> None:
        candidate = await self.get_candidate(candidate_id)
        if candidate is None:
            return
        await self.db.execute(
            "UPDATE person_profile_candidates SET"
            " evidence_memory_ids_json = ?, evidence_event_ids_json = ?,"
            " evidence_candidate_ids_json = ?, updated_at = ?"
            " WHERE candidate_id = ?",
            (
                _json(
                    sorted(
                        set(candidate["evidence_memory_ids"]) | set(memory_ids)
                    )
                ),
                _json(
                    sorted(set(candidate["evidence_event_ids"]) | set(event_ids))
                ),
                _json(
                    sorted(
                        set(candidate["evidence_candidate_ids"])
                        | set(candidate_ids)
                    )
                ),
                _iso(utcnow()),
                candidate_id,
            ),
        )

    async def reject_pending(self, person_id: str, reason: str) -> int:
        cur = await self.db.execute(
            "UPDATE person_profile_candidates SET status = 'REJECTED',"
            " updated_at = ?, reason = ? WHERE person_id = ? AND status = 'PENDING'",
            (_iso(utcnow()), reason, person_id),
        )
        return int(cur.rowcount or 0)

    @staticmethod
    def _candidate_row(row) -> dict:
        return {
            "candidate_id": row["candidate_id"],
            "person_id": row["person_id"],
            "field_key": row["field_key"],
            "candidate_value": _loads(row["candidate_value_json"], None),
            "source": row["source"],
            "confidence": float(row["confidence"]),
            "status": row["status"],
            "evidence_memory_ids": _loads(
                row["evidence_memory_ids_json"], []
            ),
            "evidence_event_ids": _loads(row["evidence_event_ids_json"], []),
            "evidence_candidate_ids": _loads(
                row["evidence_candidate_ids_json"], []
            ),
            "reason": row["reason"],
            "created_at": _dt(row["created_at"]),
            "updated_at": _dt(row["updated_at"]),
        }

    # -- revisions / evidence ---------------------------------------------

    async def insert_revision(
        self,
        *,
        person_id: str,
        revision: int,
        changes: dict,
        changed_by: str,
        source_event_id: Optional[str],
        reason: Optional[str],
    ) -> str:
        revision_id = new_id("pfr")
        await self.db.execute(
            "INSERT INTO person_profile_revisions("
            " revision_id, person_id, revision, changes_json, changed_by,"
            " source_event_id, reason, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                revision_id,
                person_id,
                int(revision),
                _json(changes),
                changed_by,
                source_event_id,
                reason,
                _iso(utcnow()),
            ),
        )
        return revision_id

    async def list_revisions(self, person_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM person_profile_revisions WHERE person_id = ?"
            " ORDER BY revision ASC",
            (person_id,),
        )
        rows = await cur.fetchall()
        return [
            {
                "revision_id": row["revision_id"],
                "person_id": row["person_id"],
                "revision": int(row["revision"]),
                "changes": _loads(row["changes_json"], {}),
                "changed_by": row["changed_by"],
                "source_event_id": row["source_event_id"],
                "reason": row["reason"],
                "created_at": _dt(row["created_at"]),
            }
            for row in rows
        ]

    async def insert_evidence(
        self,
        *,
        person_id: str,
        field_key: str,
        memory_id: Optional[str] = None,
        source_event_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        evidence_type: str = "EVENT",
    ) -> str:
        evidence_id = new_id("pfe")
        await self.db.execute(
            "INSERT INTO person_profile_evidence("
            " evidence_id, person_id, field_key, memory_id, source_event_id,"
            " candidate_id, evidence_type, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                evidence_id,
                person_id,
                field_key,
                memory_id,
                source_event_id,
                candidate_id,
                evidence_type,
                _iso(utcnow()),
            ),
        )
        return evidence_id

    async def list_evidence(self, person_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM person_profile_evidence WHERE person_id = ?"
            " ORDER BY created_at ASC",
            (person_id,),
        )
        rows = await cur.fetchall()
        return [
            {
                "evidence_id": row["evidence_id"],
                "person_id": row["person_id"],
                "field_key": row["field_key"],
                "memory_id": row["memory_id"],
                "source_event_id": row["source_event_id"],
                "candidate_id": row["candidate_id"],
                "evidence_type": row["evidence_type"],
                "created_at": _dt(row["created_at"]),
            }
            for row in rows
        ]

    async def link_candidate_memory(
        self, candidate_id: str, memory_id: str
    ) -> None:
        await self.db.execute(
            "UPDATE person_profile_evidence SET memory_id = ?"
            " WHERE candidate_id = ? AND memory_id IS NULL",
            (memory_id, candidate_id),
        )

    async def link_event_evidence(
        self, person_id: str, event_id: str, memory_id: str
    ) -> list[dict]:
        """Attach a confirmed memory to evidence recorded for one event.

        Returns the affected ``(field_key, candidate_id)`` pairs so the service
        can also refresh the denormalised evidence lists on fields/candidates.
        """
        cur = await self.db.execute(
            "SELECT evidence_id, field_key, candidate_id"
            " FROM person_profile_evidence"
            " WHERE person_id = ? AND source_event_id = ? AND memory_id IS NULL",
            (person_id, event_id),
        )
        rows = await cur.fetchall()
        affected: list[dict] = []
        for row in rows:
            await self.db.execute(
                "UPDATE person_profile_evidence SET memory_id = ?"
                " WHERE evidence_id = ?",
                (memory_id, row["evidence_id"]),
            )
            affected.append(
                {
                    "field_key": row["field_key"],
                    "candidate_id": row["candidate_id"],
                }
            )
        return affected

    async def fields_referencing_memory(self, memory_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT person_id, field_key FROM person_profile_evidence"
            " WHERE memory_id = ?",
            (memory_id,),
        )
        rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def list_all_person_ids(self) -> list[str]:
        cur = await self.db.execute(
            "SELECT person_id FROM person_interaction_profiles"
        )
        rows = await cur.fetchall()
        return [row["person_id"] for row in rows]

    # -- deletion ----------------------------------------------------------

    async def purge(self, person_id: str) -> None:
        """Delete every long-term profile artifact for one person.

        Used only when persistence/registration consent is withdrawn. The
        person row itself stays; revocation never deletes the account.
        """
        for table in (
            "person_profile_evidence",
            "person_profile_candidates",
            "person_profile_revisions",
            "person_profile_fields",
            "person_interaction_profiles",
        ):
            await self.db.execute(
                f"DELETE FROM {table} WHERE person_id = ?", (person_id,)
            )


__all__ = ["PersonProfileRepository"]
