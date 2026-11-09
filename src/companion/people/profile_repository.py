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

    async def bump_revision(
        self,
        person_id: str,
        *,
        expected_revision: int,
        updated_by: str,
        summary: Optional[str] = None,
        mark_reviewed: bool = True,
    ) -> int:
        """Optimistic-concurrency revision bump.

        Returns the new revision, or ``0`` when the row no longer matches
        ``expected_revision``. The conditional ``WHERE revision = ?`` is the
        only place a stale review may lose: it cannot overwrite a newer commit.
        """
        now = _iso(utcnow())
        new_revision = int(expected_revision) + 1
        if summary is None:
            cur = await self.db.execute(
                "UPDATE person_interaction_profiles SET revision = ?,"
                " updated_at = ?, updated_by = ?,"
                " last_reviewed_at = CASE WHEN ? THEN ? ELSE last_reviewed_at END"
                " WHERE person_id = ? AND revision = ?",
                (
                    new_revision,
                    now,
                    updated_by,
                    1 if mark_reviewed else 0,
                    now,
                    person_id,
                    int(expected_revision),
                ),
            )
        else:
            cur = await self.db.execute(
                "UPDATE person_interaction_profiles SET revision = ?,"
                " updated_at = ?, updated_by = ?, compact_summary = ?,"
                " last_reviewed_at = CASE WHEN ? THEN ? ELSE last_reviewed_at END"
                " WHERE person_id = ? AND revision = ?",
                (
                    new_revision,
                    now,
                    updated_by,
                    summary[:2000],
                    1 if mark_reviewed else 0,
                    now,
                    person_id,
                    int(expected_revision),
                ),
            )
        return new_revision if (cur.rowcount or 0) == 1 else 0

    # -- independent understanding entries (V0.9.7) ------------------------

    _ENTRY_COLUMNS = (
        "entry_id, person_id, field_key, content, status, source, confidence,"
        " evidence_event_ids_json, evidence_memory_ids_json, merged_from_json,"
        " superseded_by, valid_from, valid_until, created_at, updated_at"
    )

    async def get_entry(self, entry_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            f"SELECT {self._ENTRY_COLUMNS} FROM person_profile_entries"
            " WHERE entry_id = ?",
            (entry_id,),
        )
        row = await cur.fetchone()
        return self._entry_row(row) if row is not None else None

    async def list_entries(
        self,
        person_id: str,
        *,
        statuses: Optional[tuple[str, ...]] = None,
        field_key: Optional[str] = None,
    ) -> list[dict]:
        sql = (
            f"SELECT {self._ENTRY_COLUMNS} FROM person_profile_entries"
            " WHERE person_id = ?"
        )
        params: list[Any] = [person_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            sql += f" AND status IN ({placeholders})"
            params.extend(statuses)
        if field_key is not None:
            sql += " AND field_key = ?"
            params.append(field_key)
        sql += " ORDER BY created_at ASC, entry_id ASC"
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [self._entry_row(row) for row in rows]

    async def insert_entry(
        self,
        *,
        person_id: str,
        field_key: str,
        content: str,
        status: str,
        source: str,
        confidence: float,
        evidence_event_ids: list[str],
        evidence_memory_ids: list[str],
        merged_from: Optional[list[str]] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
    ) -> str:
        entry_id = new_id("pfe")
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO person_profile_entries("
            " entry_id, person_id, field_key, content, status, source,"
            " confidence, evidence_event_ids_json, evidence_memory_ids_json,"
            " merged_from_json, superseded_by, valid_from, valid_until,"
            " created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
            (
                entry_id,
                person_id,
                field_key,
                content,
                status,
                source,
                float(confidence),
                _json(sorted(set(evidence_event_ids))),
                _json(sorted(set(evidence_memory_ids))),
                _json(sorted(set(merged_from or []))),
                _iso(valid_from) if valid_from is not None else None,
                _iso(valid_until) if valid_until is not None else None,
                now,
                now,
            ),
        )
        return entry_id

    async def replace_entry_evidence(
        self,
        entry_id: str,
        *,
        evidence_event_ids: list[str],
        evidence_memory_ids: list[str],
    ) -> None:
        """Set the entry's evidence lists exactly (used by source invalidation)."""
        await self.db.execute(
            "UPDATE person_profile_entries SET evidence_event_ids_json = ?,"
            " evidence_memory_ids_json = ?, updated_at = ? WHERE entry_id = ?",
            (
                _json(sorted(set(evidence_event_ids))),
                _json(sorted(set(evidence_memory_ids))),
                _iso(utcnow()),
                entry_id,
            ),
        )

    async def delete_entry(self, entry_id: str) -> None:
        await self.db.execute(
            "DELETE FROM person_profile_entries WHERE entry_id = ?", (entry_id,)
        )
        await self.db.execute(
            "DELETE FROM person_profile_fact_index WHERE entry_id = ?",
            (entry_id,),
        )

    async def upsert_fact_index(self, row: dict) -> None:
        """Write one derived fact-index row.

        The index is a recall cache, never an authority. Callers must still
        read and validate the real entry before any write.
        """
        await self.db.execute(
            "INSERT INTO person_profile_fact_index("
            " entry_id, person_id, entity_key, attribute, temporal_scope,"
            " canonical_value, aliases_json, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(entry_id) DO UPDATE SET"
            " person_id = excluded.person_id,"
            " entity_key = excluded.entity_key,"
            " attribute = excluded.attribute,"
            " temporal_scope = excluded.temporal_scope,"
            " canonical_value = excluded.canonical_value,"
            " aliases_json = excluded.aliases_json,"
            " updated_at = excluded.updated_at",
            (
                row["entry_id"],
                row["person_id"],
                str(row.get("entity_key") or "self"),
                str(row.get("attribute") or ""),
                str(row.get("temporal_scope") or "STABLE"),
                str(row.get("canonical_value") or ""),
                _json(list(row.get("aliases_json") or [])),
                row.get("updated_at") or _iso(utcnow()),
            ),
        )

    async def delete_fact_index(self, entry_id: str) -> None:
        await self.db.execute(
            "DELETE FROM person_profile_fact_index WHERE entry_id = ?",
            (entry_id,),
        )

    async def list_fact_index(
        self, person_id: str, *, limit: int = 200
    ) -> list[dict]:
        cur = await self.db.execute(
            "SELECT entry_id, person_id, entity_key, attribute, temporal_scope,"
            " canonical_value, aliases_json, updated_at"
            " FROM person_profile_fact_index WHERE person_id = ?"
            " ORDER BY updated_at DESC, entry_id ASC LIMIT ?",
            (person_id, max(1, int(limit))),
        )
        rows = await cur.fetchall()
        return [
            {
                "entry_id": row["entry_id"],
                "person_id": row["person_id"],
                "entity_key": row["entity_key"],
                "attribute": row["attribute"],
                "temporal_scope": row["temporal_scope"],
                "canonical_value": row["canonical_value"],
                "aliases": _loads(row["aliases_json"], []),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def rebuild_fact_index(self, person_id: str) -> int:
        """Drop and rebuild the derived index from effective entries."""
        await self.db.execute(
            "DELETE FROM person_profile_fact_index WHERE person_id = ?",
            (person_id,),
        )
        entries = await self.list_entries(
            person_id,
            statuses=(
                "ACTIVE",
                "TENTATIVE",
            ),
        )
        for item in entries:
            field_key = str(item.get("field_key") or "")
            attribute = field_key.rsplit(".", 1)[-1] or field_key
            await self.upsert_fact_index(
                {
                    "entry_id": item["entry_id"],
                    "person_id": person_id,
                    "entity_key": "self",
                    "attribute": attribute,
                    "temporal_scope": "STABLE",
                    "canonical_value": " ".join(
                        str(item.get("content") or "").split()
                    ).casefold(),
                    "aliases_json": [],
                    "updated_at": _iso(utcnow()),
                }
            )
        return len(entries)

    async def update_entry(
        self,
        entry_id: str,
        *,
        content: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence_event_ids: Optional[list[str]] = None,
        evidence_memory_ids: Optional[list[str]] = None,
        merged_from: Optional[list[str]] = None,
        superseded_by: Optional[str] = None,
        valid_until: Optional[datetime] = None,
    ) -> None:
        entry = await self.get_entry(entry_id)
        if entry is None:
            return
        await self.db.execute(
            "UPDATE person_profile_entries SET content = ?, status = ?,"
            " source = ?, confidence = ?, evidence_event_ids_json = ?,"
            " evidence_memory_ids_json = ?, merged_from_json = ?,"
            " superseded_by = ?, valid_until = ?, updated_at = ?"
            " WHERE entry_id = ?",
            (
                content if content is not None else entry["content"],
                status if status is not None else entry["status"],
                source if source is not None else entry["source"],
                float(confidence)
                if confidence is not None
                else entry["confidence"],
                _json(
                    sorted(
                        set(entry["evidence_event_ids"])
                        | set(evidence_event_ids or [])
                    )
                ),
                _json(
                    sorted(
                        set(entry["evidence_memory_ids"])
                        | set(evidence_memory_ids or [])
                    )
                ),
                _json(
                    sorted(set(entry["merged_from"]) | set(merged_from or []))
                ),
                superseded_by
                if superseded_by is not None
                else entry["superseded_by"],
                _iso(valid_until)
                if valid_until is not None
                else (
                    _iso(entry["valid_until"])
                    if entry["valid_until"] is not None
                    else None
                ),
                _iso(utcnow()),
                entry_id,
            ),
        )

    @staticmethod
    def _entry_row(row) -> dict:
        return {
            "entry_id": row["entry_id"],
            "person_id": row["person_id"],
            "field_key": row["field_key"],
            "content": row["content"],
            "status": row["status"],
            "source": row["source"],
            "confidence": float(row["confidence"]),
            "evidence_event_ids": _loads(row["evidence_event_ids_json"], []),
            "evidence_memory_ids": _loads(row["evidence_memory_ids_json"], []),
            "merged_from": _loads(row["merged_from_json"], []),
            "superseded_by": row["superseded_by"],
            "valid_from": _dt(row["valid_from"]),
            "valid_until": _dt(row["valid_until"]),
            "created_at": _dt(row["created_at"]),
            "updated_at": _dt(row["updated_at"]),
        }

    # -- review receipts (V0.9.7) ------------------------------------------

    async def ensure_review_receipt(
        self,
        *,
        person_id: str,
        source_event_id: str,
        job_key: str,
    ) -> str:
        """Create (or return) the one receipt for a person's event.

        The unique ``(person_id, source_event_id)`` constraint makes a retried
        turn commit idempotent: it can never queue a second review of the same
        event.
        """
        review_id = new_id("pfrv")
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT OR IGNORE INTO profile_review_receipts("
            " review_id, person_id, source_event_id, job_key, status,"
            " created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'PENDING', ?, ?)",
            (review_id, person_id, source_event_id, job_key, now, now),
        )
        cur = await self.db.execute(
            "SELECT review_id FROM profile_review_receipts"
            " WHERE person_id = ? AND source_event_id = ?",
            (person_id, source_event_id),
        )
        row = await cur.fetchone()
        return row["review_id"] if row is not None else review_id

    async def get_review_receipt(
        self, person_id: str, source_event_id: str
    ) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM profile_review_receipts"
            " WHERE person_id = ? AND source_event_id = ?",
            (person_id, source_event_id),
        )
        row = await cur.fetchone()
        return self._receipt_row(row) if row is not None else None

    async def get_review_receipt_by_id(self, review_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM profile_review_receipts WHERE review_id = ?",
            (review_id,),
        )
        row = await cur.fetchone()
        return self._receipt_row(row) if row is not None else None

    async def list_review_receipts(
        self,
        person_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        sql = (
            "SELECT * FROM profile_review_receipts WHERE person_id = ?"
        )
        params: list[Any] = [person_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at ASC, rowid ASC LIMIT ?"
        params.append(max(1, int(limit)))
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [self._receipt_row(row) for row in rows]

    async def claim_review_receipt(
        self, review_id: str, execution_owner_id: Optional[str] = None
    ) -> bool:
        """Atomically take ownership of one pending review.

        Two workers may see the same pending list; only one conditional
        ``UPDATE`` can move a row out of ``PENDING``. When the caller passes
        its execution token, the row records that token and a fencing epoch so
        every later transition can be scoped to this exact execution.
        """
        cur = await self.db.execute(
            "UPDATE profile_review_receipts SET status = 'RUNNING',"
            " execution_owner_id = ?, lease_epoch = lease_epoch + 1,"
            " updated_at = ? WHERE review_id = ? AND status = 'PENDING'",
            (execution_owner_id, _iso(utcnow()), review_id),
        )
        return (cur.rowcount or 0) == 1

    async def recover_stale_review_receipts(
        self, person_id: str, *, older_than_seconds: int = 900
    ) -> int:
        """Return a crashed worker's claimed receipt to the pending queue."""
        from datetime import timedelta

        cutoff = _iso(utcnow() - timedelta(seconds=max(60, older_than_seconds)))
        cur = await self.db.execute(
            "UPDATE profile_review_receipts SET status = 'PENDING',"
            " execution_owner_id = NULL,"
            " last_error = 'stale_runner_recovered', updated_at = ?"
            " WHERE person_id = ? AND status = 'RUNNING' AND updated_at < ?",
            (_iso(utcnow()), person_id, cutoff),
        )
        return int(cur.rowcount or 0)

    async def latest_review_receipt(self, person_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM profile_review_receipts WHERE person_id = ?"
            " ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (person_id,),
        )
        row = await cur.fetchone()
        return self._receipt_row(row) if row is not None else None

    async def list_unfinished_review_receipts(
        self, person_id: Optional[str] = None, *, limit: int = 200
    ) -> list[dict]:
        """Receipts whose outcome is still unknown (PENDING or RUNNING)."""
        sql = (
            "SELECT * FROM profile_review_receipts"
            " WHERE status IN ('PENDING', 'RUNNING')"
        )
        params: list[Any] = []
        if person_id is not None:
            sql += " AND person_id = ?"
            params.append(person_id)
        sql += " ORDER BY created_at ASC, rowid ASC LIMIT ?"
        params.append(max(1, int(limit)))
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [self._receipt_row(row) for row in rows]

    async def release_review_receipt(
        self,
        review_id: str,
        *,
        reason: str,
        execution_owner_id: Optional[str] = None,
    ) -> bool:
        """Return a claimed receipt to PENDING after a lost execution lease.

        Conditional on the receipt still being ``RUNNING`` *and*, when a token
        is supplied, on that token still owning the row: a stale worker can
        never reopen a receipt a newer execution has claimed.
        """
        sql = (
            "UPDATE profile_review_receipts SET status = 'PENDING',"
            " execution_owner_id = NULL, last_error = ?, updated_at = ?"
            " WHERE review_id = ? AND status = 'RUNNING'"
        )
        params: list[Any] = [reason[:500], _iso(utcnow()), review_id]
        if execution_owner_id is not None:
            sql += " AND execution_owner_id = ?"
            params.append(execution_owner_id)
        cur = await self.db.execute(sql, tuple(params))
        return int(cur.rowcount or 0) == 1

    async def prepare_review_receipt_retry(
        self, review_id: str, *, job_key: str
    ) -> bool:
        """Clear recovery state for a manual retry with a fresh job key."""
        now = _iso(utcnow())
        cur = await self.db.execute(
            "UPDATE profile_review_receipts SET status = 'PENDING',"
            " job_key = ?, base_revision = NULL, result_revision = NULL,"
            " decision = NULL, reason = NULL, changes_json = '[]',"
            " attempt_count = 0, last_error = NULL, completed_at = NULL,"
            " execution_owner_id = NULL, updated_at = ?"
            " WHERE review_id = ? AND status != 'COMPLETED'",
            (job_key, now, review_id),
        )
        return int(cur.rowcount or 0) == 1

    async def cancel_review_receipts_for_source(
        self, person_id: str, source_event_id: str
    ) -> int:
        """Cancel any review of an event that is being privacy-deleted."""
        cur = await self.db.execute(
            "UPDATE profile_review_receipts SET status = 'CANCELLED',"
            " decision = 'CANCELLED', reason = 'source_event_deleted',"
            " execution_owner_id = NULL, completed_at = ?, updated_at = ?"
            " WHERE person_id = ? AND source_event_id = ?"
            " AND status IN ('PENDING', 'RUNNING')",
            (
                _iso(utcnow()),
                _iso(utcnow()),
                person_id,
                source_event_id,
            ),
        )
        return int(cur.rowcount or 0)

    async def scrub_receipts_for_source(
        self, person_id: str, source_event_id: str
    ) -> int:
        """Redact receipt bodies derived from a permanently deleted event."""
        cur = await self.db.execute(
            "UPDATE profile_review_receipts SET changes_json = '[]',"
            " reason = COALESCE(reason, 'source_event_deleted'), updated_at = ?"
            " WHERE person_id = ? AND source_event_id = ?"
            " AND changes_json NOT IN ('[]', '')",
            (_iso(utcnow()), person_id, source_event_id),
        )
        return int(cur.rowcount or 0)

    async def scrub_receipts_for_entries(
        self, person_id: str, entry_ids: list[str]
    ) -> int:
        affected = 0
        for entry_id in dict.fromkeys(entry_ids or []):
            cur = await self.db.execute(
                "UPDATE profile_review_receipts SET changes_json = '[]',"
                " updated_at = ? WHERE person_id = ?"
                " AND changes_json LIKE ?",
                (_iso(utcnow()), person_id, f"%{entry_id}%"),
            )
            affected += int(cur.rowcount or 0)
        return affected

    async def update_review_receipt(
        self,
        review_id: str,
        *,
        status: Optional[str] = None,
        base_revision: Optional[int] = None,
        result_revision: Optional[int] = None,
        decision: Optional[str] = None,
        reason: Optional[str] = None,
        changes: Optional[list[dict]] = None,
        increment_attempt: bool = False,
        last_error: Optional[str] = None,
        completed: bool = False,
        execution_owner_id: Optional[str] = None,
        applied_count: Optional[int] = None,
        rejected_count: Optional[int] = None,
    ) -> bool:
        """Update a receipt, optionally only while ``execution_owner_id`` owns it.

        A worker passing its token can never overwrite a receipt that a newer
        execution has since claimed. A terminal transition also clears the
        ownership token so the row is visibly finished.
        """
        receipt = await self.get_review_receipt_by_id(review_id)
        if receipt is None:
            return False
        now = _iso(utcnow())
        next_status = status if status is not None else receipt["status"]
        clear_owner = next_status in (
            "COMPLETED",
            "SKIPPED",
            "FAILED",
            "CANCELLED",
            "PENDING",
        )
        sql = (
            "UPDATE profile_review_receipts SET status = ?, base_revision = ?,"
            " result_revision = ?, decision = ?, reason = ?, changes_json = ?,"
            " attempt_count = ?, last_error = ?, updated_at = ?,"
            " completed_at = ?, execution_owner_id = ?, applied_count = ?,"
            " rejected_count = ? WHERE review_id = ?"
        )
        params: list[Any] = [
            next_status,
            base_revision
            if base_revision is not None
            else receipt["base_revision"],
            result_revision
            if result_revision is not None
            else receipt["result_revision"],
            decision if decision is not None else receipt["decision"],
            reason if reason is not None else receipt["reason"],
            _json(changes) if changes is not None else _json(receipt["changes"]),
            receipt["attempt_count"] + (1 if increment_attempt else 0),
            last_error if last_error is not None else receipt["last_error"],
            now,
            now if completed else receipt["completed_at"],
            None if clear_owner else receipt.get("execution_owner_id"),
            applied_count
            if applied_count is not None
            else int(receipt.get("applied_count") or 0),
            rejected_count
            if rejected_count is not None
            else int(receipt.get("rejected_count") or 0),
            review_id,
        ]
        if execution_owner_id is not None:
            sql += " AND execution_owner_id = ?"
            params.append(execution_owner_id)
        cur = await self.db.execute(sql, tuple(params))
        return int(cur.rowcount or 0) == 1

    @staticmethod
    def _receipt_row(row) -> dict:
        return {
            "review_id": row["review_id"],
            "person_id": row["person_id"],
            "source_event_id": row["source_event_id"],
            "job_key": row["job_key"],
            "status": row["status"],
            "base_revision": row["base_revision"],
            "result_revision": row["result_revision"],
            "decision": row["decision"],
            "reason": row["reason"],
            "changes": _loads(row["changes_json"], []),
            "applied_count": (
                int(row["applied_count"] or 0)
                if "applied_count" in row.keys()
                else 0
            ),
            "rejected_count": (
                int(row["rejected_count"] or 0)
                if "rejected_count" in row.keys()
                else 0
            ),
            "attempt_count": int(row["attempt_count"] or 0),
            "last_error": row["last_error"],
            "execution_owner_id": (
                row["execution_owner_id"]
                if "execution_owner_id" in row.keys()
                else None
            ),
            "lease_epoch": (
                int(row["lease_epoch"] or 0)
                if "lease_epoch" in row.keys()
                else 0
            ),
            "created_at": _dt(row["created_at"]),
            "updated_at": _dt(row["updated_at"]),
            "completed_at": _dt(row["completed_at"]),
        }

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

    async def redact_field(
        self, person_id: str, field_key: str, *, revision: int
    ) -> None:
        """Permanently remove a legacy field's value after a privacy delete.

        Retiring alone keeps ``value_json`` in the file, which would leave the
        private text recoverable. The row stays (history is continuous) but its
        value, source and evidence are cleared.
        """
        await self.db.execute(
            "UPDATE person_profile_fields SET status = 'RETIRED',"
            " value_json = '{\"redacted\": \"privacy_delete\"}',"
            " source = 'PRIVACY_DELETE', confidence = 0,"
            " evidence_memory_ids_json = '[]', evidence_event_ids_json = '[]',"
            " evidence_candidate_ids_json = '[]', revision = ?, updated_at = ?"
            " WHERE person_id = ? AND field_key = ?",
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

    async def scrub_revisions_for_source(
        self, person_id: str, source_event_id: str
    ) -> int:
        """Redact revision bodies that were derived from a deleted event.

        The row keeps its revision number so history stays continuous; its
        ``changes_json`` no longer retains the private before/after text.
        """
        cur = await self.db.execute(
            "UPDATE person_profile_revisions"
            " SET changes_json = '{\"redacted\": \"privacy_delete\"}'"
            " WHERE person_id = ? AND source_event_id = ?"
            " AND changes_json NOT LIKE '%\"redacted\"%'",
            (person_id, source_event_id),
        )
        return int(cur.rowcount or 0)

    async def scrub_revisions_for_entries(
        self, person_id: str, entry_ids: list[str]
    ) -> int:
        affected = 0
        for entry_id in dict.fromkeys(entry_ids or []):
            cur = await self.db.execute(
                "UPDATE person_profile_revisions"
                " SET changes_json = '{\"redacted\": \"privacy_delete\"}'"
                " WHERE person_id = ? AND changes_json LIKE ?"
                " AND changes_json NOT LIKE '%\"redacted\"%'",
                (person_id, f"%{entry_id}%"),
            )
            affected += int(cur.rowcount or 0)
        return affected

    async def delete_evidence_for_source(
        self,
        person_id: str,
        *,
        event_id: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> int:
        if event_id:
            cur = await self.db.execute(
                "DELETE FROM person_profile_evidence"
                " WHERE person_id = ? AND source_event_id = ?",
                (person_id, event_id),
            )
        elif memory_id:
            cur = await self.db.execute(
                "DELETE FROM person_profile_evidence"
                " WHERE person_id = ? AND memory_id = ?",
                (person_id, memory_id),
            )
        else:
            return 0
        return int(cur.rowcount or 0)

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

    # -- per-person review lease (V0.9.7.1) --------------------------------

    async def acquire_review_lease(
        self,
        person_id: str,
        owner_id: str,
        *,
        lease_seconds: int = 900,
    ) -> bool:
        """Atomically take the one review slot for ``person_id``.

        A row whose lease has expired (a crashed worker) may be taken over; a
        live lease held by a different owner never can. SQLite serialises the
        conditional upsert, so this works across worker processes — not just
        within one event loop.
        """
        from datetime import timedelta

        now = utcnow()
        expires = now + timedelta(seconds=max(30, int(lease_seconds)))
        cur = await self.db.execute(
            "INSERT INTO profile_review_leases("
            " person_id, owner_id, acquired_at, expires_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(person_id) DO UPDATE SET"
            " owner_id = excluded.owner_id,"
            " acquired_at = excluded.acquired_at,"
            " expires_at = excluded.expires_at,"
            " updated_at = excluded.updated_at"
            " WHERE profile_review_leases.expires_at < ?"
            " OR profile_review_leases.owner_id = ?",
            (
                person_id,
                owner_id,
                _iso(now),
                _iso(expires),
                _iso(now),
                _iso(now),
                owner_id,
            ),
        )
        return int(cur.rowcount or 0) == 1

    async def renew_review_lease(
        self,
        person_id: str,
        owner_id: str,
        *,
        lease_seconds: int = 900,
    ) -> bool:
        """Extend a lease this owner still holds; fail after it expired."""
        from datetime import timedelta

        now = utcnow()
        expires = now + timedelta(seconds=max(30, int(lease_seconds)))
        cur = await self.db.execute(
            "UPDATE profile_review_leases SET expires_at = ?, updated_at = ?"
            " WHERE person_id = ? AND owner_id = ? AND expires_at >= ?",
            (_iso(expires), _iso(now), person_id, owner_id, _iso(now)),
        )
        return int(cur.rowcount or 0) == 1

    async def release_review_lease(self, person_id: str, owner_id: str) -> bool:
        """Release only the lease this owner still holds."""
        cur = await self.db.execute(
            "DELETE FROM profile_review_leases"
            " WHERE person_id = ? AND owner_id = ?",
            (person_id, owner_id),
        )
        return int(cur.rowcount or 0) == 1

    async def get_review_lease(self, person_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT person_id, owner_id, acquired_at, expires_at, updated_at"
            " FROM profile_review_leases WHERE person_id = ?",
            (person_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {
            "person_id": row["person_id"],
            "owner_id": row["owner_id"],
            "acquired_at": _dt(row["acquired_at"]),
            "expires_at": _dt(row["expires_at"]),
            "updated_at": _dt(row["updated_at"]),
        }

    async def list_all_person_ids(self) -> list[str]:
        cur = await self.db.execute(
            "SELECT person_id FROM person_interaction_profiles"
        )
        rows = await cur.fetchall()
        return [row["person_id"] for row in rows]

    # -- legacy candidate migration (read-only archive) --------------------

    async def list_legacy_candidates(self, person_id: str) -> list[dict]:
        """Read rows from the retired candidate table for the one-time tool.

        Runtime code never calls this. The rows are kept only so the explicit
        ``profile migrate-candidates`` command can revalidate them.
        """
        cur = await self.db.execute(
            "SELECT * FROM person_profile_candidates_legacy"
            " WHERE person_id = ? ORDER BY created_at ASC, candidate_id ASC",
            (person_id,),
        )
        return [dict(row) for row in await cur.fetchall()]

    async def get_candidate_migration(self, candidate_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM profile_candidate_migrations WHERE candidate_id = ?",
            (candidate_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def record_candidate_migration(
        self,
        *,
        candidate_id: str,
        person_id: str,
        field_key: str,
        operation: str,
        candidate_value: Any,
        source_event_id: Optional[str],
        status: str,
        reason: Optional[str] = None,
        entry_id: Optional[str] = None,
        revision: Optional[int] = None,
    ) -> None:
        """Record the terminal outcome for one old candidate (idempotent)."""
        await self.db.execute(
            "INSERT OR IGNORE INTO profile_candidate_migrations("
            " migration_id, candidate_id, person_id, field_key, operation,"
            " candidate_value_json, source_event_id, status, reason, entry_id,"
            " revision, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id("pcm"),
                candidate_id,
                person_id,
                field_key,
                operation,
                _json(candidate_value),
                source_event_id,
                status,
                (reason or "")[:500] or None,
                entry_id,
                int(revision) if revision is not None else None,
                _iso(utcnow()),
            ),
        )

    # -- deletion ----------------------------------------------------------

    async def purge(self, person_id: str) -> None:
        """Delete every long-term profile artifact for one person.

        Used only when persistence/registration consent is withdrawn. The
        person row itself stays; revocation never deletes the account.
        """
        for table in (
            "person_profile_evidence",
            "person_profile_candidates_legacy",
            "profile_candidate_migrations",
            "profile_confirmations",
            "person_profile_revisions",
            "person_profile_fields",
            "person_profile_entries",
            "profile_review_receipts",
            "person_interaction_profiles",
        ):
            await self.db.execute(
                f"DELETE FROM {table} WHERE person_id = ?", (person_id,)
            )

    async def delete_legacy_candidates_for_source(
        self,
        person_id: str,
        *,
        event_id: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> int:
        """Remove legacy candidate rows that cite a privacy-deleted source.

        The legacy table is never used by runtime code. It still contains user
        text, so a deletion must remove anything derived from that source
        before the table is retired by the migration command.
        """
        cur = await self.db.execute(
            "SELECT candidate_id, evidence_event_ids_json,"
            " evidence_memory_ids_json FROM person_profile_candidates_legacy"
            " WHERE person_id = ?",
            (person_id,),
        )
        removed = 0
        for row in await cur.fetchall():
            events = _loads(row["evidence_event_ids_json"], [])
            memories = _loads(row["evidence_memory_ids_json"], [])
            if (event_id and event_id in events) or (
                memory_id and memory_id in memories
            ):
                await self.db.execute(
                    "DELETE FROM person_profile_candidates_legacy"
                    " WHERE candidate_id = ?",
                    (row["candidate_id"],),
                )
                removed += 1
        return removed


__all__ = ["PersonProfileRepository"]
