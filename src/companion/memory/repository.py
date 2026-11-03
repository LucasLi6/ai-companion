"""Data-access layer for the unified memory store.

Thin, parameterised SQL over :class:`Database`. It owns the FTS5 index and
keeps it in lock-step with ``memory_items`` inside the same transaction. No
business rules live here — visibility/privacy and optimistic-locking policy
belong to :class:`MemoryService`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
    VerificationLevel,
)
from companion.domain.models import new_id, utcnow
from companion.storage.database import Database

from .models import MemoryEvidence, MemoryItem, MemoryRevision, MemorySource


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _fts_match(text: str) -> str | None:
    """Build an AND-of-phrases FTS5 query from free text, or ``None``."""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return None
    return " AND ".join(f'"{t}"' for t in tokens)


class MemoryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -- write -------------------------------------------------------------

    async def create(
        self,
        item: MemoryItem,
        subjects: list[tuple[str, str]],
        sources: list[MemorySource],
        evidence: list[MemoryEvidence],
    ) -> None:
        """Insert an item, its subjects/sources/evidence and FTS entry, atomically."""
        async with self.db.transaction():
            await self.db.conn.execute(
                "INSERT INTO memory_items(memory_id, kind, content, status, visibility,"
                " created_by, created_at, updated_at, version, valid_from, valid_until,"
                " human_confirmed, source_supported, self_reported, third_party_reported,"
                " model_inferred, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')",
                (
                    item.memory_id,
                    item.kind.value,
                    item.content,
                    item.status.value,
                    item.visibility.value,
                    item.created_by,
                    _iso(item.created_at),
                    _iso(item.updated_at),
                    item.version,
                    _iso(item.valid_from),
                    _iso(item.valid_until),
                    int(item.human_confirmed),
                    int(item.source_supported),
                    int(item.self_reported),
                    int(item.third_party_reported),
                    int(item.model_inferred),
                ),
            )
            for subject_type, subject_id in subjects:
                await self.db.conn.execute(
                    "INSERT OR IGNORE INTO memory_subjects(memory_id, subject_type, subject_id)"
                    " VALUES (?, ?, ?)",
                    (item.memory_id, subject_type, subject_id),
                )
            for source in sources:
                await self.db.conn.execute(
                    "INSERT OR IGNORE INTO memory_sources(source_id, source_type, source_ref,"
                    " title, url, retrieved_at, published_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        source.source_id,
                        source.source_type,
                        source.source_ref,
                        source.title,
                        source.url,
                        _iso(source.retrieved_at),
                        _iso(source.published_at),
                    ),
                )
            for ev in evidence:
                await self.db.conn.execute(
                    "INSERT OR IGNORE INTO memory_evidence(memory_id, source_id, evidence_span,"
                    " verification_level, retrieved_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        item.memory_id,
                        ev.source_id,
                        ev.evidence_span,
                        ev.verification_level.value,
                        _iso(ev.retrieved_at),
                    ),
                )
            await self.db.conn.execute(
                "INSERT INTO memory_items_fts(content, memory_id) VALUES (?, ?)",
                (item.content, item.memory_id),
            )

    async def replace_item(self, item: MemoryItem, expected_version: int) -> bool:
        """Optimistic update of the live row. Returns ``False`` on version mismatch."""
        cur = await self.db.conn.execute(
            "UPDATE memory_items SET kind=?, content=?, status=?, visibility=?,"
            " updated_at=?, version=?, valid_from=?, valid_until=?, human_confirmed=?,"
            " source_supported=?, self_reported=?, third_party_reported=?, model_inferred=?"
            " WHERE memory_id=? AND version=?",
            (
                item.kind.value,
                item.content,
                item.status.value,
                item.visibility.value,
                _iso(item.updated_at),
                item.version,
                _iso(item.valid_from),
                _iso(item.valid_until),
                int(item.human_confirmed),
                int(item.source_supported),
                int(item.self_reported),
                int(item.third_party_reported),
                int(item.model_inferred),
                item.memory_id,
                expected_version,
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def sync_fts(self, memory_id: str, content: str) -> None:
        """Rewrite the FTS entry for one item (delete then insert)."""
        await self.db.conn.execute(
            "DELETE FROM memory_items_fts WHERE memory_id = ?", (memory_id,)
        )
        await self.db.conn.execute(
            "INSERT INTO memory_items_fts(content, memory_id) VALUES (?, ?)",
            (content, memory_id),
        )

    async def insert_revision(self, rev: MemoryRevision, snapshot: dict) -> None:
        await self.db.conn.execute(
            "INSERT INTO memory_revisions(revision_id, memory_id, version, kind, content,"
            " status, changed_by, changed_at, reason, snapshot)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rev.revision_id,
                rev.memory_id,
                rev.version,
                rev.kind.value,
                rev.content,
                rev.status.value,
                rev.changed_by,
                _iso(rev.changed_at),
                rev.reason,
                json.dumps(snapshot),
            ),
        )

    async def delete(self, memory_id: str) -> None:
        """Hard-delete an item and its subjects/evidence/links/revisions and FTS row."""
        async with self.db.transaction():
            await self.db.conn.execute(
                "DELETE FROM memory_items_fts WHERE memory_id = ?", (memory_id,)
            )
            await self.db.conn.execute(
                "DELETE FROM memory_items WHERE memory_id = ?", (memory_id,)
            )

    async def add_subject(self, memory_id: str, subject_type: str, subject_id: str) -> None:
        await self.db.conn.execute(
            "INSERT OR IGNORE INTO memory_subjects(memory_id, subject_type, subject_id)"
            " VALUES (?, ?, ?)",
            (memory_id, subject_type, subject_id),
        )

    async def add_link(self, memory_id: str, linked_memory_id: str, link_type: str) -> None:
        await self.db.conn.execute(
            "INSERT OR IGNORE INTO memory_links(memory_id, linked_memory_id, link_type)"
            " VALUES (?, ?, ?)",
            (memory_id, linked_memory_id, link_type),
        )

    # -- read --------------------------------------------------------------

    async def get(self, memory_id: str) -> MemoryItem | None:
        cur = await self.db.execute(
            "SELECT memory_id, kind, content, status, visibility, created_by, created_at,"
            " updated_at, version, valid_from, valid_until, human_confirmed, source_supported,"
            " self_reported, third_party_reported, model_inferred FROM memory_items"
            " WHERE memory_id = ?",
            (memory_id,),
        )
        row = await cur.fetchone()
        return self._to_item(row) if row is not None else None

    async def get_subjects(self, memory_id: str) -> list[tuple[str, str]]:
        cur = await self.db.execute(
            "SELECT subject_type, subject_id FROM memory_subjects WHERE memory_id = ?",
            (memory_id,),
        )
        return [(r["subject_type"], r["subject_id"]) for r in await cur.fetchall()]

    async def get_sources(self, memory_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT s.source_id, s.source_type, s.source_ref, s.title, s.url,"
            " s.retrieved_at, s.published_at, e.verification_level, e.evidence_span"
            " FROM memory_evidence e JOIN memory_sources s ON s.source_id = e.source_id"
            " WHERE e.memory_id = ?",
            (memory_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_links(self, memory_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT memory_id, linked_memory_id, link_type FROM memory_links"
            " WHERE memory_id = ? OR linked_memory_id = ?",
            (memory_id, memory_id),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_revisions(self, memory_id: str) -> list[MemoryRevision]:
        cur = await self.db.execute(
            "SELECT revision_id, memory_id, version, kind, content, status, changed_by,"
            " changed_at, reason FROM memory_revisions WHERE memory_id = ?"
            " ORDER BY version DESC",
            (memory_id,),
        )
        rows = await cur.fetchall()
        return [
            MemoryRevision(
                revision_id=r["revision_id"],
                memory_id=r["memory_id"],
                version=r["version"],
                kind=MemoryKind(r["kind"]),
                content=r["content"],
                status=MemoryStatus(r["status"]),
                changed_by=r["changed_by"],
                changed_at=_dt(r["changed_at"]),
                reason=r["reason"],
            )
            for r in rows
        ]

    # -- search ------------------------------------------------------------

    async def search(
        self,
        *,
        query: str | None = None,
        kinds: list[str] | None = None,
        statuses: list[str] | None = None,
        visibilities: list[str] | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
        source_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        updated_after: datetime | None = None,
        updated_before: datetime | None = None,
        visible_to_subject_id: str | None = None,
        include_retracted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        where: list[str] = []
        params: list = []

        if query:
            match = _fts_match(query)
            used_fts = False
            if match:
                try:
                    cur = await self.db.execute(
                        "SELECT memory_id FROM memory_items_fts WHERE memory_items_fts MATCH ?",
                        (match,),
                    )
                    ids = [r["memory_id"] for r in await cur.fetchall()]
                    if ids:
                        where.append(
                            "m.memory_id IN (" + ",".join("?" * len(ids)) + ")"
                        )
                        params.extend(ids)
                        used_fts = True
                except Exception:
                    used_fts = False
            if not used_fts:
                # FTS5 does not segment CJK well; fall back to substring search.
                where.append("m.content LIKE ?")
                params.append(f"%{query}%")

        if kinds:
            where.append("m.kind IN (" + ",".join("?" * len(kinds)) + ")")
            params.extend(kinds)
        if statuses:
            where.append("m.status IN (" + ",".join("?" * len(statuses)) + ")")
            params.extend(statuses)
        if visibilities:
            where.append("m.visibility IN (" + ",".join("?" * len(visibilities)) + ")")
            params.extend(visibilities)
        if subject_type and subject_id:
            where.append(
                "m.memory_id IN (SELECT memory_id FROM memory_subjects"
                " WHERE subject_type = ? AND subject_id = ?)"
            )
            params.extend([subject_type, subject_id])
        if source_id:
            where.append(
                "m.memory_id IN (SELECT memory_id FROM memory_evidence WHERE source_id = ?)"
            )
            params.append(source_id)
        if visible_to_subject_id is not None:
            # Non-admin visibility: public, or the actor is a PERSON subject.
            where.append(
                "(m.visibility = 'PUBLIC' OR m.memory_id IN ("
                " SELECT memory_id FROM memory_subjects"
                " WHERE subject_type = 'PERSON' AND subject_id = ?))"
            )
            params.append(visible_to_subject_id)
        if not include_retracted:
            where.append("m.status != 'RETRACTED'")
        if created_after:
            where.append("m.created_at >= ?")
            params.append(_iso(created_after))
        if created_before:
            where.append("m.created_at <= ?")
            params.append(_iso(created_before))
        if updated_after:
            where.append("m.updated_at >= ?")
            params.append(_iso(updated_after))
        if updated_before:
            where.append("m.updated_at <= ?")
            params.append(_iso(updated_before))

        sql = (
            "SELECT m.memory_id, m.kind, m.content, m.status, m.visibility, m.created_by,"
            " m.created_at, m.updated_at, m.version, m.human_confirmed, m.source_supported,"
            " m.self_reported, m.third_party_reported, m.model_inferred FROM memory_items m"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY m.updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        ids = [r["memory_id"] for r in rows]
        subjects_map = await self._subjects_for(ids)
        sources_map = await self._sources_for(ids)
        return [
            self._to_result(r, subjects_map.get(r["memory_id"], []), sources_map.get(r["memory_id"], []))
            for r in rows
        ]

    async def _subjects_for(self, ids: list[str]) -> dict[str, list[dict]]:
        if not ids:
            return {}
        cur = await self.db.execute(
            "SELECT memory_id, subject_type, subject_id FROM memory_subjects"
            " WHERE memory_id IN (" + ",".join("?" * len(ids)) + ")",
            tuple(ids),
        )
        out: dict[str, list[dict]] = {}
        for r in await cur.fetchall():
            out.setdefault(r["memory_id"], []).append(
                {"subject_type": r["subject_type"], "subject_id": r["subject_id"]}
            )
        return out

    async def _sources_for(self, ids: list[str]) -> dict[str, list[dict]]:
        if not ids:
            return {}
        cur = await self.db.execute(
            "SELECT e.memory_id, s.source_id, s.source_type, s.source_ref, s.title,"
            " s.url, s.retrieved_at, s.published_at, e.verification_level"
            " FROM memory_evidence e JOIN memory_sources s ON s.source_id = e.source_id"
            " WHERE e.memory_id IN (" + ",".join("?" * len(ids)) + ")",
            tuple(ids),
        )
        out: dict[str, list[dict]] = {}
        for r in await cur.fetchall():
            out.setdefault(r["memory_id"], []).append(dict(r))
        return out

    async def counts(self) -> dict:
        cur = await self.db.execute("SELECT COUNT(*) AS c FROM memory_items")
        total = (await cur.fetchone())["c"]
        cur = await self.db.execute(
            "SELECT kind, COUNT(*) AS c FROM memory_items GROUP BY kind"
        )
        by_kind = {r["kind"]: r["c"] for r in await cur.fetchall()}
        cur = await self.db.execute(
            "SELECT status, COUNT(*) AS c FROM memory_items GROUP BY status"
        )
        by_status = {r["status"]: r["c"] for r in await cur.fetchall()}
        return {
            "total": total,
            "by_kind": by_kind,
            "by_status": by_status,
            "candidates": by_status.get("CANDIDATE", 0),
            "needs_review": by_status.get("NEEDS_REVIEW", 0),
        }

    # -- helpers -----------------------------------------------------------

    def _to_item(self, row) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"],
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            status=MemoryStatus(row["status"]),
            visibility=MemoryVisibility(row["visibility"]),
            created_by=row["created_by"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            version=row["version"],
            valid_from=_dt(row["valid_from"]),
            valid_until=_dt(row["valid_until"]),
            human_confirmed=bool(row["human_confirmed"]),
            source_supported=bool(row["source_supported"]),
            self_reported=bool(row["self_reported"]),
            third_party_reported=bool(row["third_party_reported"]),
            model_inferred=bool(row["model_inferred"]),
        )

    def _to_result(self, row, subjects: list[dict], sources: list[dict]) -> dict:
        return {
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
            "subjects": subjects,
            "sources": sources,
        }
