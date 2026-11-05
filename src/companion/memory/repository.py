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

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
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
                    " title, url, retrieved_at, published_at, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        source.source_id,
                        source.source_type,
                        source.source_ref,
                        source.title,
                        source.url,
                        _iso(source.retrieved_at),
                        _iso(source.published_at),
                        json.dumps({"provider": source.provider}) if source.provider else "{}",
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
        await self.db.execute(
            "INSERT OR IGNORE INTO memory_links(memory_id, linked_memory_id, link_type)"
            " VALUES (?, ?, ?)",
            (memory_id, linked_memory_id, link_type),
        )

    async def find_derived(self, source_id: str) -> list[str]:
        """Return the ids of memories derived from ``source_id``.

        ``memory_links`` stores ``(memory_id, linked_memory_id, link_type)``,
        so a reflection derived from a source has a ``derived_from`` row where
        ``linked_memory_id`` is the source.
        """
        cur = await self.db.execute(
            "SELECT memory_id FROM memory_links"
            " WHERE linked_memory_id = ? AND link_type = 'derived_from'",
            (source_id,),
        )
        return [r["memory_id"] for r in await cur.fetchall()]

    # -- provenance / write-through ---------------------------------------

    async def add_provenance(
        self,
        memory_id: str,
        *,
        source_event_id: str | None,
        candidate_id: str | None,
        speaker_person_id: str | None,
        subject_person_id: str | None,
        source_type: str = "utterance",
        assertion_type: str = "UNKNOWN",
        certainty_category: str = "UNKNOWN",
        topic: str | None = None,
        entity_key: str | None = None,
        attribute_key: str | None = None,
        explicit_remember_request: bool = False,
        self_reported: bool = False,
        third_party_reported: bool = False,
        model_inferred: bool = False,
        new_version: int | None = None,
    ) -> None:
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT OR REPLACE INTO memory_provenance("
            " memory_id, source_event_id, candidate_id, speaker_person_id,"
            " subject_person_id, source_type, assertion_type, certainty_category,"
            " topic, entity_key, attribute_key, explicit_remember_request,"
            " self_reported, third_party_reported, model_inferred, created_at,"
            " updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory_id,
                source_event_id,
                candidate_id,
                speaker_person_id,
                subject_person_id,
                source_type,
                assertion_type,
                certainty_category,
                topic,
                entity_key,
                attribute_key,
                int(explicit_remember_request),
                int(self_reported),
                int(third_party_reported),
                int(model_inferred),
                now,
                now,
            ),
        )
        await self.add_provenance_event(
            memory_id,
            event_type="created",
            source_event_id=source_event_id,
            candidate_id=candidate_id,
            speaker_person_id=speaker_person_id,
            subject_person_id=subject_person_id,
            new_version=new_version,
            reason=None,
            payload={
                "assertion_type": assertion_type,
                "certainty_category": certainty_category,
                "topic": topic,
                "entity_key": entity_key,
                "attribute_key": attribute_key,
            },
        )

    async def add_provenance_event(
        self,
        memory_id: str,
        *,
        event_type: str,
        source_event_id: str | None = None,
        candidate_id: str | None = None,
        speaker_person_id: str | None = None,
        subject_person_id: str | None = None,
        old_version: int | None = None,
        new_version: int | None = None,
        reason: str | None = None,
        payload: dict | None = None,
    ) -> None:
        """Append one immutable provenance event.

        ``memory_provenance`` keeps the *current* origin for fast lookup; this
        table keeps every correction/promotion/retraction, including the
        versions involved, so later corrections never discard earlier history.
        """
        await self.db.execute(
            "INSERT INTO memory_provenance_events("
            " event_id, memory_id, source_event_id, candidate_id,"
            " speaker_person_id, subject_person_id, old_version, new_version,"
            " event_type, reason, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id("provev"),
                memory_id,
                source_event_id,
                candidate_id,
                speaker_person_id,
                subject_person_id,
                old_version,
                new_version,
                event_type,
                reason,
                json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                _iso(utcnow()),
            ),
        )

    async def list_provenance_events(self, memory_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM memory_provenance_events WHERE memory_id = ?"
            " ORDER BY created_at ASC, rowid ASC",
            (memory_id,),
        )
        rows = []
        for row in await cur.fetchall():
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload") or "{}")
            except (TypeError, ValueError):
                item["payload"] = {}
            rows.append(item)
        return rows

    async def get_provenance(self, memory_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM memory_provenance WHERE memory_id = ?", (memory_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def find_memory_by_candidate(self, candidate_id: str) -> str | None:
        """Return the memory already written from ``candidate_id`` (if any).

        This is the crash-recovery fallback to the write mapping: provenance
        and the memory are written in one transaction, so a committed memory
        is always discoverable even if the mapping row was lost.
        """
        cur = await self.db.execute(
            "SELECT memory_id FROM memory_provenance WHERE candidate_id = ?",
            (candidate_id,),
        )
        row = await cur.fetchone()
        return row["memory_id"] if row is not None else None

    async def list_memory_ids_for_source_event(self, event_id: str) -> list[str]:
        """Memories derived from one utterance, via committed provenance.

        A privacy deletion of the source event must also remove what was
        learned from it; this is the trusted lookup for that scope.
        """
        cur = await self.db.execute(
            "SELECT DISTINCT memory_id FROM memory_provenance"
            " WHERE source_event_id = ?",
            (event_id,),
        )
        return [row["memory_id"] for row in await cur.fetchall()]

    async def list_for_subject(
        self,
        subject_id: str,
        *,
        statuses: list[str],
        kinds: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return memories about a PERSON subject, with provenance topic.

        Used by the trusted consolidation layer to detect duplicates and
        likely corrections before writing. Never used to bypass visibility:
        the caller is the trusted local pipeline and still writes through
        :class:`MemoryService`.
        """
        if not statuses:
            return []
        params: list = [subject_id, *statuses]
        sql = (
            "SELECT m.memory_id, m.kind, m.content, m.status, m.visibility,"
            " m.created_by, m.created_at, m.updated_at, m.version, m.human_confirmed,"
            " m.source_supported, m.self_reported, m.third_party_reported,"
            " m.model_inferred, p.topic, p.assertion_type, p.entity_key,"
            " p.attribute_key"
            " FROM memory_items m"
            " JOIN memory_subjects s ON s.memory_id = m.memory_id"
            " LEFT JOIN memory_provenance p ON p.memory_id = m.memory_id"
            " WHERE s.subject_type = 'PERSON' AND s.subject_id = ?"
            " AND m.status IN (" + ",".join("?" * len(statuses)) + ")"
        )
        if kinds:
            sql += " AND m.kind IN (" + ",".join("?" * len(kinds)) + ")"
            params.extend(kinds)
        sql += " ORDER BY m.updated_at DESC LIMIT ?"
        params.append(limit)
        cur = await self.db.execute(sql, tuple(params))
        return [dict(r) for r in await cur.fetchall()]

    async def list_subject_topic(
        self,
        subject_id: str,
        topics: list[str],
        *,
        statuses: list[str],
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Deterministic lookup by subject + canonical topic (identity facts)."""
        if not topics or not statuses:
            return []
        params: list = [subject_id, *statuses, *topics]
        sql = (
            "SELECT m.memory_id, m.kind, m.content, m.status, m.visibility,"
            " m.created_by, m.created_at, m.updated_at, m.version, m.human_confirmed,"
            " m.source_supported, m.self_reported, m.third_party_reported,"
            " m.model_inferred, p.topic"
            " FROM memory_items m"
            " JOIN memory_subjects s ON s.memory_id = m.memory_id"
            " JOIN memory_provenance p ON p.memory_id = m.memory_id"
            " WHERE s.subject_type = 'PERSON' AND s.subject_id = ?"
            " AND m.status IN (" + ",".join("?" * len(statuses)) + ")"
            " AND p.topic IN (" + ",".join("?" * len(topics)) + ")"
        )
        if kinds:
            sql += " AND m.kind IN (" + ",".join("?" * len(kinds)) + ")"
            params.extend(kinds)
        sql += " ORDER BY m.updated_at DESC LIMIT ?"
        params.append(limit)
        cur = await self.db.execute(sql, tuple(params))
        return [dict(r) for r in await cur.fetchall()]

    # -- assignments (association, never access) ---------------------------

    async def assign(
        self, memory_id: str, person_id: str, *, created_by: str
    ) -> bool:
        """Associate a memory with a person. Returns False if already assigned.

        Assignment is only an association: it does not change the memory's
        content, status or visibility, and it never grants access by itself.
        """
        now = _iso(utcnow())
        cur = await self.db.execute(
            "INSERT OR IGNORE INTO memory_assignments(memory_id, person_id,"
            " created_at, created_by) VALUES (?, ?, ?, ?)",
            (memory_id, person_id, now, created_by),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def unassign(self, memory_id: str, person_id: str) -> bool:
        cur = await self.db.execute(
            "DELETE FROM memory_assignments WHERE memory_id = ? AND person_id = ?",
            (memory_id, person_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def list_assignments(self, memory_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT memory_id, person_id, created_at, created_by"
            " FROM memory_assignments WHERE memory_id = ? ORDER BY created_at ASC",
            (memory_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_assigned_memory_ids(self, person_id: str) -> list[str]:
        cur = await self.db.execute(
            "SELECT memory_id FROM memory_assignments WHERE person_id = ?"
            " ORDER BY created_at ASC",
            (person_id,),
        )
        return [r["memory_id"] for r in await cur.fetchall()]

    async def get_many(self, memory_ids: list[str]) -> list[MemoryItem]:
        if not memory_ids:
            return []
        cur = await self.db.execute(
            "SELECT memory_id, kind, content, status, visibility, created_by,"
            " created_at, updated_at, version, valid_from, valid_until,"
            " human_confirmed, source_supported, self_reported,"
            " third_party_reported, model_inferred FROM memory_items"
            " WHERE memory_id IN (" + ",".join("?" * len(memory_ids)) + ")",
            tuple(memory_ids),
        )
        return [self._to_item(r) for r in await cur.fetchall()]

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
            " s.retrieved_at, s.published_at, s.extra, e.verification_level, e.evidence_span"
            " FROM memory_evidence e JOIN memory_sources s ON s.source_id = e.source_id"
            " WHERE e.memory_id = ?",
            (memory_id,),
        )
        out = []
        for r in await cur.fetchall():
            row = dict(r)
            extra = row.pop("extra", "{}") or "{}"
            try:
                row["provider"] = json.loads(extra).get("provider")
            except (json.JSONDecodeError, AttributeError):
                row["provider"] = None
            out.append(row)
        return out

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
