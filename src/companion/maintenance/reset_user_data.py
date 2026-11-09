"""Reset one user's learned data while keeping Ava herself intact.

This command exists because the *user's* profile and the *agent's* identity are
different things. A corrupted or over-learned Personal Profile must be
repairable without wiping the persona, the emotion baseline, the relationship
presets or the schema history that Ava is built on.

Safety model
------------
* dry-run by default;
* ``--apply`` is refused unless **both** ``--preserve-identity`` and
  ``--preserve-persona`` are given;
* a full SQLite backup is taken with the engine's online backup API before any
  write;
* every delete happens inside one transaction with deferred foreign keys, so a
  failure or a failed verification rolls the whole reset back;
* unknown tables abort the command: a future migration must add its table to
  this file explicitly before the reset can run again;
* nothing is truncated silently: the retained identity values are written as
  complete sentences and the report lists every kept/deleted row count.

This module never touches the network, the model providers or the persona
configuration. It only edits one local SQLite file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from companion.config import default_db_path
from companion.domain.models import utcnow
from companion.storage.database import Database
from companion.storage.schema import SCHEMA_VERSION

# -- fixed retention policy -------------------------------------------------

# Values the user confirmed must survive the reset.
IDENTITY_VALUES: dict[str, str] = {
    "identity.name": "李约",
    "identity.preferred_name": "Lucas",
}

# Agent-owned configuration. It is not derived from this person's conversation
# history, so a profile reset must not destroy it.
_PRESERVED_TABLES = frozenset(
    {
        "cmf_manifest",
        "agent_identity",
        "agent_persona",
        "agent_persona_core",
        "persona_revisions",
        "emotion_profiles",
        "emotion_appraisal_rules",
        "relationship_presets",
        "app_settings",
        "schema_version",
        "audit_log",
        "embedding_models",
        "voice_profiles",
    }
)

# Tables whose rows are all learned/derived from this person's activity.
_PURGED_TABLES = frozenset(
    {
        "agent_actions",
        "agent_stances",
        "claim_evidence",
        "cognitive_analyses",
        "curiosity_queue",
        "data_control_operations",
        "dialogue_style_examples",
        "embedding_index_metadata",
        "emotion_events",
        "emotion_feedback",
        "emotion_state_snapshots",
        "emotion_turn_receipts",
        "goal_checkpoints",
        "goal_dependencies",
        "goals",
        "knowledge_claims",
        "knowledge_refresh_jobs",
        "learning_checkpoints",
        "learning_runs",
        "media_annotations",
        "media_assets",
        "media_links",
        "memory_assignments",
        "memory_candidate_writes",
        "memory_candidates",
        "memory_embeddings",
        "memory_evidence",
        "memory_facts",
        "memory_items",
        "memory_links",
        "memory_provenance",
        "memory_provenance_events",
        "memory_revisions",
        "memory_sources",
        "memory_subjects",
        "person_profile_candidates_legacy",
        "person_profile_fields",
        "person_profiles",
        "proactive_messages",
        "processing_jobs",
        "profile_candidate_migrations",
        "profile_confirmations",
        "profile_review_leases",
        "profile_review_receipts",
        "reflection_records",
        "relationship_history",
        "research_findings",
        "research_projects",
        "research_runs",
        "research_tasks",
        "retrieval_feedback",
        "route_decisions",
        "route_tool_calls",
        "runtime_snapshots",
        "scene_snapshots",
        "scene_transition_receipts",
        "search_queries",
        "source_excerpts",
        "source_refresh_rules",
        "turn_extractions",
        "web_sources",
    }
)

# These get row-level filtering instead of a full purge.
_PARTIAL_TABLES = frozenset(
    {
        "persons",
        "consents",
        "person_interaction_profiles",
        "person_profile_entries",
        "person_profile_evidence",
        "person_profile_revisions",
        "person_profile_fact_index",
        "utterances",
        "conversations",
    }
)

_FTS_TABLE = "memory_items_fts"

# Every table the reset understands. A new table without a classification is a
# hard stop, not an invitation to guess.
_KNOWN_TABLES = (
    _PRESERVED_TABLES
    | _PURGED_TABLES
    | _PARTIAL_TABLES
    | {"memory_items_fts"}
)

_PERSONA_TABLES = (
    "agent_identity",
    "agent_persona",
    "agent_persona_core",
    "persona_revisions",
)


class ResetRefused(Exception):
    """The reset is not safe to run; nothing was written."""


class ResetVerificationError(Exception):
    """Post-cleanup verification failed; the transaction must roll back."""


@dataclass
class TableCounts:
    kept: int = 0
    deleted: int = 0


@dataclass
class ResetReport:
    db_path: str
    dry_run: bool
    applied: bool = False
    backup_path: Optional[str] = None
    owner_person_id: Optional[str] = None
    tables: dict[str, TableCounts] = field(default_factory=dict)
    identity_entries: list[dict[str, Any]] = field(default_factory=list)
    kept_source_event_ids: list[str] = field(default_factory=list)
    integrity: str = ""
    foreign_key_violations: int = 0
    persona_matches_backup: bool = True
    reason: str = ""

    def add(self, table: str, kept: int, deleted: int) -> None:
        self.tables[table] = TableCounts(kept=max(0, kept), deleted=max(0, deleted))

    def lines(self) -> list[str]:
        head = "DRY RUN (nothing written)" if self.dry_run else "APPLIED"
        out = [
            f"{head}: {self.db_path}",
            f"owner_person_id={self.owner_person_id}",
        ]
        if self.backup_path:
            out.append(f"backup={self.backup_path}")
        for table in sorted(self.tables):
            counts = self.tables[table]
            out.append(
                f"  {table}: kept={counts.kept} deleted={counts.deleted}"
            )
        out.append(
            "identity_entries="
            + json.dumps(self.identity_entries, ensure_ascii=False, sort_keys=True)
        )
        out.append(
            "kept_source_event_ids=" + json.dumps(self.kept_source_event_ids)
        )
        if self.integrity:
            out.append(f"integrity_check={self.integrity}")
        if self.foreign_key_violations or self.integrity:
            out.append(
                f"foreign_key_violations={self.foreign_key_violations}"
            )
        out.append(f"persona_matches_backup={self.persona_matches_backup}")
        if self.reason:
            out.append(f"reason={self.reason}")
        return out


# -- small async DB helpers -------------------------------------------------


async def _count(db: Database, sql: str, params: tuple[Any, ...] = ()) -> int:
    cur = await db.execute(sql, params)
    row = await cur.fetchone()
    return int(row[0] if row is not None else 0)


async def _list_tables(db: Database) -> set[str]:
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name NOT LIKE 'sqlite_%'"
    )
    rows = await cur.fetchall()
    return {str(row[0]) for row in rows}


async def _schema_version(db: Database) -> Optional[int]:
    if "schema_version" not in await _list_tables(db):
        return None
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    row = await cur.fetchone()
    return int(row[0]) if row is not None and row[0] is not None else None


def _unknown_tables(tables: set[str]) -> list[str]:
    return sorted(
        name
        for name in tables
        if name not in _KNOWN_TABLES
        and not name.startswith(f"{_FTS_TABLE}_")
    )


async def _owner_person_id(db: Database) -> Optional[str]:
    cur = await db.execute(
        "SELECT person_id FROM persons WHERE role = 'owner'"
        " ORDER BY created_at ASC, person_id ASC LIMIT 1"
    )
    row = await cur.fetchone()
    return str(row[0]) if row is not None else None


def _json_list(value: Any) -> list[str]:
    try:
        data = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in data if item]


async def _identity_plan(db: Database, owner_id: str) -> dict[str, Any]:
    """Pick the retained identity rows and their genuinely supporting events.

    An event is kept as a source only if its stored utterance actually contains
    the retained value. Old evidence that says "我自己顺其自然" is not allowed
    to become evidence for "李约".
    """
    cur = await db.execute(
        "SELECT entry_id, field_key, content, status, evidence_event_ids_json,"
        " created_at, updated_at FROM person_profile_entries"
        " WHERE person_id = ? AND field_key IN (?, ?)"
        " ORDER BY CASE status WHEN 'ACTIVE' THEN 0 ELSE 1 END,"
        " updated_at DESC, entry_id ASC",
        (owner_id, "identity.name", "identity.preferred_name"),
    )
    rows = [dict(row) for row in await cur.fetchall()]

    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        chosen.setdefault(str(row["field_key"]), row)

    source_events: dict[str, list[str]] = {
        field_key: [] for field_key in IDENTITY_VALUES
    }
    texts: dict[str, str] = {}
    candidates: set[str] = set()
    for field_key, row in chosen.items():
        candidates.update(_json_list(row.get("evidence_event_ids_json")))
    if candidates:
        placeholders = ",".join("?" for _ in candidates)
        cur = await db.execute(
            "SELECT event_id, text FROM utterances"
            f" WHERE event_id IN ({placeholders})",
            tuple(sorted(candidates)),
        )
        texts = {
            str(row["event_id"]): str(row["text"] or "")
            for row in await cur.fetchall()
        }

    for field_key, row in chosen.items():
        wanted = IDENTITY_VALUES[field_key].casefold()
        for event_id in _json_list(row.get("evidence_event_ids_json")):
            text = texts.get(event_id, "")
            if text and wanted in text.casefold():
                source_events[field_key].append(event_id)

    kept_events = sorted(
        {event_id for ids in source_events.values() for event_id in ids}
    )
    conversation_ids: list[str] = []
    if kept_events:
        placeholders = ",".join("?" for _ in kept_events)
        cur = await db.execute(
            "SELECT DISTINCT conversation_id FROM utterances"
            f" WHERE event_id IN ({placeholders}) AND conversation_id IS NOT NULL",
            tuple(kept_events),
        )
        conversation_ids = sorted(str(row[0]) for row in await cur.fetchall())

    return {
        "chosen": chosen,
        "source_events": source_events,
        "kept_events": kept_events,
        "conversation_ids": conversation_ids,
    }


def _identity_entries_to_keep(
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for field_key, content in IDENTITY_VALUES.items():
        entries.append(
            {
                "field_key": field_key,
                "content": content,
                "evidence_event_ids": list(
                    dict.fromkeys(plan["source_events"].get(field_key, []))
                ),
            }
        )
    return entries


async def _count_plan(db: Database, owner_id: str, plan: dict[str, Any]) -> dict[str, TableCounts]:
    """Counts for every known table, computed from the pre-reset state."""
    counts: dict[str, TableCounts] = {}

    actual = await _list_tables(db)
    for table in sorted(_PRESERVED_TABLES & actual):
        counts[table] = TableCounts(
            kept=await _count(db, f"SELECT COUNT(*) FROM {table}"), deleted=0
        )
    for table in sorted(_PURGED_TABLES & actual):
        total = await _count(db, f"SELECT COUNT(*) FROM {table}")
        counts[table] = TableCounts(kept=0, deleted=total)

    if _FTS_TABLE in actual:
        total = await _count(db, f"SELECT COUNT(*) FROM {_FTS_TABLE}")
        counts[_FTS_TABLE] = TableCounts(kept=0, deleted=total)

    # persons / consents / interaction profile: owner only.
    for table in ("persons", "consents", "person_interaction_profiles"):
        if table in actual:
            total = await _count(db, f"SELECT COUNT(*) FROM {table}")
            kept = await _count(
                db,
                f"SELECT COUNT(*) FROM {table} WHERE person_id = ?",
                (owner_id,),
            )
            counts[table] = TableCounts(kept=kept, deleted=total - kept)

    if "person_profile_entries" in actual:
        total = await _count(
            db,
            "SELECT COUNT(*) FROM person_profile_entries WHERE person_id = ?",
            (owner_id,),
        )
        counts["person_profile_entries"] = TableCounts(kept=2, deleted=total)

    kept_events = plan["kept_events"]
    if "utterances" in actual:
        total = await _count(db, "SELECT COUNT(*) FROM utterances")
        counts["utterances"] = TableCounts(
            kept=len(kept_events), deleted=max(0, total - len(kept_events))
        )
    if "conversations" in actual:
        total = await _count(db, "SELECT COUNT(*) FROM conversations")
        kept = len(plan["conversation_ids"])
        counts["conversations"] = TableCounts(
            kept=kept, deleted=max(0, total - kept)
        )
    if "person_profile_evidence" in actual:
        total = await _count(
            db,
            "SELECT COUNT(*) FROM person_profile_evidence WHERE person_id = ?",
            (owner_id,),
        )
        kept = sum(
            len(ids) for ids in plan["source_events"].values() if ids
        )
        counts["person_profile_evidence"] = TableCounts(
            kept=kept, deleted=max(0, total - kept)
        )
    if "person_profile_revisions" in actual:
        total = await _count(
            db,
            "SELECT COUNT(*) FROM person_profile_revisions WHERE person_id = ?",
            (owner_id,),
        )
        counts["person_profile_revisions"] = TableCounts(kept=1, deleted=total)
    if "person_profile_fact_index" in actual:
        total = await _count(
            db,
            "SELECT COUNT(*) FROM person_profile_fact_index WHERE person_id = ?",
            (owner_id,),
        )
        counts["person_profile_fact_index"] = TableCounts(kept=2, deleted=total)
    return counts


# -- persona signature ------------------------------------------------------


async def _persona_signature(db: Database) -> dict[str, list[dict[str, Any]]]:
    signature: dict[str, list[dict[str, Any]]] = {}
    for table in _PERSONA_TABLES:
        cur = await db.execute(f"SELECT * FROM {table} ORDER BY 1")
        signature[table] = [dict(row) for row in await cur.fetchall()]
    return signature


def _persona_signature_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    import sqlite3

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        signature: dict[str, list[dict[str, Any]]] = {}
        for table in _PERSONA_TABLES:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                signature[table] = []
                continue
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            signature[table] = [dict(row) for row in rows]
        return signature
    finally:
        conn.close()


# -- process guard ----------------------------------------------------------


def _running_companion_processes() -> list[str]:
    """Return commands that look like a live Ava using this database.

    The check is intentionally conservative: it never kills anything. It only
    tells the operator to stop Ava and run the command again.
    """
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    own_pids = {os.getpid(), os.getppid()}
    hits: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in own_pids:
            continue
        lowered = command.lower()
        if "reset_user_data" in lowered or "pytest" in lowered:
            continue
        if "companion" not in lowered:
            continue
        if any(
            marker in lowered
            for marker in (
                "companion.cli",
                "companion.__main__",
                "companion.ui",
                "companion.runtime",
                "-m companion",
            )
        ):
            hits.append(f"{pid}: {command[:140]}")
    return hits


def _default_backup_path(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return db_path.parent / "backups" / f"{db_path.stem}-pre-reset-{stamp}.db"


# -- the reset --------------------------------------------------------------


async def _apply_cleanup(
    db: Database,
    owner_id: str,
    plan: dict[str, Any],
    *,
    actor: str,
    persona_before: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Perform the reset inside the caller's transaction."""
    now = utcnow()
    identity_entries = _identity_entries_to_keep(plan)
    kept_events = plan["kept_events"]
    conversation_ids = plan["conversation_ids"]

    await db.execute("PRAGMA defer_foreign_keys = ON")

    actual = await _list_tables(db)
    for table in sorted(_PURGED_TABLES & actual):
        await db.execute(f"DELETE FROM {table}")
    await db.execute(f"DELETE FROM {_FTS_TABLE}")

    await db.execute("DELETE FROM persons WHERE person_id != ?", (owner_id,))
    await db.execute("DELETE FROM consents WHERE person_id != ?", (owner_id,))
    await db.execute(
        "DELETE FROM person_interaction_profiles WHERE person_id != ?",
        (owner_id,),
    )
    await db.execute(
        "UPDATE person_interaction_profiles"
        " SET compact_summary = '', style_overrides = '{}',"
        " revision = 1, last_reviewed_at = NULL, updated_by = ?, updated_at = ?"
        " WHERE person_id = ?",
        (actor, now, owner_id),
    )

    if kept_events:
        placeholders = ",".join("?" for _ in kept_events)
        await db.execute(
            f"DELETE FROM utterances WHERE event_id NOT IN ({placeholders})",
            tuple(kept_events),
        )
    else:
        await db.execute("DELETE FROM utterances")
    if conversation_ids:
        placeholders = ",".join("?" for _ in conversation_ids)
        await db.execute(
            f"DELETE FROM conversations WHERE conversation_id NOT IN ({placeholders})",
            tuple(conversation_ids),
        )
    else:
        await db.execute("DELETE FROM conversations")

    # Exactly two identity entries, rebuilt from complete values.
    await db.execute(
        "DELETE FROM person_profile_entries WHERE person_id = ?", (owner_id,)
    )
    for entry in identity_entries:
        entry_id = f"pfe_{uuid.uuid4().hex}"
        attribute = entry["field_key"].rsplit(".", 1)[-1]
        await db.execute(
            "INSERT INTO person_profile_entries("
            " entry_id, person_id, field_key, content, status, source,"
            " confidence, evidence_event_ids_json, evidence_memory_ids_json,"
            " merged_from_json, superseded_by, valid_from, valid_until,"
            " created_at, updated_at, entity_key, attribute, temporal_scope,"
            " assertion_type, canonical_value, aliases_json"
            ") VALUES (?, ?, ?, ?, 'ACTIVE', 'DIRECT_SELF_REPORT', 1.0,"
            " ?, '[]', '[]', NULL, ?, NULL, ?, ?, 'self', ?, 'STABLE',"
            " 'DIRECT', ?, '[]')",
            (
                entry_id,
                owner_id,
                entry["field_key"],
                entry["content"],
                json.dumps(entry["evidence_event_ids"]),
                now,
                now,
                now,
                attribute,
                entry["content"].casefold(),
            ),
        )
        entry["entry_id"] = entry_id

    # Evidence: rebuilt to the minimal sources actually supporting each value.
    await db.execute(
        "DELETE FROM person_profile_evidence WHERE person_id = ?", (owner_id,)
    )
    for entry in identity_entries:
        for event_id in entry["evidence_event_ids"]:
            await db.execute(
                "INSERT INTO person_profile_evidence("
                " evidence_id, person_id, field_key, memory_id, source_event_id,"
                " candidate_id, evidence_type, created_at"
                ") VALUES (?, ?, ?, NULL, ?, NULL, 'EVENT', ?)",
                (
                    f"ppev_{uuid.uuid4().hex}",
                    owner_id,
                    entry["field_key"],
                    event_id,
                    now,
                ),
            )

    await db.execute(
        "DELETE FROM person_profile_revisions WHERE person_id = ?", (owner_id,)
    )
    await db.execute(
        "INSERT INTO person_profile_revisions("
        " revision_id, person_id, revision, changes_json, changed_by,"
        " source_event_id, reason, created_at"
        ") VALUES (?, ?, 1, ?, ?, NULL, 'reset_user_data', ?)",
        (
            f"ppr_{uuid.uuid4().hex}",
            owner_id,
            json.dumps(
                {
                    "kind": "reset_user_data",
                    "kept_identity_fields": sorted(IDENTITY_VALUES),
                    "kept_source_event_ids": sorted(kept_events),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            actor,
            now,
        ),
    )

    await db.execute(
        "DELETE FROM person_profile_fact_index WHERE person_id = ?", (owner_id,)
    )
    for entry in identity_entries:
        await db.execute(
            "INSERT INTO person_profile_fact_index("
            " entry_id, person_id, entity_key, attribute, temporal_scope,"
            " canonical_value, aliases_json, updated_at"
            ") VALUES (?, ?, 'self', ?, 'STABLE', ?, '[]', ?)",
            (
                entry["entry_id"],
                owner_id,
                entry["field_key"].rsplit(".", 1)[-1],
                entry["content"].casefold(),
                now,
            ),
        )

    # -- verification, before the commit point -------------------------
    rows = await db.execute(
        "SELECT field_key, content FROM person_profile_entries"
        " WHERE person_id = ? AND status IN ('ACTIVE', 'TENTATIVE')"
        " ORDER BY field_key",
        (owner_id,),
    )
    effective = [dict(row) for row in await rows.fetchall()]
    if len(effective) != 2:
        raise ResetVerificationError(
            f"identity_entry_count={len(effective)} expected=2"
        )
    actual_values = {row["field_key"]: row["content"] for row in effective}
    if actual_values != IDENTITY_VALUES:
        raise ResetVerificationError("identity_values_mismatch")
    identity_only = await _count(
        db,
        "SELECT COUNT(*) FROM person_profile_entries"
        " WHERE person_id = ? AND field_key LIKE 'identity.%'",
        (owner_id,),
    )
    if identity_only != 2:
        raise ResetVerificationError(
            f"identity_field_count={identity_only} expected=2"
        )

    if await _persona_signature(db) != persona_before:
        raise ResetVerificationError("persona_changed_during_reset")

    cursor = await db.execute("PRAGMA integrity_check")
    row = await cursor.fetchone()
    integrity = str(row[0]) if row is not None else "unknown"
    if integrity != "ok":
        raise ResetVerificationError(f"integrity_check={integrity}")

    cursor = await db.execute("PRAGMA foreign_key_check")
    violations = len(await cursor.fetchall())
    if violations:
        raise ResetVerificationError(
            f"foreign_key_violations={violations}"
        )

    return identity_entries


async def reset_user_data(
    db_path: str | Path,
    *,
    apply: bool = False,
    preserve_identity: bool = False,
    preserve_persona: bool = False,
    backup_path: str | Path | None = None,
    skip_process_check: bool = False,
    actor: str = "reset_user_data",
) -> ResetReport:
    """Run the guarded reset against one database file.

    Raises :class:`ResetRefused` before touching anything when the request is
    incomplete or the schema contains an unknown table.
    """
    path = Path(db_path)
    report = ResetReport(db_path=str(path), dry_run=not apply)

    if apply and not (preserve_identity and preserve_persona):
        raise ResetRefused(
            "apply requires both --preserve-identity and --preserve-persona"
        )
    if not path.exists():
        raise ResetRefused(f"database does not exist: {path}")
    if path.name == "companion(7).db":
        raise ResetRefused("refusing to operate on companion(7).db")

    if apply and not skip_process_check:
        running = _running_companion_processes()
        if running:
            raise ResetRefused(
                "Ava appears to be running; stop it first: " + "; ".join(running)
            )

    db = await Database(path).connect()
    try:
        # Dry-run must not migrate: it only inspects the file as it stands.
        # Apply takes the backup *before* initialize(), so a migration that
        # fails still has a pre-migration rollback route.
        if apply:
            target = Path(backup_path) if backup_path else _default_backup_path(path)
            report.backup_path = str(await db.backup_to(target))
            await db.initialize()
        else:
            version = await _schema_version(db)
            if version is None:
                raise ResetRefused("not a companion database (no schema_version)")
            if version > SCHEMA_VERSION:
                raise ResetRefused(
                    f"database schema {version} is newer than this build"
                    f" ({SCHEMA_VERSION})"
                )

        unknown = _unknown_tables(await _list_tables(db))
        if unknown:
            raise ResetRefused(
                "unknown table(s), add a retention rule before resetting: "
                + ", ".join(unknown)
            )
        owner_id = await _owner_person_id(db)
        if owner_id is None:
            raise ResetRefused("no owner person found")
        report.owner_person_id = owner_id

        plan = await _identity_plan(db, owner_id)
        report.tables = await _count_plan(db, owner_id, plan)
        report.identity_entries = _identity_entries_to_keep(plan)
        report.kept_source_event_ids = list(plan["kept_events"])

        if not apply:
            report.reason = "dry_run"
            return report

        persona_before = await _persona_signature(db)
        if _persona_signature_file(Path(report.backup_path)) != persona_before:
            raise ResetRefused("backup persona does not match the live database")

        try:
            async with db.transaction(immediate=True):
                report.identity_entries = await _apply_cleanup(
                    db,
                    owner_id,
                    plan,
                    actor=actor,
                    persona_before=persona_before,
                )
                report.integrity = "ok"
                report.foreign_key_violations = 0
                report.persona_matches_backup = True
        except BaseException:
            # The transaction has rolled back; the database is exactly as it
            # was before the reset, and the backup remains for inspection.
            raise

        report.applied = True
        report.reason = "applied"
        return report
    finally:
        await db.close()


# -- CLI --------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m companion.maintenance.reset_user_data",
        description=(
            "Reset learned Personal Profile / memory data while preserving the "
            "agent persona and the user's confirmed identity values. "
            "Dry-run by default."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database path (default: COMPANION_DB or data/companion.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write; requires --preserve-identity and --preserve-persona",
    )
    parser.add_argument(
        "--preserve-identity",
        action="store_true",
        help="restore identity.name=李约 and identity.preferred_name=Lucas",
    )
    parser.add_argument(
        "--preserve-persona",
        action="store_true",
        help="keep agent identity/persona/emotion/relationship configuration",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="backup target (default: data/backups/<name>-pre-reset-<ts>.db)",
    )
    parser.add_argument(
        "--skip-process-check",
        action="store_true",
        help="skip the running-Ava check (use only when you are certain)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    db_path = args.db or default_db_path()
    try:
        report = asyncio.run(
            reset_user_data(
                db_path,
                apply=args.apply,
                preserve_identity=args.preserve_identity,
                preserve_persona=args.preserve_persona,
                backup_path=args.backup,
                skip_process_check=args.skip_process_check,
            )
        )
    except ResetRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except ResetVerificationError as exc:
        print(f"FAILED (rolled back): {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "db_path": report.db_path,
            "dry_run": report.dry_run,
            "applied": report.applied,
            "backup_path": report.backup_path,
            "owner_person_id": report.owner_person_id,
            "tables": {
                name: {"kept": counts.kept, "deleted": counts.deleted}
                for name, counts in report.tables.items()
            },
            "identity_entries": report.identity_entries,
            "kept_source_event_ids": report.kept_source_event_ids,
            "integrity": report.integrity,
            "foreign_key_violations": report.foreign_key_violations,
            "persona_matches_backup": report.persona_matches_backup,
            "reason": report.reason,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\n".join(report.lines()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
