"""Repair historical assistant replies polluted by a leading ``[]``.

Before the V1.2 parser defence, the model could emit a bare ``[]`` (empty voice
markup) as the first characters of an ordinary reply, and that string was
persisted in ``agent_actions.response_text``. This tool previews and, only when
explicitly asked, removes that prefix.

Safety rules:

- dry-run is the default; ``--apply`` is required to write;
- a consistent SQLite backup is made before the first write;
- only assistant rows (``agent_actions``) are touched, user utterances never;
- only a leading empty-bracket prefix followed by whitespace/end is removed;
- code fences, arrays, Markdown links and ordinary brackets are left alone;
- updates are parameterized by ``action_id`` and idempotent.

Usage::

    python -m companion.tools.repair_speech_history --dry-run
    python -m companion.tools.repair_speech_history --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from companion.config import default_db_path
from companion.speech_expression.history import strip_leading_empty_bracket_markup
from companion.storage.database import Database

__all__ = ["RepairRow", "find_polluted_rows", "main", "repair"]


@dataclass(frozen=True)
class RepairRow:
    action_id: str
    event_id: str
    old_text: str
    new_text: str


async def find_polluted_rows(db: Database) -> list[RepairRow]:
    cur = await db.execute(
        "SELECT action_id, event_id, response_text FROM agent_actions"
        " WHERE response_text IS NOT NULL AND response_text <> ''"
        " ORDER BY created_at, action_id"
    )
    rows = await cur.fetchall()
    repairs: list[RepairRow] = []
    for row in rows:
        old = row["response_text"]
        new = strip_leading_empty_bracket_markup(old)
        if new != old and new.strip():
            repairs.append(
                RepairRow(
                    action_id=row["action_id"],
                    event_id=row["event_id"],
                    old_text=old,
                    new_text=new,
                )
            )
    return repairs


def _backup_path(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = db_path.parent / "backups"
    return backup_dir / f"{db_path.stem}-speech-repair-{stamp}.db"


async def repair(
    db_path: Path,
    *,
    apply: bool,
    limit: int = 0,
) -> int:
    db = await Database(db_path).connect()
    try:
        rows = await find_polluted_rows(db)
        if limit > 0:
            rows = rows[:limit]
        print(f"database: {db_path}")
        print(f"mode: {'APPLY' if apply else 'DRY-RUN'}")
        print(f"polluted assistant replies: {len(rows)}")
        for row in rows:
            preview_old = row.old_text[:120].replace("\n", "\\n")
            preview_new = row.new_text[:120].replace("\n", "\\n")
            print(f"  {row.action_id} event={row.event_id}")
            print(f"    before: {preview_old!r}")
            print(f"    after:  {preview_new!r}")
        if not rows:
            print("nothing to repair")
            return 0
        if not apply:
            print("dry-run only; no rows written")
            print("re-run with --apply after reviewing the preview")
            return 0

        backup = _backup_path(db_path)
        await db.backup_to(backup)
        print(f"backup: {backup}")
        async with db.transaction(immediate=True) as conn:
            for row in rows:
                await conn.execute(
                    "UPDATE agent_actions SET response_text = ?"
                    " WHERE action_id = ? AND response_text = ?",
                    (row.new_text, row.action_id, row.old_text),
                )
        print(f"updated: {len(rows)}")
        # Idempotence check: a second scan must find nothing.
        remaining = await find_polluted_rows(db)
        print(f"remaining polluted replies: {len(remaining)}")
        return 0
    finally:
        await db.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or repair leading empty-bracket markup in assistant replies."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="database path (default: data/companion.db or COMPANION_DB)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="preview only (default)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="write repairs after taking a backup",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only process the first N rows (0 = all); applies to both modes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = args.db or default_db_path()
    if not db_path.exists():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 2
    return asyncio.run(repair(db_path, apply=bool(args.apply), limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
