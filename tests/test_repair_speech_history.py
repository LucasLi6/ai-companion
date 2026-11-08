"""The historical ``[]`` repair tool is dry-run first, backed up, and narrow."""

from __future__ import annotations

import pytest

from companion.storage.database import Database
from companion.tools.repair_speech_history import repair

POLLUTED = "[] 你好呀。"
CLEAN = "正常回复。"
CODE = "items = []"


async def _seed(path):
    db = await Database(path).connect()
    try:
        await db.initialize()
        await db.execute("PRAGMA foreign_keys = OFF")
        rows = [
            ("a1", "e1", POLLUTED),
            ("a2", "e2", CLEAN),
            ("a3", "e3", CODE),
        ]
        for action_id, event_id, text in rows:
            await db.execute(
                "INSERT INTO agent_actions(action_id, event_id, action_type,"
                " response_text, reason_code, status, created_at)"
                " VALUES (?, ?, 'RESPOND', ?, 'DIRECT_AGENT_ADDRESS',"
                " 'COMPLETED', '2026-01-01T00:00:00+00:00')",
                (action_id, event_id, text),
            )
    finally:
        await db.close()


async def _texts(path):
    db = await Database(path).connect()
    try:
        rows = await (
            await db.execute(
                "SELECT action_id, response_text FROM agent_actions ORDER BY action_id"
            )
        ).fetchall()
        return {row["action_id"]: row["response_text"] for row in rows}
    finally:
        await db.close()


async def test_dry_run_changes_nothing_and_writes_no_backup(tmp_path):
    path = tmp_path / "companion.db"
    await _seed(path)

    assert await repair(path, apply=False) == 0

    assert await _texts(path) == {"a1": POLLUTED, "a2": CLEAN, "a3": CODE}
    assert not (tmp_path / "backups").exists()


async def test_apply_backs_up_and_is_idempotent(tmp_path):
    path = tmp_path / "companion.db"
    await _seed(path)

    assert await repair(path, apply=True) == 0
    assert await _texts(path) == {"a1": "你好呀。", "a2": CLEAN, "a3": CODE}
    backups = list((tmp_path / "backups").glob("*.db"))
    assert len(backups) == 1

    # Second apply finds nothing and must not write another backup.
    assert await repair(path, apply=True) == 0
    assert await _texts(path) == {"a1": "你好呀。", "a2": CLEAN, "a3": CODE}
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1


async def test_backup_contains_the_original_text(tmp_path):
    path = tmp_path / "companion.db"
    await _seed(path)
    await repair(path, apply=True)
    backup = next((tmp_path / "backups").glob("*.db"))

    assert await _texts(backup) == {"a1": POLLUTED, "a2": CLEAN, "a3": CODE}
