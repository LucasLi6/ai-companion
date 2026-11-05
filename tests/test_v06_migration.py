"""V0.6.1 schema 19: extraction markers, additive migration, backup/rollback."""

from __future__ import annotations

import pytest

from companion.storage.database import Database, _statements
from companion.storage.schema import MIGRATIONS, SCHEMA_VERSION


async def _build_version(path, version: int):
    db = await Database(path).connect()
    await db.conn.execute(
        "CREATE TABLE schema_version ("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for number, sql in MIGRATIONS[:version]:
        for statement in _statements(sql):
            await db.conn.execute(statement)
        await db.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (number, "2026-01-01T00:00:00+00:00"),
        )
    await db.conn.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at,"
        " updated_at) VALUES ('c_old', 0, '2026-01-01T00:00:00+00:00',"
        " '2026-01-01T00:00:00+00:00')"
    )
    await db.conn.execute(
        "INSERT INTO utterances(event_id, conversation_id, speaker_person_id,"
        " text, source, addressee_hint, occurred_at, processing_status,"
        " created_at) VALUES ('evt_old', 'c_old', NULL, '旧消息', 'CLI',"
        " 'agent', '2026-01-01T00:00:00+00:00', 'COMPLETED',"
        " '2026-01-01T00:00:00+00:00')"
    )
    await db.conn.commit()
    await db.close()


async def test_fresh_database_is_version_19_with_runtime_snapshots(db):
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 22
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name='runtime_snapshots'"
    )
    assert await cur.fetchone() is not None
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
        " AND name='idx_runtime_snapshots_conversation'"
    )
    assert await cur.fetchone() is not None


async def test_migration_19_is_additive_from_version_17(tmp_path):
    path = tmp_path / "v17.memory"
    await _build_version(path, 17)

    db = await Database(path).connect()
    try:
        await db.initialize()
        cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 22
        cur = await db.execute("SELECT text FROM utterances WHERE event_id='evt_old'")
        assert (await cur.fetchone())["text"] == "旧消息"
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name='runtime_snapshots'"
        )
        assert await cur.fetchone() is not None
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name='turn_extractions'"
        )
        assert await cur.fetchone() is not None
    finally:
        await db.close()


async def test_backup_restores_a_pre_migration_version(tmp_path):
    path = tmp_path / "live.memory"
    await _build_version(path, 17)
    db = await Database(path).connect()
    try:
        backup = await db.backup_to(tmp_path / "pre-migration.backup")
        assert backup.is_file()

        await db.initialize()
        cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == 22

        restored = await Database(backup).connect()
        try:
            await restored.initialize()
            # Re-opening a backup stays usable; it is itself migrated forward.
            cur = await restored.execute("SELECT MAX(version) AS v FROM schema_version")
            assert (await cur.fetchone())["v"] == SCHEMA_VERSION
            cur = await restored.execute(
                "SELECT text FROM utterances WHERE event_id='evt_old'"
            )
            assert (await cur.fetchone())["text"] == "旧消息"
        finally:
            await restored.close()
    finally:
        await db.close()


async def test_backup_never_silently_overwrites(tmp_path):
    path = tmp_path / "live.memory"
    await _build_version(path, 17)
    db = await Database(path).connect()
    try:
        target = tmp_path / "existing.backup"
        target.write_bytes(b"do not replace")
        with pytest.raises(FileExistsError):
            await db.backup_to(target)
        assert target.read_bytes() == b"do not replace"
    finally:
        await db.close()
