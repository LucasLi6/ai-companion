"""P2 — schema versioning: fresh databases reach the current version, an old
V0.0 database migrates in place without data loss, re-initialisation is
idempotent, a newer (unknown) version is refused, and a failed migration is
atomic (no half-applied DDL).
"""

from __future__ import annotations

import pytest

from companion.storage.database import Database
from companion.storage.schema import SCHEMA_SQL, SCHEMA_VERSION


async def _versions(db) -> list[int]:
    cur = await db.execute("SELECT version FROM schema_version ORDER BY version")
    return [r["version"] for r in await cur.fetchall()]


async def test_fresh_database_gets_current_version(tmp_path):
    db = await Database(tmp_path / "f.db").connect()
    await db.initialize()
    assert await _versions(db) == list(range(1, SCHEMA_VERSION + 1))
    await db.close()


async def test_v0_database_migrates_to_current(tmp_path):
    """A database stamped version 1 (the published V0.0 schema) upgrades to the
    current version on open, keeping its data and adding the new columns."""
    path = tmp_path / "old.db"
    db = await Database(path).connect()
    # Fabricate the exact published V0.0 state: schema + a single version row.
    await db.conn.executescript(SCHEMA_SQL)
    await db.conn.execute(
        "INSERT INTO schema_version(version, applied_at) VALUES (1, '2024-01-01T00:00:00+00:00')"
    )
    await db.conn.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at, updated_at)"
        " VALUES ('keep', 0, '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    try:
        assert await _versions(db2) == list(range(1, SCHEMA_VERSION + 1))
        cur = await db2.execute(
            "SELECT COUNT(*) AS c FROM conversations WHERE conversation_id = 'keep'"
        )
        assert (await cur.fetchone())["c"] == 1  # data preserved
        # New lease columns exist on processing_jobs.
        cur = await db2.execute("PRAGMA table_info(processing_jobs)")
        cols = {r["name"] for r in await cur.fetchall()}
        assert {"owner_id", "lease_expires_at", "heartbeat_at", "next_run_at"} <= cols
    finally:
        await db2.close()


async def test_reopen_idempotent_no_duplicate_versions(tmp_path):
    path = tmp_path / "e.db"
    db = await Database(path).connect()
    await db.initialize()
    await db.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at, updated_at)"
        " VALUES ('keep', 0, '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()  # no-op at the current version
    assert await _versions(db2) == list(range(1, SCHEMA_VERSION + 1))
    cur = await db2.execute(
        "SELECT COUNT(*) AS c FROM conversations WHERE conversation_id = 'keep'"
    )
    assert (await cur.fetchone())["c"] == 1
    await db2.close()


async def test_newer_schema_version_is_refused(tmp_path):
    db = await Database(tmp_path / "n.db").connect()
    await db.conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    await db.conn.execute(
        "INSERT INTO schema_version(version, applied_at) VALUES (99, '2024-01-01T00:00:00+00:00')"
    )
    with pytest.raises(RuntimeError):
        await db.initialize()
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] == 99
    await db.close()


async def test_migration_atomicity_on_failure(tmp_path, monkeypatch):
    """A migration whose SQL fails part-way rolls back completely: no partial
    DDL and no version stamp are left behind."""
    import companion.storage.database as dbmod

    path = tmp_path / "a.db"
    db = await Database(path).connect()
    await db.initialize()
    assert await _versions(db) == list(range(1, SCHEMA_VERSION + 1))

    broken = dbmod.MIGRATIONS + [
        (99, "CREATE TABLE half_done(x INTEGER); CREATE TABLE broken(")
    ]
    monkeypatch.setattr(dbmod, "MIGRATIONS", broken)

    with pytest.raises(Exception):
        await db.initialize()

    # Version unchanged; the would-be partial table does not exist.
    assert await _versions(db) == list(range(1, SCHEMA_VERSION + 1))
    cur = await db.execute("SELECT name FROM sqlite_master WHERE name = 'half_done'")
    assert await cur.fetchall() == []
    await db.close()
