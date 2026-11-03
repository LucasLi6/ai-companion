"""P2 — schema versioning: existing V0.0 databases open unchanged, the current
version is idempotent, and a newer (unknown) version is refused, never modified.
"""

from __future__ import annotations

import pytest

from companion.storage.database import Database


async def test_fresh_database_gets_current_version(tmp_path):
    db = await Database(tmp_path / "f.db").connect()
    await db.initialize()
    cur = await db.execute("SELECT version FROM schema_version")
    rows = await cur.fetchall()
    assert [r["version"] for r in rows] == [1]
    await db.close()


async def test_existing_v0_database_opens_idempotently(tmp_path):
    path = tmp_path / "e.db"
    db = await Database(path).connect()
    await db.initialize()
    # Simulate real data surviving a reopen.
    await db.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at, updated_at)"
        " VALUES ('keep', 0, '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()  # must not error, must not duplicate the version row
    cur = await db2.execute("SELECT version FROM schema_version")
    assert [r["version"] for r in await cur.fetchall()] == [1]
    cur = await db2.execute(
        "SELECT COUNT(*) AS c FROM conversations WHERE conversation_id = 'keep'"
    )
    assert (await cur.fetchone())["c"] == 1  # data preserved
    await db2.close()


async def test_newer_schema_version_is_refused(tmp_path):
    db = await Database(tmp_path / "n.db").connect()
    # Fabricate a database stamped with a version this build does not know.
    await db.conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    await db.conn.execute(
        "INSERT INTO schema_version(version, applied_at) VALUES (99, '2024-01-01T00:00:00+00:00')"
    )
    with pytest.raises(RuntimeError):
        await db.initialize()
    # The unknown version was left untouched.
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] == 99
    await db.close()
