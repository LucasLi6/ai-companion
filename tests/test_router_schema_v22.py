"""Schema v22: route rows follow their utterance, and privacy rows exist.

The repair makes route/tool history a *derived* record of an utterance. The
only truthful lifecycle is therefore ``ON DELETE CASCADE``: deleting an
utterance must not leave a route row pointing at an event that no longer
exists. The migration moves valid rows forward and drops orphans instead of
inventing parents for them.
"""

from __future__ import annotations

import json

import pytest

from companion.storage.database import Database, _statements
from companion.storage.schema import MIGRATIONS, SCHEMA_VERSION

_NOW = "2026-01-01T00:00:00+00:00"


async def _build_version_21(path):
    db = await Database(path).connect()
    await db.conn.execute(
        "CREATE TABLE schema_version ("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for version, sql in MIGRATIONS:
        if version > 21:
            break
        for statement in _statements(sql):
            await db.conn.execute(statement)
        await db.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (version, _NOW),
        )

    await db.conn.execute(
        "INSERT INTO persons(person_id, display_name, role,"
        " registration_status, created_at, updated_at)"
        " VALUES ('p1', 'Alice', 'friend', 'active', ?, ?)",
        (_NOW, _NOW),
    )
    await db.conn.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at,"
        " updated_at) VALUES ('c1', 0, ?, ?)",
        (_NOW, _NOW),
    )
    await db.conn.execute(
        "INSERT INTO utterances(event_id, conversation_id, speaker_person_id,"
        " text, source, addressee_hint, occurred_at, processing_status,"
        " created_at) VALUES ('evt_valid', 'c1', 'p1', 'hello', 'test',"
        " 'agent', ?, 'COMPLETED', ?)",
        (_NOW, _NOW),
    )
    for event_id in ("evt_valid", "evt_orphan"):
        await db.conn.execute(
            "INSERT INTO route_decisions(event_id, phase, step_index,"
            " schema_version, router_version, provider, model, decision_json,"
            " validation_status, created_at)"
            " VALUES (?, 'INITIAL', 0, 1, 'router-v1', 'fake', 'm', ?,"
            " 'VALID', ?)",
            (event_id, json.dumps({"action": "RESPOND"}), _NOW),
        )
        await db.conn.execute(
            "INSERT INTO route_tool_calls(tool_call_id, event_id,"
            " route_step_index, tool_name, arguments_json, arguments_hash,"
            " status, result_reference_json, created_at)"
            " VALUES (?, ?, 0, 'memory.search', '{}', 'hash', 'SUCCESS',"
            " '{}', ?)",
            (f"tc_{event_id}", event_id, _NOW),
        )
    await db.conn.commit()
    await db.close()


@pytest.mark.asyncio
async def test_fresh_database_is_v22_with_cascading_route_foreign_keys(db):
    assert SCHEMA_VERSION == 30
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] == 30

    for table in ("route_decisions", "route_tool_calls"):
        cur = await db.execute(f"PRAGMA foreign_key_list({table})")
        keys = [dict(row) for row in await cur.fetchall()]
        assert any(
            key["table"] == "utterances"
            and key["from"] == "event_id"
            and key["on_delete"] == "CASCADE"
            for key in keys
        ), f"{table} must cascade from utterances"

    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
        " AND name = 'data_control_operations'"
    )
    assert await cur.fetchone() is not None


@pytest.mark.asyncio
async def test_foreign_keys_are_enforced_on_every_connection(db):
    cur = await db.execute("PRAGMA foreign_keys")
    assert (await cur.fetchone())[0] == 1

    from companion.storage.repositories import make_repositories

    repos = make_repositories(db)
    from companion.routing.models import RoutePhase, ValidatedRouteDecision

    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        await repos.routing.save_decision(
            event_id="evt_missing",
            phase=RoutePhase.INITIAL,
            step_index=0,
            route=ValidatedRouteDecision(
                action="RESPOND",
                reply_necessity="REQUIRED",
                addressee_assessment="AGENT",
            ),
        )


@pytest.mark.asyncio
async def test_deleting_an_utterance_cascades_route_and_tool_rows(db):
    await db.execute(
        "INSERT INTO persons(person_id, display_name, role,"
        " registration_status, created_at, updated_at)"
        " VALUES ('p1', 'Alice', 'friend', 'active', ?, ?)",
        (_NOW, _NOW),
    )
    await db.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at,"
        " updated_at) VALUES ('c1', 0, ?, ?)",
        (_NOW, _NOW),
    )
    await db.execute(
        "INSERT INTO utterances(event_id, conversation_id, speaker_person_id,"
        " text, source, addressee_hint, occurred_at, processing_status,"
        " created_at) VALUES ('evt', 'c1', 'p1', 'hello', 'test', 'agent',"
        " ?, 'COMPLETED', ?)",
        (_NOW, _NOW),
    )
    await db.execute(
        "INSERT INTO route_decisions(event_id, phase, step_index,"
        " schema_version, router_version, provider, model, decision_json,"
        " validation_status, created_at) VALUES ('evt', 'INITIAL', 0, 1,"
        " 'r', 'p', 'm', '{}', 'VALID', ?)",
        (_NOW,),
    )
    await db.execute(
        "INSERT INTO route_tool_calls(tool_call_id, event_id,"
        " route_step_index, tool_name, arguments_json, arguments_hash,"
        " status, execution_status, result_reference_json, created_at)"
        " VALUES ('tc', 'evt', 0, 'memory.search', '{}', 'h', 'SUCCESS',"
        " 'COMPLETED', '{}', ?)",
        (_NOW,),
    )

    await db.execute("DELETE FROM utterances WHERE event_id = 'evt'")

    for table in ("route_decisions", "route_tool_calls"):
        cur = await db.execute(f"SELECT COUNT(*) AS n FROM {table}")
        assert (await cur.fetchone())["n"] == 0


@pytest.mark.asyncio
async def test_migration_from_v21_keeps_valid_rows_and_drops_orphans(tmp_path):
    path = tmp_path / "v21.memory"
    await _build_version_21(path)

    db = await Database(path).connect()
    try:
        await db.initialize()

        cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == 30

        cur = await db.execute(
            "SELECT event_id, execution_status FROM route_tool_calls"
            " ORDER BY event_id"
        )
        rows = [dict(row) for row in await cur.fetchall()]
        assert rows == [
            {"event_id": "evt_valid", "execution_status": "COMPLETED"}
        ]

        cur = await db.execute(
            "SELECT event_id FROM route_decisions ORDER BY event_id"
        )
        assert [row["event_id"] for row in await cur.fetchall()] == ["evt_valid"]

        for table in ("route_decisions", "route_tool_calls"):
            cur = await db.execute(f"PRAGMA foreign_key_list({table})")
            assert any(
                dict(row)["table"] == "utterances"
                and dict(row)["on_delete"] == "CASCADE"
                for row in await cur.fetchall()
            )

        # The rebuilt tables keep their uniqueness guarantee.
        cur = await db.execute(
            "SELECT COUNT(*) AS n FROM sqlite_master WHERE type = 'index'"
            " AND tbl_name = 'route_decisions'"
            " AND name = 'idx_route_decisions_event'"
        )
        assert (await cur.fetchone())["n"] == 1
    finally:
        await db.close()
