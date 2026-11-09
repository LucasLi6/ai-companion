"""V0.9.7 schema 25: additive profile entries, receipts and legacy conversion."""

from __future__ import annotations

import json

from companion.storage.database import Database, _statements
from companion.storage.schema import MIGRATIONS, SCHEMA_VERSION

NOW = "2026-01-01T00:00:00+00:00"


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
            (number, NOW),
        )
    await db.conn.commit()
    await db.close()


async def _seed_v24(path):
    await _build_version(path, 24)
    db = await Database(path).connect()
    try:
        await db.execute(
            "INSERT INTO persons(person_id, display_name, role, created_at, updated_at)"
            " VALUES ('person_old', 'Old', 'owner', ?, ?)",
            (NOW, NOW),
        )
        await db.execute(
            "INSERT INTO consents(person_id, allow_processing, allow_persistence,"
            " allow_memory_learning, allow_external_ai, updated_at)"
            " VALUES ('person_old', 1, 1, 1, 1, ?)",
            (NOW,),
        )
        await db.execute(
            "INSERT INTO person_interaction_profiles(person_id, relationship_tier,"
            " style_overrides, updated_at, compact_summary, revision, created_at,"
            " updated_by) VALUES ('person_old', 'FAMILIAR', '{}', ?, 'old summary',"
            " 4, ?, 'test')",
            (NOW, NOW),
        )

        def field(field_key, value, *, status="ACTIVE", source="DIRECT_SELF_REPORT",
                  confidence=0.9, events=("evt_old",), memories=("mem_old",)):
            return (
                "INSERT INTO person_profile_fields(person_id, field_key, value_json,"
                " source, confidence, status, evidence_memory_ids_json,"
                " evidence_event_ids_json, evidence_candidate_ids_json, revision,"
                " created_at, updated_at) VALUES ('person_old', ?, ?, ?, ?, ?, ?, ?,"
                " '[]', 1, ?, ?)",
                (
                    field_key,
                    json.dumps(value, ensure_ascii=False),
                    source,
                    confidence,
                    status,
                    json.dumps(list(memories)),
                    json.dumps(list(events)),
                    NOW,
                    NOW,
                ),
            )

        for statement, params in (
            field(
                "understanding.interests",
                [
                    {"label": "机器人", "memory_id": None, "event_id": None,
                     "candidate_id": None},
                    {"label": "爵士乐", "memory_id": None, "event_id": None,
                     "candidate_id": None},
                    {"label": "机器人", "memory_id": None, "event_id": None,
                     "candidate_id": None},
                    {"label": "  ", "memory_id": None, "event_id": None,
                     "candidate_id": None},
                ],
            ),
            field(
                "understanding.ongoing_projects",
                [{"label": "Ava", "memory_id": None, "event_id": None,
                  "candidate_id": None}],
                source="INTERACTION_INFERENCE",
                confidence=0.7,
            ),
            field("understanding.occupation", "engineer", confidence=0.8),
            field(
                "understanding.personal_goals",
                [{"label": "旧目标", "memory_id": None, "event_id": None,
                  "candidate_id": None}],
                status="SUPERSEDED",
            ),
        ):
            await db.execute(statement, params)

        await db.execute(
            "INSERT INTO person_profile_candidates(candidate_id, person_id,"
            " field_key, candidate_value_json, source, confidence, status,"
            " evidence_memory_ids_json, evidence_event_ids_json,"
            " evidence_candidate_ids_json, reason, created_at, updated_at)"
            " VALUES ('pfc_old', 'person_old', 'understanding.interests', ?,"
            " 'INTERACTION_INFERENCE', 0.6, 'PENDING', '[]', '[\"evt_old\"]', '[]',"
            " 'old pending', ?, ?)",
            (json.dumps([{"label": "围棋"}], ensure_ascii=False), NOW, NOW),
        )
        await db.execute(
            "INSERT INTO person_profile_revisions(revision_id, person_id, revision,"
            " changes_json, changed_by, source_event_id, reason, created_at)"
            " VALUES ('rev_old', 'person_old', 4, '{\"changes\": []}', 'test',"
            " 'evt_old', 'old', ?)",
            (NOW,),
        )
    finally:
        await db.close()


async def test_fresh_database_has_current_schema_structures(db):
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 32

    names = {
        row["name"]
        for row in await (
            await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    }
    assert {
        "person_profile_entries",
        "profile_review_receipts",
        "profile_review_leases",
    } <= names
    indexes = {
        row["name"]
        for row in await (
            await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        ).fetchall()
    }
    assert {
        "idx_profile_entries_person",
        "idx_profile_entries_effective",
        "idx_profile_receipts_person",
        "idx_profile_receipts_event",
        "idx_profile_review_leases_expiry",
    } <= indexes


async def test_migration_24_adds_columns_without_changing_old_rows(tmp_path):
    path = tmp_path / "v24.memory"
    await _seed_v24(path)

    db = await Database(path).connect()
    try:
        backup = await db.backup_to(tmp_path / "v24.backup")
        await db.initialize()

        cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 32

        cur = await db.execute(
            "SELECT * FROM person_interaction_profiles WHERE person_id='person_old'"
        )
        profile = await cur.fetchone()
        assert profile["revision"] == 4
        assert profile["compact_summary"] == "old summary"

        columns = {
            row["name"]
            for row in await (await db.execute(
                "PRAGMA table_info(person_profile_candidates_legacy)"
            )).fetchall()
        }
        assert {
            "operation",
            "target_entry_ids_json",
            "expected_revision",
            "requires_confirmation",
            "review_status",
        } <= columns

        restored = await Database(backup).connect()
        try:
            cur = await restored.execute(
                "SELECT MAX(version) AS v FROM schema_version"
            )
            assert (await cur.fetchone())["v"] == 24
            await restored.initialize()
            cur = await restored.execute(
                "SELECT value_json FROM person_profile_fields"
                " WHERE field_key='understanding.interests'"
            )
            assert "机器人" in (await cur.fetchone())["value_json"]
        finally:
            await restored.close()
    finally:
        await db.close()


async def test_legacy_lists_convert_once_with_source_and_evidence(tmp_path):
    path = tmp_path / "v24.memory"
    await _seed_v24(path)

    db = await Database(path).connect()
    try:
        await db.initialize()
        cur = await db.execute(
            "SELECT field_key, content, source, confidence,"
            " evidence_event_ids_json, evidence_memory_ids_json"
            " FROM person_profile_entries WHERE person_id='person_old'"
            " ORDER BY field_key, content"
        )
        rows = await cur.fetchall()
        assert [(row["field_key"], row["content"]) for row in rows] == [
            ("background.occupation", "engineer"),
            ("interests.topics", "机器人"),
            ("interests.topics", "爵士乐"),
            ("projects.active", "Ava"),
        ]
        by_content = {row["content"]: row for row in rows}
        assert by_content["机器人"]["source"] == "DIRECT_SELF_REPORT"
        assert by_content["机器人"]["confidence"] == 0.9
        assert json.loads(by_content["机器人"]["evidence_event_ids_json"]) == [
            "evt_old"
        ]
        assert json.loads(by_content["机器人"]["evidence_memory_ids_json"]) == [
            "mem_old"
        ]
        assert by_content["Ava"]["source"] == "INTERACTION_INFERENCE"
        assert by_content["Ava"]["confidence"] == 0.7

        # The old field rows stay readable; nothing is deleted.
        cur = await db.execute(
            "SELECT COUNT(*) AS n FROM person_profile_fields"
            " WHERE person_id='person_old'"
        )
        assert (await cur.fetchone())["n"] == 4

        # Inactive and blank values are never promoted.
        contents = {row["content"] for row in rows}
        assert "旧目标" not in contents
        assert "" not in contents
    finally:
        await db.close()


async def test_old_candidates_are_not_auto_upgraded(tmp_path):
    path = tmp_path / "v24.memory"
    await _seed_v24(path)

    db = await Database(path).connect()
    try:
        await db.initialize()
        # V0.9.9.6 renames the retired queue instead of dropping it. Nothing
        # reads or applies it implicitly.
        old = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name='person_profile_candidates'"
        )
        assert await old.fetchone() is None
        cur = await db.execute(
            "SELECT * FROM person_profile_candidates_legacy"
            " WHERE candidate_id='pfc_old'"
        )
        candidate = await cur.fetchone()
        assert candidate["status"] == "PENDING"
        assert candidate["operation"] == "ADD"
        assert candidate["target_entry_ids_json"] == "[]"
        assert candidate["requires_confirmation"] == 0
        assert candidate["review_status"] == "PENDING"
        assert candidate["expected_revision"] is None

        cur = await db.execute(
            "SELECT COUNT(*) AS n FROM person_profile_entries"
            " WHERE content='围棋'"
        )
        assert (await cur.fetchone())["n"] == 0

        cur = await db.execute(
            "SELECT revision, changes_json FROM person_profile_revisions"
            " WHERE revision_id='rev_old'"
        )
        revision = await cur.fetchone()
        assert revision["revision"] == 4
        assert revision["changes_json"] == '{"changes": []}'

        cur = await db.execute(
            "SELECT COUNT(*) AS n FROM profile_candidate_migrations"
        )
        assert (await cur.fetchone())["n"] == 0

        cur = await db.execute("SELECT COUNT(*) AS n FROM profile_review_receipts")
        assert (await cur.fetchone())["n"] == 0
    finally:
        await db.close()


async def test_migration_25_is_additive_from_23(tmp_path):
    path = tmp_path / "v23.memory"
    await _build_version(path, 23)
    db = await Database(path).connect()
    try:
        await db.initialize()
        cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == 32
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name='person_profile_entries'"
        )
        assert await cur.fetchone() is not None
        # Existing version-23 structures are untouched.
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name='person_profile_fields'"
        )
        assert await cur.fetchone() is not None
    finally:
        await db.close()
