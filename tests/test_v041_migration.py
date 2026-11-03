"""Schema 15 migration: additive, non-destructive, and backward-readable.

A version-14 database with real rows is built here, then opened by the normal
``Database.initialize`` path. No pre-existing value may change, and all new
columns must read as neutral/unknown rather than inventing a mood, tension or
hostile intent.
"""

from __future__ import annotations

import json

from companion.storage.database import Database, _statements
from companion.storage.schema import MIGRATIONS, SCHEMA_VERSION


async def _build_version_14(path):
    db = await Database(path).connect()
    await db.conn.execute(
        "CREATE TABLE schema_version ("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for version, sql in MIGRATIONS[:14]:
        for statement in _statements(sql):
            await db.conn.execute(statement)
        await db.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (version, "2026-01-01T00:00:00+00:00"),
        )
    await db.conn.execute(
        "INSERT INTO emotion_state_snapshots(agent_id, person_id, valence,"
        " arousal, intensity, emotion, half_life_seconds, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("agent_old", "person_old", -0.3, 0.4, 0.55, "FRUSTRATION", 300.0,
         "2026-01-01T00:00:00+00:00"),
    )
    await db.conn.execute(
        "INSERT INTO emotion_events(emotion_event_id, agent_id, person_id,"
        " source_event_id, emotion, valence, arousal, intensity,"
        " appraisal_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("emo_old", "agent_old", "person_old", "evt_old", "FRUSTRATION",
         -0.3, 0.4, 0.55, "legacy_rule", "2026-01-01T00:00:00+00:00"),
    )
    await db.conn.execute(
        "INSERT INTO emotion_turn_receipts(source_event_id, agent_id, person_id,"
        " outcome, processed_at) VALUES (?, ?, ?, ?, ?)",
        ("evt_old", "agent_old", "person_old", "APPLIED",
         "2026-01-01T00:00:00+00:00"),
    )
    await db.conn.commit()
    await db.close()


async def test_migration_15_is_additive_from_a_version_14_database(tmp_path):
    path = tmp_path / "v14.memory"
    await _build_version_14(path)

    db = await Database(path).connect()
    try:
        await db.initialize()
        cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 15

        # Old snapshot values are untouched; new layers start neutral.
        cur = await db.execute(
            "SELECT * FROM emotion_state_snapshots"
            " WHERE agent_id = ? AND person_id = ?",
            ("agent_old", "person_old"),
        )
        row = await cur.fetchone()
        assert row["intensity"] == 0.55
        assert row["emotion"] == "FRUSTRATION"
        assert row["activations_json"] == "{}"
        assert row["mood_intensity"] == 0.0
        assert row["mood_emotion"] == "NEUTRAL"
        assert row["relationship_tension"] == 0.0
        assert row["last_event_type"] is None
        assert row["last_event_at"] is None

        # Old emotion events keep NULL event metadata: "not recorded", never
        # "hostile", and old diagnostics are simply absent.
        cur = await db.execute(
            "SELECT * FROM emotion_events WHERE emotion_event_id = 'emo_old'"
        )
        event = await cur.fetchone()
        assert event["emotion"] == "FRUSTRATION"
        assert event["event_type"] is None
        assert event["target"] is None
        assert event["social_intent"] is None
        assert event["validation"] is None
        assert event["tension_after"] is None

        cur = await db.execute(
            "SELECT * FROM emotion_turn_receipts WHERE source_event_id = 'evt_old'"
        )
        receipt = await cur.fetchone()
        assert receipt["outcome"] == "APPLIED"
        assert receipt["reason"] is None
        assert receipt["detail"] is None

        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name='agent_persona_core'"
        )
        assert await cur.fetchone() is not None

        cur = await db.execute("PRAGMA table_info(cognitive_analyses)")
        assert "dialogue_impulse_json" in {r["name"] for r in await cur.fetchall()}

        cur = await db.execute("PRAGMA foreign_key_check")
        assert await cur.fetchall() == []
    finally:
        await db.close()


async def test_legacy_snapshot_still_reads_without_inventing_new_layers(tmp_path):
    path = tmp_path / "v14-read.memory"
    await _build_version_14(path)

    db = await Database(path).connect()
    try:
        await db.initialize()
        from companion.emotion.repository import EmotionRepository

        repo = EmotionRepository(db)
        snapshot = await repo.get_snapshot("agent_old", "person_old")
        assert snapshot is not None
        assert json.loads(snapshot["activations_json"]) == {}
        assert snapshot["mood_intensity"] == 0.0
        assert snapshot["relationship_tension"] == 0.0
    finally:
        await db.close()
