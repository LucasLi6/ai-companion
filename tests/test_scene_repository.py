"""V0.4 — durable scene snapshots, receipts, CAS and migrations 14–15.

All tests use the real SQLite database (WAL, foreign keys) in a temp directory.
They pin the two safety properties: a stale writer can never overwrite a newer
revision, and a source event is applied at most once even across retries.
"""

from __future__ import annotations

import json

import pytest

from companion.cognition.scene_models import SceneState
from companion.cognition.scene_repository import (
    SceneRepository,
    SceneRevisionConflict,
)
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.storage.database import Database, _statements
from companion.storage.schema import MIGRATIONS, SCHEMA_VERSION


async def _scope(rt, name="Alice", conversation="c1"):
    person = await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=False,
    )
    await rt.conversations.get_or_create(conversation)
    return person.person_id


async def _utterance(rt, person_id, text, *, conversation="c1"):
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    return event


def _state(person_id, event_id, *, revision, topic="旅行", source_ids=None):
    return SceneState(
        conversation_id="c1",
        person_id=person_id,
        revision=revision,
        topic=topic,
        source_event_ids=source_ids or [event_id],
        last_committed_event_id=event_id,
    )


async def test_schema_15_has_scene_tables_and_analysis_column(db):
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 32

    cur = await db.execute("PRAGMA table_info(cognitive_analyses)")
    assert "scene_proposal_json" in {r["name"] for r in await cur.fetchall()}

    curated = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN"
        " ('scene_snapshots','scene_transition_receipts','dialogue_style_examples')"
    )
    assert {r["name"] for r in await curated.fetchall()} == {
        "scene_snapshots",
        "scene_transition_receipts",
        "dialogue_style_examples",
    }


async def test_initial_transition_and_load(db, rt):
    repo = SceneRepository(db)
    person_id = await _scope(rt)
    event = await _utterance(rt, person_id, "我去了日本")

    receipt = await repo.apply_transition(
        source_event_id=event.event_id,
        conversation_id="c1",
        person_id=person_id,
        expected_revision=0,
        state=_state(person_id, event.event_id, revision=0),
        transition={"action": "RESPOND", "topic": "旅行"},
    )
    assert receipt["result_revision"] == 0
    assert receipt["source_event_id"] == event.event_id

    loaded = await repo.load("c1", person_id)
    assert loaded is not None
    assert loaded.revision == 0
    assert loaded.state.topic == "旅行"
    assert loaded.source_event_ids == [event.event_id]
    assert loaded.last_committed_event_id == event.event_id


async def test_cas_rejects_stale_writer_without_overwriting(db, rt):
    repo = SceneRepository(db)
    person_id = await _scope(rt)
    first = await _utterance(rt, person_id, "我去了日本")
    second = await _utterance(rt, person_id, "后来去了大阪")

    await repo.apply_transition(
        source_event_id=first.event_id,
        conversation_id="c1",
        person_id=person_id,
        expected_revision=0,
        state=_state(person_id, first.event_id, revision=0),
        transition={"action": "RESPOND"},
    )
    with pytest.raises(SceneRevisionConflict):
        # A writer that still thinks the scene is at initial revision.
        await repo.apply_transition(
            source_event_id=second.event_id,
            conversation_id="c1",
            person_id=person_id,
            expected_revision=0,
            state=_state(person_id, second.event_id, revision=0, topic="大阪"),
            transition={"action": "RESPOND"},
        )

    loaded = await repo.load("c1", person_id)
    assert loaded is not None
    assert loaded.revision == 0
    assert loaded.state.topic == "旅行"
    assert loaded.last_committed_event_id == first.event_id
    assert await repo.get_receipt(second.event_id) is None

    # The same transition, after reloading revision 0, applies cleanly as 1.
    await repo.apply_transition(
        source_event_id=second.event_id,
        conversation_id="c1",
        person_id=person_id,
        expected_revision=0,
        state=_state(person_id, second.event_id, revision=1, topic="大阪"),
        transition={"action": "RESPOND"},
    )
    loaded = await repo.load("c1", person_id)
    assert loaded.revision == 1
    assert loaded.state.topic == "大阪"


async def test_repeated_event_is_idempotent_and_does_not_advance(db, rt):
    repo = SceneRepository(db)
    person_id = await _scope(rt)
    event = await _utterance(rt, person_id, "我去了日本")

    first = await repo.apply_transition(
        source_event_id=event.event_id,
        conversation_id="c1",
        person_id=person_id,
        expected_revision=0,
        state=_state(person_id, event.event_id, revision=0),
        transition={"action": "RESPOND"},
    )
    # A replay with a *different* candidate must not overwrite the snapshot.
    replay = await repo.apply_transition(
        source_event_id=event.event_id,
        conversation_id="c1",
        person_id=person_id,
        expected_revision=0,
        state=_state(person_id, event.event_id, revision=0, topic="篡改"),
        transition={"action": "WAIT"},
    )
    assert replay == first

    loaded = await repo.load("c1", person_id)
    assert loaded.state.topic == "旅行"
    cur = await db.execute(
        "SELECT COUNT(*) AS n FROM scene_transition_receipts"
        " WHERE source_event_id = ?",
        (event.event_id,),
    )
    assert (await cur.fetchone())["n"] == 1


async def test_receipt_only_does_not_modify_snapshot(db, rt):
    repo = SceneRepository(db)
    person_id = await _scope(rt)
    event = await _utterance(rt, person_id, "我去了日本")

    await repo.apply_transition(
        source_event_id=event.event_id,
        conversation_id="c1",
        person_id=person_id,
        expected_revision=0,
        state=_state(person_id, event.event_id, revision=0),
        transition={"action": "RESPOND"},
    )
    # Simulate a crash after the snapshot advanced but before the receipt was
    # visible to the recovery path: delete only the receipt.
    await db.execute(
        "DELETE FROM scene_transition_receipts WHERE source_event_id = ?",
        (event.event_id,),
    )
    receipt = await repo.save_receipt_only(
        source_event_id=event.event_id,
        conversation_id="c1",
        person_id=person_id,
        result_revision=0,
        transition={"action": "RESPOND"},
    )
    assert json.loads(receipt["transition_json"])["recovered"] is True
    assert await repo.save_receipt_only(
        source_event_id=event.event_id,
        conversation_id="c1",
        person_id=person_id,
        result_revision=0,
        transition={"action": "RESPOND"},
    ) == receipt
    loaded = await repo.load("c1", person_id)
    assert loaded.revision == 0


async def test_delete_scope_is_scoped(db, rt):
    repo = SceneRepository(db)
    alice = await _scope(rt, "Alice", "c1")
    bob = await _scope(rt, "Bob", "c1")
    a_event = await _utterance(rt, alice, "A 的话")
    b_event = await _utterance(rt, bob, "B 的话")

    for person, event in ((alice, a_event), (bob, b_event)):
        await repo.apply_transition(
            source_event_id=event.event_id,
            conversation_id="c1",
            person_id=person,
            expected_revision=0,
            state=SceneState(
                conversation_id="c1",
                person_id=person,
                source_event_ids=[event.event_id],
                last_committed_event_id=event.event_id,
            ),
            transition={"action": "RESPOND"},
        )

    await repo.delete_scope("c1", alice)
    assert await repo.load("c1", alice) is None
    assert await repo.load("c1", bob) is not None
    assert await repo.get_receipt(a_event.event_id) is None
    assert await repo.get_receipt(b_event.event_id) is not None


async def test_migration_13_to_15_preserves_existing_data(tmp_path):
    """A real version-13 database upgrades in place, adding only V0.4 objects."""
    path = tmp_path / "v13.db"
    ts = "2025-01-01T00:00:00+00:00"
    db = await Database(path).connect()
    for version, sql in MIGRATIONS:
        if version > 13:
            break
        for statement in _statements(sql):
            await db.conn.execute(statement)
        await db.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (version, ts),
        )
    await db.conn.execute(
        "INSERT INTO persons(person_id, display_name, role, registration_status,"
        " created_at, updated_at) VALUES ('p1', 'Owner', 'owner', 'ACTIVE', ?, ?)",
        (ts, ts),
    )
    await db.conn.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at, updated_at)"
        " VALUES ('c1', 0, ?, ?)",
        (ts, ts),
    )
    await db.conn.execute(
        "INSERT INTO utterances(event_id, conversation_id, speaker_person_id, text,"
        " source, addressee_hint, occurred_at, processing_status, created_at)"
        " VALUES ('evt1', 'c1', 'p1', '旧对话', 'TEST', 'agent', ?, 'COMPLETED', ?)",
        (ts, ts),
    )
    await db.conn.execute(
        "INSERT INTO cognitive_analyses(analysis_id, event_id, meaning, action,"
        " reason_code, response_style, provider, created_at)"
        " VALUES ('ana1', 'evt1', 'old analysis', 'RESPOND',"
        " 'direct_agent_address', 'plain', 'fake', ?)",
        (ts,),
    )
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    try:
        cur = await db2.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == SCHEMA_VERSION
        cur = await db2.execute("SELECT event_id, meaning FROM cognitive_analyses")
        row = await cur.fetchone()
        assert (row["event_id"], row["meaning"]) == ("evt1", "old analysis")
        cur = await db2.execute("PRAGMA table_info(cognitive_analyses)")
        assert "scene_proposal_json" in {r["name"] for r in await cur.fetchall()}
        cur = await db2.execute(
            "SELECT COUNT(*) AS n FROM scene_snapshots"
        )
        assert (await cur.fetchone())["n"] == 0
        cur = await db2.execute("PRAGMA foreign_key_check")
        assert await cur.fetchall() == []

        from companion.storage.repositories import make_repositories

        legacy = await make_repositories(db2).analysis.get_analysis("evt1")
        assert legacy is not None
        assert legacy.meaning == "old analysis"
        assert legacy.scene_proposal is None
    finally:
        await db2.close()

    # Re-opening is idempotent and keeps the migrated rows.
    db3 = await Database(path).connect()
    await db3.initialize()
    cur = await db3.execute("SELECT COUNT(*) AS n FROM cognitive_analyses")
    assert (await cur.fetchone())["n"] == 1
    await db3.close()


async def test_receipt_is_cascade_deleted_with_utterance(db, rt):
    """Scene receipts are derived data: deleting the source utterance removes
    the receipt, not a dangling reference."""
    repo = SceneRepository(db)
    person_id = await _scope(rt)
    event = await _utterance(rt, person_id, "我去了日本")
    await repo.apply_transition(
        source_event_id=event.event_id,
        conversation_id="c1",
        person_id=person_id,
        expected_revision=0,
        state=_state(person_id, event.event_id, revision=0),
        transition={"action": "RESPOND"},
    )
    await db.execute("DELETE FROM utterances WHERE event_id = ?", (event.event_id,))
    assert await repo.get_receipt(event.event_id) is None
