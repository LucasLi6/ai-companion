"""V0.9.9.23: guarded reset of one user's learned data.

Every test uses a temporary SQLite file. The command must be dry-run by
default, refuse incomplete requests, keep the agent's persona and the two
confirmed identity values, and leave a database that Ava can start from.
"""

from __future__ import annotations

import sqlite3

import pytest

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event, utcnow
from companion.maintenance.reset_user_data import (
    IDENTITY_VALUES,
    ResetRefused,
    reset_user_data,
)
from companion.runtime import build_runtime
from companion.storage.database import Database


async def _seed(tmp_path):
    path = tmp_path / "reset.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db, with_worker=False)
    person = await rt.registry.register(
        "Owner",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.profile_service.ensure_profile(person.person_id, actor="test")
    await rt.agent_service.set_persona_core(
        {"character_description": "我是 Ava，一个可靠的伙伴。"},
        actor="admin",
        reason="test",
    )

    async def event(text, conversation="default"):
        await rt.conversations.get_or_create(conversation)
        item = make_utterance_event(
            conversation_id=conversation,
            text=text,
            addressee_hint="agent",
            speaker_person_id=person.person_id,
            source=EventSource.TEST,
        )
        await rt.repos.utterance.save(item)
        return item

    name_source = await event("我是李约，以后叫我 Lucas。")
    interest = await event("我喜欢爵士乐。")
    corrupted_source = await event("那是我叫我自己顺其自然。")

    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="identity.name",
        content="我自己顺其自然",
        status="ACTIVE",
        source="DIRECT_SELF_REPORT",
        confidence=1.0,
        evidence_event_ids=[corrupted_source.event_id, name_source.event_id],
        evidence_memory_ids=[],
    )
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="identity.preferred_name",
        content="自己顺其自然",
        status="ACTIVE",
        source="DIRECT_SELF_REPORT",
        confidence=1.0,
        evidence_event_ids=[corrupted_source.event_id, name_source.event_id],
        evidence_memory_ids=[],
    )
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="interests.likes",
        content="喜欢爵士乐",
        status="ACTIVE",
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_event_ids=[interest.event_id],
        evidence_memory_ids=[],
    )
    now = utcnow()
    await db.execute(
        "UPDATE person_interaction_profiles"
        " SET compact_summary = ?, revision = 7, last_reviewed_at = ?"
        " WHERE person_id = ?",
        ("旧的、被污染的摘要", now, person.person_id),
    )
    await db.execute(
        "INSERT INTO memory_items("
        " memory_id, kind, content, status, visibility, created_by,"
        " created_at, updated_at) VALUES (?, 'PERSONAL_FACT', ?, 'ACTIVE',"
        " 'PRIVATE', ?, ?, ?)",
        ("mem_seed", "用户喜欢爵士乐", person.person_id, now, now),
    )
    await db.execute(
        "INSERT INTO memory_items_fts(content, memory_id) VALUES (?, ?)",
        ("用户喜欢爵士乐", "mem_seed"),
    )
    await db.execute(
        "INSERT INTO proactive_messages("
        " conversation_id, person_id, source_event_id, text, status, created_at"
        ") VALUES ('default', ?, ?, ?, 'SENT', ?)",
        (person.person_id, interest.event_id, "你最近还好吗？", now),
    )
    await db.execute(
        "INSERT INTO processing_jobs(job_key, job_type, status, created_at)"
        " VALUES ('job_seed', 'PROFILE_REVIEW', 'PENDING', ?)",
        (now,),
    )

    persona = {
        "agent_persona": await _rows(db, "SELECT * FROM agent_persona"),
        "agent_persona_core": await _rows(
            db, "SELECT * FROM agent_persona_core"
        ),
        "persona_revisions": await _rows(db, "SELECT * FROM persona_revisions"),
    }
    await db.close()
    return {
        "path": path,
        "person_id": person.person_id,
        "persona": persona,
        "events": {
            "name": name_source.event_id,
            "interest": interest.event_id,
            "corrupted": corrupted_source.event_id,
        },
    }


async def _rows(db: Database, sql: str, params=()):
    cur = await db.execute(sql, params)
    return [dict(row) for row in await cur.fetchall()]


def _sync_rows(path, sql, params=()):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


async def test_dry_run_writes_nothing(tmp_path):
    seed = await _seed(tmp_path)

    report = await reset_user_data(seed["path"], skip_process_check=True)

    assert report.dry_run is True
    assert report.applied is False
    assert report.owner_person_id == seed["person_id"]
    assert report.identity_entries == [
        {
            "field_key": "identity.name",
            "content": "李约",
            "evidence_event_ids": [seed["events"]["name"]],
        },
        {
            "field_key": "identity.preferred_name",
            "content": "Lucas",
            "evidence_event_ids": [seed["events"]["name"]],
        },
    ]
    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM person_profile_entries")[0]["n"] == 3
    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM memory_items")[0]["n"] == 1
    summary = _sync_rows(
        seed["path"],
        "SELECT compact_summary FROM person_interaction_profiles",
    )[0]["compact_summary"]
    assert summary == "旧的、被污染的摘要"


async def test_apply_requires_both_preserve_flags(tmp_path):
    seed = await _seed(tmp_path)

    with pytest.raises(ResetRefused):
        await reset_user_data(
            seed["path"],
            apply=True,
            preserve_identity=True,
            skip_process_check=True,
        )
    with pytest.raises(ResetRefused):
        await reset_user_data(
            seed["path"],
            apply=True,
            preserve_persona=True,
            skip_process_check=True,
        )

    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM person_profile_entries")[0]["n"] == 3
    assert not (seed["path"].parent / "backups").exists()


async def test_unknown_table_aborts_the_reset(tmp_path):
    seed = await _seed(tmp_path)
    conn = sqlite3.connect(seed["path"])
    conn.execute("CREATE TABLE mystery_new_table(value TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(ResetRefused) as excinfo:
        await reset_user_data(
            seed["path"],
            apply=True,
            preserve_identity=True,
            preserve_persona=True,
            backup_path=tmp_path / "before.db",
            skip_process_check=True,
        )

    assert "unknown table" in str(excinfo.value)
    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM person_profile_entries")[0]["n"] == 3
    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM memory_items")[0]["n"] == 1


async def test_apply_keeps_persona_and_identity_and_clears_learning(tmp_path):
    seed = await _seed(tmp_path)
    backup = tmp_path / "pre-reset.db"

    report = await reset_user_data(
        seed["path"],
        apply=True,
        preserve_identity=True,
        preserve_persona=True,
        backup_path=backup,
        skip_process_check=True,
    )

    assert report.applied is True
    assert report.backup_path == str(backup)
    assert backup.exists()
    assert report.integrity == "ok"
    assert report.foreign_key_violations == 0
    assert report.persona_matches_backup is True

    entries = _sync_rows(
        seed["path"],
        "SELECT field_key, content FROM person_profile_entries"
        " WHERE status IN ('ACTIVE', 'TENTATIVE') ORDER BY field_key",
    )
    assert entries == [
        {"field_key": "identity.name", "content": "李约"},
        {"field_key": "identity.preferred_name", "content": "Lucas"},
    ]
    assert _sync_rows(
        seed["path"],
        "SELECT COUNT(*) AS n FROM person_profile_entries",
    )[0]["n"] == 2

    summary = _sync_rows(
        seed["path"],
        "SELECT compact_summary, revision FROM person_interaction_profiles",
    )[0]
    assert summary["compact_summary"] == ""
    assert summary["revision"] == 1

    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM memory_items")[0]["n"] == 0
    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM memory_items_fts")[0]["n"] == 0
    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM proactive_messages")[0]["n"] == 0
    assert _sync_rows(seed["path"], "SELECT COUNT(*) AS n FROM processing_jobs")[0]["n"] == 0

    utterances = _sync_rows(
        seed["path"], "SELECT event_id FROM utterances ORDER BY event_id"
    )
    assert [item["event_id"] for item in utterances] == [seed["events"]["name"]]
    conversations = _sync_rows(
        seed["path"], "SELECT conversation_id FROM conversations"
    )
    assert [item["conversation_id"] for item in conversations] == ["default"]

    evidence = _sync_rows(
        seed["path"],
        "SELECT field_key, source_event_id, memory_id FROM person_profile_evidence"
        " ORDER BY field_key",
    )
    assert evidence == [
        {
            "field_key": "identity.name",
            "source_event_id": seed["events"]["name"],
            "memory_id": None,
        },
        {
            "field_key": "identity.preferred_name",
            "source_event_id": seed["events"]["name"],
            "memory_id": None,
        },
    ]
    revisions = _sync_rows(
        seed["path"],
        "SELECT revision, reason FROM person_profile_revisions",
    )
    assert revisions == [{"revision": 1, "reason": "reset_user_data"}]
    assert _sync_rows(
        seed["path"],
        "SELECT COUNT(*) AS n FROM person_profile_fact_index",
    )[0]["n"] == 2

    # Agent-owned data is byte-for-byte unchanged.
    assert _sync_rows(seed["path"], "SELECT * FROM agent_persona") == seed["persona"][
        "agent_persona"
    ]
    assert _sync_rows(
        seed["path"], "SELECT * FROM agent_persona_core"
    ) == seed["persona"]["agent_persona_core"]
    assert _sync_rows(
        seed["path"], "SELECT * FROM persona_revisions"
    ) == seed["persona"]["persona_revisions"]
    assert _sync_rows(
        seed["path"], "SELECT COUNT(*) AS n FROM audit_log"
    )[0]["n"] > 0

    # The backup still contains the corrupted pre-reset state.
    assert _sync_rows(
        backup,
        "SELECT COUNT(*) AS n FROM person_profile_entries",
    )[0]["n"] == 3
    contents = {
        item["content"]
        for item in _sync_rows(
            backup,
            "SELECT content FROM person_profile_entries"
            " WHERE field_key LIKE 'identity.%'",
        )
    }
    assert contents == {"我自己顺其自然", "自己顺其自然"}


async def test_reset_keeps_owner_bindings_and_removes_others(tmp_path):
    """A speaker binding is identity: the owner's survives, others go."""
    seed = await _seed(tmp_path)
    db = await Database(seed["path"]).connect()
    try:
        rt = build_runtime(db, with_worker=False)
        guest = await rt.registry.register(
            "Guest",
            "guest",
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=False,
        )
        await rt.speaker_bindings.bind_speaker(
            seed["person_id"], "S-OWNER", "owner"
        )
        await rt.speaker_bindings.bind_speaker(
            guest.person_id, "S-GUEST", "owner"
        )
    finally:
        await db.close()

    report = await reset_user_data(
        seed["path"],
        apply=True,
        preserve_identity=True,
        preserve_persona=True,
        backup_path=tmp_path / "pre-reset.db",
        skip_process_check=True,
    )

    assert report.applied is True
    counts = report.tables["person_speaker_bindings"]
    assert counts.kept == 1
    assert counts.deleted == 1
    assert _sync_rows(
        seed["path"],
        "SELECT speaker_id, person_id FROM person_speaker_bindings",
    ) == [{"speaker_id": "S-OWNER", "person_id": seed["person_id"]}]


async def test_reset_database_starts_and_reads_persona_and_identity(tmp_path):
    seed = await _seed(tmp_path)
    await reset_user_data(
        seed["path"],
        apply=True,
        preserve_identity=True,
        preserve_persona=True,
        backup_path=tmp_path / "pre-reset.db",
        skip_process_check=True,
    )

    db = await Database(seed["path"]).connect()
    try:
        await db.initialize()
        runtime = build_runtime(db, with_worker=False)
        core = await runtime.agent_service.show_persona_core()
        assert core.character_description == "我是 Ava，一个可靠的伙伴。"

        profile = await runtime.profile_service.get(seed["person_id"])
        assert profile is not None
        assert profile.identity["name"] == IDENTITY_VALUES["identity.name"]
        assert profile.identity["preferred_name"] == IDENTITY_VALUES[
            "identity.preferred_name"
        ]
        assert profile.identity["name"] == "李约"
        assert profile.identity["preferred_name"] == "Lucas"
        assert profile.compact_summary == ""
        assert profile.entries.get("interests.likes") in (None, [])
    finally:
        await db.close()
