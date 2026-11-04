"""CMF file service and Agent Core regressions.

Covers file identity (stable agent_id, clone lineage), safe refusal of corrupt
or foreign files, legacy migration without data loss, and the effective
persona/relationship context that will enter real prompts.
"""

from __future__ import annotations

import sqlite3

import pytest

from companion.agent.service import AgentService
from companion.cmf.constants import APPLICATION_ID, REQUIRED_TABLES
from companion.cmf.service import (
    CmfError,
    CmfFileExistsError,
    CmfFileService,
)
from companion.core.policy import PolicyConfig
from companion.memory.models import AccessContext
from companion.runtime import build_runtime
from companion.storage.database import Database, _statements
from companion.storage.schema import MIGRATIONS

ADMIN = AccessContext("admin", is_admin=True)


async def test_create_inspect_validate_and_application_id(tmp_path):
    path = tmp_path / "ava.memory"
    info = await CmfFileService.create(path, display_name="Ava")
    assert info.ok
    assert info.agent_id.startswith("agent_")
    assert info.display_name == "Ava"
    assert info.format_version == "1.0"
    assert info.schema_version >= 9
    assert "cmf_manifest" in REQUIRED_TABLES

    validation = await CmfFileService.validate(path)
    assert validation.ok
    assert validation.errors == []

    conn = sqlite3.connect(path)
    try:
        (app_id,) = conn.execute("PRAGMA application_id").fetchone()
        assert app_id == APPLICATION_ID
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM cmf_manifest WHERE singleton = 1"
        ).fetchone()
        assert count == 1
    finally:
        conn.close()

    with pytest.raises(CmfFileExistsError):
        await CmfFileService.create(path)


async def test_agent_id_survives_reopen_and_restart(tmp_path):
    path = tmp_path / "ava.memory"
    first = await CmfFileService.create(path, display_name="Ava")
    db = await CmfFileService.open(path)
    service = AgentService(repos=build_runtime(db).repos)
    await service.set_name("Ava Prime")
    await db.close()

    db = await CmfFileService.open(path)
    service = AgentService(repos=build_runtime(db).repos)
    shown = await service.show()
    assert shown["agent_id"] == first.agent_id
    assert shown["display_name"] == "Ava Prime"
    revisions = await service.revision_history()
    assert len(revisions) >= 1
    await db.close()


async def test_export_preserves_identity_and_memories(tmp_path):
    source = tmp_path / "ava.memory"
    info = await CmfFileService.create(source, display_name="Ava")
    db = await CmfFileService.open(source)
    rt = build_runtime(db)
    await rt.memory_service.create(
        kind="PERSONAL_FACT",
        content="我喜欢爵士乐。",
        access=ADMIN,
        visibility="PUBLIC",
    )
    await db.close()

    destination = tmp_path / "ava-backup.memory"
    validation = await CmfFileService.export(source, destination)
    assert validation.ok

    exported = await CmfFileService.inspect(destination)
    assert exported.agent_id == info.agent_id
    assert exported.lineage_id == info.lineage_id
    assert exported.counts["memory_items"] == 1

    with pytest.raises(CmfFileExistsError):
        await CmfFileService.export(source, destination)


async def test_clone_gets_new_identity_with_lineage(tmp_path):
    source = tmp_path / "ava.memory"
    original = await CmfFileService.create(source, display_name="Ava")
    db = await CmfFileService.open(source)
    rt = build_runtime(db)
    await rt.memory_service.create(
        kind="PERSONAL_FACT",
        content="我喜欢爵士乐。",
        access=ADMIN,
        visibility="PUBLIC",
    )
    await db.close()

    clone_path = tmp_path / "ava-clone.memory"
    clone = await CmfFileService.clone(source, clone_path, display_name="Ava-2")
    assert clone.ok
    assert clone.agent_id != original.agent_id
    assert clone.lineage_parent_agent_id == original.agent_id
    assert clone.lineage_id == original.lineage_id
    assert clone.display_name == "Ava-2"
    assert clone.counts["memory_items"] == 1

    # The clone's own manifest/identity agree on the new agent id.
    conn = sqlite3.connect(clone_path)
    try:
        manifest_id = conn.execute(
            "SELECT agent_id FROM cmf_manifest WHERE singleton = 1"
        ).fetchone()[0]
        identity_id = conn.execute("SELECT agent_id FROM agent_identity").fetchone()[0]
        distinct = conn.execute(
            "SELECT COUNT(DISTINCT agent_id) FROM agent_identity"
        ).fetchone()[0]
    finally:
        conn.close()
    assert manifest_id == clone.agent_id == identity_id
    assert distinct == 1


async def test_invalid_and_foreign_files_are_refused(tmp_path):
    corrupt = tmp_path / "corrupt.memory"
    corrupt.write_bytes(b"this is not a sqlite database" * 10)
    validation = await CmfFileService.validate(corrupt)
    assert validation.ok is False
    with pytest.raises(Exception):
        await CmfFileService.open(corrupt)

    foreign = tmp_path / "foreign.db"
    conn = sqlite3.connect(foreign)
    conn.execute("CREATE TABLE notes(id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("INSERT INTO notes(body) VALUES ('not a companion')")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="not a Companion"):
        await CmfFileService.open(foreign)


async def test_newer_major_format_is_refused_without_modification(tmp_path):
    path = tmp_path / "future.memory"
    await CmfFileService.create(path)
    conn = sqlite3.connect(path)
    conn.execute("UPDATE cmf_manifest SET format_version = '2.0' WHERE singleton = 1")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="newer"):
        await CmfFileService.open(path)

    # The file was not silently rewritten to a fresh/empty companion.
    conn = sqlite3.connect(path)
    try:
        version = conn.execute(
            "SELECT format_version FROM cmf_manifest WHERE singleton = 1"
        ).fetchone()[0]
    finally:
        conn.close()
    assert version == "2.0"


async def test_legacy_db_migrates_without_data_loss(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for version, sql in MIGRATIONS:
        if version > 8:
            break
        for statement in _statements(sql):
            conn.execute(statement)
        conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (version, "2026-01-01T00:00:00+00:00"),
        )
    conn.execute(
        "INSERT INTO persons(person_id, display_name, role, registration_status,"
        " created_at, updated_at) VALUES ('p_legacy', 'Legacy', 'friend', 'active',"
        " '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    report = await CmfFileService.migrate(path)
    assert report["from_schema_version"] == 8
    assert report["to_schema_version"] >= 9
    assert report["agent_id"].startswith("agent_")

    db = await CmfFileService.open(path)
    assert await db.execute("SELECT 1") is not None
    cur = await db.execute("SELECT display_name FROM persons WHERE person_id = 'p_legacy'")
    assert (await cur.fetchone())["display_name"] == "Legacy"
    cur = await db.execute("PRAGMA application_id")
    assert (await cur.fetchone())[0] == APPLICATION_ID
    await db.close()


async def test_two_files_isolate_identity_and_persona(tmp_path):
    a_path, b_path = tmp_path / "a.memory", tmp_path / "b.memory"
    a_info = await CmfFileService.create(a_path, display_name="A")
    b_info = await CmfFileService.create(b_path, display_name="B")
    assert a_info.agent_id != b_info.agent_id

    da = await CmfFileService.open(a_path)
    sa = build_runtime(da).agent_service
    await sa.set_name("Ava")
    await sa.set_persona({"humor": 90})
    await da.close()

    db = await CmfFileService.open(b_path)
    sb = build_runtime(db).agent_service
    await sb.set_name("Bao")
    await sb.set_persona({"humor": 10})
    shown = await sb.show()
    assert shown["agent_id"] == b_info.agent_id
    assert shown["display_name"] == "Bao"
    assert shown["persona"]["humor"] == 10
    await db.close()

    da = await CmfFileService.open(a_path)
    shown = await build_runtime(da).agent_service.show()
    assert shown["agent_id"] == a_info.agent_id
    assert shown["display_name"] == "Ava"
    assert shown["persona"]["humor"] == 90
    await da.close()


async def test_persona_validation_rejects_out_of_range_and_bool(rt):
    service = rt.agent_service
    with pytest.raises(ValueError):
        await service.set_persona({"humor": 101})
    with pytest.raises(ValueError):
        await service.set_persona({"humor": -1})
    with pytest.raises(ValueError):
        await service.set_persona({"humor": True})
    with pytest.raises(ValueError):
        await service.set_persona({"nonexistent": 50})


async def test_effective_context_order_and_policy(rt, register):
    person = await register("李悦")
    service = rt.agent_service
    await service.set_persona({"humor": 40, "warmth": 50, "formality": 50})
    await service.set_relationship(person.person_id, "familiar")
    await service.set_override(person.person_id, "humor", 30)

    policy = PolicyConfig(
        response_conservatism="strict",
        interruption_threshold="high",
        curiosity_level="low",
    )
    effective = await service.effective(person.person_id, policy=policy)

    # base 40 + FAMILIAR delta (+10) then the absolute per-person override (30).
    assert effective.traits["humor"] == 30
    # base 50 + FAMILIAR warmth delta (+15).
    assert effective.traits["warmth"] == 65
    # base 50 + FAMILIAR formality delta (-10).
    assert effective.traits["formality"] == 40

    temporary = await service.effective(
        person.person_id, policy=policy, temporary={"humor": 90}
    )
    assert temporary.traits["humor"] == 90

    # Policy stays authoritative and is passed through, never merged as a trait.
    assert effective.response_conservatism == "strict"
    assert effective.interruption_threshold == "high"
    assert effective.curiosity_level == "low"

    # A relationship tier is descriptive, not a consent grant.
    assert await rt.registry.is_external_ai_authorized(person.person_id) is False


async def test_familiarity_is_not_inferred_from_message_count(rt, register, say):
    person = await register("李悦")
    for i in range(5):
        await say(person.person_id, "agent", f"普通消息 {i}")
    profile = await rt.agent_service.repo.get_profile(person.person_id)
    assert profile is None or profile["familiarity"] == "STRANGER"


async def test_prompt_includes_identity_and_style_boundary(rt, register):
    from companion.cognition.context_builder import ContextBuilder
    from companion.domain.enums import CognitiveAction, EventSource, ReasonCode
    from companion.domain.models import CognitiveResult, make_utterance_event
    from companion.prompts.response import build_response_messages

    person = await register("李悦")
    await rt.agent_service.set_name("Ava")
    builder = ContextBuilder(
        repos=rt.repos, registry=rt.registry, agent_service=rt.agent_service
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="你好",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    context = await builder.build(event, memories=[])

    analysis = CognitiveResult(
        event_id=event.event_id,
        meaning="greeting",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        response_style="plain",
    )
    messages = build_response_messages(context, analysis)
    prompt = messages[0]["content"]
    assert messages[-1]["content"] == "你好"
    assert "Ava" in prompt
    assert "权限" in prompt or "permission" in prompt.lower()
