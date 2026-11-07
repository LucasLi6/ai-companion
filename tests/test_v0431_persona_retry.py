"""V0.4.3.1: persona revision consistency on retry.

An analysis is produced under one trusted character revision. If the response
step fails and the turn is retried after an admin changed Persona Core, the
reply must be generated from the *same* snapshot the analysis saw. When that
snapshot was not recorded (an old schema-16 row) the retry is an explicit
conflict, never a silent mix of two characters.
"""

from __future__ import annotations

import json

from companion.storage.database import Database, _statements
from companion.storage.repositories import AnalysisRepository
from companion.storage.schema import MIGRATIONS, SCHEMA_VERSION
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

OLD_PERSONA = "旧的角色：说话很慢，喜欢先听完再判断。"
NEW_PERSONA = "新的角色：说话很快，喜欢立刻给出结论。"


class _FakeClient:
    """Analysis always succeeds; response calls fail ``failures`` times."""

    def __init__(self, *, failures=1, response_text="好。"):
        self.failures = failures
        self.response_text = response_text
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        if request_type == "analysis":
            content = json.dumps(
                {
                    "meaning": "用户在打招呼",
                    "action": "RESPOND",
                    "reason_code": "direct_agent_address",
                    "referenced_memory_ids": [],
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        else:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("response transport failed")
            content = self.response_text
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def count(self, request_type: str) -> int:
        return sum(1 for call in self.calls if call["request_type"] == request_type)

    def last_response_system(self) -> str:
        response = [
            call for call in self.calls if call["request_type"] == "response"
        ][-1]
        return response["messages"][0]["content"]


def _event(person_id, text, *, event_id="evt_retry"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


async def _person(rt):
    return await rt.registry.register(
        "Ada",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


async def test_same_revision_retry_reuses_analysis_and_finishes(db):
    client = _FakeClient(failures=1)
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    await rt.agent_service.set_persona_core(
        {"character_description": OLD_PERSONA}, actor="admin"
    )
    event = _event(person.person_id, "你好")

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "FAILED"
    assert client.count("analysis") == 1
    assert client.count("response") == 1

    second = await rt.orchestrator.process_utterance(event)
    assert second.status == "COMPLETED"
    # The persisted analysis is reused: no third model call.
    assert client.count("analysis") == 1
    assert client.count("response") == 2
    assert OLD_PERSONA in client.last_response_system()


async def test_retry_after_persona_change_uses_the_stored_snapshot(db):
    client = _FakeClient(failures=1)
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    before = await rt.agent_service.set_persona_core(
        {"character_description": OLD_PERSONA}, actor="admin"
    )
    event = _event(person.person_id, "你好")

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "FAILED"

    after = await rt.agent_service.set_persona_core(
        {"character_description": NEW_PERSONA}, actor="admin"
    )
    assert after.revision == before.revision + 1

    second = await rt.orchestrator.process_utterance(event)
    assert second.status == "COMPLETED"
    system = client.last_response_system()
    assert OLD_PERSONA in system
    assert NEW_PERSONA not in system
    # The analysis itself was never regenerated under the new persona.
    assert client.count("analysis") == 1


async def test_changed_revision_without_a_snapshot_is_an_explicit_conflict(db):
    client = _FakeClient(failures=1)
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    await rt.agent_service.set_persona_core(
        {"character_description": OLD_PERSONA}, actor="admin"
    )
    event = _event(person.person_id, "你好")

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "FAILED"

    await db.conn.execute(
        "UPDATE cognitive_analyses SET persona_snapshot_json = NULL"
        " WHERE event_id = ?",
        (event.event_id,),
    )
    await db.conn.commit()
    await rt.agent_service.set_persona_core(
        {"character_description": NEW_PERSONA}, actor="admin"
    )

    second = await rt.orchestrator.process_utterance(event)
    assert second.status == "CONFLICT"
    assert "snapshot is unavailable" in (second.error or "")
    # No reply was generated under a mixed character version.
    assert client.count("response") == 1
    assert client.count("analysis") == 1
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status.value == "FAILED"


async def test_legacy_row_without_persona_metadata_is_a_conflict(db):
    client = _FakeClient(failures=1)
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    event = _event(person.person_id, "你好")

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "FAILED"

    await db.conn.execute(
        "UPDATE cognitive_analyses SET persona_agent_id = NULL,"
        " persona_revision = NULL, persona_snapshot_json = NULL"
        " WHERE event_id = ?",
        (event.event_id,),
    )
    await db.conn.commit()

    second = await rt.orchestrator.process_utterance(event)
    assert second.status == "CONFLICT"
    assert "legacy row" in (second.error or "")
    assert client.count("response") == 1


async def test_completed_turn_is_a_duplicate_not_a_second_response(db):
    client = _FakeClient(failures=0)
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    event = _event(person.person_id, "你好")

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "COMPLETED"
    duplicate = await rt.orchestrator.process_utterance(event)
    assert duplicate.status == "DUPLICATE"
    assert client.count("analysis") == 1
    assert client.count("response") == 1


# -- schema-16 compatibility ------------------------------------------------


async def _build_version_16(path):
    db = await Database(path).connect()
    await db.conn.execute(
        "CREATE TABLE schema_version ("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for version, sql in MIGRATIONS[:16]:
        for statement in _statements(sql):
            await db.conn.execute(statement)
        await db.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (version, "2026-01-01T00:00:00+00:00"),
        )
    await db.conn.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at,"
        " updated_at) VALUES ('c1', 0, '2026-01-01', '2026-01-01')"
    )
    await db.conn.execute(
        "INSERT INTO utterances(event_id, conversation_id, speaker_person_id,"
        " speaker_label, text, source, addressee_hint, occurred_at,"
        " processing_status, created_at) VALUES ('evt_legacy', 'c1', NULL,"
        " 'Ada', '旧消息', 'test', 'agent', '2026-01-01', 'completed',"
        " '2026-01-01')"
    )
    await db.conn.execute(
        "INSERT INTO cognitive_analyses(analysis_id, event_id, meaning, action,"
        " reason_code, provider, referenced_memory_ids, response_style, created_at)"
        " VALUES ('ana_legacy', 'evt_legacy', '旧分析', 'RESPOND',"
        " 'direct_agent_address', 'fake', '[]', 'plain', '2026-01-01')"
    )
    await db.conn.commit()
    await db.close()


async def test_schema_16_rows_read_as_unknown_persona_not_as_matching(tmp_path):
    path = tmp_path / "v16.memory"
    await _build_version_16(path)

    db = await Database(path).connect()
    try:
        await db.initialize()
        cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 30

        repo = AnalysisRepository(db)
        stored = await repo.get_analysis_persona("evt_legacy")
        assert stored == {"agent_id": None, "revision": None, "snapshot": None}

        # Old analysis content is unchanged.
        cur = await db.execute(
            "SELECT meaning FROM cognitive_analyses WHERE event_id = 'evt_legacy'"
        )
        assert (await cur.fetchone())["meaning"] == "旧分析"

        cur = await db.execute("PRAGMA foreign_key_check")
        assert await cur.fetchall() == []
    finally:
        await db.close()
