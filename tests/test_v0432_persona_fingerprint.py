"""V0.4.3.2: retry consistency is decided by the complete snapshot.

The Persona Core revision is only one field. Traits, description, relationship
tier, stances and emotional dispositions can change without a revision bump,
so the retry path compares a deterministic SHA-256 fingerprint of the whole
CharacterSnapshot. Every test here enters through the real Orchestrator retry
path and inspects what the response provider actually received (or the exact
snapshot it was given when a field is not rendered in the prompt).
"""

from __future__ import annotations

import json
import math

import pytest

from companion.agent.character import (
    CharacterSnapshotValidationError,
    character_snapshot_fingerprint,
    validate_stored_snapshot,
)
from companion.domain.enums import EventSource
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime


def _snapshot(**updates):
    payload = {
        "agent_id": "agent_1",
        "revision": 4,
        "display_name": "Ava",
        "description": "安静的人",
        "traits": {"warmth": 55, "humor": 40},
        "persona_core": {"core_values": ["诚实"]},
        "character_description": "说话具体",
        "relationship_tier": "STRANGER",
        "relevant_stances": [
            {"topic": "工作", "stance": "我不喜欢加班。", "source": "ADMIN"}
        ],
        "emotional_dispositions": {"expressiveness": 0.5},
    }
    payload.update(updates)
    return payload


class _FakeClient:
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
                    "meaning": "用户在分享工作",
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

    def last_system(self) -> str:
        response = [
            call for call in self.calls if call["request_type"] == "response"
        ][-1]
        return response["messages"][0]["content"]


class _RecordingResponse:
    """Wraps the real provider and records the exact context it was handed."""

    def __init__(self, inner):
        self.inner = inner
        self.snapshots: list[dict] = []

    async def generate(self, event, analysis, context):
        self.snapshots.append(dict(context.character_snapshot or {}))
        return await self.inner.generate(event, analysis, context)


def _event(person_id, text="今天加班到十点", *, event_id="evt_retry"):
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


async def _runtime(db):
    client = _FakeClient(failures=1)
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    return rt, client, person


# -- fingerprint -------------------------------------------------------------


def test_identical_snapshots_have_the_same_fingerprint():
    first = _snapshot()
    second = _snapshot()
    assert character_snapshot_fingerprint(first) == character_snapshot_fingerprint(
        second
    )
    assert len(character_snapshot_fingerprint(first)) == 64


def test_dict_key_order_does_not_change_the_fingerprint():
    first = _snapshot()
    reordered = {key: first[key] for key in reversed(list(first))}
    assert character_snapshot_fingerprint(first) == character_snapshot_fingerprint(
        reordered
    )


def test_non_deterministic_values_fail_loudly():
    with pytest.raises(CharacterSnapshotValidationError):
        character_snapshot_fingerprint(
            _snapshot(emotional_dispositions={"expressiveness": math.nan})
        )
    with pytest.raises(CharacterSnapshotValidationError):
        character_snapshot_fingerprint(_snapshot(traits={"warmth": object()}))


def test_stored_snapshot_validation_checks_structure_and_metadata():
    stored = {
        "agent_id": "agent_1",
        "revision": 4,
        "snapshot": _snapshot(),
    }
    assert validate_stored_snapshot(stored)["agent_id"] == "agent_1"

    with pytest.raises(CharacterSnapshotValidationError, match="legacy row"):
        validate_stored_snapshot(
            {"agent_id": None, "revision": None, "snapshot": _snapshot()}
        )
    with pytest.raises(CharacterSnapshotValidationError, match="unavailable"):
        validate_stored_snapshot(
            {"agent_id": "agent_1", "revision": 4, "snapshot": None}
        )
    with pytest.raises(CharacterSnapshotValidationError, match="field"):
        validate_stored_snapshot(
            {
                "agent_id": "agent_1",
                "revision": 4,
                "snapshot": {**_snapshot(), "traits": "not a mapping"},
            }
        )
    with pytest.raises(CharacterSnapshotValidationError, match="agent_id"):
        validate_stored_snapshot(
            {
                "agent_id": "agent_1",
                "revision": 4,
                "snapshot": _snapshot(agent_id="agent_2"),
            }
        )
    with pytest.raises(CharacterSnapshotValidationError, match="revision"):
        validate_stored_snapshot(
            {
                "agent_id": "agent_1",
                "revision": 5,
                "snapshot": _snapshot(revision=4),
            }
        )


# -- retry: content change without a revision bump ---------------------------


async def test_retry_restores_snapshot_when_only_traits_changed(db):
    rt, client, person = await _runtime(db)
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await rt.agent_service.set_persona({"warmth": 95})

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "COMPLETED"
    system = client.last_system()
    assert "温暖=中等" in system
    assert "温暖=高" not in system
    assert client.count("analysis") == 1


async def test_retry_restores_snapshot_when_relationship_tier_changed(db):
    rt, client, person = await _runtime(db)
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await rt.agent_service.set_relationship(person.person_id, "CLOSE")

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "COMPLETED"
    system = client.last_system()
    assert "与当前说话者的关系: STRANGER" in system
    assert "与当前说话者的关系: CLOSE" not in system


async def test_retry_restores_snapshot_when_only_dispositions_changed(db):
    rt, client, person = await _runtime(db)
    await rt.repos.settings.set(
        "persona.emotional_dispositions",
        json.dumps({"expressiveness": 0.1}),
    )
    recording = _RecordingResponse(rt.orchestrator.response)
    rt.orchestrator.response = recording
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await rt.repos.settings.set(
        "persona.emotional_dispositions",
        json.dumps({"expressiveness": 0.9}),
    )

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "COMPLETED"
    assert len(recording.snapshots) == 2
    restored = recording.snapshots[-1]
    assert restored["emotional_dispositions"]["expressiveness"] == 0.1
    assert character_snapshot_fingerprint(restored) != (
        character_snapshot_fingerprint(
            {**restored, "emotional_dispositions": {"expressiveness": 0.9}}
        )
    )


async def test_retry_restores_snapshot_when_only_stances_changed(db):
    rt, client, person = await _runtime(db)
    await rt.agent_service.add_stance(
        "我不喜欢熬夜。", topic="工作", person_id=person.person_id
    )
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await rt.agent_service.add_stance(
        "我支持通宵把事做完。", topic="工作", person_id=person.person_id
    )

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "COMPLETED"
    system = client.last_system()
    assert "我不喜欢熬夜。" in system
    assert "我支持通宵把事做完。" not in system


async def test_retry_restores_snapshot_when_persona_revision_changed(db):
    rt, client, person = await _runtime(db)
    await rt.agent_service.set_persona_core(
        {"character_description": "旧人格：说话很慢。"}, actor="admin"
    )
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await rt.agent_service.set_persona_core(
        {"character_description": "新人格：说话很快。"}, actor="admin"
    )

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "COMPLETED"
    system = client.last_system()
    assert "旧人格：说话很慢。" in system
    assert "新人格：说话很快。" not in system


# -- corrupt / inconsistent snapshots ---------------------------------------


async def test_missing_stored_snapshot_is_a_conflict(db):
    rt, client, person = await _runtime(db)
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await db.conn.execute(
        "UPDATE cognitive_analyses SET persona_snapshot_json = NULL"
        " WHERE event_id = ?",
        (event.event_id,),
    )
    await db.conn.commit()

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "CONFLICT"
    assert "snapshot is unavailable" in (retry.error or "")
    assert client.count("response") == 1


async def test_corrupt_snapshot_json_is_a_conflict(db):
    rt, client, person = await _runtime(db)
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await db.conn.execute(
        "UPDATE cognitive_analyses SET persona_snapshot_json = ?"
        " WHERE event_id = ?",
        ("{not valid json", event.event_id),
    )
    await db.conn.commit()

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "CONFLICT"
    assert "snapshot is unavailable" in (retry.error or "")
    assert client.count("response") == 1


async def test_snapshot_agent_id_disagreeing_with_metadata_is_a_conflict(db):
    rt, client, person = await _runtime(db)
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    cur = await db.execute(
        "SELECT persona_agent_id, persona_revision, persona_snapshot_json"
        " FROM cognitive_analyses WHERE event_id = ?",
        (event.event_id,),
    )
    row = await cur.fetchone()
    snapshot = json.loads(row["persona_snapshot_json"])
    snapshot["agent_id"] = "agent_other"
    await db.conn.execute(
        "UPDATE cognitive_analyses SET persona_snapshot_json = ?"
        " WHERE event_id = ?",
        (json.dumps(snapshot, ensure_ascii=False), event.event_id),
    )
    await db.conn.commit()

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "CONFLICT"
    assert "agent_id" in (retry.error or "")
    assert client.count("response") == 1


# -- authorization -----------------------------------------------------------


async def test_withdrawn_external_ai_blocks_restoring_the_old_snapshot(db):
    rt, client, person = await _runtime(db)
    await rt.agent_service.set_persona_core(
        {"character_description": "旧人格：说话很慢。"}, actor="admin"
    )
    event = _event(person.person_id)

    assert (await rt.orchestrator.process_utterance(event)).status == "FAILED"
    await rt.agent_service.set_persona_core(
        {"character_description": "新人格：说话很快。"}, actor="admin"
    )
    now = utcnow()
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=False,
            updated_at=now,
        )
    )

    retry = await rt.orchestrator.process_utterance(event)
    assert retry.status == "CONFLICT"
    assert "authorization was withdrawn" in (retry.error or "")
    # The previously failed call is the only response request; the old
    # snapshot was never resent.
    assert client.count("response") == 1
