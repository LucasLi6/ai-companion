"""V0.7 person understanding profile.

Offline only: local SQLite, no provider client, no network.
"""

from __future__ import annotations

import sqlite3

import pytest

from companion.cognition.context_builder import ContextBuilder
from companion.cognition.models import ROLE_HUMAN, CognitiveContext, CognitiveTurn
from companion.domain.enums import (
    EventSource,
    MemoryKind,
    RegistrationStatus,
)
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.memory.models import AccessContext
from companion.memory.service import build_memory_service
from companion.people.profile_compiler import MAX_BLOCK_CHARS, compile_profile_context
from companion.people.profile_learning import PersonProfileLearningService
from companion.people.profile_models import (
    ProfileFieldKey,
    ProfileFieldStatus,
    ProfileSource,
)
from companion.people.profile_service import PersonProfileService
from companion.runtime import build_runtime
from companion.storage.database import Database


def learning(rt) -> PersonProfileLearningService:
    return PersonProfileLearningService(rt.profile_service)


async def test_registration_creates_empty_profile_with_persistence(rt, register):
    person = await register("Alice")
    profile = await rt.profile_service.get(person.person_id)
    assert profile is not None
    assert profile.fields == {}
    assert profile.metadata.revision == 1
    assert profile.compact_summary == ""


async def test_no_profile_without_persistence(rt, register):
    person = await register("NoPersist", persistence=False)
    assert await rt.profile_service.get(person.person_id) is None
    result = await learning(rt).observe(person.person_id, "不用这么正式。")
    assert result.authorized is False
    assert await rt.profile_service.get(person.person_id) is None


async def test_ensure_is_idempotent(rt, register):
    person = await register("Idem")
    service: PersonProfileService = rt.profile_service
    assert await service.ensure_profile(person.person_id) is False
    assert await service.ensure_profile(person.person_id) is False


async def test_explicit_preference_updates_profile_with_evidence(rt, register):
    person = await register("Pref")
    result = await learning(rt).observe(
        person.person_id, "你以后跟我说话不用这么正式。", event_id="evt-pref"
    )
    assert result.authorized
    profile = await rt.profile_service.get(person.person_id)
    field = profile.fields[ProfileFieldKey.FORMALITY.value]
    assert field.value == "casual"
    assert field.source == ProfileSource.DIRECT_SELF_REPORT
    assert field.evidence_event_ids == ["evt-pref"]
    assert profile.metadata.revision == 2

    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert len(revisions) == 1
    assert revisions[0]["changes"]["field_key"] == ProfileFieldKey.FORMALITY.value


async def test_profiles_are_isolated_per_person(rt, register):
    alice = await register("Alice2")
    bob = await register("Bob2")
    learner = learning(rt)
    await learner.observe(alice.person_id, "你说话随意一点。", event_id="e-a")
    await learner.observe(bob.person_id, "请说中文。", event_id="e-b")

    a_profile = await rt.profile_service.get(alice.person_id)
    b_profile = await rt.profile_service.get(bob.person_id)
    assert a_profile.communication_value(ProfileFieldKey.FORMALITY) == "casual"
    assert a_profile.communication_value(ProfileFieldKey.LANGUAGE) == "UNKNOWN"
    assert b_profile.communication_value(ProfileFieldKey.LANGUAGE) == "zh"
    assert b_profile.communication_value(ProfileFieldKey.FORMALITY) == "UNKNOWN"


async def test_explicit_fact_stays_a_candidate_and_never_becomes_occupation(
    rt, register
):
    person = await register("Embedded")
    await learning(rt).observe(
        person.person_id, "我现在在学习嵌入式开发。", event_id="evt-fact"
    )
    profile = await rt.profile_service.get(person.person_id)
    # It is a current activity, not an occupation.
    assert ProfileFieldKey.OCCUPATION.value not in profile.fields
    occupations = [
        candidate
        for candidate in profile.candidates
        if candidate.field_key == ProfileFieldKey.OCCUPATION.value
    ]
    assert occupations == []
    projects = [
        candidate
        for candidate in profile.candidates
        if candidate.field_key == ProfileFieldKey.ONGOING_PROJECTS.value
    ]
    assert len(projects) == 1
    assert projects[0].candidate_value[0]["label"] == "嵌入式开发"
    assert projects[0].status.value == "PENDING"

    # A single chat or a single model confidence can never confirm it.
    assert profile.fields.get(ProfileFieldKey.ONGOING_PROJECTS.value) is None


async def test_single_inference_never_confirms(rt, register):
    person = await register("Infer")
    candidate = await learning(rt).record_inference(
        person.person_id,
        ProfileFieldKey.DEPTH.value,
        "concise",
        event_id="evt-1",
        confidence=0.99,
    )
    assert candidate is not None
    profile = await rt.profile_service.get(person.person_id)
    assert ProfileFieldKey.DEPTH.value not in profile.fields

    # Three independent evidence events may confirm, not one and not confidence.
    await learning(rt).record_inference(
        person.person_id,
        ProfileFieldKey.DEPTH.value,
        "concise",
        event_id="evt-2",
        confidence=0.2,
    )
    profile = await rt.profile_service.get(person.person_id)
    assert ProfileFieldKey.DEPTH.value not in profile.fields
    await learning(rt).record_inference(
        person.person_id,
        ProfileFieldKey.DEPTH.value,
        "concise",
        event_id="evt-3",
        confidence=0.1,
    )
    profile = await rt.profile_service.get(person.person_id)
    assert profile.fields[ProfileFieldKey.DEPTH.value].value == "concise"


async def test_correction_replaces_wrong_value_and_rejects_candidates(rt, register):
    person = await register("Correct")
    service: PersonProfileService = rt.profile_service
    await service.set_field(
        person.person_id,
        ProfileFieldKey.OCCUPATION.value,
        "老师",
        source=ProfileSource.DIRECT_SELF_REPORT,
        event_id="evt-old",
    )
    await service.propose_candidate(
        person.person_id,
        ProfileFieldKey.OCCUPATION.value,
        "医生",
        source=ProfileSource.INTERACTION_INFERENCE,
        event_id="evt-guess",
    )
    revision = await learning(rt).apply_correction(
        person.person_id,
        ProfileFieldKey.OCCUPATION.value,
        "工程师",
        event_id="evt-correct",
        reason="我说错了",
    )
    assert revision == "corrected"
    profile = await rt.profile_service.get(person.person_id)
    field = profile.fields[ProfileFieldKey.OCCUPATION.value]
    assert field.value == "工程师"
    assert field.source == ProfileSource.USER_CORRECTION
    pending = await rt.repos.profile.find_pending(
        person.person_id, ProfileFieldKey.OCCUPATION.value
    )
    assert pending == []
    statuses = [
        candidate["status"]
        for candidate in await rt.repos.profile.list_candidates(
            person.person_id, status="REJECTED"
        )
        if candidate["field_key"] == ProfileFieldKey.OCCUPATION.value
    ]
    assert "REJECTED" in statuses


async def test_correction_can_retire_a_field(rt, register):
    person = await register("Retire")
    service: PersonProfileService = rt.profile_service
    await service.set_field(
        person.person_id,
        ProfileFieldKey.ADDRESS.value,
        "小明",
        source=ProfileSource.DIRECT_SELF_REPORT,
    )
    await service.correct_field(
        person.person_id,
        ProfileFieldKey.ADDRESS.value,
        revoke=True,
        reason="别再叫我小明",
    )
    profile = await rt.profile_service.get(person.person_id)
    assert ProfileFieldKey.ADDRESS.value not in profile.fields
    retired = await rt.repos.profile.list_fields(
        person.person_id, status=ProfileFieldStatus.RETIRED.value
    )
    assert retired[0]["field_key"] == ProfileFieldKey.ADDRESS.value


async def test_no_learning_without_memory_learning_consent(rt, register):
    person = await register("NoLearn", processing=True, persistence=True, learning=False)
    result = await learning(rt).observe(
        person.person_id, "不用这么正式。", event_id="e1"
    )
    assert result.authorized is False
    profile = await rt.profile_service.get(person.person_id)
    assert profile.fields == {}


async def test_revoked_learning_stops_new_updates(rt, register):
    person = await register("RevokeLearn")
    learner = learning(rt)
    await learner.observe(person.person_id, "不用这么正式。", event_id="e1")
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    result = await learner.observe(person.person_id, "请说英文。", event_id="e2")
    assert result.authorized is False
    profile = await rt.profile_service.get(person.person_id)
    assert profile.communication_value(ProfileFieldKey.LANGUAGE) == "UNKNOWN"


async def test_profile_survives_database_restart(tmp_path):
    path = tmp_path / "restart.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db)
    person = await rt.registry.register(
        "Persistent",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    await learning(rt).observe(person.person_id, "不用这么正式。", event_id="e1")
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    profile = await rt2.profile_service.get(person.person_id)
    assert profile is not None
    assert profile.communication_value(ProfileFieldKey.FORMALITY) == "casual"
    assert profile.metadata.revision == 2
    await db2.close()


async def test_revocation_purges_profile_but_not_person(rt, register):
    person = await register("WillRevoke")
    await learning(rt).observe(person.person_id, "不用这么正式。", event_id="e1")
    assert await rt.profile_service.get(person.person_id) is not None
    await rt.registry.set_status(person.person_id, RegistrationStatus.REVOKED)
    assert await rt.profile_service.get(person.person_id) is None
    assert await rt.registry.get(person.person_id) is not None


async def test_withdrawing_persistence_purges_profile(rt, register):
    person = await register("Withdraw")
    await learning(rt).observe(person.person_id, "不用这么正式。", event_id="e1")
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=False,
            allow_memory_learning=False,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    assert await rt.profile_service.get(person.person_id) is None


async def test_creating_a_profile_never_grants_consent(rt, register):
    person = await register("NoGrant", processing=True, persistence=False, learning=False)
    await rt.profile_service.ensure_profile(
        person.person_id, enforce_authorization=False
    )
    consent = await rt.registry.get_consent(person.person_id)
    assert consent.allow_persistence is False
    assert consent.allow_memory_learning is False


async def test_compiler_is_bounded_and_hides_candidates(rt, register):
    person = await register("Compile")
    learner = learning(rt)
    await learner.observe(person.person_id, "不用这么正式。", event_id="e1")
    await rt.profile_service.propose_candidate(
        person.person_id,
        ProfileFieldKey.OCCUPATION.value,
        "嵌入式工程师",
        source=ProfileSource.INTERACTION_INFERENCE,
        event_id="e2",
    )
    profile = await rt.profile_service.get(person.person_id)
    block = compile_profile_context(profile)
    assert block is not None
    assert "formality=casual" in block
    assert "occupation=unknown" in block
    assert "嵌入式工程师" not in block
    assert len(block) <= MAX_BLOCK_CHARS


async def test_context_builder_only_injects_current_speakers_profile(rt, register):
    alice = await register("CtxAlice", external_ai=True)
    bob = await register("CtxBob", external_ai=True)
    await rt.profile_service.set_field(
        alice.person_id,
        ProfileFieldKey.INTERESTS.value,
        [{"label": "机器人"}],
        source=ProfileSource.DIRECT_SELF_REPORT,
        event_id="a1",
    )
    await rt.profile_service.set_field(
        bob.person_id,
        ProfileFieldKey.INTERESTS.value,
        [{"label": "古典音乐"}],
        source=ProfileSource.DIRECT_SELF_REPORT,
        event_id="b1",
    )

    builder = ContextBuilder(
        repos=rt.repos,
        registry=rt.registry,
        agent_service=rt.agent_service,
        profile_service=rt.profile_service,
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="你好。",
        addressee_hint="agent",
        speaker_person_id=alice.person_id,
        source=EventSource.TEST,
    )
    context = await builder.build(event, memories=[])
    assert context.person_profile_context is not None
    assert "机器人" in context.person_profile_context
    assert "古典音乐" not in context.person_profile_context
    assert alice.person_id in context.person_profile_context


async def test_profile_memory_reference_is_linked(rt, register):
    person = await register("Linked")
    service = build_memory_service(rt.repos)
    memory = await service.create(
        kind=MemoryKind.PROJECT_STATE,
        content="学习嵌入式开发",
        access=AccessContext(
            actor_id=person.person_id, is_admin=False, role="cognitive"
        ),
        subjects=[("PERSON", person.person_id)],
        self_reported=True,
    )
    learner = learning(rt)
    candidate = await learner.record_inference(
        person.person_id,
        ProfileFieldKey.ONGOING_PROJECTS.value,
        "嵌入式开发",
        event_id="evt-link",
    )
    assert candidate is not None
    await learner.link_output_memory(person.person_id, "evt-link", memory.memory_id)
    candidates = await rt.repos.profile.list_candidates(person.person_id)
    assert memory.memory_id in candidates[0]["evidence_memory_ids"]
    evidence = await rt.repos.profile.list_evidence(person.person_id)
    assert any(item["memory_id"] == memory.memory_id for item in evidence)


async def test_legacy_active_persons_are_backfilled(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        INSERT INTO schema_version VALUES (19, '2026-01-01');
        CREATE TABLE persons (
            person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            role TEXT NOT NULL, registration_status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE consents (
            person_id TEXT PRIMARY KEY, allow_processing INTEGER NOT NULL,
            allow_persistence INTEGER NOT NULL,
            allow_memory_learning INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE person_interaction_profiles (
            person_id TEXT PRIMARY KEY, relationship_tier TEXT NOT NULL DEFAULT 'STRANGER',
            style_overrides TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL);
        CREATE TABLE memory_items (memory_id TEXT PRIMARY KEY);
        INSERT INTO persons VALUES
            ('p_active', 'Active', 'friend', 'active', 't', 't'),
            ('p_revoked', 'Revoked', 'friend', 'revoked', 't', 't'),
            ('p_nopersist', 'NoPersist', 'guest', 'active', 't', 't');
        INSERT INTO consents VALUES
            ('p_active', 1, 1, 1, 't'),
            ('p_revoked', 1, 1, 1, 't'),
            ('p_nopersist', 1, 0, 0, 't');
        """
    )
    conn.commit()
    conn.close()

    db = await Database(path).connect()
    await db.initialize()
    cur = await db.execute(
        "SELECT person_id FROM person_interaction_profiles ORDER BY person_id"
    )
    rows = [row["person_id"] for row in await cur.fetchall()]
    assert rows == ["p_active"]
    await db.close()
