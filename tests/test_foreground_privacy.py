"""Phase 2 — foreground privacy gateway for external AI.

Uses the real ``PersonRegistry`` (over the in-memory SQLite ``rt`` fixture) so
the tests exercise the exact consent/status semantics the orchestrator relies
on, not a hand-written stub of them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway


@pytest.fixture
def gateway(rt):
    return ForegroundPrivacyGateway(rt.registry)


def _utterance(speaker_id):
    return SimpleNamespace(speaker_person_id=speaker_id)


# -- speaker authorization --------------------------------------------------


async def test_unregistered_speaker_denied(gateway):
    result = await gateway.authorize_event(None)
    assert not result.allowed
    assert result.reason == "unregistered_speaker"


async def test_unknown_person_denied(gateway):
    result = await gateway.authorize_event("person_does_not_exist")
    assert not result.allowed
    assert result.reason == "unknown_person"


async def test_processing_without_external_ai_denied(register, gateway):
    person = await register("Alice", processing=True, external_ai=False)
    result = await gateway.authorize_event(person.person_id)
    assert not result.allowed
    assert result.reason == "external_ai_not_granted"


async def test_external_ai_without_processing_denied(register, gateway):
    person = await register("Bob", processing=False, external_ai=True)
    result = await gateway.authorize_event(person.person_id)
    assert not result.allowed
    assert result.reason == "consent_withheld"


async def test_suspended_person_denied(register, rt, gateway):
    from companion.domain.enums import RegistrationStatus

    person = await register("Carol", processing=True, external_ai=True)
    await rt.registry.set_status(person.person_id, RegistrationStatus.SUSPENDED)
    result = await gateway.authorize_event(person.person_id)
    assert not result.allowed
    assert result.reason == "person_suspended"


async def test_fully_authorized_allowed(register, gateway):
    person = await register("Dave", processing=True, external_ai=True)
    result = await gateway.authorize_event(person.person_id)
    assert result.allowed
    assert result.reason == "authorized"


async def test_non_persistence_mode_still_allowed(register, gateway):
    """allow_persistence=False + allow_external_ai=True: external OK, no DB writes.

    The gateway checks only who may be *sent* data; persistence is separate.
    """
    person = await register(
        "Erin", processing=True, persistence=False, external_ai=True
    )
    result = await gateway.authorize_event(person.person_id)
    assert result.allowed


# -- multi-person filtering -------------------------------------------------


async def test_filter_person_ids_splits(register, gateway):
    a = await register("A", processing=True, external_ai=True)
    b = await register("B", processing=True, external_ai=False)
    allowed, denied = await gateway.filter_person_ids([a.person_id, b.person_id])
    assert allowed == [a.person_id]
    assert denied == [b.person_id]


async def test_filter_utterances_keeps_only_authorized_speakers(register, gateway):
    a = await register("A", processing=True, external_ai=True)
    b = await register("B", processing=True, external_ai=False)
    kept = await gateway.filter_utterances(
        [_utterance(a.person_id), _utterance(b.person_id), _utterance(None)]
    )
    assert [u.speaker_person_id for u in kept] == [a.person_id]


async def test_filter_memories_keeps_no_subject(register, gateway):
    a = await register("A", processing=True, external_ai=True)
    public = {"memory_id": "m1", "content": "world knowledge"}

    async def subjects(m):
        return []

    kept, dropped = await gateway.filter_memories([public], subjects)
    assert kept == [public]
    assert dropped == []


async def test_filter_memories_drops_multisubject_with_one_denied(register, gateway):
    a = await register("A", processing=True, external_ai=True)
    b = await register("B", processing=True, external_ai=False)
    shared = {"memory_id": "m2", "content": "about A and B"}

    async def subjects(m):
        return [a.person_id, b.person_id]

    kept, dropped = await gateway.filter_memories([shared], subjects)
    assert kept == []
    assert dropped == [shared]


async def test_filter_memories_keeps_all_authorized_subjects(register, gateway):
    a = await register("A", processing=True, external_ai=True)
    b = await register("B", processing=True, external_ai=True)
    shared = {"memory_id": "m3", "content": "about A and B"}

    async def subjects(m):
        return [a.person_id, b.person_id]

    kept, dropped = await gateway.filter_memories([shared], subjects)
    assert kept == [shared]
    assert dropped == []
