"""Test A: multi-person independence, duplicate names, correct attribution."""

from __future__ import annotations

import pytest

from companion.domain.enums import PersonRole


async def test_person_ids_are_unique(register):
    a = await register("Alice", "owner")
    b = await register("Bob", "friend")
    c = await register("Carol", "guest")
    ids = {a.person_id, b.person_id, c.person_id}
    assert len(ids) == 3


async def test_duplicate_names_are_allowed(register):
    a = await register("Alex", "friend")
    b = await register("Alex", "friend")
    assert a.person_id != b.person_id
    assert a.display_name == b.display_name == "Alex"


async def test_roles_and_consent_flags(register, rt):
    p = await register("Owner", "owner")
    assert p.role == PersonRole.OWNER
    consent = await rt.registry.get_consent(p.person_id)
    assert consent.allow_processing is True
    assert consent.allow_persistence is True
    assert consent.allow_memory_learning is True


async def test_consent_not_defaulted_on(register, rt):
    # Registering without flags must NOT default any consent to True.
    p = await register("Guest", "guest", processing=False, persistence=False, learning=False)
    consent = await rt.registry.get_consent(p.person_id)
    assert consent.allow_processing is False
    assert consent.allow_persistence is False
    assert consent.allow_memory_learning is False


async def test_utterances_attribute_to_correct_speaker(register, say, rt):
    alice = await register("Alice")
    bob = await register("Bob")
    r1 = await say(alice.person_id, "agent", "alice speaks")
    r2 = await say(bob.person_id, "agent", "bob speaks")
    assert r1.status == "COMPLETED"
    assert r2.status == "COMPLETED"

    e1 = await rt.repos.utterance.get(r1.event_id)
    e2 = await rt.repos.utterance.get(r2.event_id)
    assert e1.speaker_person_id == alice.person_id
    assert e2.speaker_person_id == bob.person_id
    assert e1.text == "alice speaks"
    assert e2.text == "bob speaks"
    # No cross-attribution.
    assert e1.speaker_person_id != bob.person_id
    assert e2.speaker_person_id != alice.person_id
