"""V0.9.9.31.1 §8: the full Speaker -> Person -> Profile chain.

These offline integration tests join the pieces that are unit-tested
elsewhere: the matcher snapshot feeds the identity resolver, the resolved
Person ID (never the Speaker ID) labels the submitted ``UtteranceEvent``, and
the link service follows that person to their profile without mixing two
people, inventing a person for an unknown voice, or treating a missing
profile as an unknown person.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.identity_resolver import (
    IdentityResolver,
    IdentityState,
)
from companion.people.speaker_profile_link import (
    PROFILE_MISSING,
    PROFILE_NO_BINDING,
    PROFILE_READY,
    PROFILE_UNKNOWN_SPEAKER,
    SpeakerProfileLinkService,
)
from companion.speaker_id.consent import EnrollmentPolicy
from companion.speaker_id.models import (
    NEW_CANDIDATE,
    QUALITY_HIGH,
    SpeakerResult,
)
from companion.speaker_id.store import SpeakerStore


@pytest.fixture
def store(tmp_path):
    value = SpeakerStore(tmp_path / "speakers.sqlite3")
    value.initialize()
    yield value
    value.close()


def stable(store: SpeakerStore) -> str:
    speaker_id = store.promote_speaker(store.create_candidate())
    assert speaker_id is not None
    return speaker_id


def known(speaker_id: str, *, similarity=0.82, margin=0.25, uid=1):
    return SpeakerResult(
        utterance_id=uid,
        speaker_id=speaker_id,
        person_id=None,
        status="KNOWN",
        reid_quality=QUALITY_HIGH,
        similarity=similarity,
        margin=margin,
        voiced_seconds=2.2,
        provisional=False,
        model_id="fake",
        inference_ms=1.0,
        completed_at=0.0,
    )


def new_candidate(speaker_id: str, *, uid=1):
    return SpeakerResult(
        utterance_id=uid,
        speaker_id=speaker_id,
        person_id=None,
        status=NEW_CANDIDATE,
        reid_quality=QUALITY_HIGH,
        similarity=None,
        margin=None,
        voiced_seconds=2.2,
        provisional=True,
        model_id="fake",
        inference_ms=1.0,
        completed_at=0.0,
    )


def resolver(rt, store):
    return IdentityResolver(rt.repos.person, rt.speaker_bindings, store)


def link_service(rt, store):
    return SpeakerProfileLinkService(
        rt.speaker_bindings,
        rt.repos.person,
        store,
        rt.profile_service,
        audit=rt.repos.audit,
    )


async def person(rt, name, *, processing=True, persistence=True, learning=True):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=processing,
        allow_persistence=persistence,
        allow_memory_learning=learning,
    )


async def submit(rt, person_id, text, conversation="conv-a"):
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.VOICE,
    )
    report = await rt.orchestrator.process_utterance(event)
    return event, report


# ---------------------------------------------------------------- full chain
async def test_submitted_event_carries_the_person_not_the_speaker(rt, store):
    owner = await person(rt, "Owner")
    speaker_id = stable(store)
    link = link_service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)

    identity = await resolver(rt, store).resolve(
        known(speaker_id), run_id="run-a", utterance_id=1
    )
    assert identity.state is IdentityState.CONFIRMED
    assert identity.person_id == owner.person_id
    assert identity.canonical_speaker_id == speaker_id

    event, report = await submit(rt, identity.person_id, "hello Ava")
    stored = await rt.repos.utterance.get(event.event_id)

    assert report.status == "COMPLETED"
    assert stored is not None
    assert stored.speaker_person_id == owner.person_id
    assert stored.speaker_person_id != speaker_id

    resolution = await link.resolve_profile(speaker_id)
    assert resolution.state == PROFILE_READY
    assert resolution.person_id == owner.person_id
    assert resolution.profile is not None


async def test_two_people_interleaved_never_mix(rt, store):
    first = await person(rt, "First")
    second = await person(rt, "Second")
    first_voice = stable(store)
    second_voice = stable(store)
    link = link_service(rt, store)
    await link.confirm_link(first_voice, first.person_id)
    await link.confirm_link(second_voice, second.person_id)

    resolver_ = resolver(rt, store)
    sequence = [
        (first_voice, first.person_id, 1),
        (second_voice, second.person_id, 2),
        (first_voice, first.person_id, 3),
    ]
    events = []
    for speaker_id, expected_person, uid in sequence:
        identity = await resolver_.resolve(
            known(speaker_id, uid=uid), run_id="run-a", utterance_id=uid
        )
        assert identity.state is IdentityState.CONFIRMED
        assert identity.person_id == expected_person
        event, report = await submit(
            rt,
            identity.person_id,
            f"turn {uid}",
            conversation=f"conv-{uid}",
        )
        assert report.status == "COMPLETED"
        events.append((event, expected_person))

    for event, expected_person in events:
        stored = await rt.repos.utterance.get(event.event_id)
        assert stored is not None
        assert stored.speaker_person_id == expected_person

    first_profile = await link.resolve_profile(first_voice)
    second_profile = await link.resolve_profile(second_voice)
    assert first_profile.person_id == first.person_id
    assert second_profile.person_id == second.person_id
    assert first_profile.profile.person_id == first.person_id
    assert second_profile.profile.person_id == second.person_id
    assert first_profile.profile.display_name == "First"
    assert second_profile.profile.display_name == "Second"


async def test_unknown_voice_gets_neither_person_nor_profile(rt, store):
    temporary = store.create_candidate()
    link = link_service(rt, store)

    identity = await resolver(rt, store).resolve(
        new_candidate(temporary), run_id="run-a", utterance_id=1
    )
    assert identity.state is IdentityState.TENTATIVE
    assert identity.person_id is None

    resolution = await link.resolve_profile(temporary)
    assert resolution.state == PROFILE_NO_BINDING
    assert resolution.person_id is None
    assert resolution.profile is None

    preview = await link.preview_link(temporary)
    assert preview.can_link is False
    # Nothing was created for an unknown voice.
    assert await rt.speaker_bindings.get_binding(temporary) is None


async def test_an_unknown_temporary_id_has_no_profile(rt, store):
    link = link_service(rt, store)
    resolution = await link.resolve_profile("TEMP-9999")
    assert resolution.state in (PROFILE_UNKNOWN_SPEAKER, PROFILE_NO_BINDING)
    assert resolution.person_id is None


# ------------------------------------------------------- profile vs identity
async def test_a_missing_profile_does_not_make_the_person_unknown(rt, store):
    owner = await person(rt, "Owner")
    speaker_id = stable(store)
    link = link_service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)
    await rt.db.execute(
        "DELETE FROM person_interaction_profiles WHERE person_id = ?",
        (owner.person_id,),
    )

    identity = await resolver(rt, store).resolve(
        known(speaker_id), run_id="run-a", utterance_id=1
    )
    resolution = await link.resolve_profile(speaker_id)
    event, report = await submit(rt, identity.person_id, "still me")

    assert identity.state is IdentityState.CONFIRMED
    assert identity.person_id == owner.person_id
    assert resolution.state == PROFILE_MISSING
    assert resolution.person_id == owner.person_id
    assert resolution.display_name == "Owner"
    assert report.status == "COMPLETED"


async def test_consent_flags_remain_independent_after_linking(rt, store):
    owner = await person(rt, "Owner", learning=False, processing=True)
    speaker_id = stable(store)
    link = link_service(rt, store)

    await link.confirm_link(speaker_id, owner.person_id)

    consent = await rt.repos.person.get_consent(owner.person_id)
    assert consent is not None
    assert consent.allow_processing is True
    assert consent.allow_persistence is True
    assert consent.allow_memory_learning is False
    assert consent.allow_external_ai is False
    assert await rt.profile_service.can_learn(owner.person_id) is False

    identity = await resolver(rt, store).resolve(
        known(speaker_id), run_id="run-a", utterance_id=1
    )
    resolution = await link.resolve_profile(speaker_id)
    assert identity.state is IdentityState.CONFIRMED
    assert resolution.state == PROFILE_READY
    # Linking never granted voiceprint storage either.
    assert store.enrollment_consent(speaker_id) is None


async def test_withdrawn_processing_consent_refuses_the_identity(rt, store):
    owner = await person(rt, "Owner")
    speaker_id = stable(store)
    link = link_service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)
    consent = await rt.repos.person.get_consent(owner.person_id)
    consent.allow_processing = False
    await rt.registry.set_consent(consent)

    identity = await resolver(rt, store).resolve(
        known(speaker_id), run_id="run-a", utterance_id=1
    )
    resolution = await link.resolve_profile(speaker_id)

    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None
    # The person is still known; only the profile content is withheld.
    assert resolution.person_id == owner.person_id
    assert resolution.state == "consent_missing"


# ------------------------------------------------------------- persistence
async def test_alias_bound_before_promotion_resolves_after_restart(rt, store):
    owner = await person(rt, "Owner")
    temporary = store.create_candidate()
    # Give the candidate a sample so promotion is possible.
    from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config

    config = load_speaker_config(env={})
    import numpy as np

    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = 1.0
    store.add_embedding(
        temporary,
        vector,
        quality=QUALITY_HIGH,
        voiced_seconds=2.0,
        model_id=f"{config.model_id}@{config.model_revision}",
        embedding_version=config.embedding_version,
        is_verified_reference=True,
    )
    link = link_service(rt, store)
    await link.confirm_link(temporary, owner.person_id)
    canonical = store.promote_speaker(temporary)
    assert canonical is not None

    # "Restart": the main DB keeps the binding; a fresh service reads it.
    restarted = SpeakerProfileLinkService(
        rt.speaker_bindings,
        rt.repos.person,
        store,
        rt.profile_service,
        audit=rt.repos.audit,
    )
    identity = await resolver(rt, store).resolve(
        known(canonical), run_id="run-b", utterance_id=1
    )
    resolution = await restarted.resolve_profile(temporary)

    assert identity.state is IdentityState.CONFIRMED
    assert identity.person_id == owner.person_id
    assert identity.raw_speaker_id == canonical
    assert resolution.person_id == owner.person_id
    assert resolution.state == PROFILE_READY


async def test_revoked_storage_consent_survives_a_store_restart(rt, store, tmp_path):
    owner = await person(rt, "Owner")
    speaker_id = stable(store)
    link = link_service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)
    policy = EnrollmentPolicy()
    policy.authorize(speaker_id, store=store)
    assert store.enrollment_consent(speaker_id)["status"] == "GRANTED"

    policy.revoke(speaker_id, store=store)
    store.close()

    reopened = SpeakerStore(tmp_path / "speakers.sqlite3")
    reopened.initialize()
    try:
        row = reopened.enrollment_consent(speaker_id)
        assert row is not None and row["status"] == "REVOKED"
        restored = EnrollmentPolicy()
        restored.load_from_store(reopened)
        assert restored.is_authorized(speaker_id) is False
        # Recognition and the person binding are independent of storage.
        identity = await resolver(rt, reopened).resolve(
            known(speaker_id), run_id="run-c", utterance_id=1
        )
        assert identity.state is IdentityState.CONFIRMED
        assert identity.person_id == owner.person_id
    finally:
        reopened.close()


async def test_person_id_is_never_written_into_the_speaker_store(rt, store):
    owner = await person(rt, "Owner")
    speaker_id = stable(store)
    link = link_service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)
    await link.resolve_profile(speaker_id)

    speakers = store.list_speakers()
    assert speakers
    dumped = repr(speakers)
    assert owner.person_id not in dumped
    assert "Owner" not in dumped
