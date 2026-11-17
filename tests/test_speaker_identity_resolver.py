"""V0.9.9.31: conservative speaker -> person identity resolution.

These offline tests pin the four independent gates. Similarity is evidence for
the engine's matcher, never an authorisation: no binding, no person. Every
uncertain, ambiguous, provisional or failed match resolves to UNKNOWN and
never to the startup Owner.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import RegistrationStatus
from companion.people.identity_resolver import (
    IdentityResolver,
    IdentityState,
    IdentityTrustPolicy,
    REASON_AMBIGUOUS_MATCH,
    REASON_BINDING_CONFLICT,
    REASON_CONSENT_MISSING,
    REASON_MODEL_UNAVAILABLE,
    REASON_PERSON_INACTIVE,
    REASON_PROVISIONAL_MATCH,
    REASON_SPEAKER_UNBOUND,
    REASON_TRUSTED_BINDING,
)
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    PENDING,
    QUALITY_HIGH,
    QUALITY_LOW,
    QUALITY_MEDIUM,
    SpeakerResult,
)
from companion.speaker_id.store import SpeakerStore


def make_store(tmp_path) -> SpeakerStore:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    return store


def stable_speaker(store: SpeakerStore) -> str:
    temporary = store.create_candidate()
    speaker_id = store.promote_speaker(temporary)
    assert speaker_id is not None
    return speaker_id


def result(
    speaker_id="S-0001",
    *,
    status=KNOWN,
    quality=QUALITY_HIGH,
    provisional=False,
    similarity=0.62,
    margin=0.12,
):
    return SpeakerResult(
        utterance_id=1,
        speaker_id=speaker_id,
        person_id=None,
        status=status,
        reid_quality=quality,
        similarity=similarity,
        margin=margin,
        voiced_seconds=2.4,
        provisional=provisional,
        model_id="fake",
        inference_ms=1.0,
        completed_at=0.0,
    )


async def person(rt, name="Owner", **consent):
    values = dict(
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=False,
    )
    values.update(consent)
    return await rt.registry.register(name, "owner", **values)


def resolver(rt, store, policy=None) -> IdentityResolver:
    return IdentityResolver(
        rt.repos.person,
        rt.speaker_bindings,
        store,
        policy=policy or IdentityTrustPolicy(),
    )


@pytest.fixture
def store(tmp_path):
    value = make_store(tmp_path)
    yield value
    value.close()


async def resolve(rt, store, snapshot, uid=1):
    return await resolver(rt, store).resolve(
        snapshot, run_id="run-a", utterance_id=uid
    )


# ---------------------------------------------------------------- confirmed
async def test_known_reliable_match_with_binding_and_consent_confirms(rt, store):
    speaker_id = stable_speaker(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "explicit_confirmation"
    )

    identity = await resolve(rt, store, result(speaker_id))

    assert identity.state is IdentityState.CONFIRMED
    assert identity.person_id == owner.person_id
    assert identity.display_name == owner.display_name
    assert identity.canonical_speaker_id == speaker_id
    assert identity.reason == REASON_TRUSTED_BINDING
    assert identity.confirmed is True


async def test_known_match_without_binding_never_becomes_a_person(rt, store):
    speaker_id = stable_speaker(store)
    owner = await person(rt)
    assert (await rt.registry.get(owner.person_id)) is not None

    identity = await resolve(rt, store, result(speaker_id))

    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None
    assert identity.reason == REASON_SPEAKER_UNBOUND


async def test_high_similarity_is_not_authorization(rt, store):
    speaker_id = stable_speaker(store)
    await person(rt)

    identity = await resolve(
        rt, store, result(speaker_id, similarity=0.999, margin=0.5)
    )

    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None


async def test_known_provisional_or_low_quality_is_not_identity(rt, store):
    low = stable_speaker(store)
    provisional = await resolve(rt, store, result(low, provisional=True))
    assert provisional.state is IdentityState.UNKNOWN
    assert provisional.reason == REASON_PROVISIONAL_MATCH

    medium = stable_speaker(store)
    policy = IdentityTrustPolicy(minimum_quality=QUALITY_HIGH)
    tight = await resolver(rt, store, policy=policy).resolve(
        result(medium, quality=QUALITY_MEDIUM),
        run_id="run",
        utterance_id=1,
    )
    assert tight.state is IdentityState.UNKNOWN
    assert tight.reason == REASON_PROVISIONAL_MATCH

    unknown_quality = await resolve(
        rt, store, result(medium, quality=QUALITY_LOW)
    )
    assert unknown_quality.state is IdentityState.UNKNOWN


async def test_missing_similarity_or_margin_is_not_reliable(rt, store):
    speaker_id = stable_speaker(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "owner"
    )

    no_margin = await resolve(rt, store, result(speaker_id, margin=None))
    assert no_margin.state is IdentityState.UNKNOWN
    assert no_margin.reason == REASON_PROVISIONAL_MATCH

    no_similarity = await resolve(rt, store, result(speaker_id, similarity=None))
    assert no_similarity.state is IdentityState.UNKNOWN


# --------------------------------------------------------- tentative/unknown
async def test_candidate_match_is_tentative_and_grants_no_person(rt, store):
    speaker_id = stable_speaker(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "owner"
    )

    identity = await resolve(
        rt, store, result(speaker_id, status=CANDIDATE_MATCH)
    )

    # A candidate match is display-only: even a bound voice does not get a
    # private person from a weaker match.
    assert identity.state is IdentityState.TENTATIVE
    assert identity.person_id is None
    assert identity.reason == REASON_PROVISIONAL_MATCH
    assert identity.display_name == f"speaker {speaker_id}"


async def test_new_candidate_is_tentative_with_temporary_display(rt, store):
    temporary = store.create_candidate()

    identity = await resolve(
        rt, store, result(temporary, status=NEW_CANDIDATE, quality=QUALITY_MEDIUM)
    )

    assert identity.state is IdentityState.TENTATIVE
    assert identity.person_id is None
    assert identity.reason == REASON_SPEAKER_UNBOUND
    assert identity.display_name == f"speaker {temporary}"


@pytest.mark.parametrize(
    "status",
    [PENDING, ERROR, INSUFFICIENT_AUDIO, AMBIGUOUS],
)
async def test_undecided_statuses_are_unknown(rt, store, status):
    speaker_id = stable_speaker(store)
    identity = await resolve(rt, store, result(speaker_id, status=status))
    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None
    assert identity.display_name == "Unknown"
    expected = (
        REASON_AMBIGUOUS_MATCH
        if status == AMBIGUOUS
        else REASON_MODEL_UNAVAILABLE
    )
    assert identity.reason == expected


async def test_no_result_or_missing_model_is_unknown(rt, store):
    identity = await resolve(rt, store, None)
    assert identity.state is IdentityState.UNKNOWN
    assert identity.reason == REASON_MODEL_UNAVAILABLE


async def test_unknown_or_deleted_speaker_id_is_unknown(rt, store):
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, "S-9999", "owner"
    )

    identity = await resolve(rt, store, result("S-9999"))

    # A binding cannot resurrect a speaker id that has no live profile.
    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None
    assert identity.reason == REASON_SPEAKER_UNBOUND


# --------------------------------------------------------- person/consent gates
async def test_suspended_person_never_resolves(rt, store):
    speaker_id = stable_speaker(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "owner"
    )
    await rt.registry.set_status(
        owner.person_id, RegistrationStatus.SUSPENDED
    )

    identity = await resolve(rt, store, result(speaker_id))

    assert identity.state is IdentityState.UNKNOWN
    assert identity.reason == REASON_PERSON_INACTIVE


async def test_revoked_processing_consent_stops_immediately(rt, store):
    speaker_id = stable_speaker(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "owner"
    )
    consent = await rt.registry.get_consent(owner.person_id)
    consent.allow_processing = False
    await rt.registry.set_consent(consent)

    identity = await resolve(rt, store, result(speaker_id))

    assert identity.state is IdentityState.UNKNOWN
    assert identity.reason == REASON_CONSENT_MISSING


async def test_revoked_persistence_consent_stops_immediately(rt, store):
    speaker_id = stable_speaker(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "owner"
    )
    consent = await rt.registry.get_consent(owner.person_id)
    consent.allow_persistence = False
    await rt.registry.set_consent(consent)

    identity = await resolve(rt, store, result(speaker_id))
    assert identity.state is IdentityState.UNKNOWN
    assert identity.reason == REASON_CONSENT_MISSING


async def test_limit_preserves_stable_reason_codes():
    # These strings are logged/asserted; accidental renames are regressions.
    assert {
        REASON_TRUSTED_BINDING,
        REASON_SPEAKER_UNBOUND,
        REASON_PROVISIONAL_MATCH,
        REASON_AMBIGUOUS_MATCH,
        REASON_MODEL_UNAVAILABLE,
        REASON_BINDING_CONFLICT,
        REASON_CONSENT_MISSING,
        REASON_PERSON_INACTIVE,
    } == {
        "trusted_binding",
        "speaker_unbound",
        "provisional_match",
        "ambiguous_match",
        "speaker_model_unavailable",
        "binding_conflict",
        "consent_missing",
        "person_inactive",
    }
