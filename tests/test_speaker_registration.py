"""V0.9.9.31.1 §9: guided local voice registration.

The flow is offline and explicit: consent is checked before anything is
armed, only microphone observations count, the ready state never binds by
itself, and a human confirmation is the only path to an identity, a binding
and a durable voiceprint grant.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from companion.domain.enums import RegistrationStatus
from companion.people.speaker_profile_link import SpeakerProfileLinkService
from companion.people.speaker_registration import (
    BOUND,
    CANCELLED,
    COLLECTING,
    EXPIRED,
    FAILED,
    QUALITY_CHECK,
    READY_FOR_CONFIRMATION,
    SpeakerRegistrationError,
    SpeakerRegistrationFlow,
)
from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.consent import EnrollmentPolicy
from companion.speaker_id.models import (
    AMBIGUOUS,
    ERROR,
    KNOWN,
    NEW_CANDIDATE,
    PENDING,
    QUALITY_HIGH,
    QUALITY_LOW,
)
from companion.speaker_id.store import SpeakerStore


@pytest.fixture
def store(tmp_path):
    value = SpeakerStore(tmp_path / "speakers.sqlite3")
    value.initialize()
    yield value
    value.close()


def candidate_with_sample(store: SpeakerStore) -> str:
    config = load_speaker_config(env={})
    speaker_id = store.create_candidate()
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = 1.0
    outcome = store.add_embedding(
        speaker_id,
        vector,
        quality=QUALITY_HIGH,
        voiced_seconds=2.0,
        model_id=f"{config.model_id}@{config.model_revision}",
        embedding_version=config.embedding_version,
        is_verified_reference=True,
    )
    assert outcome == "ADDED"
    return speaker_id


def result(
    status,
    *,
    speaker_id=None,
    quality=QUALITY_HIGH,
    voiced=2.2,
    provisional=False,
):
    return SimpleNamespace(
        status=status,
        speaker_id=speaker_id,
        reid_quality=quality,
        voiced_seconds=voiced,
        provisional=provisional,
    )


def flow(rt, store, *, clock=None, policy=None):
    link = SpeakerProfileLinkService(
        rt.speaker_bindings,
        rt.repos.person,
        store,
        rt.profile_service,
        audit=rt.repos.audit,
    )
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return SpeakerRegistrationFlow(
        store=store,
        persons=rt.repos.person,
        link=link,
        enrollment_policy=policy or EnrollmentPolicy(),
        **kwargs,
    )


async def person(rt, name="Person", *, persistence=True, processing=True):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=processing,
        allow_persistence=persistence,
        allow_memory_learning=True,
    )


# -------------------------------------------------------------------- start
async def test_start_requires_processing_and_persistence_consent(rt, store):
    owner = await person(rt, processing=True, persistence=False)
    policy = EnrollmentPolicy()
    registration = flow(rt, store, policy=policy)

    step = await registration.start(owner.person_id)

    assert step.state == FAILED
    assert step.reason == "consent_missing"
    assert policy.armed is False
    assert registration.state == FAILED


async def test_start_requires_an_active_person(rt, store):
    owner = await person(rt)
    await rt.registry.set_status(owner.person_id, RegistrationStatus.SUSPENDED)
    registration = flow(rt, store)

    step = await registration.start(owner.person_id)

    assert step.state == FAILED
    assert step.reason == "person_inactive"


async def test_start_arms_exactly_one_identity_and_cancel_disarms(rt, store):
    owner = await person(rt)
    policy = EnrollmentPolicy()
    registration = flow(rt, store, policy=policy)

    step = await registration.start(owner.person_id)

    assert step.state == COLLECTING
    assert policy.armed is True
    cancelled = registration.cancel()
    assert cancelled.state == CANCELLED
    assert policy.armed is False


async def test_start_refuses_while_a_previous_registration_is_running(rt, store):
    owner = await person(rt)
    registration = flow(rt, store)
    await registration.start(owner.person_id)

    with pytest.raises(SpeakerRegistrationError) as error:
        await registration.start(owner.person_id)
    assert error.value.code == "registration_already_active"


# ------------------------------------------------------------------ observe
async def test_tts_audio_is_never_enrolled(rt, store):
    owner = await person(rt)
    registration = flow(rt, store)
    await registration.start(owner.person_id)
    candidate = store.create_candidate()

    step = registration.observe(
        result(NEW_CANDIDATE, speaker_id=candidate), source="tts"
    )

    assert step.state == COLLECTING
    assert step.reason == "non_microphone_audio"
    assert registration.speaker_id is None
    assert registration.state == COLLECTING


async def test_low_quality_keeps_collecting_and_high_quality_readies(rt, store):
    owner = await person(rt)
    registration = flow(rt, store)
    await registration.start(owner.person_id)
    candidate = candidate_with_sample(store)

    low = registration.observe(
        result(NEW_CANDIDATE, speaker_id=candidate, quality=QUALITY_LOW, voiced=0.4)
    )
    assert low.state == QUALITY_CHECK
    assert low.can_confirm is False

    pending = registration.observe(result(PENDING))
    assert pending.state == QUALITY_CHECK

    ready = registration.observe(
        result(
            NEW_CANDIDATE,
            speaker_id=candidate,
            quality=QUALITY_HIGH,
            voiced=2.2,
        )
    )
    assert ready.state == READY_FOR_CONFIRMATION
    assert ready.can_confirm is True
    assert ready.speaker_id == candidate


async def test_an_existing_known_voice_is_not_re_enrolled(rt, store):
    owner = await person(rt)
    registration = flow(rt, store)
    await registration.start(owner.person_id)

    step = registration.observe(result(KNOWN, speaker_id="S-0001"))

    assert step.state == FAILED
    assert step.reason == "already_known_voice"
    assert await rt.speaker_bindings.get_binding("S-0001") is None


async def test_ambiguous_or_model_error_audio_never_confirms(rt, store):
    owner = await person(rt)
    registration = flow(rt, store)
    await registration.start(owner.person_id)

    ambiguous = registration.observe(result(AMBIGUOUS))
    assert ambiguous.state == COLLECTING
    assert ambiguous.reason == "audio_not_usable_yet"

    failure = registration.observe(result(ERROR))
    assert failure.state == COLLECTING
    assert failure.reason == "speaker_model_error"
    assert failure.can_confirm is False


async def test_expired_registration_disarms_and_never_binds(rt, store):
    owner = await person(rt)
    now = [100.0]
    policy = EnrollmentPolicy()
    registration = flow(
        rt, store, policy=policy, clock=lambda: now[0]
    )
    await registration.start(owner.person_id)
    now[0] += 121.0
    candidate = candidate_with_sample(store)

    step = registration.observe(result(NEW_CANDIDATE, speaker_id=candidate))

    assert step.state == EXPIRED
    assert policy.armed is False
    with pytest.raises(SpeakerRegistrationError):
        await registration.confirm()


# ------------------------------------------------------------------ confirm
async def test_confirm_promotes_links_and_grants_storage(rt, store):
    owner = await person(rt, "Owner")
    policy = EnrollmentPolicy()
    registration = flow(rt, store, policy=policy)
    await registration.start(owner.person_id)
    candidate = candidate_with_sample(store)
    ready = registration.observe(
        result(NEW_CANDIDATE, speaker_id=candidate)
    )
    assert ready.state == READY_FOR_CONFIRMATION

    step = await registration.confirm()

    assert step.state == BOUND
    assert step.speaker_id is not None
    assert step.speaker_id.startswith("S-")
    binding = await rt.speaker_bindings.get_binding(step.speaker_id)
    assert binding is not None and binding["person_id"] == owner.person_id
    # The old temporary id keeps resolving as an alias.
    assert store.resolve_canonical_speaker_id(candidate) == step.speaker_id
    consent = store.enrollment_consent(step.speaker_id)
    assert consent is not None and consent["status"] == "GRANTED"
    assert policy.is_authorized(step.speaker_id) is True
    assert policy.armed is False


async def test_confirm_requires_the_ready_state_and_is_never_automatic(rt, store):
    owner = await person(rt)
    registration = flow(rt, store)
    await registration.start(owner.person_id)
    candidate = candidate_with_sample(store)
    registration.observe(
        result(NEW_CANDIDATE, speaker_id=candidate, quality=QUALITY_LOW, voiced=0.4)
    )

    with pytest.raises(SpeakerRegistrationError) as error:
        await registration.confirm()
    assert error.value.code == "registration_not_ready"
    assert await rt.speaker_bindings.get_binding(candidate) is None


async def test_a_failed_link_leaves_no_grant_behind(rt, store):
    owner = await person(rt)
    policy = EnrollmentPolicy()
    registration = flow(rt, store, policy=policy)
    await registration.start(owner.person_id)
    candidate = candidate_with_sample(store)
    registration.observe(result(NEW_CANDIDATE, speaker_id=candidate))
    await rt.registry.set_status(owner.person_id, RegistrationStatus.SUSPENDED)

    step = await registration.confirm()

    assert step.state == FAILED
    assert step.reason in ("person_inactive", "person_not_active")
    assert store.enrollment_consent(step.speaker_id or candidate) is None
    assert policy.armed is False


async def test_the_bound_voice_resolves_for_a_real_speaker_result(rt, store):
    owner = await person(rt, "Owner")
    registration = flow(rt, store)
    await registration.start(owner.person_id)
    candidate = candidate_with_sample(store)
    registration.observe(result(NEW_CANDIDATE, speaker_id=candidate))
    step = await registration.confirm()
    assert step.state == BOUND

    resolution = await registration.link.resolve_profile(step.speaker_id)

    assert resolution.person_id == owner.person_id
    assert resolution.display_name == "Owner"
