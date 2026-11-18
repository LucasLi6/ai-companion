"""V0.9.9.31: human-controlled promotion of a temporary voice profile.

A voice cluster created automatically by the engine is ``TEMP-xxxx`` /
``CANDIDATE``. The matcher only reports ``KNOWN`` for a profile stored as
``KNOWN``, so without an explicit, human-issued promotion a correctly bound
voice could never pass formal identity resolution. These offline tests pin the
promotion rules:

* only a live candidate with usable samples can be promoted;
* the old id becomes an alias, so an earlier binding still resolves;
* promotion never creates a person binding;
* a storage grant follows the voiceprint data to the new id;
* the resolver confirms the promoted voice only while the binding agrees.
"""

from __future__ import annotations

import numpy as np

from companion.people.identity_resolver import IdentityResolver, IdentityState
from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.models import KNOWN, QUALITY_HIGH, SpeakerResult
from companion.speaker_id.promotion import (
    ALREADY_KNOWN,
    NO_SAMPLES,
    PROMOTED,
    UNKNOWN_SPEAKER,
    promote_voice_profile,
)
from companion.speaker_id.store import SpeakerStore


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def store_config():
    return load_speaker_config(env={})


def model_key(config) -> str:
    return f"{config.model_id}@{config.model_revision}"


def make_store(tmp_path) -> SpeakerStore:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    return store


def candidate_with_sample(store: SpeakerStore) -> str:
    config = store_config()
    speaker_id = store.create_candidate()
    outcome = store.add_embedding(
        speaker_id,
        unit(1, 0),
        quality=QUALITY_HIGH,
        voiced_seconds=2.0,
        model_id=model_key(config),
        embedding_version=config.embedding_version,
        is_verified_reference=True,
    )
    assert outcome == "ADDED"
    return speaker_id


def known_result(speaker_id: str) -> SpeakerResult:
    return SpeakerResult(
        utterance_id=1,
        speaker_id=speaker_id,
        person_id=None,
        status=KNOWN,
        reid_quality=QUALITY_HIGH,
        similarity=0.82,
        margin=0.25,
        voiced_seconds=2.2,
        provisional=False,
        model_id="fake",
        inference_ms=1.0,
        completed_at=0.0,
    )


# --------------------------------------------------------------- promotion
def test_candidate_with_samples_becomes_a_stable_known_profile(tmp_path):
    store = make_store(tmp_path)
    try:
        temporary = candidate_with_sample(store)

        result = promote_voice_profile(store, temporary)

        assert result.outcome == PROMOTED
        assert result.previous_id == temporary
        assert result.speaker_id is not None
        assert result.speaker_id.startswith("S-")
        assert result.speaker_id != temporary
        assert store.resolve_canonical_speaker_id(temporary) == result.speaker_id
        assert (
            store.resolve_canonical_speaker_id(result.speaker_id)
            == result.speaker_id
        )
        profile = store.get_profile(result.speaker_id)
        assert profile is not None
        assert profile["status"] == "KNOWN"
        listed = {
            row["speaker_id"]: int(row["samples"])
            for row in store.list_speakers()
        }
        assert listed[result.speaker_id] == 1
        candidates, _ = store.load_candidates(
            model_id=profile["model_id"],
            embedding_version=profile["embedding_version"],
        )
        promoted = next(
            item for item in candidates if item.speaker_id == result.speaker_id
        )
        assert len(promoted.samples) == 1
    finally:
        store.close()


def test_second_promotion_reports_already_known(tmp_path):
    store = make_store(tmp_path)
    try:
        temporary = candidate_with_sample(store)
        first = promote_voice_profile(store, temporary)

        second = promote_voice_profile(store, temporary)

        assert second.outcome == ALREADY_KNOWN
        assert second.speaker_id == first.speaker_id
        assert second.previous_id == first.speaker_id
    finally:
        store.close()


def test_candidate_without_samples_is_refused(tmp_path):
    store = make_store(tmp_path)
    try:
        temporary = store.create_candidate()

        result = promote_voice_profile(store, temporary)

        assert result.outcome == NO_SAMPLES
        assert result.previous_id == temporary
        # Nothing was renamed or deleted.
        assert store.resolve_canonical_speaker_id(temporary) == temporary
        assert store.get_profile(temporary)["status"] == "CANDIDATE"
        assert store.aliases() == []
    finally:
        store.close()


def test_missing_or_deleted_id_is_refused(tmp_path):
    store = make_store(tmp_path)
    try:
        assert (
            promote_voice_profile(store, "TEMP-9999").outcome
            == UNKNOWN_SPEAKER
        )
        temporary = candidate_with_sample(store)
        assert store.delete_speaker(temporary)
        assert (
            promote_voice_profile(store, temporary).outcome
            == UNKNOWN_SPEAKER
        )
        assert (
            promote_voice_profile(store, "  ").outcome == UNKNOWN_SPEAKER
        )
    finally:
        store.close()


def test_storage_grant_follows_the_voiceprint_data(tmp_path):
    store = make_store(tmp_path)
    try:
        temporary = candidate_with_sample(store)
        store.set_enrollment_consent(temporary, granted_by="local_cli")

        result = promote_voice_profile(store, temporary)

        assert result.outcome == PROMOTED
        grants = {
            row["speaker_id"]: row["granted_by"]
            for row in store.enrollment_consents()
        }
        assert grants == {result.speaker_id: "local_cli"}
    finally:
        store.close()


# -------------------------------------------------------- resolver scenario
async def test_binding_made_before_promotion_still_confirms(tmp_path, rt):
    """The exact user flow: bind TEMP-xxxx, then promote it."""
    store = make_store(tmp_path)
    try:
        temporary = candidate_with_sample(store)
        owner = await rt.registry.register(
            "Owner",
            "owner",
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=True,
        )
        await rt.speaker_bindings.bind_speaker(
            owner.person_id, temporary, "local_cli_confirmation"
        )

        result = promote_voice_profile(store, temporary)
        assert result.outcome == PROMOTED

        identity = await IdentityResolver(
            rt.repos.person, rt.speaker_bindings, store
        ).resolve(
            known_result(result.speaker_id),
            run_id="run",
            utterance_id=1,
        )

        assert identity.state is IdentityState.CONFIRMED
        assert identity.person_id == owner.person_id
        assert identity.canonical_speaker_id == result.speaker_id
        # Promotion is a voiceprint decision only; it wrote no binding.
        bindings = await rt.speaker_bindings.list_bindings()
        assert [row["speaker_id"] for row in bindings] == [temporary]
    finally:
        store.close()
