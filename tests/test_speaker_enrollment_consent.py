"""V0.9.9.31: voiceprint storage consent is independent from recognition.

Recognition may analyse an unauthorized voice transiently, but no permanent
identity or embedding may be written without an explicit local grant. These
offline tests drive the real engine with a deterministic fake model and pin:

* an unarmed strict policy creates no profile and writes no sample;
* arming authorises exactly one new identity, then the arm is consumed;
* the engine's single write point refuses unauthorized identities;
* only an explicit ``GRANTED`` row survives a reload; historical samples are
  recognisable but read-only, and ``REVOKED`` stays revoked across restarts;
* revocation stops later learning without deleting rows, and deleted or
  cleared identities leave no orphan grants;
* merging two voices keeps the more restrictive consent.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.consent import EnrollmentPolicy
from companion.speaker_id.embedding import pcm_to_float32
from companion.speaker_id.engine import (
    LEARN_NOT_AUTHORIZED,
    SpeakerEngine,
)
from companion.speaker_id.models import NEW_CANDIDATE, QUALITY_HIGH
from companion.speaker_id.store import SpeakerStore

SAMPLE_RATE = 16000


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def tone(seconds: float, frequency: float = 220.0) -> bytes:
    n = int(seconds * SAMPLE_RATE)
    samples = 0.2 * np.sin(
        2 * np.pi * frequency * np.arange(n, dtype=np.float32) / SAMPLE_RATE
    )
    return (samples * 32767).astype("<i2").tobytes()


class FakeEmbedding:
    def __init__(self) -> None:
        self.vectors = {
            220.0: unit(1, 0),
            # Same speaker, a genuinely different recording: high match but
            # below the near-duplicate cutoff, so an authorized policy can
            # keep it as a new sample.
            221.0: unit(
                float(np.cos(np.deg2rad(8.0))),
                float(np.sin(np.deg2rad(8.0))),
            ),
            440.0: unit(0, 1),
        }

    def encode(self, pcm: bytes) -> np.ndarray:
        audio = pcm_to_float32(pcm)
        tail = audio[-SAMPLE_RATE:] if audio.size > SAMPLE_RATE else audio
        spectrum = np.abs(np.fft.rfft(tail))
        frequency = float(np.argmax(spectrum) * SAMPLE_RATE / tail.size)
        key = min(self.vectors, key=lambda candidate: abs(candidate - frequency))
        return self.vectors[key].copy()


class Collector:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.results = []

    def __call__(self, result) -> None:
        with self._condition:
            self.results.append(result)
            self._condition.notify_all()

    def wait_for(self, predicate, *, timeout: float = 3.0):
        with self._condition:
            self._condition.wait_for(
                lambda: any(predicate(result) for result in self.results),
                timeout=timeout,
            )
            matches = [
                result for result in self.results if predicate(result)
            ]
        assert matches, self.results
        return matches[-1]


def wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


class Harness:
    def __init__(self, store: SpeakerStore, policy: EnrollmentPolicy) -> None:
        self.store = store
        self.policy = policy
        self.collector = Collector()
        self.engine = SpeakerEngine(
            load_speaker_config(env={}),
            embedding_model=FakeEmbedding(),
            store=store,
            on_result=self.collector,
            enrollment_policy=policy,
        )

    def start(self):
        self.engine.start()
        return self

    def close(self) -> None:
        self.engine.shutdown()

    def finish(self, uid: int, seconds: float, frequency: float):
        self.engine.on_start(uid, tone(seconds, frequency))
        result = self.collector.wait_for(
            lambda item: item.utterance_id == uid
            and item.status != "PENDING"
        )
        self.engine.on_end(uid)
        self.engine.on_finalized(uid, True)
        wait_until(lambda: self.engine.is_finished(uid))
        return result

    def samples(self) -> int:
        return sum(int(item["samples"]) for item in self.store.list_speakers())


def _store(tmp_path) -> SpeakerStore:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    return store


def _seed_existing_profile(store: SpeakerStore) -> str:
    """Create a promoted identity with one sample as if collected earlier."""
    store_config = load_speaker_config(env={})
    speaker_id = store.promote_speaker(store.create_candidate())
    assert speaker_id is not None
    store.add_embedding(
        speaker_id,
        unit(1, 0),
        quality=QUALITY_HIGH,
        voiced_seconds=2.0,
        model_id=(
            f"{store_config.model_id}@{store_config.model_revision}"
        ),
        embedding_version=store_config.embedding_version,
        is_verified_reference=True,
    )
    return speaker_id


# ------------------------------------------------- recognition without consent
def test_unarmed_strict_policy_creates_no_identity_and_no_sample(tmp_path):
    store = _store(tmp_path)
    policy = EnrollmentPolicy(strict=True)
    harness = Harness(store, policy).start()
    try:
        result = harness.finish(1, 2.0, 220.0)

        # The engine honestly reports "new voice", but the permanent store is
        # untouched: no profile, no embedding, and no arm was consumed.
        assert result.status == NEW_CANDIDATE
        assert result.speaker_id is None
        assert store.list_speakers() == []
        assert harness.samples() == 0
        assert policy.armed is False
    finally:
        harness.close()
        store.close()


def test_arming_authorizes_exactly_one_new_identity(tmp_path):
    store = _store(tmp_path)
    policy = EnrollmentPolicy(strict=True)
    policy.arm_new_identity("P-person")
    harness = Harness(store, policy).start()
    try:
        first = harness.finish(1, 2.0, 220.0)
        assert first.status == NEW_CANDIDATE
        assert first.speaker_id is not None
        created = first.speaker_id
        assert policy.is_authorized(created) is True
        assert policy.armed is False
        assert harness.samples() == 1

        # A different voice in the same session has no grant of its own.
        second = harness.finish(2, 2.0, 440.0)
        assert second.status == NEW_CANDIDATE
        assert second.speaker_id is None
        assert len(store.list_speakers()) == 1
        assert harness.samples() == 1
    finally:
        harness.close()
        store.close()


def test_unauthorized_existing_identity_cannot_receive_a_new_sample(tmp_path):
    store = _store(tmp_path)
    speaker_id = _seed_existing_profile(store)
    policy = EnrollmentPolicy(strict=True)  # deliberately not loaded
    harness = Harness(store, policy).start()
    try:
        harness.finish(1, 2.0, 220.0)

        # The match itself is fine, but the single write point refuses the
        # sample because this identity was never authorized in this policy.
        assert harness.engine.stats()["learning"][
            LEARN_NOT_AUTHORIZED.lower()
        ] >= 1
        assert harness.samples() == 1
        profiles = store.list_speakers()
        assert [row["speaker_id"] for row in profiles] == [speaker_id]
        # A durable observation row would itself be a long-term record of an
        # unauthorized voice, so none is written for the identity either.
        assert int(profiles[0]["observation_count"]) == 0
    finally:
        harness.close()
        store.close()


# ---------------------------------------------------- persisted local grants
def test_loading_the_store_never_infers_a_grant_from_samples(tmp_path):
    store = _store(tmp_path)
    try:
        existing = _seed_existing_profile(store)
        empty = store.create_candidate()
        policy = EnrollmentPolicy(strict=True)
        policy.load_from_store(store)

        # Sample count is history, not consent: the profile can still be
        # recognised, but it cannot keep learning until a human grants it.
        assert policy.is_authorized(existing) is False
        assert policy.status_of(existing) == "LEGACY_READ_ONLY"
        assert policy.status_of(empty) is None
        assert policy.allow_new_identity() is False
        assert policy.snapshot()["loaded"] is True

        config = load_speaker_config(env={})
        candidates, _ = store.load_candidates(
            model_id=f"{config.model_id}@{config.model_revision}",
            embedding_version=config.embedding_version,
        )
        assert [item.speaker_id for item in candidates] == [existing]
    finally:
        store.close()


def test_revocation_is_durable_and_never_deletes_samples(tmp_path):
    store = _store(tmp_path)
    try:
        speaker_id = _seed_existing_profile(store)
        first = EnrollmentPolicy(strict=True)
        first.authorize(speaker_id, store=store, actor="local_cli")
        assert first.is_authorized(speaker_id) is True

        reloaded = EnrollmentPolicy(strict=True)
        reloaded.load_from_store(store)
        assert reloaded.is_authorized(speaker_id) is True

        reloaded.revoke(speaker_id, store=store)
        assert reloaded.is_authorized(speaker_id) is False
        assert reloaded.status_of(speaker_id) == "REVOKED"

        # The refusal is a row, not a deletion: a restart must not read the
        # missing row as "never decided anything".
        rows = store.enrollment_consents()
        assert [(row["speaker_id"], row["status"]) for row in rows] == [
            (speaker_id, "REVOKED")
        ]

        third = EnrollmentPolicy(strict=True)
        third.load_from_store(store)
        assert third.is_authorized(speaker_id) is False
        assert third.status_of(speaker_id) == "REVOKED"
        assert third.allow_new_identity() is False

        # Historical biometrics are untouched and recognition still works.
        profile = store.get_profile(speaker_id)
        assert profile is not None
        config = load_speaker_config(env={})
        candidates, _ = store.load_candidates(
            model_id=f"{config.model_id}@{config.model_revision}",
            embedding_version=config.embedding_version,
        )
        assert [item.speaker_id for item in candidates] == [speaker_id]
    finally:
        store.close()


def test_delete_and_clear_leave_no_orphan_grants(tmp_path):
    store = _store(tmp_path)
    try:
        speaker_id = _seed_existing_profile(store)
        store.set_enrollment_consent(speaker_id, granted_by="local_cli")

        assert store.delete_speaker(speaker_id) is True
        assert store.list_speakers() == []
        assert store.enrollment_consents() == []

        other = _seed_existing_profile(store)
        store.set_enrollment_consent(other, granted_by="local_cli")
        store.clear()
        assert store.enrollment_consents() == []
    finally:
        store.close()


def test_merge_keeps_the_most_restrictive_consent(tmp_path):
    store = _store(tmp_path)
    config = load_speaker_config(env={})
    model = f"{config.model_id}@{config.model_revision}"
    try:
        canonical = _seed_existing_profile(store)
        duplicate = store.create_candidate()
        store.add_embedding(
            duplicate,
            unit(2, 0),
            quality=QUALITY_HIGH,
            voiced_seconds=2.0,
            model_id=model,
            embedding_version=config.embedding_version,
            is_verified_reference=True,
        )

        store.set_enrollment_consent(canonical, granted_by="local_cli")
        store.revoke_enrollment_consent(duplicate, actor="local_cli")

        assert store.merge_speakers(canonical, duplicate) is True

        statuses = {
            row["speaker_id"]: row["status"]
            for row in store.enrollment_consents()
        }
        # Merging two voices must never widen what may be learned.
        assert statuses == {canonical: "REVOKED"}
        policy = EnrollmentPolicy(strict=True)
        policy.load_from_store(store)
        assert policy.is_authorized(canonical) is False
    finally:
        store.close()


def test_merge_of_a_grant_with_unrecorded_history_is_read_only(tmp_path):
    store = _store(tmp_path)
    config = load_speaker_config(env={})
    model = f"{config.model_id}@{config.model_revision}"
    try:
        canonical = _seed_existing_profile(store)
        duplicate = store.create_candidate()
        store.add_embedding(
            duplicate,
            unit(2, 0),
            quality=QUALITY_HIGH,
            voiced_seconds=2.0,
            model_id=model,
            embedding_version=config.embedding_version,
            is_verified_reference=True,
        )
        store.set_enrollment_consent(canonical, granted_by="local_cli")
        # ``duplicate`` has samples but no explicit row: LEGACY_READ_ONLY.

        assert store.merge_speakers(canonical, duplicate) is True

        statuses = {
            row["speaker_id"]: row["status"]
            for row in store.enrollment_consents()
        }
        assert statuses == {canonical: "LEGACY_READ_ONLY"}
    finally:
        store.close()


def test_revoking_an_identity_stops_all_further_learning(tmp_path):
    store = _store(tmp_path)
    speaker_id = _seed_existing_profile(store)
    policy = EnrollmentPolicy(strict=True)
    policy.authorize(speaker_id, store=store)
    harness = Harness(store, policy).start()
    try:
        harness.finish(1, 2.0, 221.0)
        assert harness.engine.stats()["learning"]["added"] >= 1

        policy.revoke(speaker_id, store=store)
        harness.finish(2, 2.0, 221.0)
        assert harness.engine.stats()["learning"][
            LEARN_NOT_AUTHORIZED.lower()
        ] >= 1
        # Revocation never deletes the existing biometric row by itself.
        assert harness.samples() >= 1
        assert any(
            row["speaker_id"] == speaker_id for row in store.list_speakers()
        )
    finally:
        harness.close()
        store.close()
