"""V0.9.9.31: voiceprint storage consent is independent from recognition.

Recognition may analyse an unauthorized voice transiently, but no permanent
identity or embedding may be written without an explicit local grant. These
offline tests drive the real engine with a deterministic fake model and pin:

* an unarmed strict policy creates no profile and writes no sample;
* arming authorises exactly one new identity, then the arm is consumed;
* the engine's single write point refuses unauthorized identities;
* existing profiles stay usable without re-consent, and grants survive a
  policy reload;
* revocation stops later learning without deleting rows.
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
    finally:
        harness.close()
        store.close()


# ---------------------------------------------------- persisted local grants
def test_loading_the_store_authorizes_only_existing_profiles(tmp_path):
    store = _store(tmp_path)
    try:
        existing = _seed_existing_profile(store)
        empty = store.create_candidate()
        policy = EnrollmentPolicy(strict=True)
        policy.load_from_store(store)

        assert policy.is_authorized(existing) is True
        # A profile with no samples is not prior collected data.
        assert policy.is_authorized(empty) is False
        assert policy.allow_new_identity() is False
        assert policy.snapshot()["loaded"] is True
    finally:
        store.close()


def test_explicit_grant_survives_a_policy_reload_and_revoke_removes_it(tmp_path):
    store = _store(tmp_path)
    try:
        speaker_id = _seed_existing_profile(store)
        first = EnrollmentPolicy(strict=True)
        first.revoke(speaker_id, store=store)
        assert first.is_authorized(speaker_id) is False
        assert store.enrollment_consents() == []

        first.authorize(speaker_id, store=store, actor="local_cli")
        assert first.is_authorized(speaker_id) is True

        # A new REPL session loads the durable grant, not just the samples.
        second = EnrollmentPolicy(strict=True)
        second.load_from_store(store)
        assert second.is_authorized(speaker_id) is True
        assert any(
            row["speaker_id"] == speaker_id
            and row["granted_by"] == "local_cli"
            for row in store.enrollment_consents()
        )

        second.revoke(speaker_id, store=store)
        third = EnrollmentPolicy(strict=True)
        third.load_from_store(store)
        # Existing samples still make the profile usable, but the explicit
        # grant row is gone and no new identity may be armed.
        assert store.enrollment_consents() == []
        assert third.allow_new_identity() is False
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
