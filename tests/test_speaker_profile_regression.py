"""V2 regression tests against the legacy behaviour and the live engine.

The first group pins that ``legacy`` mode is still the default and still
behaves exactly as before, that ``shadow`` changes no verdict and no row, and
that ``variants`` delays a permanent identity until the tracker has enough
evidence. The rest cover configuration tolerance, concurrency and a loose
real-time bound.

All embeddings are deterministic synthetic vectors; nothing here measures
recognition accuracy.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np

from companion.speaker_id.config import (
    DEFAULT_NEW_ID_MIN_TURNS,
    DEFAULT_PENDING_MAX_COUNT,
    DEFAULT_PROFILE_MODE,
    DEFAULT_SINGLETON_PENALTY,
    DEFAULT_TRACK_EXPIRY_SECONDS,
    DEFAULT_VARIANT_MAX_COUNT,
    DEFAULT_VARIANT_MERGE_THRESHOLD,
    EMBEDDING_DIM,
    load_speaker_config,
)
from companion.speaker_id.engine import SpeakerEngine
from companion.speaker_id.embedding import pcm_to_float32
from companion.speaker_id.matcher import SpeakerCandidate, SpeakerMatcher
from companion.speaker_id.quality import VoiceQuality
from companion.speaker_id.store import SpeakerStore
from companion.speaker_id.tracker import SpeakerTrackManager

SAMPLE_RATE = 16000


# ------------------------------------------------------------------ helpers
def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return (vector / np.linalg.norm(vector)).astype(np.float32)


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
            440.0: unit(0, 1),
            660.0: unit(0, 0, 1),
        }
        self._lock = threading.Lock()
        self.calls = 0

    def encode(self, pcm: bytes) -> np.ndarray:
        with self._lock:
            self.calls += 1
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

    def wait_for(self, predicate, *, timeout: float = 3.0, nth: int = 1):
        with self._condition:
            matched = lambda: [
                result for result in self.results if predicate(result)
            ]
            self._condition.wait_for(
                lambda: len(matched()) >= nth, timeout=timeout
            )
            results = matched()
        assert len(results) >= nth, results
        return results[nth - 1]


def wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


class Harness:
    def __init__(self, tmp_path, env=None, embedding=None) -> None:
        self.store = SpeakerStore(tmp_path / "speakers.sqlite3")
        self.store.initialize()
        self.collector = Collector()
        self.engine = SpeakerEngine(
            load_speaker_config(env=env or {}),
            embedding_model=embedding or FakeEmbedding(),
            store=self.store,
            on_result=self.collector,
        )
        self.engine.start()

    def finish(self, uid: int, success: bool = True) -> None:
        self.engine.on_end(uid)
        self.collector.wait_for(
            lambda result: result.utterance_id == uid
            and result.status != "PENDING"
        )
        self.engine.on_finalized(uid, success)
        wait_until(lambda: self.engine.is_finished(uid))

    def speakers(self) -> list[tuple[str, str, int]]:
        return [
            (
                row["speaker_id"],
                row["status"],
                int(row["samples"]),
            )
            for row in self.store.list_speakers()
        ]

    def samples(self) -> int:
        return sum(row[2] for row in self.speakers())

    def close(self) -> None:
        self.engine.shutdown()
        self.store.close()


def run_voice(harness: Harness, uid: int, seconds: float = 2.0) -> list:
    harness.engine.on_start(uid, tone(seconds))
    harness.collector.wait_for(
        lambda result: result.utterance_id == uid
        and result.status != "PENDING"
    )
    harness.finish(uid)
    return [
        (result.status, result.speaker_id)
        for result in harness.collector.results
        if result.utterance_id == uid and result.status != "PENDING"
    ]


def db_snapshot(harness: Harness) -> dict:
    """Every durable row that recognition could have written."""
    with sqlite3.connect(harness.store.path) as connection:
        profiles = connection.execute(
            "SELECT speaker_id, status, model_id, embedding_version"
            " FROM speaker_profiles ORDER BY speaker_id"
        ).fetchall()
        embeddings = connection.execute(
            "SELECT speaker_id, vector, quality, origin, quarantined"
            " FROM speaker_embeddings ORDER BY embedding_id"
        ).fetchall()
        observations = connection.execute(
            "SELECT utterance_id, speaker_id, match_status, similarity,"
            " margin FROM speaker_observations ORDER BY observation_id"
        ).fetchall()
    return {
        "profiles": profiles,
        "embeddings": embeddings,
        "observations": observations,
    }


# ------------------------------------------------------------- legacy default
def test_default_configuration_is_still_legacy() -> None:
    config = load_speaker_config(env={})
    assert config.profile_mode == DEFAULT_PROFILE_MODE == "legacy"
    assert config.match_strategy == "legacy_max_sample"
    assert config.variant_max_count == DEFAULT_VARIANT_MAX_COUNT == 4
    assert config.variant_merge_threshold == DEFAULT_VARIANT_MERGE_THRESHOLD
    assert config.singleton_penalty == DEFAULT_SINGLETON_PENALTY == 0.0
    assert config.track_expiry_seconds == DEFAULT_TRACK_EXPIRY_SECONDS == 15.0
    assert config.new_id_min_turns == DEFAULT_NEW_ID_MIN_TURNS == 2
    assert config.pending_max_count == DEFAULT_PENDING_MAX_COUNT == 12
    assert config.auto_learn_enabled is True


def test_legacy_engine_has_no_tracker_and_keeps_the_old_flow(tmp_path) -> None:
    harness = Harness(tmp_path)
    try:
        assert harness.engine._tracker is None
        assert harness.engine.stats()["tracker"] == {"enabled": False}

        first = run_voice(harness, 1)
        # Same as before V2: the first utterance creates TEMP-0001 at once.
        assert ("NEW_CANDIDATE", "TEMP-0001") in first
        second = run_voice(harness, 2)
        assert ("CANDIDATE_MATCH", "TEMP-0001") in second
        assert harness.speakers() == [("TEMP-0001", "CANDIDATE", 1)]
        assert harness.engine.stats()["profile"]["mode"] == "legacy"
    finally:
        harness.close()


# -------------------------------------------------------------------- shadow
def test_shadow_mode_returns_legacy_verdicts_and_writes_the_same_rows(
    tmp_path,
) -> None:
    legacy = Harness(tmp_path / "legacy")
    shadow = Harness(
        tmp_path / "shadow", env={"AVA_SPEAKER_PROFILE_MODE": "shadow"}
    )
    try:
        legacy_first = run_voice(legacy, 1)
        shadow_first = run_voice(shadow, 1)
        legacy_second = run_voice(legacy, 2)
        shadow_second = run_voice(shadow, 2)

        assert shadow_first == legacy_first
        assert shadow_second == legacy_second
        assert shadow.speakers() == legacy.speakers()
        # The shadow comparison itself is read-only: profiles, embeddings
        # (including origin), and observations must be row-for-row identical.
        assert db_snapshot(shadow) == db_snapshot(legacy)

        stats = shadow.engine.stats()
        assert stats["profile"]["mode"] == "shadow"
        # Shadow returns the max-over-samples verdict even though its active
        # strategy is the variant one when variants mode is selected.
        assert stats["profile"]["match_strategy"] == "legacy_max_sample"
        assert stats["shadow"]["agree"] >= 2
        assert stats["shadow"]["disagree"] == 0
        assert stats["shadow"]["examples"] == []
        # Legacy never pays the shadow comparison.
        assert legacy.engine.stats()["shadow"]["agree"] == 0
    finally:
        legacy.close()
        shadow.close()


# ------------------------------------------------------------------ variants
def test_variants_defers_the_first_identity_until_two_turns(tmp_path) -> None:
    harness = Harness(
        tmp_path, env={"AVA_SPEAKER_PROFILE_MODE": "variants"}
    )
    try:
        # Turn 1: a new voice, but one 2 s turn is not enough. Nothing is
        # written and the public result carries no speaker id.
        harness.engine.on_start(1, tone(2.0))
        first = harness.collector.wait_for(
            lambda result: result.utterance_id == 1
            and result.status == "NEW_CANDIDATE"
        )
        assert first.speaker_id is None
        assert first.person_id is None
        harness.finish(1)
        assert harness.store.list_speakers() == []
        assert harness.engine.stats()["tracker"]["tracks"] == 1
        assert harness.engine.stats()["pending_samples"]["count"] == 1

        # Turn 2: the same voice continues the tentative track; the tracker now
        # has 1 finished turn + the current turn and enough voiced audio.
        harness.engine.on_start(2, tone(2.0))
        second = harness.collector.wait_for(
            lambda result: result.utterance_id == 2
            and result.status == "NEW_CANDIDATE"
        )
        assert second.speaker_id == "TEMP-0001"
        harness.finish(2)
        assert harness.speakers() == [("TEMP-0001", "CANDIDATE", 1)]
        # The staged pending sample is dropped once the identity is real.
        assert harness.engine.stats()["pending_samples"]["count"] == 0
        assert harness.engine.stats()["profile"]["match_strategy"] == (
            "variant_centroid"
        )
    finally:
        harness.close()


def test_variants_still_respects_the_voiced_seconds_gate(tmp_path) -> None:
    harness = Harness(
        tmp_path,
        env={
            "AVA_SPEAKER_PROFILE_MODE": "variants",
            "AVA_SPEAKER_NEW_ID_MIN_VOICED_SECONDS": "5.0",
        },
    )
    try:
        for uid in (1, 2):
            harness.engine.on_start(uid, tone(2.0))
            result = harness.collector.wait_for(
                lambda item, uid=uid: item.utterance_id == uid
                and item.status == "NEW_CANDIDATE"
            )
            assert result.speaker_id is None
            harness.finish(uid)
        # 2 turns x 2 s = 4 s, below the 5 s gate.
        assert harness.store.list_speakers() == []
    finally:
        harness.close()


def test_auto_learn_disabled_creates_an_identity_but_no_samples(tmp_path) -> None:
    harness = Harness(
        tmp_path,
        env={
            "AVA_SPEAKER_PROFILE_MODE": "variants",
            "AVA_SPEAKER_AUTO_LEARN_ENABLED": "0",
        },
    )
    try:
        for uid in (1, 2):
            harness.engine.on_start(uid, tone(2.0))
            harness.collector.wait_for(
                lambda item, uid=uid: item.utterance_id == uid
                and item.status == "NEW_CANDIDATE"
            )
            harness.finish(uid)
        assert harness.samples() == 0
        assert harness.engine.stats()["profile"]["auto_learn_enabled"] is False
    finally:
        harness.close()


def test_tracking_can_be_disabled_without_disabling_variant_scoring(
    tmp_path,
) -> None:
    harness = Harness(
        tmp_path,
        env={
            "AVA_SPEAKER_PROFILE_MODE": "variants",
            "AVA_SPEAKER_TRACK_ENABLED": "0",
        },
    )
    try:
        assert harness.engine._tracker is None
        first = run_voice(harness, 1)
        # Without the tracker there is no deferral: the legacy timing returns.
        assert ("NEW_CANDIDATE", "TEMP-0001") in first
        assert harness.engine.stats()["profile"]["match_strategy"] == (
            "variant_centroid"
        )
    finally:
        harness.close()


# ------------------------------------------------------------- configuration
def test_bad_v2_values_never_break_voice_input() -> None:
    config = load_speaker_config(
        env={
            "AVA_SPEAKER_PROFILE_MODE": "nonsense",
            "AVA_SPEAKER_MATCH_STRATEGY": "nonsense",
            "AVA_SPEAKER_VARIANT_MAX_COUNT": "abc",
            "AVA_SPEAKER_VARIANT_MERGE_THRESHOLD": "nan",
            "AVA_SPEAKER_SINGLETON_PENALTY": "999",
            "AVA_SPEAKER_TRACK_EXPIRY_SECONDS": "-4",
            "AVA_SPEAKER_TRACK_RECENT_TURNS": "0",
            "AVA_SPEAKER_NEW_ID_MIN_TURNS": "1000",
            "AVA_SPEAKER_NEW_ID_MIN_VOICED_SECONDS": "oops",
            "AVA_SPEAKER_PENDING_MAX_COUNT": "100000",
        }
    )
    assert config.profile_mode == "legacy"
    assert config.match_strategy == "legacy_max_sample"
    assert 1 <= config.variant_max_count <= 8
    assert 0.0 <= config.variant_merge_threshold <= 1.0
    assert 0.0 <= config.singleton_penalty <= 0.5
    assert 1.0 <= config.track_expiry_seconds <= 600.0
    assert config.track_recent_turns == 1
    assert config.new_id_min_turns <= 10
    assert 0.0 <= config.new_id_min_voiced_seconds <= 60.0
    assert config.pending_max_count <= 64


def test_explicit_variants_mode_selects_variant_scoring() -> None:
    config = load_speaker_config(
        env={"AVA_SPEAKER_PROFILE_MODE": "variants"}
    )
    assert config.match_strategy == "variant_centroid"
    overridden = load_speaker_config(
        env={
            "AVA_SPEAKER_PROFILE_MODE": "variants",
            "AVA_SPEAKER_MATCH_STRATEGY": "legacy_max_sample",
        }
    )
    assert overridden.match_strategy == "legacy_max_sample"


# -------------------------------------------------------------- concurrency
def test_concurrent_tracker_observations_stay_bounded_and_consistent() -> None:
    tracks = SpeakerTrackManager(
        expiry_seconds=15.0,
        recent_turns=4,
        min_turns=2,
        min_voiced_seconds=3.0,
        continuity_threshold=0.43,
        max_tracks=8,
    )
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=3.0)
            tracks.observe(
                index,
                speaker_id=None,
                status="NEW_CANDIDATE",
                similarity=None,
                margin=None,
                voiced_seconds=2.0,
                quality_rank=3,
                vector=unit(1, 0),
            )
            tracks.finalize(index, success=True)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert not errors
    assert len(tracks.tracks()) <= 8
    assert tracks.stats()["tracks"] <= 8


# ------------------------------------------------------------- real-time bound
def _quality(seconds: float = 2.0) -> VoiceQuality:
    return VoiceQuality(
        usable=True,
        voiced_seconds=seconds,
        quality="HIGH",
        reason=None,
    )


def test_variant_scoring_stays_well_inside_a_real_time_budget() -> None:
    # 20 people x 8 samples: a deliberately larger set than the default cap.
    candidates = []
    for person in range(20):
        vectors = []
        for offset in range(8):
            angle = np.deg2rad(((person * 13 + offset * 7) % 360))
            vectors.append(unit(np.cos(angle), np.sin(angle)))
        candidates.append(
            SpeakerCandidate(
                speaker_id=f"TEMP-{person:04d}",
                status="CANDIDATE",
                vectors=tuple(vectors),
            )
        )
    # No sample metadata: both strategies fall back to max-over-vectors, which
    # is exactly the worst case for the matcher overhead measurement itself.
    query = unit(1, 0)
    legacy_matcher = SpeakerMatcher(
        load_speaker_config(env={})
    )
    variant_matcher = SpeakerMatcher(
        load_speaker_config(env={"AVA_SPEAKER_PROFILE_MODE": "variants"})
    )

    def timed(matcher: SpeakerMatcher) -> float:
        matcher.match(query, candidates, quality=_quality())
        started = time.perf_counter()
        for _ in range(200):
            matcher.match(query, candidates, quality=_quality())
        return (time.perf_counter() - started) * 1000.0 / 200.0

    legacy_ms = timed(legacy_matcher)
    variant_ms = timed(variant_matcher)
    # A very loose sanity ceiling, not a benchmark result: both paths must be
    # far below the ~1 s analysis cadence even with 160 samples.
    assert legacy_ms < 5.0
    assert variant_ms < 5.0
