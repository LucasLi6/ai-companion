"""V0.9.9.24 regression tests: safe learning, finalisation and migrations.

Everything runs offline with a deterministic fake embedding model. The fake
can gate loading, block inference and return a sequence of vectors, so the
thread and state-machine transitions are exercised without sleeps that hide
races.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np
import pytest

from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.engine import SpeakerEngine
from companion.speaker_id.runtime import DeferredSpeakerObserver
from companion.speaker_id.store import SpeakerStore

SAMPLE_RATE = 16000
MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common@v1.0.1"
VERSION = "eres2netv2-192-v1"


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def tone(seconds: float, *, amplitude: float = 0.25) -> bytes:
    n = int(seconds * SAMPLE_RATE)
    samples = amplitude * np.sin(
        2 * np.pi * 220 * np.arange(n, dtype=np.float32) / SAMPLE_RATE
    )
    return (samples * 32767).astype("<i2").tobytes()


def wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


class SequencedModel:
    """Fake ERes2NetV2: returns vectors in order, reusing the last one."""

    def __init__(
        self,
        vectors,
        *,
        load_gate: threading.Event | None = None,
        load_error: Exception | None = None,
        encode_gate: threading.Event | None = None,
    ) -> None:
        self.vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        self.load_gate = load_gate
        self.load_error = load_error
        self.encode_gate = encode_gate
        self.calls = 0
        self.encoded = threading.Event()
        self.closed = False
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.load_gate is not None:
            assert self.load_gate.wait(timeout=5.0)
        if self.load_error is not None:
            raise self.load_error

    def warmup(self) -> None:
        return None

    def encode(self, pcm: bytes) -> np.ndarray:
        with self._lock:
            index = min(self.calls, len(self.vectors) - 1)
            self.calls += 1
            vector = self.vectors[index]
        self.encoded.set()
        if self.encode_gate is not None:
            assert self.encode_gate.wait(timeout=5.0)
        return vector.copy()

    def close(self) -> None:
        self.closed = True


def make_engine(tmp_path, model) -> tuple[SpeakerEngine, SpeakerStore]:
    config = load_speaker_config(env={}, db_path=tmp_path / "speakers.sqlite3")
    store = SpeakerStore(
        config.db_path,
        max_embeddings_per_speaker=config.max_embeddings_per_speaker,
        duplicate_sample_similarity=config.duplicate_sample_similarity,
        minimum_enrollment_quality=config.minimum_enrollment_quality,
    )
    store.initialize()
    engine = SpeakerEngine(config, embedding_model=model, store=store)
    engine.start()
    return engine, store


def count_source(store: SpeakerStore, session_id: str, uid: int) -> int:
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM speaker_embeddings"
            " WHERE source_session_id = ? AND source_utterance_id = ?",
            (session_id, uid),
        ).fetchone()
    return int(row[0])


def count_samples(store: SpeakerStore) -> int:
    with sqlite3.connect(store.path) as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM speaker_embeddings"
            ).fetchone()[0]
        )


def seed_identity(store: SpeakerStore, vector) -> str:
    speaker = store.create_candidate()
    store.add_embedding(
        speaker,
        vector,
        quality="HIGH",
        voiced_seconds=3.0,
        model_id=MODEL,
        embedding_version=VERSION,
    )
    return speaker


# ---------------------------------------------------------------- learning
def test_successful_utterance_adds_at_most_one_sample(tmp_path) -> None:
    model = SequencedModel([unit(1, 0), unit(1, 0.6)])
    engine, store = make_engine(tmp_path, model)
    try:
        speaker = seed_identity(store, unit(1, 0))
        before = count_samples(store)

        engine.on_start(7, tone(1.5))
        wait_until(lambda: 7 in engine._results)
        engine.on_audio(7, tone(2.0))  # second analysis at ~2.7 s
        wait_until(lambda: engine._results[7].similarity is not None and engine._results[7].similarity < 1.0)
        engine.on_end(7)
        engine.on_finalized(7, True)
        wait_until(lambda: 7 not in engine._states)

        assert count_source(store, engine.session_id, 7) == 1
        assert count_samples(store) - before == 1
        assert store.get_profile(speaker) is not None
    finally:
        engine.shutdown()
        store.close()


def test_failed_asr_never_writes_an_identity_sample(tmp_path) -> None:
    model = SequencedModel([unit(1, 0.05)])
    engine, store = make_engine(tmp_path, model)
    try:
        speaker = seed_identity(store, unit(1, 0))
        engine.on_start(8, tone(2.0))
        wait_until(lambda: 8 in engine._results)
        engine.on_end(8)
        engine.on_finalized(8, False)
        wait_until(lambda: 8 not in engine._states)

        assert count_source(store, engine.session_id, 8) == 0
        assert count_samples(store) == 1
        assert store.get_profile(speaker)["observation_count"] >= 1
    finally:
        engine.shutdown()
        store.close()


def test_cancel_during_encode_commits_nothing(tmp_path) -> None:
    gate = threading.Event()
    model = SequencedModel([unit(1, 0)], encode_gate=gate)
    engine, store = make_engine(tmp_path, model)
    try:
        engine.on_start(9, tone(2.0))
        assert model.encoded.wait(timeout=5.0)
        engine.on_cancel(9)
        gate.set()
        wait_until(lambda: 9 not in engine._states, timeout=5.0)
        assert count_source(store, engine.session_id, 9) == 0
        assert count_samples(store) == 0
    finally:
        engine.shutdown()
        store.close()


def test_early_success_then_later_conflict_writes_nothing(tmp_path) -> None:
    # First window is the enrolled voice; every later window is a different
    # voice, so the identity evidence contradicts itself.
    model = SequencedModel([unit(1, 0), unit(0, 1)])
    engine, store = make_engine(tmp_path, model)
    try:
        speaker = seed_identity(store, unit(1, 0))
        before = count_samples(store)

        engine.on_start(11, tone(1.5))
        wait_until(lambda: 11 in engine._results)
        engine.on_audio(11, tone(2.0))  # second analysis at ~3.5 s
        wait_until(
            lambda: 11 in engine._results
            and engine._results[11].status == "AMBIGUOUS"
        )
        engine.on_end(11)
        engine.on_finalized(11, True)
        wait_until(lambda: 11 not in engine._states)

        assert count_samples(store) == before
        assert count_source(store, engine.session_id, 11) == 0
        assert store.get_profile(speaker) is not None
    finally:
        engine.shutdown()
        store.close()


def test_duplicate_finalized_is_idempotent(tmp_path) -> None:
    model = SequencedModel([unit(1, 0.05), unit(1, 0.6)])
    engine, store = make_engine(tmp_path, model)
    try:
        seed_identity(store, unit(1, 0))
        before = count_samples(store)
        engine.on_start(12, tone(1.5))
        wait_until(lambda: 12 in engine._results)
        engine.on_audio(12, tone(2.0))
        wait_until(lambda: engine._results[12].similarity is not None and engine._results[12].similarity < 1.0)
        engine.on_end(12)
        engine.on_finalized(12, True)
        engine.on_finalized(12, True)
        wait_until(lambda: 12 not in engine._states)

        assert count_source(store, engine.session_id, 12) == 1
        assert count_samples(store) - before == 1
    finally:
        engine.shutdown()
        store.close()


def test_end_before_or_after_finalized_converges(tmp_path) -> None:
    model = SequencedModel([unit(1, 0.6), unit(1, 1.2), unit(1, -1)])
    engine, store = make_engine(tmp_path, model)
    try:
        seed_identity(store, unit(1, 0))
        before = count_samples(store)

        # ASR result first, VAD end later.
        engine.on_start(13, tone(1.5))
        wait_until(lambda: 13 in engine._results)
        engine.on_audio(13, tone(2.0))
        wait_until(lambda: engine._results[13].similarity is not None and engine._results[13].similarity < 1.0)
        engine.on_finalized(13, True)
        engine.on_end(13)
        wait_until(lambda: 13 not in engine._states)

        # VAD end first, ASR result later.
        engine.on_start(14, tone(1.5))
        wait_until(lambda: 14 in engine._results)
        engine.on_audio(14, tone(2.0))
        wait_until(lambda: engine._results[14].similarity is not None and engine._results[14].similarity < 1.0)
        engine.on_end(14)
        engine.on_finalized(14, True)
        wait_until(lambda: 14 not in engine._states)

        assert count_source(store, engine.session_id, 13) == 1
        assert count_source(store, engine.session_id, 14) == 1
        assert count_samples(store) - before == 2
    finally:
        engine.shutdown()
        store.close()


# --------------------------------------------------------------- deferred
def test_deferred_observer_skips_an_utterance_opened_before_ready(tmp_path) -> None:
    gate = threading.Event()
    model = SequencedModel([unit(1, 0)], load_gate=gate)
    config = load_speaker_config(env={}, db_path=tmp_path / "speakers.sqlite3")
    store = SpeakerStore(config.db_path)
    observer = DeferredSpeakerObserver(
        config, embedding_model=model, store=store
    )
    results = []
    observer.set_on_result(results.append)
    try:
        observer.start_loading()
        assert observer.snapshot(1).status == "PENDING"
        observer.on_start(1, tone(2.0))
        observer.on_audio(1, tone(1.0))

        gate.set()
        wait_until(lambda: observer.ready)
        observer.on_audio(1, tone(1.0))
        observer.on_end(1)
        observer.on_finalized(1, True)
        wait_until(lambda: observer.engine is not None and not observer.engine._states)

        # The pre-READY utterance is skipped entirely: no identity, no sample.
        assert store.list_speakers() == []
        assert count_samples(store) == 0

        # A later utterance is analysed normally.
        observer.on_start(2, tone(2.0))
        wait_until(
            lambda: any(
                getattr(result, "utterance_id", 0) == 2
                and result.status != "PENDING"
                for result in results
            )
        )
        observer.on_end(2)
        observer.on_finalized(2, True)
    finally:
        observer.shutdown()
        store.close()


def test_deferred_observer_load_failure_never_raises(tmp_path) -> None:
    model = SequencedModel([unit(1, 0)], load_error=RuntimeError("boom"))
    config = load_speaker_config(env={}, db_path=tmp_path / "speakers.sqlite3")
    store = SpeakerStore(config.db_path)
    observer = DeferredSpeakerObserver(
        config, embedding_model=model, store=store
    )
    try:
        observer.start_loading()
        wait_until(lambda: observer.status == "UNAVAILABLE")
        assert observer.error is not None
        assert observer.snapshot(1).status == "ERROR"
        observer.on_start(1, tone(2.0))
        observer.on_audio(1, tone(1.0))
        observer.on_end(1)
        observer.on_finalized(1, False)
        observer.on_cancel(1)
    finally:
        observer.shutdown()


def test_deferred_observer_shutdown_ignores_late_callbacks(tmp_path) -> None:
    model = SequencedModel([unit(1, 0)])
    config = load_speaker_config(env={}, db_path=tmp_path / "speakers.sqlite3")
    store = SpeakerStore(config.db_path)
    observer = DeferredSpeakerObserver(
        config, embedding_model=model, store=store
    )
    observer.start_loading()
    wait_until(lambda: observer.ready)
    observer.shutdown()
    assert observer.status == "CLOSED"
    observer.on_start(5, tone(2.0))
    observer.on_audio(5, tone(1.0))
    observer.on_end(5)
    observer.on_finalized(5, True)
    observer.on_cancel(5)
    assert observer.snapshot(5).status in ("PENDING", "ERROR")


# --------------------------------------------------------------- storage
def test_observations_are_scoped_by_session(tmp_path) -> None:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    try:
        store.record_observation(
            1, session_id="session-a", speaker_id=None,
            match_status="NEW_CANDIDATE", similarity=None, margin=None,
        )
        store.record_observation(
            1, session_id="session-b", speaker_id=None,
            match_status="NEW_CANDIDATE", similarity=None, margin=None,
        )
        store.record_observation(
            1, session_id="session-a", speaker_id=None,
            match_status="AMBIGUOUS", similarity=0.4, margin=0.1,
        )
        with sqlite3.connect(store.path) as connection:
            rows = connection.execute(
                "SELECT session_id, match_status FROM speaker_observations"
                " ORDER BY session_id"
            ).fetchall()
        assert rows == [
            ("session-a", "AMBIGUOUS"),
            ("session-b", "NEW_CANDIDATE"),
        ]
    finally:
        store.close()


def test_candidate_ids_are_monotonic_after_deletion(tmp_path) -> None:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    try:
        assert store.create_candidate() == "TEMP-0001"
        second = store.create_candidate()
        assert second == "TEMP-0002"
        assert store.delete_speaker(second) is True
        assert store.create_candidate() == "TEMP-0003"
    finally:
        store.close()


def test_promoted_ids_are_monotonic_after_deletion(tmp_path) -> None:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    try:
        first = store.create_candidate()
        stable = store.promote_speaker(first)
        assert stable == "S-0001"
        assert store.delete_speaker(stable) is True
        second = store.create_candidate()
        assert store.promote_speaker(second) == "S-0002"
    finally:
        store.close()


def test_legacy_store_migrates_without_losing_rows(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE speaker_profiles (
            speaker_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 0,
            model_id TEXT NOT NULL,
            embedding_version TEXT NOT NULL
        );
        CREATE TABLE speaker_embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_id TEXT NOT NULL,
            vector BLOB NOT NULL,
            quality TEXT NOT NULL,
            voiced_seconds REAL NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE speaker_observations (
            observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            utterance_id INTEGER NOT NULL UNIQUE,
            speaker_id TEXT,
            match_status TEXT NOT NULL,
            similarity REAL,
            margin REAL,
            created_at REAL NOT NULL
        );
        CREATE TABLE speaker_aliases (
            alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_speaker_id TEXT NOT NULL,
            alias_speaker_id TEXT NOT NULL UNIQUE,
            created_at REAL NOT NULL
        );
        """
    )
    vector = unit(1, 0).astype("<f4").tobytes()
    connection.execute(
        "INSERT INTO speaker_profiles VALUES"
        " ('TEMP-0004', 'CANDIDATE', 1.0, 1.0, 1, ?, ?)",
        (MODEL, VERSION),
    )
    connection.execute(
        "INSERT INTO speaker_embeddings"
        " (speaker_id, vector, quality, voiced_seconds, created_at)"
        " VALUES ('TEMP-0004', ?, 'HIGH', 2.0, 1.0)",
        (vector,),
    )
    connection.execute(
        "INSERT INTO speaker_observations"
        " (utterance_id, speaker_id, match_status, similarity, margin,"
        "  created_at)"
        " VALUES (3, 'TEMP-0004', 'CANDIDATE_MATCH', 0.9, 0.2, 1.0)"
    )
    connection.commit()
    connection.close()

    store = SpeakerStore(path)
    store.initialize()
    try:
        with sqlite3.connect(path) as check:
            observation = check.execute(
                "SELECT session_id, utterance_id, match_status"
                " FROM speaker_observations"
            ).fetchone()
            sample = check.execute(
                "SELECT is_verified_reference, source_session_id,"
                " source_utterance_id FROM speaker_embeddings"
            ).fetchone()
            version = check.execute(
                "SELECT value FROM speaker_meta"
                " WHERE key = 'schema_version'"
            ).fetchone()[0]
        assert observation == ("legacy", 3, "CANDIDATE_MATCH")
        assert sample == (0, None, None)
        assert version == "2"
        assert store.create_candidate() == "TEMP-0005"
    finally:
        store.close()

    # A second open must be a no-op, not a re-migration.
    reopened = SpeakerStore(path)
    reopened.initialize()
    try:
        with sqlite3.connect(path) as check:
            assert check.execute(
                "SELECT COUNT(*) FROM speaker_observations"
            ).fetchone()[0] == 1
    finally:
        reopened.close()


def test_sample_retention_keeps_verified_reference_and_diversity(tmp_path) -> None:
    store = SpeakerStore(
        tmp_path / "speakers.sqlite3", max_embeddings_per_speaker=3
    )
    store.initialize()
    try:
        speaker = store.create_candidate()
        store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=3.0,
            model_id=MODEL,
            embedding_version=VERSION,
            is_verified_reference=True,
        )
        for index in range(1, 6):
            store.add_embedding(
                speaker,
                unit(*([0.0] * index + [1.0])),
                quality="HIGH",
                voiced_seconds=3.0,
                model_id=MODEL,
                embedding_version=VERSION,
            )
        with sqlite3.connect(store.path) as connection:
            rows = connection.execute(
                "SELECT embedding_id, is_verified_reference"
                " FROM speaker_embeddings WHERE speaker_id = ?",
                (speaker,),
            ).fetchall()
        assert len(rows) == 3
        assert any(int(row[1]) == 1 for row in rows)
    finally:
        store.close()


def test_duplicate_and_better_quality_sample_rules(tmp_path) -> None:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    try:
        speaker = store.create_candidate()
        first = store.add_embedding(
            speaker,
            unit(1, 0),
            quality="LOW",
            voiced_seconds=1.0,
            model_id=MODEL,
            embedding_version=VERSION,
        )
        duplicate = store.add_embedding(
            speaker,
            unit(1, 0),
            quality="LOW",
            voiced_seconds=1.0,
            model_id=MODEL,
            embedding_version=VERSION,
        )
        assert first is not None
        assert duplicate is None
        assert count_samples(store) == 1

        better = store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=3.0,
            model_id=MODEL,
            embedding_version=VERSION,
        )
        assert better is not None
        with sqlite3.connect(store.path) as connection:
            qualities = connection.execute(
                "SELECT quality FROM speaker_embeddings"
            ).fetchall()
        assert qualities == [("HIGH",)]
    finally:
        store.close()


def test_verified_sample_is_never_replaced_by_a_duplicate(tmp_path) -> None:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    try:
        speaker = store.create_candidate()
        store.add_embedding(
            speaker,
            unit(1, 0),
            quality="LOW",
            voiced_seconds=1.0,
            model_id=MODEL,
            embedding_version=VERSION,
            is_verified_reference=True,
        )
        assert store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=3.0,
            model_id=MODEL,
            embedding_version=VERSION,
        ) is None
        with sqlite3.connect(store.path) as connection:
            row = connection.execute(
                "SELECT quality, is_verified_reference"
                " FROM speaker_embeddings"
            ).fetchone()
        assert row == ("LOW", 1)
    finally:
        store.close()


def test_source_idempotency_and_revocation(tmp_path) -> None:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    try:
        speaker = store.create_candidate()
        first = store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=3.0,
            model_id=MODEL,
            embedding_version=VERSION,
            source_session_id="session",
            source_utterance_id=4,
        )
        again = store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=3.0,
            model_id=MODEL,
            embedding_version=VERSION,
            source_session_id="session",
            source_utterance_id=4,
        )
        assert first == again
        assert count_samples(store) == 1

        assert store.revoke_embedding(
            speaker, source_session_id="session", source_utterance_id=4
        ) is True
        assert count_samples(store) == 0
    finally:
        store.close()
