"""V0.9.9.25: continuous learning outcomes for the local speaker engine.

These offline tests use a deterministic fake embedding model. They pin the
rules that make repeated speech useful without letting one utterance, one
noisy window or one unreliable match pollute a voice print.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import numpy as np

from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.embedding import pcm_to_float32
from companion.speaker_id.engine import (
    LEARN_ADDED,
    LEARN_ASR_FAILED,
    LEARN_DUPLICATE,
    LEARN_IDENTITY_CONFLICT,
    LEARN_LOW_MARGIN,
    LEARN_LOW_SIMILARITY,
    SpeakerEngine,
)
from companion.speaker_id.store import SpeakerStore

SAMPLE_RATE = 16000
LEARNING_KEYS = {
    "added",
    "duplicate",
    "low_quality",
    "low_similarity",
    "low_margin",
    "identity_conflict",
    "asr_failed",
    "store_error",
}


# ------------------------------------------------------------------ helpers
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
    def __init__(self, vectors=None) -> None:
        self.vectors = dict(
            vectors
            or {220.0: unit(1, 0), 440.0: unit(0, 1), 660.0: unit(0, 0, 1)}
        )
        self.calls = 0
        self._lock = threading.Lock()

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


class EngineHarness:
    def __init__(self, store: SpeakerStore, embedding=None) -> None:
        self.store = store
        self.collector = Collector()
        self.engine = SpeakerEngine(
            load_speaker_config(env={}),
            embedding_model=embedding or FakeEmbedding(),
            store=store,
            on_result=self.collector,
        )
        self.engine.start()

    def close(self) -> None:
        self.engine.shutdown()

    def finish(self, uid: int, success: bool = True) -> None:
        self.engine.on_end(uid)
        self.collector.wait_for(
            lambda result: result.utterance_id == uid
            and result.status != "PENDING"
        )
        self.engine.on_finalized(uid, success)
        wait_until(lambda: self.engine.is_finished(uid))

    def samples(self) -> int:
        return sum(int(item["samples"]) for item in self.store.list_speakers())


def _engine(tmp_path, embedding=None) -> EngineHarness:
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    return EngineHarness(store, embedding)


def _learning(engine: SpeakerEngine) -> dict:
    return engine.stats()["learning"]


def _count(engine: SpeakerEngine, outcome: str) -> int:
    """Read a stat by its outcome constant (stats keys are lowercase)."""
    return _learning(engine)[outcome.lower()]


# ------------------------------------------------------- explicit outcomes
def test_stats_expose_every_learning_outcome() -> None:
    engine = SpeakerEngine(
        load_speaker_config(env={}),
        embedding_model=FakeEmbedding(),
        store=object(),
    )
    assert set(engine.stats()["learning"]) == LEARNING_KEYS


def test_one_utterance_adds_at_most_one_sample(tmp_path) -> None:
    harness = _engine(tmp_path)
    try:
        engine = harness.engine
        engine.on_start(1, tone(2.0, 220.0))
        harness.collector.wait_for(
            lambda r: r.utterance_id == 1 and r.status == "NEW_CANDIDATE"
        )
        engine.on_audio(1, tone(1.2, 220.0))
        engine.on_end(1)
        engine.on_finalized(1, True)
        wait_until(lambda: engine.is_finished(1))

        assert harness.samples() == 1
        assert _count(engine, LEARN_ADDED) == 1
        assert sum(_learning(engine).values()) == 1
    finally:
        harness.close()


def test_a_high_quality_trusted_sample_extends_the_identity(tmp_path) -> None:
    harness = _engine(tmp_path)
    try:
        engine = harness.engine
        engine.on_start(1, tone(2.0, 220.0))
        harness.finish(1)

        # Same voice, a genuinely different recording (0.99 < duplicate 0.995)
        # still passes the enrollment thresholds, so it is kept.
        angle = np.deg2rad(8.0)
        engine._embedding.vectors[221.0] = unit(
            float(np.cos(angle)), float(np.sin(angle))
        )
        engine.on_start(2, tone(2.0, 221.0))
        harness.finish(2)

        assert harness.samples() == 2
        assert _count(engine, LEARN_ADDED) == 2
    finally:
        harness.close()


def test_repeated_audio_does_not_grow_the_sample_set(tmp_path) -> None:
    harness = _engine(tmp_path)
    try:
        engine = harness.engine
        for uid in range(1, 6):
            engine.on_start(uid, tone(2.0, 220.0))
            harness.finish(uid)

        assert harness.samples() == 1
        learning = _learning(engine)
        assert learning[LEARN_ADDED.lower()] == 1
        assert learning[LEARN_DUPLICATE.lower()] == 4
    finally:
        harness.close()


def test_low_margin_competition_does_not_learn(tmp_path) -> None:
    # Two identities and a query that matches one of them above the enrollment
    # score but is too close to the other for a safe permanent sample.
    vectors = {
        220.0: unit(1, 0),
        440.0: unit(0, 1),
        660.0: unit(0.65, 0.57, 0.502),  # dot(A)=0.65, dot(B)=0.57
    }
    harness = _engine(tmp_path, FakeEmbedding(vectors))
    try:
        engine = harness.engine
        engine.on_start(1, tone(2.0, 220.0))
        harness.finish(1)
        engine.on_start(2, tone(2.0, 440.0))
        harness.finish(2)
        before = harness.samples()

        engine.on_start(3, tone(2.0, 660.0))
        harness.finish(3)

        assert harness.samples() == before
        assert _count(engine, LEARN_LOW_MARGIN) == 1
        assert _count(engine, LEARN_ADDED) == 2
    finally:
        harness.close()


def test_low_similarity_does_not_learn(tmp_path) -> None:
    # 0.50 clears the recognition threshold (0.43) but not enrollment (0.55).
    vectors = {
        220.0: unit(1, 0),
        440.0: unit(0, 1),
        660.0: unit(0.50, 0.30, 0.812),
    }
    harness = _engine(tmp_path, FakeEmbedding(vectors))
    try:
        engine = harness.engine
        engine.on_start(1, tone(2.0, 220.0))
        harness.finish(1)
        engine.on_start(2, tone(2.0, 440.0))
        harness.finish(2)
        before = harness.samples()

        engine.on_start(3, tone(2.0, 660.0))
        harness.finish(3)

        assert harness.samples() == before
        assert _count(engine, LEARN_LOW_SIMILARITY) == 1
    finally:
        harness.close()


def test_identity_conflict_permanently_blocks_the_utterance(tmp_path) -> None:
    vectors = {220.0: unit(1, 0), 440.0: unit(0, 1)}
    harness = _engine(tmp_path, FakeEmbedding(vectors))
    try:
        engine = harness.engine
        engine.on_start(1, tone(2.0, 220.0))
        harness.finish(1)
        engine.on_start(2, tone(2.0, 440.0))
        harness.finish(2)
        before = harness.samples()

        # One window says A, the next says B: no sample from this utterance.
        engine.on_start(3, tone(1.5, 220.0))
        harness.collector.wait_for(
            lambda r: r.utterance_id == 3 and r.status != "PENDING"
        )
        engine.on_audio(3, tone(1.5, 440.0))
        harness.collector.wait_for(
            lambda r: r.utterance_id == 3 and r.status != "PENDING", nth=2
        )
        harness.finish(3)

        assert harness.samples() == before
        assert _count(engine, LEARN_IDENTITY_CONFLICT) == 1
    finally:
        harness.close()


def test_asr_failure_keeps_no_sample_and_records_the_reason(tmp_path) -> None:
    harness = _engine(tmp_path)
    try:
        engine = harness.engine
        engine.on_start(1, tone(2.0, 220.0))
        harness.finish(1, success=False)

        assert harness.samples() == 0
        assert _count(engine, LEARN_ASR_FAILED) == 1

        engine.on_start(2, tone(2.0, 220.0))
        harness.finish(2, success=True)
        engine.on_start(3, tone(2.0, 220.0))
        harness.finish(3, success=False)

        assert harness.samples() == 1
        assert _count(engine, LEARN_ASR_FAILED) == 2
    finally:
        harness.close()


def test_cancel_keeps_no_sample(tmp_path) -> None:
    harness = _engine(tmp_path)
    try:
        engine = harness.engine
        engine.on_start(1, tone(2.0, 220.0))
        harness.finish(1)

        engine.on_start(2, tone(2.0, 220.0))
        harness.collector.wait_for(
            lambda r: r.utterance_id == 2 and r.status != "PENDING"
        )
        engine.on_cancel(2)
        wait_until(lambda: engine.is_finished(2))

        assert harness.samples() == 1
        # Cancellation is not a finalized utterance, so it has no learning
        # outcome; the retained sample set is what matters.
        assert sum(_learning(engine).values()) == 1
    finally:
        harness.close()


def test_a_sample_evicted_by_the_trim_is_not_reported_as_added(tmp_path) -> None:
    from companion.speaker_id.store import SAMPLE_ADDED, SAMPLE_DUPLICATE

    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    try:
        speaker = store.create_candidate()
        for index in range(8):
            vector = unit(*([0.0] * index + [1.0]))
            assert (
                store.add_embedding(
                    speaker,
                    vector,
                    quality="HIGH",
                    voiced_seconds=3.0,
                    model_id="fake",
                    embedding_version="v",
                )
                == SAMPLE_ADDED
            )

        # A ninth, weaker sample is inserted but the trim immediately evicts
        # it: the identity did not gain a reference vector, so this is not a
        # successful learn.
        weak = unit(*([0.0] * 8 + [1.0]))
        outcome = store.add_embedding(
            speaker,
            weak,
            quality="LOW",
            voiced_seconds=3.0,
            model_id="fake",
            embedding_version="v",
        )

        assert outcome == SAMPLE_DUPLICATE
        assert sum(int(item["samples"]) for item in store.list_speakers()) == 8
    finally:
        store.close()


# ------------------------------------------------------------- benchmark
def _benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_speaker.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_speaker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_run_waits_for_the_learning_commit(tmp_path) -> None:
    bench = _benchmark_module()
    engine = bench.TempEngine(
        load_speaker_config(env={}), FakeEmbedding(), tmp_path
    )
    try:
        engine.run(1, tone(2.0, 220.0))
        assert engine.engine.stats()["learning"][LEARN_ADDED.lower()] == 1

        # The count is only correct because run() waited for the commit; the
        # next utterance starts from the already-committed sample set.
        engine.run(2, tone(2.0, 220.0))
        assert engine.engine.stats()["learning"][LEARN_DUPLICATE.lower()] == 1
        samples = sum(
            int(item["samples"]) for item in engine.store.list_speakers()
        )
        assert samples == 1
    finally:
        engine.close()
