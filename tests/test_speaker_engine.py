"""Offline tests for the non-blocking speaker engine.

The embedding model is a deterministic fake: it maps the dominant frequency of
the newest part of the window to a fixed 192-d vector. No weights, no network,
no random sleeps — every wait is a ``Condition``/``Event`` signalled by the
engine itself.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np
import pytest

from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.embedding import pcm_to_float32
from companion.speaker_id.engine import SpeakerEngine
from companion.speaker_id.models import final_tag, status_tag
from companion.speaker_id.store import SpeakerStore

SAMPLE_RATE = 16000


# ------------------------------------------------------------------ helpers
def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def tone(seconds: float, frequency: float = 220.0, *, amplitude: float = 0.2) -> bytes:
    n = int(seconds * SAMPLE_RATE)
    samples = amplitude * np.sin(
        2 * np.pi * frequency * np.arange(n, dtype=np.float32) / SAMPLE_RATE
    )
    return (samples * 32767).astype("<i2").tobytes()


DEFAULT_VECTORS = {
    220.0: unit(1, 0),
    440.0: unit(0, 1),
    660.0: unit(0, 0, 1),
}


class FakeEmbedding:
    """Deterministic stand-in for ERes2NetV2.

    Uses the dominant frequency of the most recent second of the window, so a
    window whose ending changed speaker maps to that speaker — exactly the
    hysteresis case the engine must survive.
    """

    def __init__(
        self,
        vectors: dict[float, np.ndarray] | None = None,
        *,
        fail_times: int = 0,
        block: bool = False,
    ) -> None:
        self.vectors = dict(vectors or DEFAULT_VECTORS)
        self.fail_times = fail_times
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self._lock = threading.Lock()

    def encode(self, pcm: bytes) -> np.ndarray:
        with self._lock:
            self.calls += 1
            if self.fail_times:
                self.fail_times -= 1
                raise RuntimeError("fake model failure")
        audio = pcm_to_float32(pcm)
        tail = audio[-SAMPLE_RATE:] if audio.size > SAMPLE_RATE else audio
        spectrum = np.abs(np.fft.rfft(tail))
        frequency = float(np.argmax(spectrum) * SAMPLE_RATE / tail.size)
        key = min(self.vectors, key=lambda candidate: abs(candidate - frequency))
        self.started.set()
        assert self.release.wait(timeout=3.0)
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


@pytest.fixture
def store(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def run_engine(store):
    engines = []

    def factory(embedding=None, *, on_result=None):
        engine = SpeakerEngine(
            load_speaker_config(env={}),
            embedding_model=embedding or FakeEmbedding(),
            store=store,
            on_result=on_result,
        )
        engine.start()
        engines.append(engine)
        return engine

    yield factory
    for engine in engines:
        engine.shutdown()


def identity(result) -> bool:
    return result.status != "PENDING"


def wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


# ------------------------------------------------------------------- tests
def test_first_utterance_creates_one_temporary_identity(run_engine, store) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(2.0))

    result = collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    assert result.utterance_id == 1
    assert result.speaker_id == "TEMP-0001"
    assert result.person_id is None
    assert result.reid_quality == "HIGH"
    assert result.provisional is True
    assert len(store.list_speakers()) == 1
    engine.on_end(1)


def test_same_voice_reuses_the_temporary_identity(run_engine, store) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(2.0))
    first = collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    engine.on_end(1)

    engine.on_start(2, tone(2.0))
    second = collector.wait_for(
        lambda r: r.utterance_id == 2 and identity(r)
    )
    engine.on_end(2)

    assert second.speaker_id == first.speaker_id == "TEMP-0001"
    assert second.status == "CANDIDATE_MATCH"
    assert 0.99 <= (second.similarity or 0.0) <= 1.0
    assert len(store.list_speakers()) == 1


def test_an_utterance_is_never_matched_against_its_own_samples(
    run_engine, store
) -> None:
    # Two recordings of one voice: cosine 0.766 with each other, 1.0 with
    # themselves. If interim windows were enrolled immediately, the second
    # analysis of utterance 2 would "match" its own first window at 1.0.
    import math

    angle = math.radians(40)
    first_voice = unit(1, 0)
    second_voice = unit(math.cos(angle), math.sin(angle))
    collector = Collector()
    engine = run_engine(
        embedding=FakeEmbedding({220.0: first_voice, 221.0: second_voice}),
        on_result=collector,
    )

    engine.on_start(1, tone(2.0, 220.0))
    first = collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    engine.on_end(1)
    engine.on_finalized(1, True)

    engine.on_start(2, tone(1.5, 221.0))
    collector.wait_for(lambda r: r.utterance_id == 2 and identity(r))
    engine.on_audio(2, tone(1.2, 221.0))  # second analysis at ~2.7 s
    second = collector.wait_for(
        lambda r: r.utterance_id == 2 and identity(r), nth=2
    )
    engine.on_end(2)
    engine.on_finalized(2, True)
    # The worker finalises an ended utterance before it analyses the next one,
    # so utterance 3's first result guarantees utterance 2 has been enrolled.
    engine.on_start(3, tone(1.5, 221.0))
    collector.wait_for(lambda r: r.utterance_id == 3 and identity(r))
    engine.on_end(3)
    engine.on_finalized(3, True)

    assert second.speaker_id == first.speaker_id == "TEMP-0001"
    assert second.status == "CANDIDATE_MATCH"
    assert second.similarity == pytest.approx(math.cos(angle), abs=0.01)
    # Exactly one independent sample per utterance, added when it ends.
    assert store.list_speakers()[0]["samples"] == 2


def test_different_people_are_not_merged(run_engine, store) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(2.0, 220.0))
    first = collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    engine.on_end(1)

    engine.on_start(2, tone(2.0, 440.0))
    second = collector.wait_for(
        lambda r: r.utterance_id == 2 and identity(r)
    )
    engine.on_end(2)

    assert second.status == "NEW_CANDIDATE"
    assert second.speaker_id == "TEMP-0002"
    assert second.speaker_id != first.speaker_id
    assert len(store.list_speakers()) == 2


def test_multiple_analyses_of_one_utterance_keep_one_identity(
    run_engine, store
) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(1.5))
    collector.wait_for(identity)
    engine.on_audio(1, tone(1.2))  # second analysis at ~2.7 s
    second = collector.wait_for(identity, nth=2)
    engine.on_audio(1, tone(1.0))  # refresh at ~3.7 s
    third = collector.wait_for(identity, nth=3)
    engine.on_end(1)

    assert second.speaker_id == "TEMP-0001"
    assert third.speaker_id == "TEMP-0001"
    # A new identity stays "new" for the whole first utterance; later windows
    # must not report a self-match as evidence of a known person.
    assert second.status == "NEW_CANDIDATE"
    assert second.similarity is None
    assert third.status == "NEW_CANDIDATE"
    assert third.similarity is None
    assert len(store.list_speakers()) == 1


def test_a_changed_window_mid_utterance_cannot_spawn_a_second_identity(
    run_engine, store
) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(1.5, 220.0))
    collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    # The recent second is now a clearly different voice; the lock keeps the
    # original identity from being replaced, but the honest verdict for this
    # window is AMBIGUOUS — not a verified match for TEMP-0001.
    engine.on_audio(1, tone(2.0, 440.0))
    second = collector.wait_for(identity, nth=2)
    engine.on_end(1)
    engine.on_finalized(1, True)

    assert second.status == "AMBIGUOUS"
    assert second.speaker_id is None
    # No second temporary identity was created, and the contradiction means
    # the utterance's provisional sample is not kept as a voice print.
    assert [row["speaker_id"] for row in store.list_speakers()] == [
        "TEMP-0001"
    ]
    wait_until(lambda: store.list_speakers()[0]["samples"] == 0)


def test_short_utterance_is_insufficient_and_creates_nothing(
    run_engine, store
) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(0.4))
    engine.on_end(1)

    result = collector.wait_for(identity)
    assert result.status == "INSUFFICIENT_AUDIO"
    assert result.speaker_id is None
    assert result.person_id is None
    assert result.provisional is False
    assert result.voiced_seconds < 0.9
    assert store.list_speakers() == []


def test_low_quality_audio_never_creates_an_identity(run_engine, store) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(1.0))
    engine.on_end(1)

    result = collector.wait_for(identity)
    assert result.status == "INSUFFICIENT_AUDIO"
    assert result.reid_quality == "LOW"
    assert store.list_speakers() == []


def test_score_in_the_uncertainty_band_is_ambiguous(
    run_engine, store
) -> None:
    config = load_speaker_config(env={})
    speaker = store.create_candidate()
    store.add_embedding(
        speaker,
        unit(1, 0),
        quality="HIGH",
        voiced_seconds=3.0,
        model_id=f"{config.model_id}@{config.model_revision}",
        embedding_version=config.embedding_version,
    )
    collector = Collector()
    engine = run_engine(
        FakeEmbedding({220.0: unit(0.35, 0.93675)}), on_result=collector
    )

    engine.on_start(1, tone(2.0))
    result = collector.wait_for(identity)
    engine.on_end(1)

    assert result.status == "AMBIGUOUS"
    assert result.speaker_id is None
    assert result.similarity == pytest.approx(0.35, abs=0.01)
    assert len(store.list_speakers()) == 1
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT match_status FROM speaker_observations"
            " WHERE utterance_id = 1"
        ).fetchone()
    assert row == ("AMBIGUOUS",)


def test_model_exception_does_not_kill_the_worker(run_engine, store) -> None:
    collector = Collector()
    embedding = FakeEmbedding(fail_times=1)
    engine = run_engine(embedding, on_result=collector)

    engine.on_start(1, tone(1.5))
    error = collector.wait_for(lambda r: r.status == "ERROR")
    assert error.speaker_id is None

    engine.on_audio(1, tone(1.2))  # next scheduled analysis succeeds
    result = collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    engine.on_end(1)

    assert engine.running is True
    assert result.speaker_id == "TEMP-0001"
    assert len(store.list_speakers()) == 1


def test_incompatible_model_versions_are_refused(run_engine, store) -> None:
    speaker = store.create_candidate()
    store.add_embedding(
        speaker,
        unit(1, 0),
        quality="HIGH",
        voiced_seconds=3.0,
        model_id="other-model@v0",
        embedding_version="v0",
    )
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(2.0))
    result = collector.wait_for(identity)
    engine.on_end(1)

    assert result.status == "ERROR"
    assert result.speaker_id is None
    assert [row["speaker_id"] for row in store.list_speakers()] == [
        speaker
    ]


def test_snapshot_never_waits_for_inference(run_engine) -> None:
    collector = Collector()
    embedding = FakeEmbedding(block=True)
    engine = run_engine(embedding, on_result=collector)

    engine.on_start(1, tone(2.0))
    assert embedding.started.wait(timeout=3.0), "analysis never started"

    started = time.monotonic()
    snapshot = engine.snapshot(1)
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert snapshot.status == "PENDING"
    assert snapshot.utterance_id == 1

    embedding.release.set()
    assert collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    engine.on_end(1)


def test_on_end_never_waits_for_inference(run_engine) -> None:
    collector = Collector()
    embedding = FakeEmbedding(block=True)
    engine = run_engine(embedding, on_result=collector)

    engine.on_start(1, tone(2.0))
    assert embedding.started.wait(timeout=3.0)

    started = time.monotonic()
    engine.on_end(1)
    elapsed = time.monotonic() - started
    assert elapsed < 0.1

    embedding.release.set()
    assert collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")


def test_cancel_discards_state_and_never_enrolls(run_engine, store) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(0.5))
    engine.on_cancel(1)
    engine.on_end(1)
    engine.shutdown()

    assert engine.snapshot(1).status == "PENDING"
    assert store.list_speakers() == []


def test_buffer_is_bounded_to_the_recent_window(run_engine) -> None:
    engine = run_engine()
    audio = tone(10.0)

    engine.on_start(1, audio)
    with engine._lock:
        state = engine._states[1]
        buffered = len(state.buffer)
        total = state.total_bytes

    assert total == len(audio)
    assert buffered <= engine.config.window_bytes
    assert buffered == engine.config.window_bytes


def test_shutdown_leaves_no_inference_thread(run_engine) -> None:
    engine = run_engine()
    assert engine.running is True
    engine.shutdown()
    assert engine.running is False
    assert [
        thread.name
        for thread in threading.enumerate()
        if thread.name == "speaker-id" and thread.is_alive()
    ] == []


def test_restart_reuses_a_persisted_identity(tmp_path) -> None:
    path = tmp_path / "restart.sqlite3"
    config = load_speaker_config(env={})
    first_store = SpeakerStore(path)
    first_store.initialize()
    first_engine = SpeakerEngine(
        config, embedding_model=FakeEmbedding(), store=first_store
    )
    first_engine.start()
    first_collector = Collector()
    first_engine.set_on_result(first_collector)
    first_engine.on_start(1, tone(2.0))
    first = first_collector.wait_for(lambda r: r.status == "NEW_CANDIDATE")
    first_engine.shutdown()
    first_store.close()

    second_store = SpeakerStore(path)
    second_store.initialize()
    second_engine = SpeakerEngine(
        config, embedding_model=FakeEmbedding(), store=second_store
    )
    second_engine.start()
    second_collector = Collector()
    second_engine.set_on_result(second_collector)
    second_engine.on_start(1, tone(2.0))
    second = second_collector.wait_for(identity)
    second_engine.shutdown()
    second_store.close()

    assert first.speaker_id == "TEMP-0001"
    assert second.status == "CANDIDATE_MATCH"
    assert second.speaker_id == "TEMP-0001"


def test_results_are_per_utterance_and_never_cross_used(run_engine) -> None:
    collector = Collector()
    engine = run_engine(on_result=collector)

    engine.on_start(1, tone(2.0, 220.0))
    collector.wait_for(lambda r: r.utterance_id == 1 and identity(r))
    engine.on_end(1)
    engine.on_start(2, tone(2.0, 440.0))
    collector.wait_for(lambda r: r.utterance_id == 2 and identity(r))
    engine.on_end(2)

    assert engine.snapshot(1).utterance_id == 1
    assert engine.snapshot(2).utterance_id == 2
    assert engine.snapshot(1).speaker_id != engine.snapshot(2).speaker_id
    assert all(result.person_id is None for result in collector.results)


def test_tags_describe_state_without_claiming_certainty() -> None:
    from companion.speaker_id.models import SpeakerResult, pending_result

    assert status_tag(pending_result(1)) == "[ID: PENDING | 语音质量: 评估中]"

    new = SpeakerResult(
        utterance_id=1,
        speaker_id="TEMP-0001",
        person_id=None,
        status="NEW_CANDIDATE",
        reid_quality="HIGH",
        similarity=None,
        margin=None,
        voiced_seconds=2.4,
        provisional=False,
        model_id="m",
        inference_ms=12.0,
        completed_at=1.0,
    )
    assert status_tag(new) == "[ID: TEMP-0001 | NEW | 可复识: HIGH]"
    assert final_tag(new) == "[Speaker: TEMP-0001 | NEW | ReID: HIGH]"

    known = SpeakerResult(
        **{**new.__dict__, "status": "KNOWN", "speaker_id": "S-0001",
           "similarity": 0.89}
    )
    assert final_tag(known) == (
        "[Speaker: S-0001 | KNOWN | ReID: HIGH | sim: 0.89]"
    )

    ambiguous = SpeakerResult(
        **{**new.__dict__, "status": "AMBIGUOUS", "speaker_id": None,
           "similarity": 0.60, "reid_quality": "MEDIUM"}
    )
    assert final_tag(ambiguous) == (
        "[Speaker: UNKNOWN | AMBIGUOUS | ReID: MEDIUM | sim: 0.60]"
    )
