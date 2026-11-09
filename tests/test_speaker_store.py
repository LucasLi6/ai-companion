"""Offline tests for the separate speaker SQLite store."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from companion.speaker_id.config import EMBEDDING_DIM
from companion.speaker_id.embedding import SpeakerEmbeddingError
from companion.speaker_id.store import (
    IncompatibleModelError,
    SpeakerStore,
)

MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common@v1.0.1"
VERSION = "eres2netv2-192-v1"


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def open_store(tmp_path, **kwargs) -> SpeakerStore:
    kwargs.setdefault("max_embeddings_per_speaker", 4)
    store = SpeakerStore(tmp_path / "speakers.sqlite3", **kwargs)
    store.initialize()
    return store


def enroll(store, speaker_id, vector=None, quality="HIGH") -> None:
    store.add_embedding(
        speaker_id,
        vector if vector is not None else unit(1, 0),
        quality=quality,
        voiced_seconds=3.0,
        model_id=MODEL,
        embedding_version=VERSION,
    )


def test_temporary_ids_are_allocated_sequentially_by_the_database(tmp_path) -> None:
    store = open_store(tmp_path)
    assert store.create_candidate() == "TEMP-0001"
    assert store.create_candidate() == "TEMP-0002"
    store.close()

    reopened = open_store(tmp_path)
    assert reopened.create_candidate() == "TEMP-0003"
    reopened.close()


def test_embeddings_round_trip_and_reuse_a_profile(tmp_path) -> None:
    store = open_store(tmp_path)
    speaker = store.create_candidate()
    enroll(store, speaker, unit(1, 0))

    candidates, incompatible = store.load_candidates(
        model_id=MODEL, embedding_version=VERSION
    )
    assert incompatible == 0
    assert [c.speaker_id for c in candidates] == [speaker]
    assert candidates[0].status == "CANDIDATE"
    assert np.allclose(candidates[0].vectors[0], unit(1, 0), atol=1e-6)
    store.close()


def test_embedding_with_wrong_dimension_is_rejected(tmp_path) -> None:
    store = open_store(tmp_path)
    speaker = store.create_candidate()
    with pytest.raises(SpeakerEmbeddingError):
        store.add_embedding(
            speaker,
            np.ones(10, dtype=np.float32),
            quality="HIGH",
            voiced_seconds=1.0,
            model_id=MODEL,
            embedding_version=VERSION,
        )
    store.close()


def test_incompatible_model_is_never_mixed(tmp_path) -> None:
    store = open_store(tmp_path)
    speaker = store.create_candidate()
    enroll(store, speaker)
    with pytest.raises(IncompatibleModelError):
        store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=1.0,
            model_id="another-model",
            embedding_version="v2",
        )

    candidates, incompatible = store.load_candidates(
        model_id="another-model", embedding_version="v2"
    )
    assert candidates == []
    assert incompatible == 1
    store.close()


def test_sample_count_is_bounded_and_keeps_quality(tmp_path) -> None:
    store = open_store(tmp_path, max_embeddings_per_speaker=3)
    speaker = store.create_candidate()
    for index in range(3):
        enroll(store, speaker, unit(1, index * 0.1), quality="LOW")
    enroll(store, speaker, unit(0, 1), quality="HIGH")

    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT quality FROM speaker_embeddings WHERE speaker_id = ?",
            (speaker,),
        ).fetchall()
    assert len(rows) == 3
    assert any(row[0] == "HIGH" for row in rows)
    store.close()


def test_observation_is_one_row_per_utterance(tmp_path) -> None:
    store = open_store(tmp_path)
    speaker = store.create_candidate()
    store.record_observation(
        7, speaker_id=speaker, match_status="NEW_CANDIDATE",
        similarity=None, margin=None,
    )
    store.record_observation(
        7, speaker_id=speaker, match_status="CANDIDATE_MATCH",
        similarity=0.91, margin=0.2,
    )
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT match_status, similarity FROM speaker_observations"
            " WHERE utterance_id = 7"
        ).fetchall()
    assert rows == [("CANDIDATE_MATCH", 0.91)]
    assert store.get_profile(speaker)["observation_count"] == 1
    store.close()


def test_low_information_verdict_does_not_overwrite_a_match(tmp_path) -> None:
    store = open_store(tmp_path)
    speaker = store.create_candidate()
    store.record_observation(
        9, speaker_id=speaker, match_status="CANDIDATE_MATCH",
        similarity=0.9, margin=0.2,
    )
    store.record_observation(
        9, speaker_id=None, match_status="INSUFFICIENT_AUDIO",
        similarity=None, margin=None,
    )
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT speaker_id, match_status FROM speaker_observations"
            " WHERE utterance_id = 9"
        ).fetchone()
    assert row == (speaker, "CANDIDATE_MATCH")
    store.close()


def test_delete_removes_profile_samples_and_observations(tmp_path) -> None:
    store = open_store(tmp_path)
    speaker = store.create_candidate()
    enroll(store, speaker)
    store.record_observation(
        1, speaker_id=speaker, match_status="NEW_CANDIDATE",
        similarity=None, margin=None,
    )
    assert store.delete_speaker(speaker) is True
    assert store.delete_speaker(speaker) is False
    candidates, _ = store.load_candidates(
        model_id=MODEL, embedding_version=VERSION
    )
    assert candidates == []
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM speaker_embeddings"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM speaker_observations"
        ).fetchone()[0] == 0
    store.close()


def test_clear_removes_everything(tmp_path) -> None:
    store = open_store(tmp_path)
    enroll(store, store.create_candidate())
    enroll(store, store.create_candidate())
    assert store.clear() == 2
    assert store.list_speakers() == []
    store.close()


def test_merge_moves_samples_and_keeps_an_alias(tmp_path) -> None:
    store = open_store(tmp_path)
    first = store.create_candidate()
    second = store.create_candidate()
    enroll(store, first, unit(1, 0))
    enroll(store, second, unit(0, 1))

    assert store.merge_speakers(first, second) is True
    candidates, _ = store.load_candidates(
        model_id=MODEL, embedding_version=VERSION
    )
    assert [c.speaker_id for c in candidates] == [first]
    assert len(candidates[0].vectors) == 2
    aliases = store.aliases()
    assert aliases[0]["canonical_speaker_id"] == first
    assert aliases[0]["alias_speaker_id"] == second
    store.close()


def test_promote_temporary_identity_keeps_mapping(tmp_path) -> None:
    store = open_store(tmp_path)
    temporary = store.create_candidate()
    enroll(store, temporary)
    stable = store.promote_speaker(temporary)
    assert stable == "S-0001"
    assert store.get_profile(temporary) is None
    assert store.get_profile(stable)["status"] == "KNOWN"
    candidates, _ = store.load_candidates(
        model_id=MODEL, embedding_version=VERSION
    )
    assert [c.speaker_id for c in candidates] == [stable]
    assert store.aliases()[0]["alias_speaker_id"] == temporary
    store.close()
