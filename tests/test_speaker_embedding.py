"""Offline tests for the speaker embedding adapter.

The real ERes2NetV2 model is not downloaded in CI. These tests lock down the
pure preprocessing/validation contract and the failure modes of the adapter;
a separate opt-in test can exercise real weights with
``AVA_SPEAKER_REAL_TESTS=1``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from companion.speaker_id.embedding import (
    EMBEDDING_DIM,
    SpeakerEmbeddingError,
    SpeakerEmbeddingModel,
    pcm_to_float32,
    validate_embedding,
)


def test_pcm_to_float32_normalizes_and_ignores_odd_tail() -> None:
    pcm = np.array([0, 32767, -32768], dtype="<i2").tobytes() + b"\x01"
    audio = pcm_to_float32(pcm)
    assert audio.dtype == np.float32
    assert audio.shape == (3,)
    assert audio[0] == 0.0
    assert audio[1] == pytest.approx(32767 / 32768)
    assert audio[2] == pytest.approx(-1.0)


def test_pcm_to_float32_empty() -> None:
    assert pcm_to_float32(b"").size == 0


def test_validate_embedding_l2_normalizes() -> None:
    vector = np.arange(EMBEDDING_DIM, dtype=np.float32) + 1.0
    result = validate_embedding(vector)
    assert result.shape == (EMBEDDING_DIM,)
    assert np.isclose(float(np.linalg.norm(result)), 1.0, atol=1e-5)


@pytest.mark.parametrize(
    "vector",
    [
        np.zeros(EMBEDDING_DIM, dtype=np.float32),
        np.zeros(EMBEDDING_DIM - 1, dtype=np.float32),
        np.array([np.nan] * EMBEDDING_DIM, dtype=np.float32),
        np.array([np.inf] * EMBEDDING_DIM, dtype=np.float32),
    ],
)
def test_validate_embedding_rejects_invalid_vectors(vector) -> None:
    with pytest.raises(SpeakerEmbeddingError):
        validate_embedding(vector)


def test_encode_before_load_raises() -> None:
    model = SpeakerEmbeddingModel()
    with pytest.raises(SpeakerEmbeddingError):
        model.encode(b"\x00\x10" * 1600)
    assert model.loaded is False


def test_extract_embedding_accepts_numpy_and_torch_like_objects() -> None:
    model = SpeakerEmbeddingModel()
    array = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    array[0, 0] = 1.0
    assert model._extract_embedding({"embs": array}, np).shape == (EMBEDDING_DIM,)

    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return array

    assert model._extract_embedding(
        {"embs": TensorLike()}, np
    ).shape == (EMBEDDING_DIM,)


def test_extract_embedding_rejects_missing_payload() -> None:
    model = SpeakerEmbeddingModel()
    with pytest.raises(SpeakerEmbeddingError):
        model._extract_embedding({"score": 0.9}, np)


def test_close_resets_loaded_state() -> None:
    model = SpeakerEmbeddingModel()
    model._loaded = True
    model._pipeline = object()
    model.close()
    assert model.loaded is False
    assert model._pipeline is None


@pytest.mark.skipif(
    not os.environ.get("AVA_SPEAKER_REAL_TESTS"),
    reason="real model test is opt-in and downloads weights",
)
def test_real_model_returns_normalized_192d_embedding() -> None:  # pragma: no cover
    # The suite blocks the network. Point the adapter at the already-downloaded
    # snapshot so the opt-in test stays offline; download it first with
    # `python scripts/benchmark_speaker.py --skip-identity`.
    from modelscope.hub.snapshot_download import snapshot_download

    from companion.speaker_id.config import (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
    )

    try:
        local = snapshot_download(
            DEFAULT_MODEL_ID,
            revision=DEFAULT_MODEL_REVISION,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001 - cache miss is a skip, not a failure
        pytest.skip("real model weights are not in the local ModelScope cache")
    model = SpeakerEmbeddingModel(model_path=local)
    model.load()
    model.warmup()
    audio = (
        np.sin(
            2
            * np.pi
            * 220
            * np.arange(16000, dtype=np.float32)
            / 16000
        )
        * 8000
    ).astype("<i2")
    vector = model.encode(audio.tobytes())
    assert vector.shape == (EMBEDDING_DIM,)
    assert np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-4)
    model.close()
