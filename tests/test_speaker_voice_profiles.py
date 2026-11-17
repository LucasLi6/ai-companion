"""Offline tests for V2 voice-profile mathematics.

All vectors are deterministic synthetic embeddings. They test geometry and
determinism, never real recognition accuracy: no result here is an accuracy
claim about the ERes2NetV2 model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from companion.speaker_id.config import EMBEDDING_DIM
from companion.speaker_id.voice_profiles import (
    STRATEGY_LEGACY_MAX_SAMPLE,
    STRATEGY_VARIANT_CENTROID,
    VoiceProfileError,
    VoiceProfileIndex,
    VoiceSample,
    build_variants,
    make_profile,
    normalize_embedding,
    score_profile,
)


def unit_angle(degrees: float) -> np.ndarray:
    """A unit vector in the first two dimensions (easy to reason about)."""
    radians = math.radians(degrees)
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = math.cos(radians)
    vector[1] = math.sin(radians)
    return vector


def sample(sample_id: int, degrees: float, *, created_at: float = 0.0) -> VoiceSample:
    return VoiceSample(
        sample_id=sample_id,
        vector=unit_angle(degrees),
        created_at=created_at,
    )


# ------------------------------------------------------------------ validation
def test_normalize_embedding_returns_a_unit_vector() -> None:
    vector = normalize_embedding([3.0, 4.0], expected_dim=2)
    assert np.allclose(vector, [0.6, 0.8], atol=1e-6)
    assert np.isclose(float(np.linalg.norm(vector)), 1.0)


@pytest.mark.parametrize(
    "vector",
    [
        np.ones(10, dtype=np.float32),
        np.array([np.nan, 1.0], dtype=np.float32),
        np.zeros(EMBEDDING_DIM, dtype=np.float32),
    ],
)
def test_normalize_embedding_rejects_bad_vectors(vector) -> None:
    with pytest.raises(VoiceProfileError):
        normalize_embedding(vector, expected_dim=EMBEDDING_DIM)


# ----------------------------------------------------------------------- variants
def test_close_samples_form_one_variant() -> None:
    variants = build_variants(
        [sample(1, 0), sample(2, 10), sample(3, 20)],
        merge_threshold=0.8,
        max_variants=4,
    )
    assert len(variants) == 1
    assert variants[0].support == 3
    assert variants[0].sample_ids == (1, 2, 3)
    # The centroid of 0/10/20 degrees lies at 10 degrees.
    assert np.allclose(variants[0].centroid, unit_angle(10), atol=1e-5)


def test_condition_variants_stay_separate_when_the_threshold_says_so() -> None:
    variants = build_variants(
        [sample(1, 0), sample(2, 0), sample(3, 60), sample(4, 60)],
        merge_threshold=0.8,
        max_variants=4,
    )
    assert len(variants) == 2
    assert [variant.support for variant in variants] == [2, 2]


def test_variant_build_is_deterministic_regardless_of_input_order() -> None:
    samples = [
        sample(3, 60, created_at=3.0),
        sample(1, 0, created_at=1.0),
        sample(2, 5, created_at=2.0),
        sample(4, 65, created_at=4.0),
    ]
    forward = build_variants(samples, merge_threshold=0.9, max_variants=4)
    backward = build_variants(
        list(reversed(samples)), merge_threshold=0.9, max_variants=4
    )
    assert [v.sample_ids for v in forward] == [v.sample_ids for v in backward]
    assert [v.index for v in forward] == [0, 1]


def test_max_variants_is_always_honoured() -> None:
    samples = [sample(index, index * 20) for index in range(1, 9)]
    variants = build_variants(samples, merge_threshold=0.99, max_variants=3)
    assert len(variants) == 3
    assert sum(variant.support for variant in variants) == 8


def test_quarantined_samples_do_not_build_variants() -> None:
    quarantined = VoiceSample(
        sample_id=9,
        vector=unit_angle(180),
        created_at=9.0,
        quarantined=True,
    )
    variants = build_variants(
        [sample(1, 0), quarantined], merge_threshold=0.8, max_variants=4
    )
    assert len(variants) == 1
    assert variants[0].sample_ids == (1,)


# ------------------------------------------------------------------------ scores
def test_variant_centroid_averages_samples_and_beats_the_max_sample() -> None:
    # Two samples at -15/+15 degrees: max-sample score for the 0-degree query
    # is cos(15)=0.966, but their centroid is exactly the query (1.0).
    profile = make_profile(
        "TEMP-0001",
        [sample(1, -15), sample(2, 15)],
        merge_threshold=0.8,
        max_variants=4,
    )
    query = unit_angle(0)
    legacy = score_profile(query, profile, strategy=STRATEGY_LEGACY_MAX_SAMPLE)
    variant = score_profile(query, profile, strategy=STRATEGY_VARIANT_CENTROID)

    assert legacy.raw_max_similarity == pytest.approx(math.cos(math.radians(15)), abs=1e-5)
    assert legacy.effective_similarity == pytest.approx(legacy.raw_max_similarity)
    assert variant.variant_max_similarity == pytest.approx(1.0, abs=1e-5)
    assert variant.effective_similarity > legacy.effective_similarity
    assert variant.best_variant_support == 2


def test_legacy_strategy_reproduces_the_old_max_sample_score() -> None:
    profile = make_profile(
        "TEMP-0001",
        [sample(1, 0), sample(2, 60)],
        merge_threshold=0.9,
        max_variants=4,
    )
    score = score_profile(
        unit_angle(55), profile, strategy=STRATEGY_LEGACY_MAX_SAMPLE
    )
    expected = max(
        float(unit_angle(55) @ sample(1, 0).vector),
        float(unit_angle(55) @ sample(2, 60).vector),
    )
    assert score.effective_similarity == pytest.approx(expected, abs=1e-6)


def test_singleton_penalty_is_opt_in_and_only_hits_singletons() -> None:
    profile = make_profile(
        "TEMP-0001",
        [sample(1, 0, created_at=1), sample(2, 0, created_at=2), sample(3, 60, created_at=3)],
        merge_threshold=0.9,
        max_variants=4,
    )
    # A query close to the singleton (60 degrees) but far from the pair.
    query = unit_angle(60)
    without = score_profile(
        query, profile, strategy=STRATEGY_VARIANT_CENTROID, singleton_penalty=0.0
    )
    with_penalty = score_profile(
        query, profile, strategy=STRATEGY_VARIANT_CENTROID, singleton_penalty=0.05
    )
    assert without.best_variant_support == 1
    assert with_penalty.effective_similarity == pytest.approx(
        without.effective_similarity - 0.05, abs=1e-6
    )
    assert with_penalty.singleton_penalty_applied == pytest.approx(0.05)


def test_best_sample_and_variant_are_reported() -> None:
    profile = make_profile(
        "TEMP-0001", [sample(11, 0), sample(12, 90)], max_variants=4
    )
    score = score_profile(
        unit_angle(80), profile, strategy=STRATEGY_VARIANT_CENTROID
    )
    assert score.best_sample_id == 12
    assert score.best_variant_index is not None
    assert score.speaker_id == "TEMP-0001"
    assert score.strategy == STRATEGY_VARIANT_CENTROID


def test_empty_profile_scores_as_no_evidence() -> None:
    profile = make_profile("TEMP-0001", [])
    score = score_profile(unit_angle(0), profile)
    assert score.raw_max_similarity == -1.0
    assert score.variant_max_similarity == -1.0
    assert score.effective_similarity == -1.0
    assert score.best_sample_id is None
    assert score.best_variant_index is None


# ------------------------------------------------------------------------- cache
def test_profile_cache_hits_and_invalidates() -> None:
    index = VoiceProfileIndex(merge_threshold=0.8, max_variants=4)
    samples = [sample(1, 0), sample(2, 10)]
    first = index.profile_for("TEMP-0001", samples)
    second = index.profile_for("TEMP-0001", samples)
    assert first is second
    stats = index.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1

    index.invalidate("TEMP-0001")
    third = index.profile_for("TEMP-0001", samples)
    assert third is not first
    assert index.stats()["misses"] == 2


def test_profile_cache_is_bounded() -> None:
    index = VoiceProfileIndex(merge_threshold=0.8, max_variants=4, max_entries=2)
    for index_value in range(5):
        index.profile_for(
            f"TEMP-{index_value:04d}",
            [sample(index_value + 1, index_value * 20)],
        )
    stats = index.stats()
    assert stats["entries"] == 2
    assert stats["evictions"] == 3


def test_cache_fingerprint_changes_when_a_sample_is_added() -> None:
    index = VoiceProfileIndex(merge_threshold=0.8, max_variants=4)
    index.profile_for("TEMP-0001", [sample(1, 0)])
    expanded = index.profile_for(
        "TEMP-0001", [sample(1, 0), sample(2, 60)]
    )
    assert len(expanded.samples) == 2
    assert index.stats()["misses"] == 2
