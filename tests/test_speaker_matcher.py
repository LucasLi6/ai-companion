"""Offline tests for open-set matching, uncertainty and hysteresis."""

from __future__ import annotations

import numpy as np
import pytest

from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.matcher import SpeakerCandidate, SpeakerMatcher
from companion.speaker_id.quality import VoiceQuality


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


HIGH = VoiceQuality(True, 3.0, "HIGH", None)
LOW = VoiceQuality(True, 1.0, "LOW", None)
BAD = VoiceQuality(False, 0.2, "LOW", "有效语音时长不足")


def matcher() -> SpeakerMatcher:
    return SpeakerMatcher(load_speaker_config(env={}))


def test_new_speaker_with_high_quality_creates_a_candidate() -> None:
    decision = matcher().match(unit(1, 0), [], quality=HIGH)
    assert decision.status == "NEW_CANDIDATE"
    assert decision.speaker_id is None


def test_new_speaker_with_low_quality_is_not_enrolled() -> None:
    decision = matcher().match(unit(1, 0), [], quality=LOW)
    assert decision.status == "INSUFFICIENT_AUDIO"


def test_unusable_audio_never_matches_or_creates() -> None:
    candidate = SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),))
    decision = matcher().match(unit(1, 0), [candidate], quality=BAD)
    assert decision.status == "INSUFFICIENT_AUDIO"
    assert decision.speaker_id is None


def test_close_voice_reuses_a_temporary_identity() -> None:
    candidate = SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),))
    query = unit(1.0, 0.1)
    decision = matcher().match(query, [candidate], quality=HIGH)
    assert decision.status == "CANDIDATE_MATCH"
    assert decision.speaker_id == "TEMP-0001"
    assert decision.similarity is not None and decision.similarity > 0.9


def test_close_voice_matches_a_known_identity() -> None:
    candidate = SpeakerCandidate("S-0001", "KNOWN", (unit(1, 0),))
    decision = matcher().match(unit(1, 0.05), [candidate], quality=HIGH)
    assert decision.status == "KNOWN"
    assert decision.speaker_id == "S-0001"


def test_score_inside_the_uncertainty_band_is_ambiguous() -> None:
    candidate = SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),))
    query = unit(0.35, 0.93675)  # cosine ≈ 0.35, between 0.30 and 0.43
    decision = matcher().match(query, [candidate], quality=HIGH)
    assert decision.status == "AMBIGUOUS"
    assert decision.speaker_id is None
    assert decision.similarity == pytest.approx(0.35, abs=0.01)


def test_small_margin_between_two_identities_is_ambiguous() -> None:
    candidates = [
        SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),)),
        SpeakerCandidate("TEMP-0002", "CANDIDATE", (unit(0, 1),)),
    ]
    query = unit(0.9, 0.9)
    decision = matcher().match(query, candidates, quality=HIGH)
    assert decision.status == "AMBIGUOUS"
    assert decision.margin is not None and decision.margin < 0.06


def test_far_voice_creates_a_new_temporary_identity() -> None:
    candidate = SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),))
    decision = matcher().match(unit(0, 1), [candidate], quality=HIGH)
    assert decision.status == "NEW_CANDIDATE"
    assert decision.speaker_id is None


def test_locked_identity_loses_to_a_clearly_better_other_match() -> None:
    """A lock keeps the candidate stable; it is not proof of identity.

    When a later window is clearly closer to another identity, the honest
    answer is AMBIGUOUS — never "KNOWN TEMP-0001" based on the lock alone.
    """
    candidates = [
        SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),)),
        SpeakerCandidate("TEMP-0002", "CANDIDATE", (unit(0, 1),)),
    ]
    query = unit(0.2, 1.0)  # closer to TEMP-0002
    decision = matcher().match(
        query, candidates, quality=HIGH, locked_speaker_id="TEMP-0001"
    )
    assert decision.speaker_id is None
    assert decision.status == "AMBIGUOUS"
    assert decision.similarity == pytest.approx(0.196, abs=0.01)


def test_locked_identity_still_matches_when_the_score_holds_up() -> None:
    candidates = [
        SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),)),
        SpeakerCandidate("TEMP-0002", "CANDIDATE", (unit(0, 1),)),
    ]
    decision = matcher().match(
        unit(1, 0.05),
        candidates,
        quality=HIGH,
        locked_speaker_id="TEMP-0001",
    )
    assert decision.status == "CANDIDATE_MATCH"
    assert decision.speaker_id == "TEMP-0001"
    assert decision.margin is not None and decision.margin >= 0.06


def test_a_locked_identity_that_disappeared_uses_the_real_quality() -> None:
    """The fallback must not fabricate LOW quality for a usable window."""
    decision = matcher().match(
        unit(1, 0),
        [],
        quality=HIGH,
        locked_speaker_id="TEMP-0001",
    )
    assert decision.status == "NEW_CANDIDATE"

    low = VoiceQuality(False, 0.2, "LOW", "有效语音时长不足")
    decision = matcher().match(
        unit(1, 0),
        [],
        quality=low,
        locked_speaker_id="TEMP-0001",
    )
    assert decision.status == "INSUFFICIENT_AUDIO"


def test_a_locked_identity_can_recover_after_a_bad_window() -> None:
    candidates = [
        SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),)),
        SpeakerCandidate("TEMP-0002", "CANDIDATE", (unit(0, 1),)),
    ]
    noisy = matcher().match(
        unit(0.2, 1.0),
        candidates,
        quality=HIGH,
        locked_speaker_id="TEMP-0001",
    )
    assert noisy.status == "AMBIGUOUS"

    recovered = matcher().match(
        unit(1, 0.02),
        candidates,
        quality=HIGH,
        locked_speaker_id="TEMP-0001",
    )
    assert recovered.status == "CANDIDATE_MATCH"
    assert recovered.speaker_id == "TEMP-0001"


def test_multiple_samples_use_the_best_match() -> None:
    candidate = SpeakerCandidate(
        "TEMP-0001", "CANDIDATE", (unit(0, 1), unit(1, 0))
    )
    decision = matcher().match(unit(1, 0.05), [candidate], quality=HIGH)
    assert decision.status == "CANDIDATE_MATCH"


# ------------------------------------------------------- V2 profile strategies
import math

from companion.speaker_id.voice_profiles import VoiceSample


def voice_sample(sample_id: int, degrees: float) -> VoiceSample:
    radians = math.radians(degrees)
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = math.cos(radians)
    vector[1] = math.sin(radians)
    return VoiceSample(sample_id=sample_id, vector=vector, created_at=sample_id)


def angle(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = math.cos(radians)
    vector[1] = math.sin(radians)
    return vector


def variant_matcher(**env) -> SpeakerMatcher:
    settings = {"AVA_SPEAKER_MATCH_STRATEGY": "variant_centroid"}
    settings.update(env)
    return SpeakerMatcher(load_speaker_config(env=settings))


def test_legacy_strategy_still_uses_the_best_single_sample() -> None:
    candidate = SpeakerCandidate(
        "TEMP-0001",
        "CANDIDATE",
        (angle(-15), angle(15)),
        (voice_sample(1, -15), voice_sample(2, 15)),
    )
    decision = matcher().match(angle(15), [candidate], quality=HIGH)
    assert decision.similarity == pytest.approx(
        math.cos(math.radians(0)), abs=1e-5
    )


def test_variant_strategy_uses_the_cluster_centroid() -> None:
    candidate = SpeakerCandidate(
        "TEMP-0001",
        "CANDIDATE",
        (angle(-15), angle(15)),
        (voice_sample(1, -15), voice_sample(2, 15)),
    )
    decision = variant_matcher().match(angle(0), [candidate], quality=HIGH)
    assert decision.status == "CANDIDATE_MATCH"
    assert decision.similarity == pytest.approx(1.0, abs=1e-5)


def test_variants_mode_selects_variant_scoring_by_default() -> None:
    config = load_speaker_config(env={"AVA_SPEAKER_PROFILE_MODE": "variants"})
    assert config.match_strategy == "variant_centroid"
    matcher_v2 = SpeakerMatcher(config)
    candidate = SpeakerCandidate(
        "TEMP-0001",
        "CANDIDATE",
        (angle(-15), angle(15)),
        (voice_sample(1, -15), voice_sample(2, 15)),
    )
    decision = matcher_v2.match(angle(0), [candidate], quality=HIGH)
    assert decision.similarity == pytest.approx(1.0, abs=1e-5)


def test_shadow_mode_returns_the_legacy_verdict_and_records_the_difference() -> None:
    shadow = SpeakerMatcher(
        load_speaker_config(env={"AVA_SPEAKER_PROFILE_MODE": "shadow"})
    )
    # 0.40 is inside the uncertainty band for the raw max (0.40), while the
    # centroid of the two samples sits at 15 degrees and scores ~0.966.
    candidate = SpeakerCandidate(
        "TEMP-0001",
        "CANDIDATE",
        (angle(-15), angle(15)),
        (voice_sample(1, -15), voice_sample(2, 15)),
    )
    decision = shadow.match(angle(0), [candidate], quality=HIGH)
    # Legacy verdict is returned unchanged: raw max is 0.966, so match.
    assert decision.status == "CANDIDATE_MATCH"
    stats = shadow.shadow_stats()
    assert stats["agree"] + stats["disagree"] == 1
    assert stats["mode"] == "shadow"


def test_shadow_mode_never_changes_the_threshold_band() -> None:
    shadow = SpeakerMatcher(
        load_speaker_config(env={"AVA_SPEAKER_PROFILE_MODE": "shadow"})
    )
    candidate = SpeakerCandidate("TEMP-0001", "CANDIDATE", (unit(1, 0),))
    decision = shadow.match(
        unit(0.35, 0.93675), [candidate], quality=HIGH
    )
    assert decision.status == "AMBIGUOUS"
    assert decision.speaker_id is None


def test_second_place_is_always_a_different_person_under_variants() -> None:
    first = SpeakerCandidate(
        "TEMP-0001",
        "CANDIDATE",
        (angle(-15), angle(15)),
        (voice_sample(1, -15), voice_sample(2, 15)),
    )
    second = SpeakerCandidate(
        "TEMP-0002",
        "CANDIDATE",
        (angle(-15), angle(15)),
        (voice_sample(3, -15), voice_sample(4, 15)),
    )
    decision = variant_matcher().match(angle(0), [first, second], quality=HIGH)
    # Equal scores: the margin is zero, so the honest answer is AMBIGUOUS.
    assert decision.status == "AMBIGUOUS"
    assert decision.margin == pytest.approx(0.0, abs=1e-6)


def test_singleton_penalty_can_be_configured_without_touching_thresholds() -> None:
    strict = variant_matcher(
        AVA_SPEAKER_SINGLETON_PENALTY="0.10",
    )
    candidate = SpeakerCandidate(
        "TEMP-0001",
        "CANDIDATE",
        (angle(0),),
        (voice_sample(1, 0),),
    )
    # Raw/variant score is 0.50 in the uncertainty band; a 0.10 singleton
    # penalty pushes it below the new-candidate gate.
    decision = strict.match(angle(60), [candidate], quality=HIGH)
    assert decision.similarity == pytest.approx(0.40, abs=1e-6)
    assert decision.status == "AMBIGUOUS"


def test_variant_scoring_uses_the_profile_cache() -> None:
    matcher_v2 = variant_matcher()
    candidate = SpeakerCandidate(
        "TEMP-0001",
        "CANDIDATE",
        (angle(-15), angle(15)),
        (voice_sample(1, -15), voice_sample(2, 15)),
    )
    for _ in range(3):
        matcher_v2.match(angle(0), [candidate], quality=HIGH)
    stats = matcher_v2.profile_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 2


def test_locked_variant_match_still_needs_a_margin() -> None:
    matcher_v2 = variant_matcher()
    candidates = [
        SpeakerCandidate(
            "TEMP-0001", "CANDIDATE", (angle(0),), (voice_sample(1, 0),)
        ),
        SpeakerCandidate(
            "TEMP-0002", "CANDIDATE", (angle(10),), (voice_sample(2, 10),)
        ),
    ]
    decision = matcher_v2.match(
        angle(5), candidates, quality=HIGH, locked_speaker_id="TEMP-0001"
    )
    assert decision.status == "AMBIGUOUS"
    assert decision.speaker_id is None


# ------------------------------------------------- singleton margin semantics
def test_singleton_known_match_is_decided_without_a_margin() -> None:
    """One stored identity means there is no runner-up to compare with."""
    candidate = SpeakerCandidate("S-0001", "KNOWN", (unit(1, 0),))

    decision = matcher().match(unit(1, 0.03), [candidate], quality=HIGH)

    assert decision.status == "KNOWN"
    assert decision.speaker_id == "S-0001"
    assert decision.margin is None
    assert decision.singleton_match is True


def test_two_identities_produce_a_real_margin_not_a_singleton_claim() -> None:
    candidates = [
        SpeakerCandidate("S-0001", "KNOWN", (unit(1, 0),)),
        SpeakerCandidate("S-0002", "KNOWN", (unit(0, 1),)),
    ]

    decision = matcher().match(unit(1, 0.05), candidates, quality=HIGH)

    assert decision.status == "KNOWN"
    assert decision.singleton_match is False
    assert decision.margin is not None and decision.margin > 0.9


def test_locked_singleton_match_is_also_marked_as_singleton() -> None:
    candidate = SpeakerCandidate("S-0001", "KNOWN", (unit(1, 0),))

    decision = matcher().match(
        unit(1, 0.02),
        [candidate],
        quality=HIGH,
        locked_speaker_id="S-0001",
    )

    assert decision.status == "KNOWN"
    assert decision.margin is None
    assert decision.singleton_match is True


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_scores_are_an_error_not_a_new_speaker(value) -> None:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = value
    candidate = SpeakerCandidate("S-0001", "KNOWN", (vector,))

    decision = matcher().match(unit(1, 0), [candidate], quality=HIGH)

    assert decision.status == "ERROR"
    assert decision.speaker_id is None
    assert decision.similarity is None
