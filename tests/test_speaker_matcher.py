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
