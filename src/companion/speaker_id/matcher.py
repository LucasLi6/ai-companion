"""Open-set identity matching with hysteresis and an uncertainty band.

Matching never uses a single fixed threshold:

* a strong score that is clearly ahead of the runner-up reuses the identity;
* a score inside the uncertainty band (``new_candidate_threshold`` ..
  ``candidate_match_threshold``) is reported ``AMBIGUOUS`` instead of
  inventing a person;
* a new temporary identity is only created when *every* existing identity
  scores below ``new_candidate_threshold`` and the audio quality is at least
  ``minimum_enrollment_quality``;
* once an utterance has been assigned an identity, later verdicts for the same
  utterance are locked to it, so one noisy window cannot spawn a second
  temporary ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from companion.speaker_id.config import SpeakerConfig
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    QUALITY_RANK,
)
from companion.speaker_id.quality import VoiceQuality


@dataclass(frozen=True)
class SpeakerCandidate:
    """One stored identity with all of its enrolled samples."""

    speaker_id: str
    status: str  # "KNOWN" or "CANDIDATE"
    vectors: tuple


@dataclass(frozen=True)
class MatchDecision:
    status: str
    speaker_id: Optional[str]
    similarity: Optional[float]
    margin: Optional[float]


def _cosine(left, right) -> float:
    # Both vectors are L2-normalized when they enter the store, so the dot
    # product is the cosine similarity. Clamp for numerical noise.
    value = float(left @ right)
    return max(-1.0, min(1.0, value))


class SpeakerMatcher:
    def __init__(self, config: SpeakerConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ public
    def match(
        self,
        embedding,
        candidates: Sequence[SpeakerCandidate],
        *,
        quality: VoiceQuality,
        locked_speaker_id: Optional[str] = None,
    ) -> MatchDecision:
        if not quality.usable:
            return MatchDecision(INSUFFICIENT_AUDIO, None, None, None)
        if locked_speaker_id is not None:
            return self._match_locked(
                embedding, candidates, locked_speaker_id
            )
        if not candidates:
            return self._new_or_insufficient(quality, similarity=None)

        scored = sorted(
            (
                (self._score(embedding, candidate), candidate)
                for candidate in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else None
        margin = (
            best_score - second_score if second_score is not None else None
        )

        cfg = self.config
        if best_score >= cfg.candidate_match_threshold:
            if margin is not None and margin < cfg.minimum_match_margin:
                return MatchDecision(
                    AMBIGUOUS, None, best_score, margin
                )
            if (
                best_score >= cfg.known_match_threshold
                and best.status == "KNOWN"
            ):
                return MatchDecision(
                    KNOWN, best.speaker_id, best_score, margin
                )
            return MatchDecision(
                CANDIDATE_MATCH, best.speaker_id, best_score, margin
            )
        if best_score >= cfg.new_candidate_threshold:
            # Close to an existing identity but not close enough: this is
            # exactly the case where a second temporary ID must not appear.
            return MatchDecision(AMBIGUOUS, None, best_score, margin)
        return self._new_or_insufficient(quality, similarity=best_score)

    # ----------------------------------------------------------------- helpers
    def _new_or_insufficient(
        self, quality: VoiceQuality, *, similarity: Optional[float]
    ) -> MatchDecision:
        minimum = self.config.minimum_enrollment_quality
        required = QUALITY_RANK.get(minimum, 2)
        if quality.rank < required:
            return MatchDecision(
                INSUFFICIENT_AUDIO, None, similarity, None
            )
        return MatchDecision(NEW_CANDIDATE, None, similarity, None)

    def _match_locked(
        self,
        embedding,
        candidates: Sequence[SpeakerCandidate],
        locked_speaker_id: str,
    ) -> MatchDecision:
        locked = next(
            (c for c in candidates if c.speaker_id == locked_speaker_id),
            None,
        )
        if locked is None:
            # The identity vanished (deleted/merged) mid-utterance. Fall back
            # to one fresh evaluation rather than keeping a dead reference.
            return self.match(embedding, candidates, quality=_usable())
        locked_score = self._score(embedding, locked)
        others = [
            (self._score(embedding, candidate), candidate)
            for candidate in candidates
            if candidate.speaker_id != locked_speaker_id
        ]
        other_best = max((score for score, _ in others), default=None)
        margin = (
            locked_score - other_best if other_best is not None else None
        )
        status = KNOWN if locked.status == "KNOWN" else CANDIDATE_MATCH
        return MatchDecision(status, locked.speaker_id, locked_score, margin)

    @staticmethod
    def _score(embedding, candidate: SpeakerCandidate) -> float:
        if not candidate.vectors:
            return -1.0
        return max(_cosine(embedding, vector) for vector in candidate.vectors)


def _usable() -> VoiceQuality:
    # Placeholder used only when a locked identity disappeared; the caller
    # already passed the quality gate for the current utterance.
    return VoiceQuality(True, 0.0, "LOW", None)


__all__ = [
    "MatchDecision",
    "SpeakerCandidate",
    "SpeakerMatcher",
]
