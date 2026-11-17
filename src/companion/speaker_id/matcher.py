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
  temporary ID — but a locked verdict must still re-pass the same threshold and
  margin checks, otherwise it is reported ``AMBIGUOUS``.

V2 adds a second, optional way to score a person without changing any
threshold: ``variant_centroid`` first groups the person's samples into
deterministic variants and compares the query with each variant centroid.
``legacy_max_sample`` (the default) keeps the old max-over-samples score.
``shadow`` mode computes both and returns the legacy verdict, recording the
comparison in :meth:`SpeakerMatcher.shadow_stats` so the change can be
evaluated without touching recognition behaviour.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from companion.speaker_id.config import SpeakerConfig
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    QUALITY_RANK,
)
from companion.speaker_id.quality import VoiceQuality
from companion.speaker_id.voice_profiles import (
    STRATEGY_LEGACY_MAX_SAMPLE,
    STRATEGY_VARIANT_CENTROID,
    VoiceProfileIndex,
    cosine,
    score_profile,
)

_MAX_SHADOW_EXAMPLES = 32


@dataclass(frozen=True)
class SpeakerCandidate:
    """One stored identity with all of its enrolled samples.

    ``samples`` is optional so existing callers that only have vectors keep
    working; when present it carries the ids/timestamps needed to cache the
    variant structure and to diagnose which sample won.
    """

    speaker_id: str
    status: str  # "KNOWN" or "CANDIDATE"
    vectors: tuple
    samples: tuple = ()


@dataclass(frozen=True)
class MatchDecision:
    status: str
    speaker_id: Optional[str]
    similarity: Optional[float]
    margin: Optional[float]
    # True only when no second identity existed, so ``margin`` is None
    # because there is nothing to compare — not because the value was lost.
    # Downstream identity code must not reject this legitimate case, while a
    # ``margin=None`` without this flag stays malformed evidence.
    singleton_match: bool = False


def _cosine(left, right) -> float:
    # Both vectors are L2-normalized when they enter the store, so the dot
    # product is the cosine similarity. Clamp for numerical noise.
    return cosine(left, right)


class SpeakerMatcher:
    def __init__(self, config: SpeakerConfig) -> None:
        self.config = config
        self._profiles = VoiceProfileIndex(
            merge_threshold=config.variant_merge_threshold,
            max_variants=config.variant_max_count,
        )
        self._shadow_lock = threading.Lock()
        self._shadow_agree = 0
        self._shadow_disagree = 0
        self._shadow_examples: "deque[dict]" = deque(
            maxlen=_MAX_SHADOW_EXAMPLES
        )
        self.logger = logging.getLogger("companion.speaker")

    # ------------------------------------------------------------- strategy
    @property
    def active_strategy(self) -> str:
        if self.config.match_strategy == STRATEGY_VARIANT_CENTROID:
            return STRATEGY_VARIANT_CENTROID
        return STRATEGY_LEGACY_MAX_SAMPLE

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
        if self.config.profile_mode == "shadow":
            legacy = self._match_with_strategy(
                embedding,
                candidates,
                quality=quality,
                locked_speaker_id=locked_speaker_id,
                strategy=STRATEGY_LEGACY_MAX_SAMPLE,
            )
            variant = self._match_with_strategy(
                embedding,
                candidates,
                quality=quality,
                locked_speaker_id=locked_speaker_id,
                strategy=STRATEGY_VARIANT_CENTROID,
            )
            self._record_shadow(legacy, variant)
            return legacy
        return self._match_with_strategy(
            embedding,
            candidates,
            quality=quality,
            locked_speaker_id=locked_speaker_id,
            strategy=self.active_strategy,
        )

    def _match_with_strategy(
        self,
        embedding,
        candidates: Sequence[SpeakerCandidate],
        *,
        quality: VoiceQuality,
        locked_speaker_id: Optional[str],
        strategy: str,
    ) -> MatchDecision:
        if not quality.usable:
            return MatchDecision(INSUFFICIENT_AUDIO, None, None, None)
        if locked_speaker_id is not None:
            return self._match_locked(
                embedding,
                candidates,
                locked_speaker_id,
                quality,
                strategy=strategy,
            )
        if not candidates:
            return self._new_or_insufficient(quality, similarity=None)

        scored = sorted(
            (
                (self._score(embedding, candidate, strategy=strategy), candidate)
                for candidate in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not all(math.isfinite(score) for score, _ in scored):
            # NaN/Infinity would make every threshold comparison silently
            # false and could turn a broken vector into a "new speaker".
            return MatchDecision(ERROR, None, None, None)
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else None
        margin = (
            best_score - second_score if second_score is not None else None
        )
        singleton = second_score is None

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
                    KNOWN,
                    best.speaker_id,
                    best_score,
                    margin,
                    singleton_match=singleton,
                )
            return MatchDecision(
                CANDIDATE_MATCH,
                best.speaker_id,
                best_score,
                margin,
                singleton_match=singleton,
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
        quality: VoiceQuality,
        *,
        strategy: Optional[str] = None,
    ) -> MatchDecision:
        """Re-evaluate a locked identity with the same proof as a fresh match.

        A lock only keeps the *candidate* from changing mid-utterance; it is
        never proof that the current window belongs to that identity. The
        locked score must still clear the candidate threshold, and a competing
        identity must not come too close. Otherwise the honest answer is
        ``AMBIGUOUS`` — not ``KNOWN``.
        """
        selected = strategy or self.active_strategy
        locked = next(
            (c for c in candidates if c.speaker_id == locked_speaker_id),
            None,
        )
        if locked is None:
            # The identity vanished (deleted/merged) mid-utterance. Evaluate
            # afresh with the *real* quality of this window; a fabricated LOW
            # quality would silently turn every later window into a new or
            # unknown identity.
            return self._match_with_strategy(
                embedding,
                candidates,
                quality=quality,
                locked_speaker_id=None,
                strategy=selected,
            )
        locked_score = self._score(embedding, locked, strategy=selected)
        others = [
            self._score(embedding, candidate, strategy=selected)
            for candidate in candidates
            if candidate.speaker_id != locked_speaker_id
        ]
        if not math.isfinite(locked_score) or not all(
            math.isfinite(score) for score in others
        ):
            return MatchDecision(ERROR, None, None, None)
        other_best = max(others) if others else None
        margin = (
            locked_score - other_best if other_best is not None else None
        )

        cfg = self.config
        if locked_score < cfg.candidate_match_threshold:
            return MatchDecision(AMBIGUOUS, None, locked_score, margin)
        if margin is not None and margin < cfg.minimum_match_margin:
            return MatchDecision(AMBIGUOUS, None, locked_score, margin)
        status = (
            KNOWN
            if locked.status == "KNOWN"
            and locked_score >= cfg.known_match_threshold
            else CANDIDATE_MATCH
        )
        return MatchDecision(
            status,
            locked.speaker_id,
            locked_score,
            margin,
            singleton_match=other_best is None,
        )

    def _score(
        self,
        embedding,
        candidate: SpeakerCandidate,
        *,
        strategy: Optional[str] = None,
    ) -> float:
        selected = strategy or self.active_strategy
        if not candidate.vectors and not candidate.samples:
            return -1.0
        if (
            selected == STRATEGY_VARIANT_CENTROID
            and candidate.samples
        ):
            profile = self._profiles.profile_for(
                candidate.speaker_id,
                candidate.samples,
                status=candidate.status,
            )
            return score_profile(
                embedding,
                profile,
                strategy=selected,
                singleton_penalty=self.config.singleton_penalty,
            ).effective_similarity
        if not candidate.vectors:
            vectors = tuple(sample.vector for sample in candidate.samples)
            if not vectors:
                return -1.0
            return max(_cosine(embedding, vector) for vector in vectors)
        return max(_cosine(embedding, vector) for vector in candidate.vectors)

    # ------------------------------------------------------------- shadow
    def _record_shadow(
        self, legacy: MatchDecision, variant: MatchDecision
    ) -> None:
        agrees = (
            legacy.status == variant.status
            and legacy.speaker_id == variant.speaker_id
        )
        if not agrees:
            self.logger.debug(
                "speaker_shadow_disagree legacy=%s/%s variant=%s/%s",
                legacy.status,
                legacy.speaker_id,
                variant.status,
                variant.speaker_id,
            )
        with self._shadow_lock:
            if agrees:
                self._shadow_agree += 1
            else:
                self._shadow_disagree += 1
            if not agrees:
                self._shadow_examples.append(
                    {
                        "legacy_status": legacy.status,
                        "legacy_speaker_id": legacy.speaker_id,
                        "legacy_similarity": legacy.similarity,
                        "legacy_margin": legacy.margin,
                        "variant_status": variant.status,
                        "variant_speaker_id": variant.speaker_id,
                        "variant_similarity": variant.similarity,
                        "variant_margin": variant.margin,
                    }
                )

    def shadow_stats(self) -> dict:
        with self._shadow_lock:
            return {
                "mode": self.config.profile_mode,
                "agree": self._shadow_agree,
                "disagree": self._shadow_disagree,
                "examples": list(self._shadow_examples),
            }

    def profile_stats(self) -> dict:
        return self._profiles.stats()


__all__ = [
    "MatchDecision",
    "SpeakerCandidate",
    "SpeakerMatcher",
]
