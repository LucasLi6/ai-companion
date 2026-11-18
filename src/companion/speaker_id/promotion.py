"""Human-controlled upgrade of a temporary voice cluster to a stable id.

The engine only reports ``KNOWN`` for a profile whose stored status is
``KNOWN``; every automatically created cluster starts as ``TEMP-xxxx`` /
``CANDIDATE``. Promotion is therefore the *only* way a recognised voice can
reach the ``KNOWN`` gate that formal identity resolution requires, and it is a
decision a human makes at the terminal (``/speaker promote``). It never runs
from speech, the Router or an LLM.

Promotion changes voiceprint trust only:

* it renames ``TEMP-xxxx`` to ``S-xxxx``, moves samples/observations and
  leaves the old id as an alias so earlier bindings keep resolving;
* it never creates or moves a person binding;
* it never grants or revokes storage consent.

Because a profile can be promoted only once and the rename is transactional,
the function is safe to call from a slow CLI path; it never guesses when the
store and the caller disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

PROMOTED = "promoted"
ALREADY_KNOWN = "already_known"
UNKNOWN_SPEAKER = "unknown_speaker"
UNSUPPORTED_STATUS = "unsupported_status"
NO_SAMPLES = "no_samples"
PROMOTION_FAILED = "promotion_failed"


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of one promotion request; ``outcome`` is a stable code."""

    outcome: str
    requested_id: str
    previous_id: Optional[str] = None
    speaker_id: Optional[str] = None


def promote_voice_profile(store: Any, speaker_id: str) -> PromotionResult:
    """Promote one temporary profile, or explain why it cannot be.

    ``store`` is a :class:`~companion.speaker_id.store.SpeakerStore`. The call
    is read-only unless the profile is a live ``CANDIDATE`` with at least one
    usable sample, so an empty or already-stable id is never rewritten.
    """
    requested = str(speaker_id or "").strip()
    if not requested:
        return PromotionResult(UNKNOWN_SPEAKER, requested)

    canonical = store.resolve_canonical_speaker_id(requested)
    if canonical is None:
        return PromotionResult(UNKNOWN_SPEAKER, requested)
    profile = store.get_profile(canonical)
    if profile is None:
        return PromotionResult(UNKNOWN_SPEAKER, requested)

    status = str(profile.get("status") or "").upper()
    if status == "KNOWN":
        return PromotionResult(
            ALREADY_KNOWN, requested, canonical, canonical
        )
    if status != "CANDIDATE":
        return PromotionResult(UNSUPPORTED_STATUS, requested, canonical)

    model_id = str(profile.get("model_id") or "")
    embedding_version = str(profile.get("embedding_version") or "")
    if not model_id or not embedding_version:
        return PromotionResult(NO_SAMPLES, requested, canonical)
    candidates, _ = store.load_candidates(
        model_id=model_id, embedding_version=embedding_version
    )
    candidate = next(
        (item for item in candidates if item.speaker_id == canonical), None
    )
    if candidate is None or not candidate.samples:
        # A profile without usable samples has nothing to recognise; calling
        # it KNOWN would create an identity that can never match.
        return PromotionResult(NO_SAMPLES, requested, canonical)

    promoted = store.promote_speaker(canonical)
    if promoted is None:
        # A concurrent promotion/merge moved the id between the checks. Report
        # what the store now says instead of inventing a canonical id.
        current = store.resolve_canonical_speaker_id(requested)
        if current is not None and current != canonical:
            current_profile = store.get_profile(current) or {}
            if str(current_profile.get("status") or "").upper() == "KNOWN":
                return PromotionResult(
                    ALREADY_KNOWN, requested, canonical, current
                )
        return PromotionResult(PROMOTION_FAILED, requested, canonical)

    verified = store.resolve_canonical_speaker_id(requested)
    if verified != promoted:
        return PromotionResult(PROMOTION_FAILED, requested, canonical)
    return PromotionResult(PROMOTED, requested, canonical, promoted)


__all__ = [
    "ALREADY_KNOWN",
    "NO_SAMPLES",
    "PROMOTED",
    "PROMOTION_FAILED",
    "PromotionResult",
    "UNKNOWN_SPEAKER",
    "UNSUPPORTED_STATUS",
    "promote_voice_profile",
]
