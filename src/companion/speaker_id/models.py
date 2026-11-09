"""Immutable identity snapshots and their terminal representation.

``SpeakerResult`` is the only value that leaves the speaker engine. It is
frozen so the ASR control thread can read the latest snapshot while the
recognition worker publishes a newer one, without either side observing a
half-updated object.

The result deliberately keeps four different notions apart, because collapsing
them is what makes "who is speaking" look more certain than it is:

* ``status``        what the matcher decided for this utterance
* ``reid_quality``  whether this audio is worth re-identifying the speaker by
* ``similarity``    the raw cosine score of the best match
* ``margin``        how far the best match is ahead of the second one

None of these is an accuracy percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Match statuses. These strings are part of the demo contract and appear in
# the terminal output; do not rename them without updating the renderer.
PENDING = "PENDING"
KNOWN = "KNOWN"
CANDIDATE_MATCH = "CANDIDATE_MATCH"
NEW_CANDIDATE = "NEW_CANDIDATE"
AMBIGUOUS = "AMBIGUOUS"
INSUFFICIENT_AUDIO = "INSUFFICIENT_AUDIO"
ERROR = "ERROR"

STATUSES = (
    PENDING,
    KNOWN,
    CANDIDATE_MATCH,
    NEW_CANDIDATE,
    AMBIGUOUS,
    INSUFFICIENT_AUDIO,
    ERROR,
)

# Per-utterance re-identification quality of the audio itself.
QUALITY_HIGH = "HIGH"
QUALITY_MEDIUM = "MEDIUM"
QUALITY_LOW = "LOW"
QUALITY_UNKNOWN = "UNKNOWN"

QUALITY_RANK = {
    QUALITY_UNKNOWN: 0,
    QUALITY_LOW: 1,
    QUALITY_MEDIUM: 2,
    QUALITY_HIGH: 3,
}

MATCHED_STATUSES = (KNOWN, CANDIDATE_MATCH, NEW_CANDIDATE)


@dataclass(frozen=True)
class SpeakerResult:
    """One immutable identity snapshot for one utterance."""

    utterance_id: int

    speaker_id: Optional[str]
    # Always ``None`` in the standalone demo. A future, authorized
    # identity binding is the only thing allowed to fill it; voice matching
    # alone must never grant access to another person's private memory.
    person_id: Optional[str]

    status: str
    reid_quality: str

    similarity: Optional[float]
    margin: Optional[float]

    voiced_seconds: float
    provisional: bool

    model_id: str
    inference_ms: Optional[float]
    completed_at: float


def pending_result(
    utterance_id: int,
    *,
    model_id: str = "",
    voiced_seconds: float = 0.0,
    at: float = 0.0,
) -> SpeakerResult:
    """Snapshot returned while no analysis has completed yet."""
    return SpeakerResult(
        utterance_id=int(utterance_id),
        speaker_id=None,
        person_id=None,
        status=PENDING,
        reid_quality=QUALITY_UNKNOWN,
        similarity=None,
        margin=None,
        voiced_seconds=float(voiced_seconds),
        provisional=True,
        model_id=model_id,
        inference_ms=None,
        completed_at=float(at),
    )


def _status_label(result: SpeakerResult) -> str:
    if result.status == NEW_CANDIDATE:
        return "NEW"
    return result.status


def _display_quality(result: SpeakerResult) -> str:
    if result.reid_quality in QUALITY_RANK:
        return result.reid_quality
    return QUALITY_UNKNOWN


def status_tag(result: SpeakerResult) -> str:
    """The one-line identity state shown under a live partial."""
    if result.status == PENDING:
        return "[ID: PENDING | 语音质量: 评估中]"
    quality = _display_quality(result)
    if result.status in MATCHED_STATUSES and result.speaker_id:
        label = _status_label(result)
        return f"[ID: {result.speaker_id} | {label} | 可复识: {quality}]"
    return f"[ID: UNKNOWN | {_status_label(result)} | 可复识: {quality}]"


def final_tag(result: SpeakerResult) -> str:
    """The identity line printed under a confirmed transcript."""
    if result.status == PENDING:
        # The transcript was confirmed before the voice print finished. Never
        # present that as "UNKNOWN person"; it is simply not decided yet.
        return "[Speaker: PENDING | ReID: 评估中]"
    quality = _display_quality(result)
    if result.status in MATCHED_STATUSES and result.speaker_id:
        speaker = result.speaker_id
        label = _status_label(result)
    else:
        speaker = "UNKNOWN"
        label = _status_label(result)
    parts = [f"Speaker: {speaker}", label, f"ReID: {quality}"]
    if result.similarity is not None:
        parts.append(f"sim: {result.similarity:.2f}")
    return "[" + " | ".join(parts) + "]"


__all__ = [
    "AMBIGUOUS",
    "CANDIDATE_MATCH",
    "ERROR",
    "INSUFFICIENT_AUDIO",
    "KNOWN",
    "MATCHED_STATUSES",
    "NEW_CANDIDATE",
    "PENDING",
    "QUALITY_HIGH",
    "QUALITY_LOW",
    "QUALITY_MEDIUM",
    "QUALITY_RANK",
    "QUALITY_UNKNOWN",
    "STATUSES",
    "SpeakerResult",
    "final_tag",
    "pending_result",
    "status_tag",
]
