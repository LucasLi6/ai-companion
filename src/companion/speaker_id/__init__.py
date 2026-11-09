"""Optional local speaker identification for the independent ASR demo.

This package is deliberately separate from Ava's Person, Router, Speaker,
Memory and TTS systems. It only turns microphone audio into an anonymous,
structured identity snapshot that a future Router may consume. Nothing here
writes to the application database.

The heavy dependencies (PyTorch, ModelScope, torchaudio) are imported lazily
by :mod:`companion.speaker_id.embedding`, so importing this package or the
plain ASR demo never loads a model.
"""

from __future__ import annotations

from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    PENDING,
    SpeakerResult,
    final_tag,
    pending_result,
    status_tag,
)

__all__ = [
    "AMBIGUOUS",
    "CANDIDATE_MATCH",
    "ERROR",
    "INSUFFICIENT_AUDIO",
    "KNOWN",
    "NEW_CANDIDATE",
    "PENDING",
    "SpeakerResult",
    "final_tag",
    "pending_result",
    "status_tag",
]
