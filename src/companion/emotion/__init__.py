"""Persistent emotion framework (V0.3).

Computational affect and expression only — never a claim about a human's real
emotional state. See :mod:`companion.emotion.models` for the four-layer model.
"""

from __future__ import annotations

from companion.emotion.appraisal import AppraisalEngine
from companion.emotion.dynamics import decay, decay_factor, label_for
from companion.emotion.models import (
    AppraisalResult,
    AppraisalRule,
    EmotionProfile,
    EmotionState,
    SpeechStylePlan,
    VoiceProfile,
)
from companion.emotion.repository import EmotionRepository
from companion.emotion.service import EmotionService
from companion.emotion.speech import SpeechPlanner

__all__ = [
    "AppraisalEngine",
    "AppraisalResult",
    "AppraisalRule",
    "EmotionProfile",
    "EmotionRepository",
    "EmotionService",
    "EmotionState",
    "SpeechPlanner",
    "SpeechStylePlan",
    "VoiceProfile",
    "decay",
    "decay_factor",
    "label_for",
]
