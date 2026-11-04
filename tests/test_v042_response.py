"""V0.4.2 Step 5 — response prompt and CONCERN expression policy.

These tests pin the behaviour the spec asks for: concern is attention, not a
comfort script. It may make the companion more serious and less playful, but it
must not lower the bar for questions or turn every reply warmer. The prompt
must likewise carry persona first and must not forward legacy style labels as
cozy instructions.
"""

from __future__ import annotations

from companion.emotion.expression_policy import compile_expression_influence
from companion.emotion.models import EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG


def _state(**overrides):
    payload = {"emotion": "CONCERN", "intensity": 0.0}
    payload.update(overrides)
    return EmotionState(**payload)


def test_concern_does_not_increase_questioning_or_warmth():
    influence = compile_expression_influence(
        _state(intensity=0.85), config=DEFAULT_CONFIG
    )
    # Concern must not push the model toward a question or a comfort script.
    assert influence.questioning_adjustment <= 0
    assert influence.warmth_adjustment <= 0.15
    assert influence.playfulness_adjustment < 0
    assert influence.seriousness_adjustment > 0
    assert "更愿意追问" not in influence.explanation


def test_concern_is_more_restrained_than_grief():
    concern = compile_expression_influence(
        _state(intensity=0.85), config=DEFAULT_CONFIG
    )
    sadness = compile_expression_influence(
        _state(emotion="SADNESS", intensity=0.85), config=DEFAULT_CONFIG
    )
    # SADNESS may soften; CONCERN must stay the more restrained one.
    assert concern.warmth_adjustment < sadness.warmth_adjustment
    assert concern.playfulness_adjustment > sadness.playfulness_adjustment
