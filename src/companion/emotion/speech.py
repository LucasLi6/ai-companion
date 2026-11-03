"""Emotion-aware speech planning (provider-neutral, no real TTS).

The planner turns the current :class:`EmotionState` into a
:class:`SpeechStylePlan`: concrete delivery instructions (speaking rate,
delivery note, voice profile) that shape *how* a reply sounds. It deliberately
cannot change *whether* a reply is sent — permission, privacy, silence, tools
and facts are owned by the policy layer and the orchestrator, never by emotion.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

from companion.emotion.models import EmotionState, SpeechStylePlan

# Delivery instructions by emotion. The rate here is the *target multiplier at
# full intensity*; the plan interpolates from neutral toward it by the current
# intensity, so a weak emotion barely changes the voice.
_STYLE = {
    "NEUTRAL": ("平稳、自然地说，语速适中。", 1.0),
    "JOY": ("语气轻快、温暖，带一点笑意，句子可以稍短。", 1.10),
    "EXCITEMENT": ("语气更兴奋、有活力，适当提高语速，但不要夸张。", 1.20),
    "RELIEF": ("语气舒缓、放松，像松了一口气。", 0.90),
    "CURIOSITY": ("语气好奇、温和，多用开放式提问。", 1.02),
    "CONCERN": ("语气关切、平稳，放慢语速，措辞更谨慎。", 0.80),
    "SADNESS": ("语气低沉、温和，语速放慢，给予陪伴感。", 0.70),
    "FRUSTRATION": ("语气克制、认真，语气平稳但坚定。", 0.90),
    "SERIOUS": ("语气严肃、郑重，语速放慢，避免玩笑。", 0.80),
}

_NEUTRAL_INSTRUCTION = "平稳、自然地说，语速适中。"


def _instruction_for(emotion: str, factor: float) -> str:
    """Soften delivery guidance at low intensity; stay natural at high.

    The text never asserts an inner feeling in the first person — it describes
    how to speak, not what the companion "is" feeling.
    """
    full, _rate = _STYLE.get(emotion, _STYLE["NEUTRAL"])
    if factor < 0.2 or emotion == "NEUTRAL":
        return _NEUTRAL_INSTRUCTION
    if factor < 0.6:
        return "语气自然，可以带出一点细微的情绪变化，不要夸张。"
    return full


class SpeechPlanner:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("companion.emotion.speech")

    def plan(
        self,
        *,
        state: EmotionState,
        voice_profile: Optional[Mapping] = None,
        text: str = "",
    ) -> SpeechStylePlan:
        """Build a delivery plan from the (already decayed) emotional state.

        ``voice_profile`` may supply a base speaking rate and provider-neutral
        ``voice_profile_id``; emotion then modulates the rate within bounds.
        """
        _instruction, target_rate = _STYLE.get(state.emotion, _STYLE["NEUTRAL"])
        base_rate = 1.0
        voice_profile_id = None
        if voice_profile:
            base_rate = float(voice_profile.get("speaking_rate") or 1.0)
            voice_profile_id = voice_profile.get("voice_profile_id")

        # Intensity scales the distance from a natural rate; a weak emotion is
        # nearly inaudible in the delivery, a strong one reaches the target.
        factor = max(0.0, min(1.0, float(state.intensity)))
        rate_multiplier = 1.0 + (target_rate - 1.0) * factor
        speaking_rate = round(
            max(0.6, min(1.6, base_rate * rate_multiplier)), 4
        )

        return SpeechStylePlan(
            text=text,
            emotion=state.emotion,
            intensity=round(state.intensity, 4),
            speaking_rate=speaking_rate,
            delivery_instruction=_instruction_for(state.emotion, factor),
            voice_profile_id=voice_profile_id,
        )

    def instruction_for(self, emotion: str, intensity: float = 1.0) -> str:
        return _instruction_for(emotion, max(0.0, min(1.0, float(intensity))))
