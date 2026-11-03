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

# A small, stable mapping from categorical emotion to a delivery instruction
# and a rate multiplier. These are presentation hints only.
_STYLE = {
    "NEUTRAL": ("平稳、自然地说，语速适中。", 1.0),
    "JOY": ("语气轻快、温暖，带一点笑意，句子可以稍短。", 1.05),
    "EXCITEMENT": ("语气更兴奋、有活力，适当提高语速，但不要夸张。", 1.15),
    "RELIEF": ("语气舒缓、放松，像松了一口气。", 0.95),
    "CURIOSITY": ("语气好奇、温和，多用开放式提问。", 1.0),
    "CONCERN": ("语气关切、平稳，放慢语速，措辞更谨慎。", 0.9),
    "SADNESS": ("语气低沉、温和，语速放慢，给予陪伴感。", 0.85),
    "FRUSTRATION": ("语气克制、认真，语气平稳但坚定。", 0.95),
    "SERIOUS": ("语气严肃、郑重，语速放慢，避免玩笑。", 0.9),
}


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
        instruction, rate = _STYLE.get(state.emotion, _STYLE["NEUTRAL"])
        base_rate = 1.0
        voice_profile_id = None
        if voice_profile:
            base_rate = float(voice_profile.get("speaking_rate") or 1.0)
            voice_profile_id = voice_profile.get("voice_profile_id")

        # Emotion modulates the base rate, bounded to a safe, natural range.
        speaking_rate = round(max(0.6, min(1.6, base_rate * rate)), 4)

        return SpeechStylePlan(
            text=text,
            emotion=state.emotion,
            intensity=round(state.intensity, 4),
            speaking_rate=speaking_rate,
            delivery_instruction=instruction,
            voice_profile_id=voice_profile_id,
        )

    def instruction_for(self, emotion: str) -> str:
        instruction, _rate = _STYLE.get(emotion, _STYLE["NEUTRAL"])
        return instruction
