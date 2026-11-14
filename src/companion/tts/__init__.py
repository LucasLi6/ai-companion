"""Independent Qwen TTS output (Phase A + Phase B).

The package deliberately has no imports from Router, Speaker, Memory or
Emotion: speech is an output-only side channel for the text reply Ava is
already producing, never an input to Ava's thinking.

Phase A submits one complete reply; Phase B streams display deltas through a
bidirectional session so the first sentence can play before the reply ends.
Both paths share one generation token, one bounded audio queue and one player.
"""

from companion.tts.manager import (
    PlaybackEvent,
    PlaybackTiming,
    SpeechJob,
    TTSManager,
    TTSStatus,
    TTSStream,
)

__all__ = [
    "PlaybackEvent",
    "PlaybackTiming",
    "SpeechJob",
    "TTSManager",
    "TTSStatus",
    "TTSStream",
]
