"""Independent Qwen TTS output (Phase A).

The package deliberately has no imports from Router, Speaker, Memory or
Emotion: speech is an output-only side channel for an already completed text
reply, never an input to Ava's thinking.
"""

from companion.tts.manager import SpeechJob, TTSManager, TTSStatus

__all__ = ["SpeechJob", "TTSManager", "TTSStatus"]
