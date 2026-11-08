"""Independent real-time speech recognition demo (V0.9.9.18).

This package is a standalone microphone → Silero VAD → Qwen streaming ASR →
terminal pipeline. It deliberately has no imports from Ava's Speaker, Router,
TTS, Memory, Orchestrator or proactive systems. ``python -m
companion.asr.demo`` is the only intended entry point.
"""

from companion.asr.config import ASRConfig, ASRConfigError, ASRError
from companion.asr.manager import ASRManager, ASRStats

__all__ = [
    "ASRConfig",
    "ASRConfigError",
    "ASRError",
    "ASRManager",
    "ASRStats",
]
