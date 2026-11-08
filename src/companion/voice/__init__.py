"""Voice input bridge for the formal conversation surface.

``bridge`` moves completed ASR turns from ASR threads to the asyncio loop
without blocking; ``coordinator`` owns the half-duplex mic state machine.
The ASR and TTS packages stay independent: only this package imports both.
"""

from companion.voice.bridge import (
    BridgeFailure,
    TurnClosed,
    VoiceBridge,
)
from companion.voice.coordinator import (
    VoiceCoordinator,
    VoiceState,
    VoiceStats,
)

__all__ = [
    "BridgeFailure",
    "TurnClosed",
    "VoiceBridge",
    "VoiceCoordinator",
    "VoiceState",
    "VoiceStats",
]
