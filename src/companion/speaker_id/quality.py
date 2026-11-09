"""Audio suitability for voice-print matching.

The VAD already tells us a region contains speech, but a voice print is only
useful when enough *voiced* audio is present and the recording is neither
clipped nor mostly silence. This module measures that cheaply on 20 ms frames
with the standard library + NumPy; it never claims to separate overlapping
speakers.

Conservative policy: audio that is merely loud but clipped, too short, too
quiet or too sparse is reported ``LOW``/unusable, and the matcher will not use
it to create a new identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from companion.speaker_id.models import (
    QUALITY_HIGH,
    QUALITY_LOW,
    QUALITY_MEDIUM,
)

FRAME_SECONDS = 0.02

# Absolute frame floor (~ -48 dBFS) so a quiet recording is not treated as
# speech merely because it has a non-zero peak.
_MIN_FRAME_RMS = 0.004
_MIN_PEAK = 0.02
_MAX_CLIPPING_RATIO = 0.02
_MAX_SILENCE_RATIO = 0.80

# Quality bands. HIGH requires a comfortably long, clean sample; MEDIUM is the
# minimum worth enrolling; LOW audio may still *match* an existing identity
# when the score is strong, but never creates one.
_HIGH_VOICED_SECONDS = 2.0
_MEDIUM_VOICED_SECONDS = 1.2
_HIGH_SILENCE_RATIO = 0.35
_HIGH_CLIPPING_RATIO = 0.005


@dataclass(frozen=True)
class VoiceQuality:
    usable: bool
    voiced_seconds: float
    quality: str
    reason: Optional[str] = None

    @property
    def rank(self) -> int:
        return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(self.quality, 0)


def _as_int16(pcm: bytes):
    import numpy as np

    usable = len(pcm) - (len(pcm) % 2)
    return np.frombuffer(pcm[:usable], dtype="<i2")


def evaluate_voice_quality(
    pcm: bytes,
    *,
    sample_rate: int = 16000,
    min_voiced_seconds: float = 0.9,
) -> VoiceQuality:
    """Measure how suitable ``pcm`` (mono 16-bit) is for a voice print."""
    import numpy as np

    samples = _as_int16(pcm)
    if samples.size == 0:
        return VoiceQuality(False, 0.0, QUALITY_LOW, "没有音频")

    audio = samples.astype(np.float32) / 32768.0
    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    frame_count = audio.size // frame
    if frame_count == 0:
        voiced_seconds = 0.0
        silence_ratio = 1.0
    else:
        frames = audio[: frame_count * frame].reshape(frame_count, frame)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        voiced = frame_rms >= _MIN_FRAME_RMS
        voiced_seconds = float(voiced.sum()) * (frame / sample_rate)
        silence_ratio = float(1.0 - voiced.mean())

    peak = float(np.max(np.abs(audio)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.985))

    if voiced_seconds < min_voiced_seconds:
        return VoiceQuality(
            False, voiced_seconds, QUALITY_LOW, "有效语音时长不足"
        )
    if peak < _MIN_PEAK:
        return VoiceQuality(False, voiced_seconds, QUALITY_LOW, "音量过低")
    if clipping_ratio > _MAX_CLIPPING_RATIO:
        # Hard clipping usually means closeness/overload, and often overlap;
        # never enrol such audio and never claim the speakers were separated.
        return VoiceQuality(
            False, voiced_seconds, QUALITY_LOW, "声音明显削波"
        )
    if silence_ratio > _MAX_SILENCE_RATIO:
        return VoiceQuality(
            False, voiced_seconds, QUALITY_LOW, "静音或噪声占比过高"
        )

    if (
        voiced_seconds >= _HIGH_VOICED_SECONDS
        and silence_ratio <= _HIGH_SILENCE_RATIO
        and clipping_ratio <= _HIGH_CLIPPING_RATIO
    ):
        quality = QUALITY_HIGH
    elif voiced_seconds >= _MEDIUM_VOICED_SECONDS:
        quality = QUALITY_MEDIUM
    else:
        quality = QUALITY_LOW
    return VoiceQuality(True, voiced_seconds, quality, None)


__all__ = ["FRAME_SECONDS", "VoiceQuality", "evaluate_voice_quality"]
