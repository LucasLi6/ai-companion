"""Audio suitability for voice-print matching.

The VAD already tells us a region contains speech, but a voice print is only
useful when enough *voiced* audio is present and the recording is neither
clipped nor mostly silence. This module measures that cheaply on 20 ms frames
with the standard library + NumPy; it never claims to separate overlapping
speakers.

V0.9.9.24 splits the decision into three explicit steps so that "too quiet"
is an explained result rather than a silent path to a new identity:

1. ``inspect_raw_audio`` records clipping, peak, frame RMS and the noise floor
   of the original PCM;
2. ``prepare_speaker_audio`` may apply a *bounded* gain for the voice-print
   branch only (never for ASR), and only when the raw recording is not
   severely clipped;
3. ``inspect_speaker_audio`` decides usability/quality from the raw facts plus
   the prepared waveform, always with a stable reason code.

Gain preprocessing is disabled by default: it changes the waveform the model
sees and therefore its embeddings, so enabling it changes ``model_key`` and
old/new vectors are never compared. The default path preserves compatibility
with existing voice prints.
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

# Raw clipping that no gain stage may hide. Kept much stricter than the
# matching gate: a severely overloaded recording is never rescuable.
_SEVERE_CLIPPING_RATIO = 0.10

# Quality bands. HIGH requires a comfortably long, clean sample; MEDIUM is the
# minimum worth enrolling; LOW audio may still *match* an existing identity
# when the score is strong, but never creates one.
_HIGH_VOICED_SECONDS = 2.0
_MEDIUM_VOICED_SECONDS = 1.2
_HIGH_SILENCE_RATIO = 0.35
_HIGH_CLIPPING_RATIO = 0.005

# Gain stage: aim for this frame RMS after gain, but never exceed the caller's
# decibel cap and never amplify a recording whose noise floor would rise above
# the speech. Values are conservative on purpose.
_TARGET_VOICED_RMS = 0.05


@dataclass(frozen=True)
class RawAudioQuality:
    """Measured facts about the original PCM; no decision is made yet."""

    frames: int
    voiced_frames: int
    voiced_seconds: float
    silence_ratio: float
    peak: float
    clipping_ratio: float
    voiced_rms: float
    noise_rms: float
    severe_clipping: bool
    very_quiet: bool
    reason: Optional[str] = None


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


def inspect_raw_audio(
    pcm: bytes,
    *,
    sample_rate: int = 16000,
    min_voiced_seconds: float = 0.9,
) -> RawAudioQuality:
    """Measure the original recording without deciding whether to use it."""
    import numpy as np

    samples = _as_int16(pcm)
    if samples.size == 0:
        return RawAudioQuality(
            0, 0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, True, "没有音频"
        )

    audio = samples.astype(np.float32) / 32768.0
    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    frame_count = audio.size // frame
    if frame_count == 0:
        return RawAudioQuality(
            0, 0, 0.0, 1.0, float(np.max(np.abs(audio))), 0.0,
            0.0, 0.0, False, True, "没有音频",
        )

    frames = audio[: frame_count * frame].reshape(frame_count, frame)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    voiced_mask = frame_rms >= _MIN_FRAME_RMS
    voiced_frames = int(voiced_mask.sum())
    voiced_seconds = float(voiced_frames) * (frame / sample_rate)
    silence_ratio = float(1.0 - voiced_mask.mean())

    peak = float(np.max(np.abs(audio)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.985))

    if voiced_frames:
        voiced_rms = float(np.mean(frame_rms[voiced_mask]))
    else:
        voiced_rms = 0.0
    if voiced_frames < frame_count:
        noise_rms = float(np.mean(frame_rms[~voiced_mask]))
    else:
        noise_rms = 0.0

    severe = clipping_ratio > _SEVERE_CLIPPING_RATIO
    very_quiet = peak < _MIN_PEAK or (
        voiced_frames > 0 and voiced_rms < _MIN_FRAME_RMS * 1.5
    )

    reason: Optional[str] = None
    if voiced_seconds < min_voiced_seconds:
        reason = "有效语音时长不足"
    elif severe:
        reason = "声音明显削波"
    elif very_quiet:
        reason = "音量过低"
    elif silence_ratio > _MAX_SILENCE_RATIO:
        reason = "静音或噪声占比过高"

    return RawAudioQuality(
        frames=frame_count,
        voiced_frames=voiced_frames,
        voiced_seconds=voiced_seconds,
        silence_ratio=silence_ratio,
        peak=peak,
        clipping_ratio=clipping_ratio,
        voiced_rms=voiced_rms,
        noise_rms=noise_rms,
        severe_clipping=severe,
        very_quiet=very_quiet,
        reason=reason,
    )


def prepare_speaker_audio(
    pcm: bytes,
    raw: RawAudioQuality,
    *,
    enable_gain: bool = False,
    max_gain_db: float = 12.0,
) -> bytes:
    """Return the PCM that should be sent to the voice-print model.

    With ``enable_gain=False`` (the default) the bytes are returned untouched,
    preserving the embedding version used by every existing voice print.
    Bounded gain is never applied to severely clipped audio, and the noise
    floor is lifted with the signal — a whisper is not made to look like a
    shout.
    """
    if not enable_gain or not pcm or raw.severe_clipping:
        return pcm
    if raw.voiced_rms <= 0.0:
        return pcm
    gain = _TARGET_VOICED_RMS / raw.voiced_rms
    if gain <= 1.0:
        return pcm
    limit = 10.0 ** (max(0.0, float(max_gain_db)) / 20.0)
    gain = min(gain, limit)
    if gain <= 1.0:
        return pcm
    import numpy as np

    samples = _as_int16(pcm).astype(np.float32)
    scaled = np.clip(samples * gain, -32768.0, 32767.0)
    return scaled.astype("<i2").tobytes()


def inspect_speaker_audio(
    prepared: bytes,
    raw: RawAudioQuality,
    *,
    sample_rate: int = 16000,
    min_voiced_seconds: float = 0.9,
    gain_applied: bool = False,
) -> VoiceQuality:
    """Decide usability/quality with an explicit reason for every failure."""
    if raw.frames == 0:
        return VoiceQuality(False, 0.0, QUALITY_LOW, raw.reason or "没有音频")
    if raw.voiced_seconds < min_voiced_seconds:
        return VoiceQuality(
            False, raw.voiced_seconds, QUALITY_LOW, "有效语音时长不足"
        )
    if raw.severe_clipping:
        # The raw recording is overloaded; gain cannot turn it into a clean
        # sample and lowering the amplitude must not disguise that.
        return VoiceQuality(
            False, raw.voiced_seconds, QUALITY_LOW, "声音明显削波"
        )
    if raw.very_quiet and not gain_applied:
        # Compatibility path: keep the old vectors untouched and state why.
        return VoiceQuality(
            False, raw.voiced_seconds, QUALITY_LOW, "音量过低"
        )

    prepared_facts = inspect_raw_audio(
        prepared,
        sample_rate=sample_rate,
        min_voiced_seconds=min_voiced_seconds,
    )
    if prepared_facts.severe_clipping or (
        prepared_facts.clipping_ratio > _MAX_CLIPPING_RATIO
    ):
        return VoiceQuality(
            False, raw.voiced_seconds, QUALITY_LOW, "声音明显削波"
        )
    if prepared_facts.silence_ratio > _MAX_SILENCE_RATIO:
        return VoiceQuality(
            False, raw.voiced_seconds, QUALITY_LOW, "静音或噪声占比过高"
        )

    voiced_seconds = prepared_facts.voiced_seconds
    if (
        voiced_seconds >= _HIGH_VOICED_SECONDS
        and prepared_facts.silence_ratio <= _HIGH_SILENCE_RATIO
        and prepared_facts.clipping_ratio <= _HIGH_CLIPPING_RATIO
    ):
        quality = QUALITY_HIGH
    elif voiced_seconds >= _MEDIUM_VOICED_SECONDS:
        quality = QUALITY_MEDIUM
    else:
        quality = QUALITY_LOW
    return VoiceQuality(True, voiced_seconds, quality, None)


def evaluate_voice_quality(
    pcm: bytes,
    *,
    sample_rate: int = 16000,
    min_voiced_seconds: float = 0.9,
    enable_gain: bool = False,
    max_gain_db: float = 12.0,
) -> VoiceQuality:
    """Compatibility wrapper: inspect → prepare → inspect."""
    raw = inspect_raw_audio(
        pcm, sample_rate=sample_rate, min_voiced_seconds=min_voiced_seconds
    )
    prepared = prepare_speaker_audio(
        pcm, raw, enable_gain=enable_gain, max_gain_db=max_gain_db
    )
    return inspect_speaker_audio(
        prepared,
        raw,
        sample_rate=sample_rate,
        min_voiced_seconds=min_voiced_seconds,
        gain_applied=prepared is not pcm,
    )


__all__ = [
    "FRAME_SECONDS",
    "RawAudioQuality",
    "VoiceQuality",
    "evaluate_voice_quality",
    "inspect_raw_audio",
    "inspect_speaker_audio",
    "prepare_speaker_audio",
]
