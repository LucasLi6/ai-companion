"""Offline tests for voice-print audio quality gating."""

from __future__ import annotations

import numpy as np

from companion.speaker_id.quality import (
    evaluate_voice_quality,
    inspect_raw_audio,
    inspect_speaker_audio,
    prepare_speaker_audio,
)


def tone(seconds: float, *, amplitude: float = 0.25, sample_rate: int = 16000) -> bytes:
    n = int(seconds * sample_rate)
    samples = amplitude * np.sin(
        2 * np.pi * 220 * np.arange(n, dtype=np.float32) / sample_rate
    )
    return (samples * 32767).astype("<i2").tobytes()


def noise(seconds: float, *, amplitude: float = 0.1, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000)
    samples = rng.normal(0.0, amplitude, n)
    return np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes()


def test_long_clean_tone_is_high_quality() -> None:
    quality = evaluate_voice_quality(tone(2.5))
    assert quality.usable is True
    assert quality.quality == "HIGH"
    assert quality.reason is None
    assert quality.voiced_seconds >= 2.0


def test_one_and_a_half_seconds_is_medium() -> None:
    quality = evaluate_voice_quality(tone(1.5))
    assert quality.usable is True
    assert quality.quality == "MEDIUM"


def test_one_second_is_low_but_usable() -> None:
    quality = evaluate_voice_quality(tone(1.0))
    assert quality.usable is True
    assert quality.quality == "LOW"


def test_short_utterance_is_not_usable() -> None:
    quality = evaluate_voice_quality(tone(0.4))
    assert quality.usable is False
    assert quality.reason == "有效语音时长不足"
    assert quality.voiced_seconds < 0.9


def test_quiet_audio_is_not_usable() -> None:
    # Frames are loud enough to count as voiced, but the peak stays below the
    # -34 dBFS floor: a voice print from this would be noise-dominated.
    quality = evaluate_voice_quality(tone(2.0, amplitude=0.012))
    assert quality.usable is False
    assert quality.reason == "音量过低"


def test_clipped_audio_is_not_usable() -> None:
    n = 16000 * 2
    samples = np.where(np.arange(n) % 2 == 0, 0.99, -0.99)
    pcm = (samples * 32767).astype("<i2").tobytes()
    quality = evaluate_voice_quality(pcm)
    assert quality.usable is False
    assert quality.reason == "声音明显削波"


def test_mostly_silent_audio_is_not_usable() -> None:
    # 1 s of tone inside 6 s of silence: enough voiced audio, too sparse to
    # enrol from. Never claim overlap separation — just refuse.
    silence = b"\x00\x00" * (16000 * 5)
    quality = evaluate_voice_quality(tone(1.0) + silence)
    assert quality.usable is False
    assert quality.reason == "静音或噪声占比过高"


def test_empty_audio_is_not_usable() -> None:
    quality = evaluate_voice_quality(b"")
    assert quality.usable is False
    assert quality.voiced_seconds == 0.0


def test_voiced_seconds_is_frame_based_not_total_length() -> None:
    # 3 s tone + 1 s silence: total 4 s, voiced ≈ 3 s.
    pcm = tone(3.0) + b"\x00\x00" * 16000
    quality = evaluate_voice_quality(pcm)
    assert 2.8 <= quality.voiced_seconds <= 3.2
    assert quality.quality == "HIGH"


def test_loud_but_unclipped_audio_is_usable() -> None:
    quality = evaluate_voice_quality(tone(2.0, amplitude=0.6))
    assert quality.usable is True
    assert quality.quality == "HIGH"
    assert quality.reason is None


def test_noisy_audio_has_an_explicit_verdict() -> None:
    # The gate cannot separate overlapping voices; it must at least state a
    # stable verdict instead of treating noise as a clean enrollment sample.
    quality = evaluate_voice_quality(noise(2.0, amplitude=0.05))
    assert isinstance(quality.usable, bool)
    if not quality.usable:
        assert quality.reason is not None


def test_default_path_does_not_change_the_model_waveform() -> None:
    pcm = tone(2.0, amplitude=0.012)
    raw = inspect_raw_audio(pcm)
    prepared = prepare_speaker_audio(pcm, raw)
    assert prepared is pcm


def test_bounded_gain_can_rescue_quiet_audio_for_enrollment() -> None:
    pcm = tone(2.0, amplitude=0.012)
    raw = inspect_raw_audio(pcm)
    prepared = prepare_speaker_audio(pcm, raw, enable_gain=True, max_gain_db=24)
    assert prepared is not pcm
    quality = inspect_speaker_audio(prepared, raw, gain_applied=True)
    assert quality.usable is True
    assert quality.quality in ("MEDIUM", "HIGH")


def test_gain_never_disguises_severe_clipping() -> None:
    n = 16000 * 2
    samples = np.where(np.arange(n) % 2 == 0, 0.99, -0.99)
    pcm = (samples * 32767).astype("<i2").tobytes()
    raw = inspect_raw_audio(pcm)
    prepared = prepare_speaker_audio(pcm, raw, enable_gain=True)
    assert prepared is pcm
    quality = inspect_speaker_audio(prepared, raw, gain_applied=False)
    assert quality.usable is False
    assert quality.reason == "声音明显削波"
