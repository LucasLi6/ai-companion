"""Offline tests for the independent ASR Silero-VAD segmentation state machine.

No microphone, network, NumPy, ONNX Runtime or model file is needed: a fake
probability function drives the exact same window-by-window logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from companion.asr.config import ASRConfig, VAD_WINDOW_SAMPLES
from companion.asr.vad import (
    AUDIO,
    SPEECH_ACTIVE,
    SPEECH_ENDED,
    SPEECH_STARTED,
    VADError,
    VADSegmenter,
    WAITING_FOR_END,
    WAITING_FOR_SPEECH,
    resolve_vad_model_path,
)

WINDOW_BYTES = VAD_WINDOW_SAMPLES * 2


def window(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * VAD_WINDOW_SAMPLES


SILENCE = window(0)
SPEECH = window(1)


def config(**overrides) -> ASRConfig:
    base = dict(api_key="unit-test-key", workspace_id="unit-test-workspace")
    base.update(overrides)
    return ASRConfig(**base)


def content_probability(pcm: bytes) -> float:
    return 0.9 if pcm[0] else 0.1


def segmenter(**overrides) -> VADSegmenter:
    return VADSegmenter(content_probability, config(**overrides))


def feed(seg: VADSegmenter, windows: list[bytes]):
    events = []
    for item in windows:
        events.extend(seg.process(item))
    return events


def kinds(events) -> list[str]:
    return [event.kind for event in events]


def test_window_math_matches_spec_defaults() -> None:
    seg = segmenter()
    assert seg.min_speech_windows == 5  # 160 ms
    assert seg.end_silence_windows == 22  # 700 ms rounded up to 704 ms
    assert seg.pre_roll_windows == 11  # 350 ms rounded up to 352 ms
    assert seg.min_speech_ms == 160
    assert seg.end_silence_ms == 704
    assert seg.pre_roll_ms == 352


def test_custom_durations_round_to_windows() -> None:
    seg = segmenter(min_speech_ms=100, end_silence_ms=500, pre_roll_ms=200)
    assert seg.min_speech_windows == 4
    assert seg.end_silence_windows == 16
    assert seg.pre_roll_windows == 7


def test_pre_roll_is_prepended_to_first_syllable() -> None:
    seg = segmenter()
    events = feed(seg, [SILENCE] * 11 + [SPEECH] * 5)
    starts = [event for event in events if event.kind == SPEECH_STARTED]
    assert len(starts) == 1
    audio = starts[0].audio
    assert len(audio) == (11 + 5) * WINDOW_BYTES
    assert audio[: 11 * WINDOW_BYTES] == SILENCE * 11
    assert audio[11 * WINDOW_BYTES: 12 * WINDOW_BYTES] == SPEECH
    assert seg.state == SPEECH_ACTIVE


def test_short_noise_does_not_start_or_end_an_utterance() -> None:
    seg = segmenter()
    events = feed(seg, [SPEECH] * 3 + [SILENCE] * 22)
    assert events == []
    assert seg.state == WAITING_FOR_SPEECH


def test_short_pause_keeps_the_same_utterance_alive() -> None:
    seg = segmenter()
    events = feed(seg, [SPEECH] * 5)
    assert kinds(events) == [SPEECH_STARTED]
    events = feed(seg, [SILENCE] * 10)  # 320 ms < 700 ms
    assert SPEECH_ENDED not in kinds(events)
    assert seg.state == WAITING_FOR_END
    events = feed(seg, [SPEECH])  # speech resumes
    assert seg.state == SPEECH_ACTIVE
    assert SPEECH_ENDED not in kinds(events)
    events = feed(seg, [SILENCE] * 22)
    assert kinds(events).count(SPEECH_ENDED) == 1
    assert seg.state == WAITING_FOR_SPEECH


def test_end_silence_emits_exactly_one_end_event() -> None:
    seg = segmenter()
    feed(seg, [SPEECH] * 5)
    events = feed(seg, [SILENCE] * 40)
    assert kinds(events).count(SPEECH_ENDED) == 1
    assert kinds(events).count(SPEECH_STARTED) == 0


def test_multiple_utterances_are_independent() -> None:
    seg = segmenter()
    events = []
    for _ in range(2):
        events.extend(feed(seg, [SILENCE] * 11 + [SPEECH] * 5))
        events.extend(feed(seg, [SILENCE] * 22))
    assert kinds(events).count(SPEECH_STARTED) == 2
    assert kinds(events).count(SPEECH_ENDED) == 2
    starts = [e for e in events if e.kind == SPEECH_STARTED]
    ends = [e for e in events if e.kind == SPEECH_ENDED]
    start_positions = [
        index for index, event in enumerate(events)
        if event.kind == SPEECH_STARTED
    ]
    end_positions = [
        index for index, event in enumerate(events)
        if event.kind == SPEECH_ENDED
    ]
    assert len(starts) == len(ends) == 2
    assert start_positions[0] < end_positions[0]
    assert end_positions[0] < start_positions[1]
    assert start_positions[1] < end_positions[1]


def test_arbitrary_byte_chunk_boundaries_preserve_window_order() -> None:
    seg = segmenter()
    pcm = SILENCE * 11 + SPEECH * 5 + SILENCE * 22
    sizes = [1, 3, 7, 1023, 1025, 333, 2048, 17]
    events = []
    offset = 0
    index = 0
    while offset < len(pcm):
        size = sizes[index % len(sizes)]
        index += 1
        events.extend(seg.process(pcm[offset: offset + size]))
        offset += size
    assert kinds(events).count(SPEECH_STARTED) == 1
    assert kinds(events).count(SPEECH_ENDED) == 1
    start = next(event for event in events if event.kind == SPEECH_STARTED)
    assert len(start.audio) == 16 * WINDOW_BYTES
    assert start.audio[:11 * WINDOW_BYTES] == SILENCE * 11


def test_audio_events_carry_every_post_start_window() -> None:
    seg = segmenter()
    events = feed(seg, [SPEECH] * 6)
    audio = [event for event in events if event.kind == AUDIO]
    # The fifth speech window completes activation; the sixth is streamed.
    assert len(audio) == 1
    assert audio[0].audio == SPEECH


def test_flush_closes_an_active_utterance_once() -> None:
    seg = segmenter()
    feed(seg, [SPEECH] * 5)
    events = seg.flush()
    assert kinds(events) == [SPEECH_ENDED]
    assert seg.flush() == []
    assert seg.state == WAITING_FOR_SPEECH


def test_flush_does_not_promote_short_noise() -> None:
    seg = segmenter()
    feed(seg, [SPEECH] * 3)
    assert seg.flush() == []


def test_reset_clears_buffered_partial_window() -> None:
    seg = segmenter()
    assert seg.process(SILENCE[:500]) == []
    assert seg.buffered_bytes == 500
    seg.reset()
    assert seg.buffered_bytes == 0
    assert seg.state == WAITING_FOR_SPEECH


def test_threshold_boundary_counts_as_speech() -> None:
    seg = VADSegmenter(lambda _pcm: 0.50, config(vad_threshold=0.50))
    events = feed(seg, [SPEECH] * 5)
    assert kinds(events) == [SPEECH_STARTED]


def test_resolve_vad_model_path_explicit_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.onnx"
    explicit.write_bytes(b"model")
    assert resolve_vad_model_path(str(explicit)) == explicit

    env_path = tmp_path / "env.onnx"
    env_path.write_bytes(b"model")
    monkeypatch.setenv("AVA_ASR_VAD_MODEL", str(env_path))
    assert resolve_vad_model_path("") == env_path


def test_resolve_vad_model_path_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AVA_ASR_VAD_MODEL", str(tmp_path / "missing.onnx"))
    monkeypatch.setenv("HOME", str(tmp_path))
    # The optional silero-vad package is not installed in the offline test
    # environment; if it were, this test would still be valid only when its
    # data file is absent. Guard for that possibility.
    try:
        resolved = resolve_vad_model_path("")
    except VADError as exc:
        assert "silero_vad.onnx" in str(exc)
    else:  # pragma: no cover - only when silero-vad is installed
        assert resolved.name == "silero_vad.onnx"
