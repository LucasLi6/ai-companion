"""Offline latency regressions: same monotonic clock, no raw voice/text."""

from __future__ import annotations

import pytest

from companion.asr.manager import FinalUtterance
from companion.cli import _format_tts_metrics
from companion.tts.config import TTSConfigError, load_tts_config
from companion.tts.manager import TTSManager
from companion.tts.segmenter import TextSegmenter


def test_asr_final_keeps_backward_compatible_optional_vad_mark():
    old = FinalUtterance(1, "hello", 10.0)
    assert old.vad_end_at is None
    measured = FinalUtterance(2, "hello", 10.5, vad_end_at=10.0)
    assert measured.vad_end_at == 10.0


def test_first_clause_mode_is_off_by_default():
    segmenter = TextSegmenter(6, 35)
    text = "这是一个很长很长很长的开头，接下来才继续说明"
    assert segmenter.feed(text) == []
    assert segmenter.finish() == [text]


def test_opt_in_first_clause_flushes_once_and_keeps_remaining_sentence():
    segmenter = TextSegmenter(6, 35, first_clause_chars=10)
    first = "这是一个很长很长很长的开头，"
    second = "接下来的内容，也要连在同一句话里。"
    assert segmenter.feed(first) == [first]
    assert segmenter.feed(second) == [second]
    assert segmenter.finish() == []


def test_first_clause_waits_for_threshold_and_never_splits_numeric_comma():
    segmenter = TextSegmenter(2, 35, first_clause_chars=10)
    assert segmenter.feed("这个数量为 1,000") == []
    assert segmenter.feed("，后续还有解释。") == ["这个数量为 1,000，", "后续还有解释。"]
    assert segmenter.finish() == []


def test_first_clause_setting_validation():
    base = {
        "DASHSCOPE_API_KEY": "fake",
        "DASHSCOPE_WORKSPACE_ID": "fake",
    }
    assert load_tts_config(base).first_clause_chars == 0
    assert load_tts_config({
        **base, "AVA_TTS_FIRST_CLAUSE_CHARS": "12"
    }).first_clause_chars == 12
    for bad in ("-1", "x", "161"):
        with pytest.raises(TTSConfigError):
            load_tts_config({
                **base, "AVA_TTS_FIRST_CLAUSE_CHARS": bad
            })


def test_voice_latency_reports_include_asr_and_identity_before_router():
    manager = TTSManager()
    manager._timings[7] = {
        "user_submitted_at": 10.0,
        "voice_asr_final_at": 10.4,
        "voice_identity_confirmed_at": 10.52,
        "router_completed_at": 11.82,
        "response_started_at": 12.0,
        "first_text_at": 12.3,
        "first_segment_ready_at": 12.7,
        "tts_connect_started_at": 12.0,
        "tts_connect_finished_at": 12.5,
        "first_segment_sent_at": 12.72,
        "first_audio_received_at": 13.1,
        "first_audio_write_started_at": 13.12,
        "first_audio_played_at": 13.15,
        "response_completed_at": 13.2,
    }
    result = manager.metrics(7)
    assert result is not None
    assert result["is_voice"]
    assert result["vad_to_asr_ms"] == pytest.approx(400)
    assert result["asr_to_identity_ms"] == pytest.approx(120)
    assert result["identity_to_router_ms"] == pytest.approx(1300)
    assert result["vad_to_first_audio_played_ms"] == pytest.approx(3150)
    assert result["first_text_to_tts_connected_ms"] == pytest.approx(200)
    assert result["tts_connected_to_first_segment_ms"] == pytest.approx(200)
    assert result["response_to_tts_connected_ms"] == pytest.approx(500)
    report = _format_tts_metrics(result)
    assert "Connected → first segment" in report
    assert "VAD end" in report
    assert "ASR final" in report
    assert "Identity" in report
    assert "3150 ms" in report


def test_live_generation_takes_priority_over_last_finished_generation():
    manager = TTSManager()
    manager._timings[1] = {
        "response_started_at": 1.0, "first_text_at": 2.0
    }
    manager._timings[2] = {
        "response_started_at": 3.0, "first_text_at": 3.2
    }
    manager._last_finished_generation = 1
    assert manager.metrics()["generation"] == 2
