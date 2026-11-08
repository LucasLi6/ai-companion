"""Offline tests for TTS Phase B (streaming text -> speech).

Every test uses a fake bidirectional session and a fake player: no network, no
audio device, no credentials. Real Qwen synthesis is exercised separately by
``python -m companion.tts.demo`` and a live REPL run.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from companion.core.response_events import ResponseEvent
from companion.tts import TTSManager
from companion.tts.client import TTSClient, TTSClientError
from companion.tts.config import (
    TTSConfig,
    TTSConfigError,
    load_tts_config,
    tts_enabled_by_default,
)
from companion.tts.manager import TTSStream
from companion.tts.segmenter import TextSegmenter

CONFIG = TTSConfig(
    api_key="unit-test-key",
    workspace_id="unit-test-workspace",
    segment_min_chars=10,
    segment_max_chars=60,
)

_SESSION_END = object()


# --------------------------------------------------------------------- fakes
class EchoSession:
    """Bidirectional fake: every sent segment produces two PCM chunks."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.flushes = 0
        self.stop_calls = 0
        self.close_calls = 0
        self._audio: queue.Queue = queue.Queue()
        self._send_lock = threading.Lock()

    def send_text(self, text: str) -> None:
        with self._send_lock:
            self.sent.append(text)
        self._audio.put(b"chunk-1")
        self._audio.put(b"chunk-2")

    def flush(self) -> None:
        self.flushes += 1

    def audio_chunks(self):
        while True:
            item = self._audio.get(timeout=5)
            if item is _SESSION_END:
                return
            yield item

    def stop(self) -> None:
        self.stop_calls += 1
        self._audio.put(_SESSION_END)

    def close(self) -> None:
        self.close_calls += 1
        self._audio.put(_SESSION_END)


class FailingSendSession(EchoSession):
    def send_text(self, text: str) -> None:
        raise RuntimeError("provider rejected the segment")


class GatedSession:
    """Audio only appears after the test opens the gate (late audio)."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.sent: list[str] = []
        self.stop_calls = 0
        self.close_calls = 0

    def send_text(self, text: str) -> None:
        self.sent.append(text)

    def flush(self) -> None:
        pass

    def audio_chunks(self):
        self.gate.wait(5)
        yield b"late-1"
        yield b"late-2"

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FakeStreamClient:
    def __init__(self, session_factory=EchoSession) -> None:
        self._factory = session_factory
        self.sessions: list = []
        self.open_calls = 0
        self.closed = False
        self.stream_pcm_calls = 0

    def open_stream(self):
        self.open_calls += 1
        session = self._factory()
        self.sessions.append(session)
        return session

    def stream_pcm(self, text):
        self.stream_pcm_calls += 1
        raise AssertionError("Phase B must not use the complete-text path")
        yield b""  # pragma: no cover - generator signature

    def close(self) -> None:
        self.closed = True


class FakePlayer:
    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.interrupted = 0
        self.writes: list[bytes] = []
        self.write_event = threading.Event()

    def open(self) -> None:
        self.opened = True

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)
        self.write_event.set()

    def interrupt(self) -> None:
        self.interrupted += 1

    def close(self) -> None:
        self.closed = True


def wait_for(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def make_manager(client=None, player=None, **kwargs):
    client = client if client is not None else FakeStreamClient()
    player = player if player is not None else FakePlayer()
    manager = TTSManager(
        config_loader=lambda: CONFIG,
        client_factory=lambda cfg: client,
        player_factory=lambda cfg: player,
        join_timeout=2.0,
        **kwargs,
    )
    manager.start()
    return manager, client, player


def tts_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("ava-tts-")
    ]


# ------------------------------------------------------------------ config
def test_tts_is_enabled_by_default_unless_explicitly_off():
    assert tts_enabled_by_default({}) is True
    assert tts_enabled_by_default({"AVA_TTS_ENABLED": "true"}) is True
    for value in ("0", "false", "off", "no", "disabled", "FALSE"):
        assert tts_enabled_by_default({"AVA_TTS_ENABLED": value}) is False


def test_segment_lengths_come_from_environment():
    config = load_tts_config(
        {
            "DASHSCOPE_API_KEY": "k",
            "DASHSCOPE_WORKSPACE_ID": "w",
            "AVA_TTS_SEGMENT_MIN_CHARS": "8",
            "AVA_TTS_SEGMENT_MAX_CHARS": "32",
        }
    )
    assert (config.segment_min_chars, config.segment_max_chars) == (8, 32)


def test_invalid_segment_lengths_are_rejected():
    base = {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_WORKSPACE_ID": "w"}
    with pytest.raises(TTSConfigError):
        load_tts_config(
            {**base, "AVA_TTS_SEGMENT_MIN_CHARS": "40", "AVA_TTS_SEGMENT_MAX_CHARS": "10"}
        )
    with pytest.raises(TTSConfigError):
        load_tts_config({**base, "AVA_TTS_SEGMENT_MIN_CHARS": "0"})


# --------------------------------------------------------------- segmenter
def test_sentence_boundaries_split_without_character_fragments():
    segmenter = TextSegmenter(10, 60)
    assert segmenter.feed("第一句话已经足够长了。第二") == [
        "第一句话已经足够长了。"
    ]
    assert segmenter.feed("句话也足够长了。") == ["第二句话也足够长了。"]


def test_short_fragment_waits_but_a_complete_sentence_speaks():
    segmenter = TextSegmenter(10, 60)
    # No punctuation yet: never synthesise one character at a time.
    assert segmenter.feed("嗯") == []
    assert segmenter.feed("，那我们继续聊吧。") == ["嗯，那我们继续聊吧。"]
    # A complete two-character sentence is speakable on its own.
    assert segmenter.feed("好。") == ["好。"]


def test_comma_cuts_early_only_with_enough_content():
    segmenter = TextSegmenter(10, 60)
    assert segmenter.feed("短句，") == []
    assert segmenter.feed("这是一个比较长的句子，") == [
        "短句，这是一个比较长的句子，"
    ]


def test_max_length_split_prefers_whitespace_then_hard_cut():
    segmenter = TextSegmenter(2, 5)
    assert segmenter.feed("ab cd ef") == ["ab cd"]
    assert segmenter.pending == "ef"
    assert segmenter.feed("ghijkl") == ["efghi"]
    assert segmenter.finish() == ["jkl"]

    hard = TextSegmenter(2, 4)
    assert hard.feed("abcdefgh") == ["abcd", "efgh"]


def test_decimals_and_abbreviations_are_not_split():
    decimal = TextSegmenter(2, 100)
    assert decimal.feed("圆周率约为3.14对吗。") == ["圆周率约为3.14对吗。"]

    abbreviation = TextSegmenter(2, 100)
    assert abbreviation.feed("Mr. Smith is here. He waits.") == [
        "Mr. Smith is here.",
        "He waits.",
    ]


def test_urls_links_and_code_are_never_split():
    url = TextSegmenter(2, 100)
    assert url.feed("详情见 https://example.com/a?b=1。") == [
        "详情见 https://example.com/a?b=1。"
    ]

    link = TextSegmenter(2, 100)
    assert link.feed("看这个[文档](https://example.com/a.b)就好。") == [
        "看这个[文档](https://example.com/a.b)就好。"
    ]

    code = TextSegmenter(2, 100)
    assert code.feed("使用 `a.b(c)` 即可。") == ["使用 `a.b(c)` 即可。"]


def test_incomplete_tag_and_url_are_held_until_complete():
    tag = TextSegmenter(2, 100)
    assert tag.feed("[gig") == []
    assert tag.pending == "[gig"
    assert tag.feed("gles]你好呀。") == ["[giggles]你好呀。"]

    url = TextSegmenter(2, 100)
    assert url.feed("打开 https://example") == []
    assert url.feed(".com/a。") == ["打开 https://example.com/a。"]


def test_tag_is_atomic_and_never_duplicated_or_alone():
    segmenter = TextSegmenter(2, 8)
    pieces = segmenter.feed("[giggles]你好世界。")
    pieces += segmenter.finish()
    assert "".join(pieces) == "[giggles]你好世界。"
    assert sum(piece.count("[giggles]") for piece in pieces) == 1
    # A leading tag must not be emitted as its own speakable segment.
    assert all(piece.strip() != "[giggles]" for piece in pieces)


def test_finish_flushes_the_remainder_once():
    segmenter = TextSegmenter(10, 60)
    assert segmenter.feed("还没说完") == []
    assert segmenter.finish() == ["还没说完"]
    assert segmenter.finish() == []


# ------------------------------------------------------------------- client
def test_open_stream_uses_connect_with_voice_and_format():
    calls: list[dict] = []

    def connect(**kwargs):
        calls.append(kwargs)
        return object()

    sdk = SimpleNamespace(
        audio=SimpleNamespace(tts=SimpleNamespace(connect=connect))
    )
    client = TTSClient(CONFIG, client_factory=lambda: sdk)
    session = client.open_stream()
    assert session is not None
    assert calls == [
        {
            "model": "qwen-audio-3.1-tts-flash",
            "voice": "xuyanchu_v3.1",
            "format": "pcm",
            "sample_rate": 24000,
        }
    ]


def test_open_stream_reports_unsupported_sdk_and_failures():
    sdk = SimpleNamespace(audio=SimpleNamespace(tts=SimpleNamespace()))
    with pytest.raises(TTSClientError) as excinfo:
        TTSClient(CONFIG, client_factory=lambda: sdk).open_stream()
    assert "connect" in str(excinfo.value)

    def broken(**kwargs):
        raise RuntimeError("boom")

    sdk = SimpleNamespace(
        audio=SimpleNamespace(tts=SimpleNamespace(connect=broken))
    )
    with pytest.raises(TTSClientError) as excinfo:
        TTSClient(CONFIG, client_factory=lambda: sdk).open_stream()
    assert "RuntimeError" in str(excinfo.value)


# ------------------------------------------------------------------ manager
def test_audio_plays_before_the_reply_completes():
    manager, client, player = make_manager()
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        assert isinstance(stream, TTSStream)
        stream.feed("[giggles]第一句话已经足够长了。")

        assert wait_for(lambda: bool(player.writes))
        # Still open: the reply has not finished but speech already started.
        assert not stream.done
        assert manager.status().state == "playing"
        assert client.stream_pcm_calls == 0

        stream.feed("第二句话也足够长了。")
        stream.finish()
        assert wait_for(lambda: (manager.metrics() or {}).get("playback_finished"))
        assert wait_for(lambda: stream.done)
    finally:
        manager.shutdown()

    session = client.sessions[0]
    assert session.sent[0].startswith("[giggles]")
    assert "".join(session.sent) == (
        "[giggles]第一句话已经足够长了。第二句话也足够长了。"
    )
    assert client.stream_pcm_calls == 0


def test_each_segment_is_sent_once_and_in_order():
    manager, client, _player = make_manager()
    text = "第一句话已经足够长了。第二句话也足够长了。第三句收尾。"
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed(text)
        stream.finish()
        assert wait_for(lambda: stream.done)
    finally:
        manager.shutdown()

    sent = client.sessions[0].sent
    assert "".join(sent) == text
    assert len(sent) == 3
    assert len(set(sent)) == len(sent)
    assert all(len(segment) > 1 for segment in sent)


def test_metrics_expose_streaming_latencies():
    manager, _client, _player = make_manager()
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed("第一句话已经足够长了。")
        stream.finish()
        assert wait_for(lambda: (manager.metrics() or {}).get("playback_finished"))
        metrics = manager.metrics()
    finally:
        manager.shutdown()

    assert metrics["response_started_to_first_audio_played_ms"] is not None
    assert metrics["first_display_delta_to_first_audio_played_ms"] is not None
    assert metrics["first_segment_sent_to_first_audio_played_ms"] is not None
    assert metrics["response_to_playback_done_ms"] is not None
    assert metrics["first_audio_received_to_played_ms"] is not None


def test_send_failure_isolated_and_stops_without_playing_more():
    client = FakeStreamClient(FailingSendSession)
    manager, _client, player = make_manager(client=client)
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed("第一句话已经足够长了。")
        stream.finish()
        assert wait_for(lambda: manager.status().state == "error")
        assert "RuntimeError" in (manager.last_error or "")
        assert wait_for(lambda: stream.done)
        played_at_error = len(player.writes)
        time.sleep(0.2)
        assert len(player.writes) == played_at_error
    finally:
        manager.shutdown()


def test_open_stream_failure_does_not_break_text_or_keep_threads():
    class BrokenClient(FakeStreamClient):
        def open_stream(self):
            self.open_calls += 1
            raise TTSClientError("TTS 会话建立失败: ConnectionError")

    manager, client, _player = make_manager(client=BrokenClient())
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed("你好。")
        stream.finish()
        assert wait_for(lambda: manager.status().state == "error")
        assert "会话建立失败" in (manager.last_error or "")
        assert wait_for(lambda: stream.done)
    finally:
        manager.shutdown()
    assert client.open_calls == 1


def test_cancel_discards_late_audio_from_the_old_reply():
    client = FakeStreamClient(GatedSession)
    manager, _client, player = make_manager(client=client)
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed("第一句话已经足够长了。")
        assert wait_for(lambda: client.sessions)
        session = client.sessions[0]
        assert wait_for(lambda: session.sent)

        manager.cancel_stream(stream)
        assert manager.status().state == "idle"
        session.gate.set()  # provider releases audio *after* cancellation
        assert wait_for(lambda: stream.done)
        time.sleep(0.2)
        assert player.writes == []
        assert session.stop_calls + session.close_calls >= 1
        # A cancelled stream can no longer accept text.
        stream.feed("迟到的第二句。")
        time.sleep(0.1)
        assert session.sent == ["第一句话已经足够长了。"]
    finally:
        manager.shutdown()


def test_disable_stops_the_active_stream_and_blocks_new_ones():
    manager, client, _player = make_manager()
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed("第一句话已经足够长了。")
        assert wait_for(lambda: bool(client.sessions[0].sent))
        manager.disable()
        assert wait_for(lambda: stream.done)
        assert manager.enabled is False
        assert manager.start_stream("resp-2", "conv-1") is None
        assert manager.submit(_speech_job("你好。")) is False
    finally:
        manager.shutdown()


def test_start_stream_is_none_when_tts_is_off():
    manager, _client, _player = make_manager()
    manager.disable()
    try:
        assert manager.start_stream("resp-1", "conv-1") is None
    finally:
        manager.shutdown()


def test_speech_expression_switch_does_not_control_tts():
    manager, _client, _player = make_manager()
    orchestrator = SimpleNamespace(speech_expression_enabled=True)
    try:
        # The REPL's ``/speech off`` branch only flips this flag.
        orchestrator.speech_expression_enabled = False
        assert manager.enabled is True
        stream = manager.start_stream("resp-1", "conv-1")
        assert stream is not None
        stream.finish()
        assert wait_for(lambda: stream.done)
    finally:
        manager.shutdown()


def test_no_tts_threads_are_left_after_shutdown():
    manager, _client, _player = make_manager()
    stream = manager.start_stream("resp-1", "conv-1")
    stream.feed("第一句话已经足够长了。")
    stream.finish()
    assert wait_for(lambda: (manager.metrics() or {}).get("playback_finished"))
    manager.shutdown()
    assert wait_for(lambda: not tts_threads(), timeout=3.0)


def _speech_job(text: str):
    from companion.tts import SpeechJob

    return SpeechJob(response_id="resp", conversation_id="conv", text=text)


# ---------------------------------------------------------------------- CLI
class FakeStreamingOrchestrator:
    def __init__(self, events) -> None:
        self._events = events

    def process_utterance_stream(self, event):
        async def _stream():
            for item in self._events:
                await asyncio.sleep(0)
                yield item

        return _stream()


def _started(sequence=0):
    return ResponseEvent(
        type="response_started",
        event_id="event-1",
        conversation_id="conv-1",
        response_id="resp-1",
        sequence=sequence,
        status="STREAMING",
    )


def _delta(text, display_delta, sequence=1):
    return ResponseEvent(
        type="response_delta",
        event_id="event-1",
        conversation_id="conv-1",
        response_id="resp-1",
        sequence=sequence,
        status="STREAMING",
        delta=text,
        display_delta=display_delta,
    )


def _completed(text, display_text, status="COMPLETED", sequence=9):
    return ResponseEvent(
        type="response_completed",
        event_id="event-1",
        conversation_id="conv-1",
        response_id="resp-1",
        sequence=sequence,
        status=status,
        text=text,
        display_text=display_text,
        action="RESPOND",
    )


def _failed(sequence=2):
    return ResponseEvent(
        type="response_failed",
        event_id="event-1",
        conversation_id="conv-1",
        response_id="resp-1",
        sequence=sequence,
        status="FAILED",
        error="provider down",
    )


def _run_cli(events, manager):
    from companion.cli import _run_streaming_turn

    rt = SimpleNamespace(orchestrator=FakeStreamingOrchestrator(events))
    return asyncio.run(
        _run_streaming_turn(rt, object(), tts_manager=manager)
    )


def test_cli_streams_display_deltas_and_flushes_only_the_tail(capsys):
    manager, client, _player = make_manager()
    display = "[softly]第一句话已经足够长了。第二句话也足够长了。"
    events = [
        _started(0),
        _delta("", "[softly]第一句话", 1),
        _delta("已经足够长了。", "已经足够长了。第二句话", 2),
        _delta("也足够长了。", "也足够长了。", 3),
        _completed(display, display, sequence=4),
    ]
    try:
        result = _run_cli(events, manager)
        assert result["status"] == "COMPLETED"
        assert wait_for(
            lambda: (manager.metrics() or {}).get("playback_finished")
        )
    finally:
        manager.shutdown()

    sent = client.sessions[0].sent
    assert "".join(sent) == display
    assert sum(segment.count("[softly]") for segment in sent) == 1
    # Exactly-one-synthesis: the completed event did not resend the reply.
    assert sum(len(segment) for segment in sent) == len(display)
    assert client.stream_pcm_calls == 0
    capsys.readouterr()


def test_cli_falls_back_to_full_text_when_no_delta_arrives(capsys):
    manager, client, _player = make_manager()
    display = "[sad]你好呀。"
    events = [
        _started(0),
        _completed(display, display, sequence=1),
    ]
    try:
        _run_cli(events, manager)
        assert wait_for(
            lambda: (manager.metrics() or {}).get("playback_finished")
        )
    finally:
        manager.shutdown()
    assert "".join(client.sessions[0].sent) == display
    capsys.readouterr()


def test_cli_cancels_stream_on_failure(capsys):
    manager, client, player = make_manager()
    events = [
        _started(0),
        _delta("第一句话已经足够长了。", "第一句话已经足够长了。", 1),
        _failed(2),
    ]
    try:
        _run_cli(events, manager)
        session = client.sessions[0]
        assert wait_for(lambda: session.stop_calls + session.close_calls >= 1)
        assert wait_for(lambda: manager.status().state == "idle")
        writes = len(player.writes)
        time.sleep(0.2)
        assert len(player.writes) == writes
    finally:
        manager.shutdown()
    capsys.readouterr()


@pytest.mark.parametrize("status", ["TRUNCATED", "FAILED"])
def test_cli_does_not_speak_an_unsuccessful_reply_to_the_end(status, capsys):
    manager, client, _player = make_manager()
    events = [
        _started(0),
        _delta("第一句话已经足够长了。", "第一句话已经足够长了。", 1),
        _completed("第一句话已经足够长了。", "第一句话已经足够长了。", status=status, sequence=2),
    ]
    try:
        _run_cli(events, manager)
        assert wait_for(lambda: manager.status().state == "idle")
        # Never the full Phase A submission path.
        assert client.stream_pcm_calls == 0
    finally:
        manager.shutdown()
    capsys.readouterr()
