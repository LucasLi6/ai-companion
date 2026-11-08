"""Offline tests for TTS Phase B (streaming text -> speech).

Every test uses a fake bidirectional session and a fake player: no network, no
audio device, no credentials. Real Qwen synthesis is exercised separately by
``python -m companion.tts.demo`` and a live REPL run.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from companion.core.response_events import ResponseEvent
from companion.tts import TTSManager
from companion.tts.client import TTSClient, TTSClientError
from companion.tts.config import (
    DEFAULT_LONG_FALLBACK_CHARS,
    DEFAULT_TTS_INSTRUCTION,
    TTSConfig,
    TTSConfigError,
    load_local_env,
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
    instruction="test-instruction",
)

_SESSION_END = object()


# --------------------------------------------------------------------- fakes
class EchoSession:
    """Bidirectional fake: every sent segment produces two PCM chunks."""

    def __init__(self) -> None:
        self.started = False
        self.sent: list[str] = []
        self.flushes = 0
        self.stop_calls = 0
        self.close_calls = 0
        self._audio: queue.Queue = queue.Queue()
        self._send_lock = threading.Lock()

    def start(self):
        # Mirrors dashscope's SynthesisSession: send_text/audio_chunks are
        # only valid after the handshake has completed.
        assert not self.started, "session started twice"
        self.started = True
        return self

    def send_text(self, text: str) -> None:
        assert self.started, "send_text before start()"
        with self._send_lock:
            self.sent.append(text)
        self._audio.put(b"chunk-1")
        self._audio.put(b"chunk-2")

    def flush(self) -> None:
        assert self.started, "flush before start()"
        self.flushes += 1

    def audio_chunks(self):
        assert self.started, "audio_chunks before start()"
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
        self.started = False
        self.gate = threading.Event()
        self.sent: list[str] = []
        self.stop_calls = 0
        self.close_calls = 0

    def start(self):
        self.started = True
        return self

    def send_text(self, text: str) -> None:
        assert self.started, "send_text before start()"
        self.sent.append(text)

    def flush(self) -> None:
        assert self.started, "flush before start()"

    def audio_chunks(self):
        assert self.started, "audio_chunks before start()"
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
        # Real TTSClient.open_stream() performs the start() handshake before
        # returning; the fake must follow the same contract.
        starter = getattr(session, "start", None)
        if callable(starter):
            starter()
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


def test_speech_rate_defaults_to_1_07_and_is_configurable():
    base = {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_WORKSPACE_ID": "w"}
    assert load_tts_config(base).speech_rate == 1.07
    assert load_tts_config({**base, "AVA_TTS_RATE": "1.15"}).speech_rate == 1.15
    for bad in ("0.4", "2.5", "fast"):
        with pytest.raises(TTSConfigError):
            load_tts_config({**base, "AVA_TTS_RATE": bad})


def test_instruction_flush_strategy_and_fallback_are_configurable():
    base = {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_WORKSPACE_ID": "w"}
    default = load_tts_config(base)
    assert default.instruction == DEFAULT_TTS_INSTRUCTION
    assert default.flush_strategy == "auto"
    assert default.long_fallback_chars == DEFAULT_LONG_FALLBACK_CHARS

    overridden = load_tts_config(
        {
            **base,
            "AVA_TTS_INSTRUCTION": "请更温柔地说话。",
            "AVA_TTS_FLUSH_STRATEGY": "sentence_flush",
            "AVA_TTS_LONG_FALLBACK_CHARS": "120",
        }
    )
    assert overridden.instruction == "请更温柔地说话。"
    assert overridden.flush_strategy == "sentence_flush"
    assert overridden.long_fallback_chars == 120

    with pytest.raises(TTSConfigError):
        load_tts_config({**base, "AVA_TTS_FLUSH_STRATEGY": "bogus"})
    with pytest.raises(TTSConfigError):
        load_tts_config({**base, "AVA_TTS_LONG_FALLBACK_CHARS": "0"})


def test_output_device_defaults_to_system_and_is_configurable():
    base = {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_WORKSPACE_ID": "w"}
    assert load_tts_config(base).output_device == ""
    assert (
        load_tts_config(
            {**base, "AVA_TTS_OUTPUT_DEVICE": "WH-1000XM6"}
        ).output_device
        == "WH-1000XM6"
    )


def test_empty_instruction_disables_the_prompt():
    base = {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_WORKSPACE_ID": "w"}
    # Absent -> default; explicitly empty -> no instruction (for A/B listening).
    assert load_tts_config(base).instruction == DEFAULT_TTS_INSTRUCTION
    assert load_tts_config({**base, "AVA_TTS_INSTRUCTION": ""}).instruction == ""


def test_local_env_file_supplies_missing_values_only(tmp_path, monkeypatch):
    env_file = tmp_path / "ava.env"
    env_file.write_text(
        "# local settings\n"
        "DASHSCOPE_API_KEY=from-file\n"
        "export DASHSCOPE_WORKSPACE_ID='ws-file'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AVA_ENV_FILE", str(env_file))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-exported")

    loaded = load_local_env()

    assert "DASHSCOPE_API_KEY" in loaded
    assert "DASHSCOPE_WORKSPACE_ID" not in loaded
    assert os.environ["DASHSCOPE_API_KEY"] == "from-file"
    # An explicitly exported value always wins over the file.
    assert os.environ["DASHSCOPE_WORKSPACE_ID"] == "ws-exported"


# --------------------------------------------------------------- segmenter
def test_sentence_boundaries_split_without_character_fragments():
    segmenter = TextSegmenter(10, 60)
    assert segmenter.feed("第一句话已经足够长了。第二") == [
        "第一句话已经足够长了。"
    ]
    assert segmenter.feed("句话也足够长了。") == ["第二句话也足够长了。"]


def test_commas_are_not_a_synthesis_boundary():
    segmenter = TextSegmenter(10, 60)
    # Three commas but only one period: a single continuous sentence.
    assert segmenter.feed("第一句，中间，还有，终于说完了。") == [
        "第一句，中间，还有，终于说完了。"
    ]
    # A complete short sentence is still speakable on its own.
    assert segmenter.feed("好。") == ["好。"]


def test_commas_accumulate_until_a_sentence_end():
    segmenter = TextSegmenter(10, 60)
    assert segmenter.feed("短句，") == []
    assert segmenter.feed("好的，") == []
    assert segmenter.feed("这是一个比较长的句子。") == [
        "短句，好的，这是一个比较长的句子。"
    ]


def test_no_sentence_end_means_no_early_cut():
    segmenter = TextSegmenter(6, 35)
    # No punctuation yet: never synthesise a fragment. With no sentence end the
    # text stays buffered instead of being cut at a space or comma.
    assert segmenter.feed("hello there my friend") == []
    assert segmenter.pending == "hello there my friend"
    assert segmenter.finish() == ["hello there my friend"]


def test_chinese_english_and_portuguese_sentences():
    segmenter = TextSegmenter(6, 35)
    pieces = segmenter.feed("你好。Hello world. Olá mundo!") + segmenter.finish()
    assert pieces == ["你好。", "Hello world.", "Olá mundo!"]


def test_newline_is_not_an_unconditional_boundary():
    segmenter = TextSegmenter(6, 35)
    assert segmenter.feed("第一行还没结束\n第二行继续。") == [
        "第一行还没结束\n第二行继续。"
    ]


def test_long_fallback_cuts_at_a_safe_clause_boundary_without_loss():
    segmenter = TextSegmenter(6, 35, long_fallback_chars=15)
    head = segmenter.feed("一二三四五六七八九十，甲乙丙丁戊己庚辛壬癸")
    tail = segmenter.finish()
    assert head == ["一二三四五六七八九十，"]
    assert tail == ["甲乙丙丁戊己庚辛壬癸"]
    assert "".join(head) + "".join(tail) == (
        "一二三四五六七八九十，甲乙丙丁戊己庚辛壬癸"
    )


def test_long_fallback_preserves_protected_tokens_whole():
    segmenter = TextSegmenter(6, 35, long_fallback_chars=12)
    text = (
        "请看 https://example.com/very/long/path?query=1 和数字 1234567890，"
        "还有标签 [mischievously] 以及 `code span`，最后结束。"
    )
    pieces = segmenter.feed(text) + segmenter.finish()
    # Fallback cuts may trim a separating space at the cut point, but every
    # word, number, tag, URL and code span must survive once and in order.
    joined = "".join(pieces)
    assert joined.replace(" ", "") == text.replace(" ", "")
    assert joined.count("[mischievously]") == 1
    for token in (
        "https://example.com/very/long/path?query=1",
        "1234567890",
        "[mischievously]",
        "`code span`",
    ):
        assert any(token in piece for piece in pieces), token


def test_decimals_and_abbreviations_are_not_split():
    decimal = TextSegmenter(2, 100)
    assert decimal.feed("圆周率约为3.14对吗。") == ["圆周率约为3.14对吗。"]

    abbreviation = TextSegmenter(2, 100)
    # The final "." is deferred: it may be an abbreviation, a decimal point
    # or a sentence end, so the segmenter waits for the next delta/finish().
    assert abbreviation.feed("Mr. Smith is here. He waits.") == [
        "Mr. Smith is here."
    ]
    assert abbreviation.finish() == ["He waits."]


def test_decimal_and_thousands_separator_survive_delta_boundaries():
    decimal = TextSegmenter(2, 100)
    # "3." at the end of a delta must not be cut: the next delta completes
    # it as 3.14. The comma no longer cuts, so the whole sentence is emitted
    # at the "？".
    assert decimal.feed("圆周率约等于3.") == []
    assert decimal.feed("14，你知道吗？") == ["圆周率约等于3.14，你知道吗？"]

    thousands = TextSegmenter(1, 100)
    assert thousands.feed("一共有1000000000,") == []
    assert thousands.feed("000人。") == ["一共有1000000000,000人。"]


def test_character_by_character_feeding_never_loses_text():
    text = "[giggles]圆周率约等于3.14，一共有1000000000,000人。你记住了吗？"
    whole = TextSegmenter(2, 100)
    expected = whole.feed(text) + whole.finish()

    split = TextSegmenter(2, 100)
    pieces: list[str] = []
    for char in text:
        pieces.extend(split.feed(char))
    pieces.extend(split.finish())

    assert "".join(expected) == text
    assert "".join(pieces) == text
    # Decimal/thousands grouping survives every token boundary.
    assert "3.14" in "".join(pieces)
    assert "1000000000,000" in "".join(pieces)


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
    started: list[bool] = []

    class FakeSession:
        def start(self):
            started.append(True)
            return self

    def connect(**kwargs):
        calls.append(kwargs)
        return FakeSession()

    sdk = SimpleNamespace(
        audio=SimpleNamespace(tts=SimpleNamespace(connect=connect))
    )
    client = TTSClient(CONFIG, client_factory=lambda: sdk)
    session = client.open_stream()
    assert isinstance(session, FakeSession)
    assert started == [True]
    assert calls == [
        {
            "model": "qwen-audio-3.1-tts-flash",
            "voice": "xuyanchu_v3.1",
            "format": "pcm",
            "sample_rate": 24000,
            "rate": 1.07,
            "parameters": {"instruction": "test-instruction"},
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


def test_open_stream_start_failure_closes_the_session():
    closed: list[bool] = []

    class BrokenSession:
        def start(self):
            raise RuntimeError("handshake failed")

        def close(self):
            closed.append(True)

    sdk = SimpleNamespace(
        audio=SimpleNamespace(
            tts=SimpleNamespace(connect=lambda **kwargs: BrokenSession())
        )
    )
    with pytest.raises(TTSClientError) as excinfo:
        TTSClient(CONFIG, client_factory=lambda: sdk).open_stream()
    assert "RuntimeError" in str(excinfo.value)
    assert closed == [True]


def test_open_stream_omits_instruction_when_disabled():
    calls: list[dict] = []

    class FakeSession:
        def start(self):
            return self

    def connect(**kwargs):
        calls.append(kwargs)
        return FakeSession()

    sdk = SimpleNamespace(
        audio=SimpleNamespace(tts=SimpleNamespace(connect=connect))
    )
    config = TTSConfig(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        instruction="",
    )
    TTSClient(config, client_factory=lambda: sdk).open_stream()
    assert calls == [
        {
            "model": "qwen-audio-3.1-tts-flash",
            "voice": "xuyanchu_v3.1",
            "format": "pcm",
            "sample_rate": 24000,
            "rate": 1.07,
            "parameters": {},
        }
    ]


# ------------------------------------------------------------------ manager
def test_status_reports_rate_without_credentials():
    manager, _client, _player = make_manager()
    try:
        lines = manager.status().lines()
    finally:
        manager.shutdown()
    text = "\n".join(lines)
    assert "rate: 1.07" in text
    assert "output: system default" in text
    assert "unit-test-key" not in text


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


def test_auto_strategy_never_flushes_within_a_sentence():
    manager, client, _player = make_manager()
    text = "这句话里有，逗号，但只到句号才算一句。"
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed(text)
        stream.finish()
        assert wait_for(lambda: stream.done)
    finally:
        manager.shutdown()
    session = client.sessions[0]
    assert "".join(session.sent) == text
    # One sentence -> one send_text, no per-fragment flush.
    assert len(session.sent) == 1
    assert session.flushes == 0


def test_sentence_flush_strategy_flushes_once_per_sentence():
    config = TTSConfig(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        segment_min_chars=10,
        segment_max_chars=60,
        instruction="test-instruction",
        flush_strategy="sentence_flush",
    )
    client = FakeStreamClient()
    manager = TTSManager(
        config_loader=lambda: config,
        client_factory=lambda cfg: client,
        player_factory=lambda cfg: FakePlayer(),
        join_timeout=2.0,
    )
    manager.start()
    text = "第一句话结束了。第二句话也结束了。"
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed(text)
        stream.finish()
        assert wait_for(lambda: stream.done)
    finally:
        manager.shutdown()
    session = client.sessions[0]
    assert "".join(session.sent) == text
    assert len(session.sent) == 2
    assert session.flushes == 2


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


def test_starting_a_new_reply_does_not_wait_for_a_stuck_previous_session():
    opening = threading.Event()
    release = threading.Event()

    class SlowClient(FakeStreamClient):
        def open_stream(self):
            self.open_calls += 1
            opening.set()
            release.wait(2)
            session = EchoSession()
            session.start()
            self.sessions.append(session)
            return session

    manager, _client, _player = make_manager(client=SlowClient())
    try:
        assert manager.start_stream("resp-1", "conv-1") is not None
        assert wait_for(opening.is_set)
        started_at = time.monotonic()
        second = manager.start_stream("resp-2", "conv-1")
        elapsed = time.monotonic() - started_at
        assert second is not None
        # The chat thread must not block on the old session's handshake.
        assert elapsed < 0.5
    finally:
        release.set()
        manager.shutdown()


def test_stream_text_backlog_is_bounded_by_reply_limit():
    opening = threading.Event()
    release = threading.Event()

    class SlowClient(FakeStreamClient):
        def open_stream(self):
            self.open_calls += 1
            opening.set()
            release.wait(2)
            session = EchoSession()
            session.start()
            self.sessions.append(session)
            return session

    config = TTSConfig(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        max_text_chars=20,
        segment_min_chars=1,
        segment_max_chars=60,
    )
    client = SlowClient()
    manager = TTSManager(
        config_loader=lambda: config,
        client_factory=lambda cfg: client,
        player_factory=lambda cfg: FakePlayer(),
        join_timeout=1.0,
    )
    manager.start()
    try:
        stream = manager.start_stream("resp-1", "conv-1")
        assert stream is not None
        assert wait_for(opening.is_set)
        stream.feed("a" * 12)
        # This delta would push the backlog past one reply's allowance: it is
        # dropped, the already-buffered text is still spoken, and memory is
        # bounded.
        stream.feed("b" * 12)
        release.set()
        stream.finish()
        assert wait_for(lambda: stream.done)
        assert client.sessions[0].sent == ["a" * 12]
    finally:
        release.set()
        manager.shutdown()


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


def test_cli_prints_one_timing_line_per_completed_reply(capsys):
    manager, _client, _player = make_manager()
    display = "[softly]第一句话已经足够长了。第二句话也足够长了。"
    events = [
        _started(0),
        _delta("", "[softly]第一句话", 1),
        _delta("已经足够长了。", "已经足够长了。第二句话", 2),
        _delta("也足够长了。", "也足够长了。", 3),
        _completed(display, display, sequence=4),
    ]
    try:
        _run_cli(events, manager)
    finally:
        manager.shutdown()
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith("[timing]")]
    assert len(lines) == 1
    assert "first_text=" in lines[0]
    assert "text_done=" in lines[0]
    assert "first_audio=" in lines[0]


def test_cli_reports_first_audio_off_when_tts_is_disabled(capsys):
    manager, _client, _player = make_manager()
    manager.disable()
    events = [
        _started(0),
        _delta("你好呀。", "你好呀。", 1),
        _completed("你好呀。", "你好呀。", sequence=2),
    ]
    try:
        _run_cli(events, manager)
    finally:
        manager.shutdown()
    output = capsys.readouterr().out
    assert "[timing] first_text=" in output
    assert "first_audio=off" in output


def test_cli_prints_no_timing_for_wait_or_ignore(capsys):
    manager, _client, _player = make_manager()
    try:
        _run_cli([], manager)
    finally:
        manager.shutdown()
    assert "[timing]" not in capsys.readouterr().out


def test_metrics_are_relative_to_the_user_submission():
    manager, _client, _player = make_manager()
    try:
        now = time.monotonic()
        stream = manager.start_stream(
            "resp-1",
            "conv-1",
            submitted_at=now - 2.0,
            router_completed_at=now - 1.5,
        )
        assert stream is not None
        stream.feed("第一句话已经足够长了。")
        stream.note_first_text()
        stream.finish()
        assert wait_for(
            lambda: (manager.metrics(stream.generation) or {}).get(
                "user_to_first_audio_ms"
            )
            is not None
        )
        metrics = manager.metrics(stream.generation)
        assert metrics["user_to_response_started_ms"] >= 1900
        assert metrics["user_to_router_ms"] >= 450
        assert metrics["user_to_first_text_ms"] is not None
        assert metrics["user_to_text_done_ms"] is not None
        assert metrics["tts_connect_ms"] is not None
        assert metrics["first_segment_to_first_pcm_ms"] is not None

        from companion.cli import _format_tts_metrics

        report = _format_tts_metrics(metrics)
        assert "Last response timing:" in report
        assert "WebSocket handshake:" in report
        assert "User → First text:" in report
        assert "User → First audio:" in report
        assert "User → Text complete:" in report
    finally:
        manager.shutdown()


def test_timing_stays_pending_until_audio_is_written(capsys):
    manager, client, _player = make_manager(
        client=FakeStreamClient(session_factory=GatedSession)
    )
    events = [
        _started(0),
        _delta("你好呀。", "你好呀。", 1),
        _completed("你好呀。", "你好呀。", sequence=2),
    ]
    try:
        _run_cli(events, manager)
        output = capsys.readouterr().out
        timing = [
            line for line in output.splitlines() if line.startswith("[timing]")
        ]
        assert len(timing) == 1
        assert "first_audio=pending" in timing[0]

        assert wait_for(lambda: bool(client.sessions))
        client.sessions[0].gate.set()
        assert wait_for(
            lambda: (manager.metrics() or {}).get("user_to_first_audio_ms")
            is not None
        )
        # The background audio start never prints into the prompt.
        assert capsys.readouterr().out == ""
    finally:
        manager.shutdown()


def test_started_event_carries_the_router_completion_mark():
    from companion.core.response_events import ResponseEventSink

    emitted = []
    sink = ResponseEventSink("event-1", "conv-1", emitted.append)
    sink.started("resp-1", router_completed_at=123.0)
    assert emitted[0].type == "response_started"
    assert emitted[0].router_completed_at == 123.0


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
