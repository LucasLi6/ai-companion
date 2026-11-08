"""Offline tests for TTS Phase A (independent Qwen speech output).

Every test uses a fake DashScope client and a fake player: no network, no
audio device, no credentials. Real Qwen synthesis is exercised separately by
``python -m companion.tts.demo``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from companion.core.response_events import ResponseEvent
from companion.tts import SpeechJob, TTSManager
from companion.tts.client import (
    TTSClient,
    TTSClientError,
    TTSEmptyAudioError,
    extract_audio_bytes,
)
from companion.tts.config import (
    TTSConfig,
    TTSConfigError,
    load_tts_config,
)
from companion.tts.player import PyAudioPlayer

CONFIG = TTSConfig(
    api_key="unit-test-key",
    workspace_id="unit-test-workspace",
    instruction="test-instruction",
)


# --------------------------------------------------------------------- fakes
class FakePlayer:
    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.interrupted = 0
        self.writes: list[bytes] = []
        self.write_started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def open(self) -> None:
        self.opened = True

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)
        self.write_started.set()
        # Tests that need back-pressure clear ``release`` explicitly.
        self.release.wait(0.2)

    def interrupt(self) -> None:
        self.interrupted += 1

    def close(self) -> None:
        self.closed = True


class FakeTTSClient:
    def __init__(self, behavior, log: list | None = None) -> None:
        self.behavior = behavior
        self.log = log if log is not None else []
        self.closed = False

    def stream_pcm(self, text):
        self.log.append(("request", text))
        yield from self.behavior(text)

    def close(self) -> None:
        self.closed = True


def make_manager(client, player, **kwargs) -> TTSManager:
    return TTSManager(
        config_loader=lambda: CONFIG,
        client_factory=lambda cfg: client,
        player_factory=lambda cfg: player,
        join_timeout=2.0,
        **kwargs,
    )


def wait_for(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# -------------------------------------------------------------------- config
def test_missing_credentials_are_reported_by_name_only():
    with pytest.raises(TTSConfigError) as excinfo:
        load_tts_config({})
    message = str(excinfo.value)
    assert "DASHSCOPE_API_KEY" in message
    assert "DASHSCOPE_WORKSPACE_ID" in message


def test_config_defaults_and_overrides():
    config = load_tts_config(
        {
            "DASHSCOPE_API_KEY": "k",
            "DASHSCOPE_WORKSPACE_ID": "w",
            "AVA_TTS_MODEL": "m",
            "AVA_TTS_VOICE": "v",
            "AVA_TTS_REGION": "r",
            "AVA_TTS_SAMPLE_RATE": "16000",
        }
    )
    assert (config.model, config.voice, config.region) == ("m", "v", "r")
    assert config.sample_rate == 16000


def test_config_default_voice_is_xuyanchu():
    config = load_tts_config(
        {
            "DASHSCOPE_API_KEY": "k",
            "DASHSCOPE_WORKSPACE_ID": "w",
        }
    )
    assert config.voice == "xuyanchu_v3.1"
    assert config.model == "qwen-audio-3.1-tts-flash"


def test_enable_without_credentials_stays_disabled_and_never_connects():
    manager = TTSManager(
        config_loader=lambda: load_tts_config({}),
        client_factory=lambda cfg: pytest.fail("client must not be created"),
        player_factory=lambda cfg: pytest.fail("player must not be created"),
    )
    ok, message = manager.enable()
    assert ok is False
    assert manager.enabled is False
    assert "DASHSCOPE_API_KEY" in message
    manager.shutdown()


def test_disabled_manager_does_not_read_config_on_submit():
    def fail_loader():
        raise AssertionError("config must not be read while TTS is off")

    manager = TTSManager(config_loader=fail_loader)
    assert manager.submit(SpeechJob("r", "c", "你好")) is False
    assert manager.status().state == "disabled"
    manager.shutdown()


def test_player_failure_keeps_tts_disabled_with_a_clear_message():
    from companion.tts.errors import TTSPlayerError

    class BrokenPlayer(FakePlayer):
        def open(self):
            raise TTSPlayerError(
                "pyaudio 未安装：请执行 pip install 'ai-companion[tts]'"
            )

    client = FakeTTSClient(lambda text: iter([b"x"]))
    manager = make_manager(client, BrokenPlayer())
    try:
        ok, message = manager.enable()
        assert ok is False
        assert manager.enabled is False
        assert "pyaudio" in message
        assert client.closed is True
    finally:
        manager.shutdown()


# -------------------------------------------------------------------- client
class _FakeStream:
    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return iter(self.responses)


def _fake_sdk(responses, calls):
    stream = _FakeStream(responses, calls)
    return SimpleNamespace(audio=SimpleNamespace(tts=SimpleNamespace(stream=stream)))


def test_client_passes_configured_voice_and_settings_lazily():
    calls = []
    responses = [SimpleNamespace(output=SimpleNamespace(audio=b"pcm-1"))]
    sdk = _fake_sdk(responses, calls)
    created = []

    def factory():
        created.append(True)
        return sdk

    client = TTSClient(CONFIG, client_factory=factory)
    assert created == []
    assert list(client.stream_pcm("[sad]你好")) == [b"pcm-1"]
    text, kwargs = calls[0]
    assert text == "[sad]你好"
    assert kwargs == {
        "model": CONFIG.model,
        "voice": CONFIG.voice,
        "format": "pcm",
        "sample_rate": 24000,
        "rate": 1.07,
        "parameters": {"instruction": "test-instruction"},
    }
    assert created == [True]


def test_client_accepts_top_level_bytes_chunks():
    """The official SDK yields raw PCM bytes, not response objects."""
    calls = []
    sdk = _fake_sdk([b"pcm-1", b"pcm-2"], calls)
    client = TTSClient(CONFIG, client_factory=lambda: sdk)
    assert list(client.stream_pcm("你好")) == [b"pcm-1", b"pcm-2"]


def test_manager_plays_top_level_bytes_without_empty_audio_error():
    calls = []
    sdk = _fake_sdk([b"pcm-1", b"pcm-2"], calls)
    player = FakePlayer()
    manager = TTSManager(
        config_loader=lambda: CONFIG,
        client_factory=lambda cfg: TTSClient(
            cfg, client_factory=lambda: sdk
        ),
        player_factory=lambda cfg: player,
        join_timeout=2.0,
    )
    try:
        ok, _ = manager.enable()
        assert ok is True
        assert manager.submit(SpeechJob("r", "c", "你好"))
        assert wait_for(lambda: len(player.writes) == 2)
        assert player.writes == [b"pcm-1", b"pcm-2"]
        assert wait_for(
            lambda: manager.metrics() is not None
            and manager.metrics()["playback_finished"]
        )
        assert manager.last_error is None
    finally:
        manager.shutdown()


def test_extract_audio_supports_base64_and_dict_shapes():
    encoded = base64.b64encode(b"hello").decode("ascii")
    assert extract_audio_bytes(
        SimpleNamespace(output=SimpleNamespace(audio=encoded))
    ) == b"hello"
    assert extract_audio_bytes(
        SimpleNamespace(output={"audio": encoded})
    ) == b"hello"
    assert extract_audio_bytes(
        SimpleNamespace(output=SimpleNamespace(audio_data=b"raw"))
    ) == b"raw"


def test_client_provider_error_and_empty_audio_are_tts_errors():
    calls = []
    bad = _fake_sdk(
        [
            SimpleNamespace(
                status_code=401, code="InvalidApiKey", message="bad key"
            )
        ],
        calls,
    )
    with pytest.raises(TTSClientError) as excinfo:
        list(TTSClient(CONFIG, client_factory=lambda: bad).stream_pcm("hi"))
    assert "401" in str(excinfo.value)

    empty = _fake_sdk([], calls)
    with pytest.raises(TTSEmptyAudioError):
        list(TTSClient(CONFIG, client_factory=lambda: empty).stream_pcm("hi"))


# ------------------------------------------------------------------- manager
def test_completed_reply_plays_pcm_in_order_and_records_latency():
    log: list = []
    client = FakeTTSClient(
        lambda text: iter([b"a", b"b", b"c"]), log
    )
    player = FakePlayer()
    manager = make_manager(client, player)
    try:
        ok, message = manager.enable()
        assert ok is True
        assert "unit-test-key" not in message
        assert player.opened is True

        assert manager.submit(
            SpeechJob("r1", "c1", "[sad]你好", reply_completed_at=time.monotonic())
        )
        assert wait_for(
            lambda: manager.metrics() is not None
            and manager.metrics()["playback_finished"]
        )
        assert player.writes == [b"a", b"b", b"c"]
        assert log[0] == ("request", "[sad]你好")

        metrics = manager.metrics()
        for key in (
            "reply_to_speech_start_ms",
            "reply_to_request_start_ms",
            "request_to_first_audio_ms",
            "first_audio_received_to_played_ms",
            "first_audio_to_playback_done_ms",
            "reply_to_playback_done_ms",
        ):
            assert metrics[key] is not None
    finally:
        manager.shutdown()
    assert player.closed is True
    assert client.closed is True


def test_submit_is_non_blocking_and_newest_reply_wins():
    gate = threading.Event()
    started = threading.Event()
    requested: list[str] = []
    log: list = []

    def behavior(text):
        requested.append(text)
        if text == "one":
            started.set()
            gate.wait(3)
            yield b"1"
        else:
            yield text.encode("utf-8")

    player = FakePlayer()
    manager = make_manager(FakeTTSClient(behavior, log), player)
    try:
        manager.enable()
        assert manager.submit(SpeechJob("r1", "c", "one"))
        assert started.wait(2)

        began = time.monotonic()
        assert manager.submit(SpeechJob("r2", "c", "two"))
        elapsed = time.monotonic() - began
        assert elapsed < 0.2  # chat thread is never blocked by synthesis
        assert manager.submit(SpeechJob("r3", "c", "three"))

        gate.set()
        assert wait_for(lambda: b"three" in player.writes)
        assert b"two" not in player.writes
        assert b"one" not in player.writes  # stale generation dropped
        assert "two" not in requested
    finally:
        gate.set()
        manager.shutdown()


def test_stop_invalidates_synthesis_and_playback_but_keeps_tts_enabled():
    started = threading.Event()
    player = FakePlayer()
    player.release.clear()
    requested: list[str] = []

    def behavior(text):
        requested.append(text)
        if text == "first":
            started.set()
            for pcm in (b"1", b"2", b"3", b"4"):
                yield pcm
        else:
            yield b"new"

    manager = make_manager(FakeTTSClient(behavior), player)
    try:
        manager.enable()
        assert manager.submit(SpeechJob("r1", "c", "first"))
        assert started.wait(2)
        assert wait_for(lambda: b"1" in player.writes)

        manager.stop()
        assert manager.enabled is True
        assert manager.status().state == "idle"
        time.sleep(0.3)
        assert b"3" not in player.writes and b"4" not in player.writes
        assert player.interrupted >= 1

        # Speech remains available after stop.
        player.release.set()
        assert manager.submit(SpeechJob("r2", "c", "second"))
        assert wait_for(lambda: b"new" in player.writes)
    finally:
        player.release.set()
        manager.shutdown()


def test_text_validation_rejects_empty_too_long_and_protocol_headers():
    player = FakePlayer()
    client = FakeTTSClient(lambda text: iter([b"x"]))
    manager = make_manager(client, player)
    try:
        manager.enable()
        assert manager.submit(SpeechJob("r", "c", "   ")) is False
        assert manager.last_error == "TTS_TEXT_EMPTY"

        assert manager.submit(SpeechJob("r", "c", "x" * 2100)) is False
        assert manager.last_error.startswith("TTS_TEXT_TOO_LONG")

        assert manager.submit(
            SpeechJob("r", "c", "VOICE=natural\nTEXT=hello")
        ) is False
        assert manager.last_error == "TTS_PROTOCOL_HEADER"
        assert manager.submit(
            SpeechJob("r", "c", "TEXT=hello")
        ) is False
        assert manager.last_error == "TTS_PROTOCOL_HEADER"

        assert manager.submit(SpeechJob("r", "c", "[sad]hello")) is True
    finally:
        manager.shutdown()


def test_tts_error_is_isolated_and_does_not_leak_text_or_disable_tts():
    def behavior(text):
        if text == "boom":
            raise RuntimeError("private-provider-detail")
        yield b"ok"

    player = FakePlayer()
    manager = make_manager(FakeTTSClient(behavior), player)
    try:
        manager.enable()
        assert manager.submit(SpeechJob("r1", "c", "boom"))
        assert wait_for(lambda: manager.status().state == "error")
        assert manager.enabled is True
        error = manager.last_error
        assert "RuntimeError" in error
        assert "private-provider-detail" not in error

        assert manager.submit(SpeechJob("r2", "c", "good"))
        assert wait_for(lambda: b"ok" in player.writes)
    finally:
        manager.shutdown()


def test_audio_queue_is_bounded_and_worker_blocks_not_chat_thread():
    hold = threading.Event()
    started = threading.Event()
    player = FakePlayer()
    player.release.clear()

    def behavior(text):
        for index in range(200):
            yield bytes([index % 256])

    player.write_started = started
    manager = make_manager(FakeTTSClient(behavior), player, audio_queue_size=2)
    try:
        manager.enable()
        assert manager.submit(SpeechJob("r", "c", "long"))
        assert started.wait(2)
        # The player is blocked in write(); the worker must stop at the bound.
        assert wait_for(
            lambda: manager.status().queued_audio_chunks <= 2
        )
        began = time.monotonic()
        assert manager.submit(SpeechJob("r2", "c", "next")) is True
        assert time.monotonic() - began < 0.2
    finally:
        hold.set()
        player.release.set()
        manager.shutdown()


def test_shutdown_leaves_no_tts_threads_or_streams():
    player = FakePlayer()
    manager = make_manager(FakeTTSClient(lambda text: iter([b"a"])), player)
    manager.enable()
    manager.submit(SpeechJob("r", "c", "hi"))
    assert wait_for(lambda: b"a" in player.writes)
    worker = manager._worker
    player_thread = manager._player_thread

    manager.shutdown()
    assert manager.enabled is False
    assert not worker.is_alive()
    assert not player_thread.is_alive()
    assert player.closed is True


def test_speech_package_has_no_router_speaker_memory_or_emotion_imports():
    import companion.tts.client as client_module
    import companion.tts.config as config_module
    import companion.tts.manager as manager_module
    import companion.tts.player as player_module

    for module in (
        config_module,
        client_module,
        player_module,
        manager_module,
    ):
        source = inspect.getsource(module)
        for forbidden in (
            "companion.core",
            "companion.agent",
            "companion.memory",
            "companion.emotion",
            "companion.people",
        ):
            assert forbidden not in source


# -------------------------------------------------------------------- player
class _FakeAudioStream:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.stopped = 0
        self.started = 0
        self.closed = 0

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    def stop_stream(self) -> None:
        self.stopped += 1

    def start_stream(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1


class _FakePyAudio:
    """Deliberately has no ``paInt16`` attribute, like a real instance."""

    def __init__(self) -> None:
        self.stream = _FakeAudioStream()
        self.format_widths: list[int] = []
        self.open_kwargs: dict | None = None
        self.terminated = 0
        self.default_index = 0
        self.devices = [
            {
                "index": 0,
                "name": "MacBook Air Speakers",
                "maxOutputChannels": 2,
                "defaultSampleRate": 48000.0,
            },
            {
                "index": 1,
                "name": "WH-1000XM6",
                "maxOutputChannels": 2,
                "defaultSampleRate": 44100.0,
            },
        ]

    def get_format_from_width(self, width: int) -> int:
        self.format_widths.append(width)
        return 8  # a sentinel for paInt16

    def get_default_output_device_info(self) -> dict:
        return self.devices[self.default_index]

    def get_device_count(self) -> int:
        return len(self.devices)

    def get_device_info_by_index(self, index: int) -> dict:
        return self.devices[index]

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        return self.stream

    def terminate(self) -> None:
        self.terminated += 1


def test_player_uses_instance_format_method_not_module_constant():
    fake = _FakePyAudio()
    player = PyAudioPlayer(CONFIG, pyaudio_factory=lambda: fake)

    player.open()
    assert fake.format_widths == [2]  # 16-bit PCM
    assert fake.open_kwargs == {
        "format": 8,
        "channels": 1,
        "rate": 24000,
        "output": True,
        "output_device_index": 0,
    }
    assert player.device_name == "MacBook Air Speakers"

    player.open()  # idempotent: one stream for the whole session
    assert fake.format_widths == [2]

    player.write(b"pcm")
    assert fake.stream.writes == [b"pcm"]

    player.interrupt()
    assert (fake.stream.stopped, fake.stream.started) == (1, 1)

    player.close()
    assert (fake.stream.closed, fake.terminated) == (1, 1)


def test_player_open_failure_is_a_tts_error_and_terminates():
    from companion.tts.errors import TTSPlayerError

    class BrokenPyAudio(_FakePyAudio):
        def open(self, **kwargs):
            raise OSError("no output device")

    fake = BrokenPyAudio()
    player = PyAudioPlayer(CONFIG, pyaudio_factory=lambda: fake)
    with pytest.raises(TTSPlayerError) as excinfo:
        player.open()
    assert "OSError" in str(excinfo.value)
    assert fake.terminated == 1


def test_player_follows_default_output_and_refreshes_on_change():
    fake = _FakePyAudio()
    player = PyAudioPlayer(CONFIG, pyaudio_factory=lambda: fake)
    player.open()
    assert fake.open_kwargs["output_device_index"] == 0
    assert player.device_name == "MacBook Air Speakers"

    # Bluetooth connected and selected as the system default.
    fake.default_index = 1
    player.refresh()
    assert fake.open_kwargs["output_device_index"] == 1
    assert player.device_name == "WH-1000XM6"

    # A refresh with no device change must not reopen anything.
    player.refresh()
    assert fake.open_kwargs["output_device_index"] == 1
    player.close()


def test_player_can_pin_a_device_by_name_or_index():
    for spec in ("WH-1000", "1"):
        fake = _FakePyAudio()
        config = dataclasses.replace(CONFIG, output_device=spec)
        player = PyAudioPlayer(config, pyaudio_factory=lambda: fake)
        player.open()
        assert fake.open_kwargs["output_device_index"] == 1
        assert player.device_name == "WH-1000XM6"
        player.close()

    fake = _FakePyAudio()
    config = dataclasses.replace(CONFIG, output_device="Nonexistent")
    player = PyAudioPlayer(config, pyaudio_factory=lambda: fake)
    from companion.tts.errors import TTSPlayerError

    with pytest.raises(TTSPlayerError):
        player.open()
    assert fake.terminated == 1


# ---------------------------------------------------------------------- demo
def test_demo_prints_last_error_when_playback_fails(monkeypatch, capsys):
    import companion.tts.demo as demo

    class StubManager:
        generation = 1
        last_error = "TTS 请求失败: TTSClientError"

        def enable(self):
            return True, "tts: on"

        def submit(self, job):
            return True

        def status(self):
            return SimpleNamespace(state="error")

        def metrics(self):
            return None

        def shutdown(self):
            pass

    monkeypatch.setattr(demo, "TTSManager", StubManager)
    assert demo.main(["你好"]) == 0
    output = capsys.readouterr().out
    assert "finished=False" in output
    assert "last_error: TTS 请求失败" in output


# ----------------------------------------------------------------------- CLI
class FakeStreamingOrchestrator:
    def __init__(self, events):
        self._events = events

    def process_utterance_stream(self, event):
        async def _stream():
            for item in self._events:
                await asyncio.sleep(0)
                yield item

        return _stream()


class RecordingManager:
    enabled = True
    generation = 41

    def __init__(self):
        self.jobs = []

    def submit(self, job):
        self.jobs.append(job)
        return True


def _event(status="COMPLETED", text="正文", display_text=None):
    return ResponseEvent(
        type="response_completed",
        event_id="event-1",
        conversation_id="conv-1",
        response_id="resp-1",
        sequence=0,
        status=status,
        text=text,
        display_text=display_text,
        action="RESPOND",
    )


def _run_turn(events, manager):
    from companion.cli import _run_streaming_turn

    rt = SimpleNamespace(orchestrator=FakeStreamingOrchestrator(events))
    return asyncio.run(
        _run_streaming_turn(rt, object(), tts_manager=manager)
    )


def test_cli_submits_display_text_with_legal_tags(capsys):
    manager = RecordingManager()
    _run_turn([_event(display_text="[sad]你好。")], manager)
    assert len(manager.jobs) == 1
    job = manager.jobs[0]
    assert job.text == "[sad]你好。"
    assert job.response_id == "resp-1"
    assert job.conversation_id == "conv-1"
    assert job.reply_completed_at is not None


def test_cli_falls_back_to_canonical_text(capsys):
    manager = RecordingManager()
    _run_turn([_event(display_text=None, text="你好。")], manager)
    assert [job.text for job in manager.jobs] == ["你好。"]


@pytest.mark.parametrize("status", ["TRUNCATED", "FAILED"])
def test_cli_never_submits_unsuccessful_replies(status, capsys):
    manager = RecordingManager()
    _run_turn([_event(status=status, display_text="部分")], manager)
    assert manager.jobs == []


def test_cli_never_submits_failed_cancelled_or_silent_turns(capsys):
    manager = RecordingManager()
    failed = ResponseEvent(
        type="response_failed",
        event_id="e",
        conversation_id="c",
        response_id="r",
        sequence=1,
        status="FAILED",
        error="provider down",
    )
    cancelled = ResponseEvent(
        type="response_cancelled",
        event_id="e",
        conversation_id="c",
        response_id="r",
        sequence=1,
        status="CANCELLED",
    )
    _run_turn([], manager)  # WAIT / IGNORE: intentional silence
    _run_turn([failed], manager)
    _run_turn([cancelled], manager)
    assert manager.jobs == []


def test_cli_does_not_submit_when_tts_disabled(capsys):
    manager = RecordingManager()
    manager.enabled = False
    _run_turn([_event(display_text="你好。")], manager)
    assert manager.jobs == []


class StubCommandManager:
    def __init__(self):
        self.enabled = False
        self.stopped = 0
        self.enable_result = (True, "tts: on  model=m  voice=v")

    def enable(self):
        self.enabled = True
        return self.enable_result

    def disable(self):
        self.enabled = False

    def stop(self):
        self.stopped += 1

    def status(self):
        return TTSStatusStub()

    def metrics(self):
        return None


class TTSStatusStub:
    def lines(self):
        return ["tts: off (disabled)", "last_error: -"]


def test_tts_on_warns_about_alibaba_before_enabling():
    from companion.cli import _handle_tts_command

    manager = StubCommandManager()
    lines = _handle_tts_command(manager, ["/tts", "on"])
    assert manager.enabled is True
    assert any("阿里云" in line for line in lines)
    assert lines[-1].startswith("tts: on")


def test_tts_off_disables_and_stop_keeps_enabled():
    from companion.cli import _handle_tts_command

    manager = StubCommandManager()
    manager.enabled = True
    assert _handle_tts_command(manager, ["/tts", "off"]) == ["tts: off"]
    assert manager.enabled is False

    manager.enabled = True
    lines = _handle_tts_command(manager, ["/tts", "stop"])
    assert manager.stopped == 1
    assert manager.enabled is True
    assert "still" not in lines[0]  # enabled: no "(tts off)" suffix


def test_tts_status_never_contains_credentials():
    from companion.tts.manager import TTSStatus

    status = TTSStatus(
        enabled=True,
        state="idle",
        model="qwen-audio-3.1-tts-flash",
        voice="xuyanchu_v3.1",
        region="cn-beijing",
        sample_rate=24000,
        waiting_jobs=0,
        queued_audio_chunks=0,
        generation=3,
        last_error=None,
    )
    text = "\n".join(status.lines())
    assert "model" in text and "voice" in text
    assert "api_key" not in text.lower()
    assert "DASHSCOPE_API_KEY" not in text


def test_tts_usage_for_unknown_action():
    from companion.cli import _handle_tts_command

    manager = StubCommandManager()
    assert _handle_tts_command(manager, ["/tts"]) == [
        "usage: /tts on|off|stop|status"
    ]
