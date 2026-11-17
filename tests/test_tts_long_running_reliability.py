"""V0.9.9.29 — long-running TTS reliability.

The previous behavior could report a reply as ``finished`` while the speaker
stayed silent, could churn the output device on every user input, and could
prune a cancelled session out of the manager while its sender thread was still
blocked in ``open_stream()``. These offline tests pin the repaired contract:

* an idle ``stop()`` never touches the PortAudio stream;
* a stream that PortAudio says is not active is restarted or rebuilt before
  the next write, and a failed rebuild raises instead of dropping audio;
* a generation that received no PCM (or submitted no text) ends as
  ``FAILED``/``TTS_EMPTY_AUDIO``, never as ``FINISHED``;
* a session stays tracked until its threads exited and its socket closed;
* exactly one terminal result is reported per generation.

No network, no audio device, no credentials.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from companion.tts import TTSManager
from companion.tts.config import TTSConfig
from companion.tts.errors import TTSPlayerError
from companion.tts.player import PyAudioPlayer

CONFIG = TTSConfig(
    api_key="unit-test-key",
    workspace_id="unit-test-workspace",
    segment_min_chars=10,
    segment_max_chars=60,
    instruction="test-instruction",
)

_SESSION_END = object()
_TEXT = "第一句话已经足够长了。"


# --------------------------------------------------------------------- fakes
class ScriptedSession:
    """One fake bidirectional session with scripted audio.

    ``audio_chunks()`` blocks until ``stop()``/``close()`` signals the end, so
    the receiver can never finish before the sender submitted its text.
    """

    def __init__(
        self,
        chunks: tuple[bytes, ...] = (b"AAAA", b"BBBB"),
        *,
        stop_raises: bool = False,
        close_raises: bool = False,
    ) -> None:
        self.chunks = chunks
        self.stop_raises = stop_raises
        self.close_raises = close_raises
        self.started = False
        self.sent: list[str] = []
        self.stop_calls = 0
        self.close_calls = 0
        self._audio: queue.Queue = queue.Queue()

    def start(self):
        assert not self.started
        self.started = True
        return self

    def send_text(self, text: str) -> None:
        assert self.started
        self.sent.append(text)
        for chunk in self.chunks:
            self._audio.put(chunk)

    def flush(self) -> None:
        pass

    def audio_chunks(self):
        assert self.started
        while True:
            item = self._audio.get(timeout=5)
            if item is _SESSION_END:
                return
            yield item

    def stop(self) -> None:
        self.stop_calls += 1
        self._audio.put(_SESSION_END)
        if self.stop_raises:
            raise RuntimeError("stop exploded")

    def close(self) -> None:
        self.close_calls += 1
        self._audio.put(_SESSION_END)
        if self.close_raises:
            raise RuntimeError("close exploded")


class GatedSession(ScriptedSession):
    """Audio stays blocked until the test opens the gate."""

    def __init__(self, gate: threading.Event, chunk: bytes = b"OLD") -> None:
        super().__init__(chunks=(chunk,))
        self._gate = gate

    def send_text(self, text: str) -> None:
        if text not in self.sent:
            self.sent.append(text)

    def audio_chunks(self):
        self._gate.wait(5)
        for chunk in self.chunks:
            yield chunk


class BlockingOpenClient:
    """The first ``open_stream()`` blocks until the gate opens."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.opened = 0
        self.sessions: list[ScriptedSession] = []

    def open_stream(self):
        self.opened += 1
        if self.opened == 1:
            self.gate.wait(5)
        session = ScriptedSession()
        session.start()
        self.sessions.append(session)
        return session

    def close(self) -> None:
        pass


class FakeClient:
    def __init__(self, factory=lambda: ScriptedSession()) -> None:
        self._factory = factory
        self.sessions: list[ScriptedSession] = []
        self.closed = False

    def open_stream(self):
        session = self._factory()
        session.start()
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


class RecordingPlayer:
    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0
        self.interrupts = 0
        self.writes: list[bytes] = []
        self.write_event = threading.Event()

    def open(self) -> None:
        self.opened += 1

    def write(self, pcm: bytes) -> None:
        self.writes.append(bytes(pcm))
        self.write_event.set()

    def interrupt(self) -> None:
        self.interrupts += 1

    def close(self) -> None:
        self.closed += 1


class BlockingPlayer(RecordingPlayer):
    """Blocks inside ``write`` so ``stop()`` runs while audio is in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.write_started = threading.Event()

    def write(self, pcm: bytes) -> None:
        self.write_started.set()
        self.release.wait(3)
        super().write(pcm)


class _FakeStream:
    def __init__(self) -> None:
        self.active = True
        self.stopped = 0
        self.started = 0
        self.closed = 0
        self.writes: list[bytes] = []

    def is_active(self) -> bool:
        return self.active

    def write(self, pcm: bytes) -> None:
        if not self.active:
            raise OSError("stream is not active")
        self.writes.append(bytes(pcm))

    def stop_stream(self) -> None:
        self.active = False
        self.stopped += 1

    def start_stream(self) -> None:
        self.active = True
        self.started += 1

    def close(self) -> None:
        self.closed += 1


class _FakePyAudio:
    def __init__(self) -> None:
        self.streams: list[_FakeStream] = []

    def open(self, **kwargs) -> _FakeStream:
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    def get_format_from_width(self, width: int) -> int:
        return 8

    def get_default_output_device_info(self) -> dict:
        return {"index": 0, "name": "Fake Speakers"}

    def terminate(self) -> None:
        pass


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def make_manager(client=None, player=None, **kwargs):
    client = client if client is not None else FakeClient()
    player = player if player is not None else RecordingPlayer()
    manager = TTSManager(
        config_loader=lambda: CONFIG,
        client_factory=lambda cfg: client,
        player_factory=lambda cfg: player,
        join_timeout=2.0,
        **kwargs,
    )
    manager.start()
    return manager, client, player


def collect_events(manager):
    events = []
    manager.add_playback_listener(events.append)
    return events


def terminal_events(events, generation):
    return [
        event
        for event in events
        if event.generation == generation
        and event.kind in ("finished", "failed", "cancelled")
    ]


def play_once(manager, response_id: str = "resp-1"):
    stream = manager.start_stream(response_id, "conv-1")
    assert stream is not None
    stream.feed(_TEXT)
    stream.finish()
    return stream


# ----------------------------------------------------------------- 50 turns
def test_fifty_consecutive_replies_play_through_one_manager():
    manager, client, player = make_manager()
    try:
        written = 0
        for turn in range(50):
            stream = play_once(manager, f"resp-{turn}")
            assert wait_for(
                lambda: manager.playback_outcome(stream.generation)
                == "finished",
                timeout=5.0,
            ), f"turn {turn} never finished"
            assert wait_for(lambda: stream.done)
            assert len(player.writes) >= written
            written = len(player.writes)
        assert written >= 50
        assert manager.status().state == "idle"
        assert manager.status().last_outcome == "FINISHED"
        assert len(client.sessions) == 50
    finally:
        manager.shutdown()
    # Only the two long-lived manager threads remain; every session thread
    # has exited.
    lingering = [
        thread
        for thread in threading.enumerate()
        if thread.name in ("ava-tts-sender", "ava-tts-receiver")
        and thread.is_alive()
    ]
    assert lingering == []


def test_fifty_idle_stops_never_touch_the_output_stream():
    manager, _client, player = make_manager()
    try:
        stream = play_once(manager)
        assert wait_for(
            lambda: manager.playback_outcome(stream.generation) == "finished"
        )
        assert wait_for(lambda: manager.status().state == "idle")

        for _ in range(50):
            manager.stop()

        assert player.interrupts == 0
        assert player.opened == 1
        assert manager.enabled is True
        assert manager.status().state == "idle"
    finally:
        manager.shutdown()


def test_stop_while_playing_flushes_once_and_a_new_reply_still_plays():
    player = BlockingPlayer()
    manager, _client, _player = make_manager(player=player)
    try:
        stream = manager.start_stream("resp-old", "conv-1")
        assert stream is not None
        stream.feed(_TEXT + _TEXT)
        assert player.write_started.wait(3)

        manager.stop()
        player.release.set()
        assert player.interrupts >= 1
        assert manager.enabled is True

        before = len(player.writes)
        stream2 = play_once(manager, "resp-new")
        assert wait_for(
            lambda: manager.playback_outcome(stream2.generation)
            == "finished"
        )
        assert len(player.writes) > before
    finally:
        player.release.set()
        manager.shutdown()


def test_a_new_reply_reports_exactly_one_terminal_result_per_generation():
    manager, _client, _player = make_manager()
    events = collect_events(manager)
    try:
        stream = play_once(manager)
        generation = stream.generation
        assert wait_for(
            lambda: manager.playback_outcome(generation) == "finished"
        )
        # A later stop of the same (already finished) generation must not add
        # a second terminal notice.
        manager.stop()
        manager.stop()
        assert wait_for(lambda: len(terminal_events(events, generation)) >= 1)
        assert len(terminal_events(events, generation)) == 1
        assert terminal_events(events, generation)[0].kind == "finished"
    finally:
        manager.shutdown()


# -------------------------------------------------------- zero-audio truth
def test_zero_pcm_is_reported_as_empty_audio_never_finished():
    client = FakeClient(lambda: ScriptedSession(chunks=()))
    manager, _client, player = make_manager(client=client)
    events = collect_events(manager)
    try:
        stream = play_once(manager)
        generation = stream.generation
        assert wait_for(
            lambda: manager.playback_outcome(generation) == "failed"
        )
        assert "TTS_EMPTY_AUDIO" in (manager.last_error or "")
        assert player.writes == []
        terminal = terminal_events(events, generation)
        assert len(terminal) == 1
        assert terminal[0].kind == "failed"
        assert terminal[0].stage == "SESSION"
        status = manager.status()
        assert status.last_outcome == "FAILED"
        assert status.last_failure_stage == "SESSION"
        assert status.pcm_received_bytes == 0
        # A failed reply is never counted as the last finished one.
        assert (manager.metrics(generation) or {}).get("playback_finished") is not True
    finally:
        manager.shutdown()


def test_a_reply_without_submitted_text_is_not_a_successful_playback():
    manager, _client, _player = make_manager()
    events = collect_events(manager)
    try:
        stream = manager.start_stream("resp-empty", "conv-1")
        generation = stream.generation
        stream.finish()
        assert wait_for(
            lambda: manager.playback_outcome(generation) == "failed"
        )
        terminal = terminal_events(events, generation)
        assert len(terminal) == 1
        assert terminal[0].kind == "failed"
        assert "TTS 未提交任何可合成文本" in (terminal[0].error or "")
    finally:
        manager.shutdown()


def test_status_lines_show_the_last_failure_and_pcm_counters():
    client = FakeClient(lambda: ScriptedSession(chunks=()))
    manager, _client, _player = make_manager(client=client)
    try:
        stream = play_once(manager)
        assert wait_for(
            lambda: manager.playback_outcome(stream.generation) == "failed"
        )
        text = "\n".join(manager.status().lines())
        assert f"last_generation: {stream.generation}" in text
        assert "last_outcome: FAILED" in text
        assert "last_failure_stage: SESSION" in text
        assert "pcm_received_bytes: 0" in text
    finally:
        manager.shutdown()


def test_pcm_written_bytes_match_what_the_device_accepted():
    manager, _client, player = make_manager()
    try:
        stream = play_once(manager)
        assert wait_for(
            lambda: manager.playback_outcome(stream.generation) == "finished"
        )
        status = manager.status()
        assert status.pcm_written_bytes == sum(len(w) for w in player.writes)
        assert status.pcm_received_bytes >= status.pcm_written_bytes
    finally:
        manager.shutdown()


# --------------------------------------------------------- session failure
def test_session_stop_failure_is_recorded_without_faking_a_failure():
    client = FakeClient(lambda: ScriptedSession(stop_raises=True))
    manager, _client, _player = make_manager(client=client)
    try:
        stream = play_once(manager)
        # Audio still arrived, so playback is genuinely finished.
        assert wait_for(
            lambda: manager.playback_outcome(stream.generation) == "finished"
        )
        # ... but the stop() failure is never silently dropped.
        assert "TTS_SESSION_STOP_FAILED" in (
            manager.status().last_error or ""
        )
        assert client.sessions[0].stop_calls == 1
    finally:
        manager.shutdown()


def test_session_close_failure_is_recorded_on_its_own_generation():
    client = FakeClient(lambda: ScriptedSession(close_raises=True))
    manager, _client, _player = make_manager(client=client)
    try:
        stream = play_once(manager)
        assert wait_for(
            lambda: manager.playback_outcome(stream.generation) == "finished"
        )
        assert wait_for(lambda: stream.done)
        assert "TTS_SESSION_CLOSE_FAILED" in (manager.last_error or "")
    finally:
        manager.shutdown()


# -------------------------------------------------------- session tracking
def test_cancel_while_open_stream_blocks_keeps_the_session_tracked():
    client = BlockingOpenClient()
    manager, _client, player = make_manager(client=client)
    try:
        stuck = manager.start_stream("resp-stuck", "conv-1")
        assert stuck is not None
        assert wait_for(lambda: client.opened == 1)

        manager.stop()
        # The sender is still inside open_stream(): the stream is NOT done and
        # must remain tracked rather than leaking an untracked thread.
        assert stuck.done is False
        assert stuck in manager._streams

        stream2 = play_once(manager, "resp-new")
        assert wait_for(
            lambda: manager.playback_outcome(stream2.generation)
            == "finished",
            timeout=3.0,
        )
        assert stream2 is not None

        client.gate.set()
        assert wait_for(lambda: stuck.done, timeout=5.0)
        assert player.opened == 1
    finally:
        client.gate.set()
        manager.shutdown()


def test_late_audio_from_a_previous_generation_is_never_played():
    gate = threading.Event()
    sessions: list[ScriptedSession] = []

    def factory():
        if not sessions:
            session = GatedSession(gate)
        else:
            session = ScriptedSession(chunks=(b"NEW",))
        sessions.append(session)
        return session

    manager, _client, player = make_manager(client=FakeClient(factory))
    try:
        old = manager.start_stream("resp-old", "conv-1")
        old.feed(_TEXT)
        manager.stop()
        new = play_once(manager, "resp-new")
        assert wait_for(
            lambda: manager.playback_outcome(new.generation) == "finished"
        )
        gate.set()
        time.sleep(0.2)
        assert b"OLD" not in player.writes
        assert b"NEW" in player.writes
    finally:
        gate.set()
        manager.shutdown()


def test_a_failure_in_an_old_generation_cannot_fail_the_new_one():
    manager, _client, player = make_manager()
    try:
        stream = play_once(manager)
        generation = stream.generation
        assert wait_for(
            lambda: manager.playback_outcome(generation) == "finished"
        )
        manager.stop()
        # Audio tagged with the stale generation is dropped, not played and
        # not reported.
        assert manager._put_audio(generation, b"STALE") is False
        stream2 = play_once(manager, "resp-next")
        assert wait_for(
            lambda: manager.playback_outcome(stream2.generation)
            == "finished"
        )
        assert b"STALE" not in player.writes
    finally:
        manager.shutdown()


# ------------------------------------------------------------ player health
def _player_with_fake_pyaudio():
    pyaudio = _FakePyAudio()
    player = PyAudioPlayer(CONFIG, pyaudio_factory=lambda: pyaudio)
    return player, pyaudio


def test_a_stream_portaudio_reports_inactive_is_restarted_before_write():
    player, pyaudio = _player_with_fake_pyaudio()
    player.open()
    stream = pyaudio.streams[0]
    stream.active = False

    player.write(b"pcm")

    assert stream.started == 1
    assert stream.writes == [b"pcm"]
    assert player.last_error is None


def test_a_stream_that_cannot_restart_is_rebuilt_then_errors_cleanly():
    player, pyaudio = _player_with_fake_pyaudio()
    player.open()
    stream = pyaudio.streams[0]
    stream.active = False

    def broken_start():
        raise OSError("device lost")

    stream.start_stream = broken_start

    def broken_open(**kwargs):
        raise OSError("no device")

    pyaudio.open = broken_open
    with pytest.raises(TTSPlayerError):
        player.write(b"pcm")
    assert player._stream is None
    assert "OUTPUT_DEVICE_OPEN_FAILED" in (player.last_error or "")
    assert stream.closed == 1


def test_interrupting_an_already_stopped_stream_does_not_stop_or_start_it():
    player, pyaudio = _player_with_fake_pyaudio()
    player.open()
    stream = pyaudio.streams[0]
    stream.active = False

    player.interrupt()

    assert (stream.stopped, stream.started) == (0, 0)
    assert stream.closed == 1
    # The next write rebuilds a working stream instead of reusing the dead one.
    player.write(b"pcm")
    assert pyaudio.streams[-1].writes == [b"pcm"]


def test_refresh_without_a_device_change_never_rebuilds_the_stream():
    player, pyaudio = _player_with_fake_pyaudio()
    player.open()
    assert len(pyaudio.streams) == 1

    player.refresh()
    player.refresh()

    assert len(pyaudio.streams) == 1
    assert pyaudio.streams[0].closed == 0


def test_a_failed_device_open_is_reported_and_leaves_nothing_playable():
    class BrokenPyAudio(_FakePyAudio):
        def open(self, **kwargs):
            raise OSError("no output device")

    player = PyAudioPlayer(CONFIG, pyaudio_factory=BrokenPyAudio)
    with pytest.raises(TTSPlayerError) as excinfo:
        player.open()
    assert "OSError" in str(excinfo.value)
    assert player._stream is None


def test_start_fails_cleanly_when_the_output_device_cannot_open():
    class BrokenPlayer(RecordingPlayer):
        def open(self) -> None:
            raise TTSPlayerError("无法打开音频输出设备: OSError")

    manager = TTSManager(
        config_loader=lambda: CONFIG,
        client_factory=lambda cfg: FakeClient(),
        player_factory=lambda cfg: BrokenPlayer(),
        join_timeout=2.0,
    )
    ok, message = manager.enable()
    assert ok is False
    assert "无法打开音频输出设备" in message
    assert manager.enabled is False
    manager.shutdown()


# ------------------------------------------------------------- diagnostics
def test_cli_reports_a_failed_generation_and_stays_quiet_otherwise(capsys):
    from companion.cli import _register_tts_failure_reporter

    class Manager:
        def __init__(self) -> None:
            self.listeners = []

        def add_playback_listener(self, listener) -> None:
            self.listeners.append(listener)

    manager = Manager()
    _register_tts_failure_reporter(manager)
    assert len(manager.listeners) == 1

    manager.listeners[0](
        SimpleNamespace(
            kind="failed",
            generation=42,
            stage="PLAYBACK",
            error="OUTPUT_STREAM_INACTIVE",
        )
    )
    captured = capsys.readouterr()
    assert (
        "[tts error] generation=42 stage=PLAYBACK "
        "reason=OUTPUT_STREAM_INACTIVE"
    ) in captured.err

    manager.listeners[0](
        SimpleNamespace(kind="finished", generation=43, stage=None, error=None)
    )
    assert capsys.readouterr().err == ""


# --------------------------------------------------------- timing / threads
def test_first_audio_is_not_delayed_by_the_health_checks():
    manager, _client, player = make_manager()
    try:
        began = time.monotonic()
        stream = manager.start_stream("resp-1", "conv-1")
        stream.feed(_TEXT)
        assert player.write_event.wait(2)
        elapsed = time.monotonic() - began
        assert elapsed < 1.0, f"first audio took {elapsed:.3f}s"
        stream.finish()
        assert wait_for(
            lambda: manager.playback_outcome(stream.generation) == "finished"
        )
    finally:
        manager.shutdown()


def test_repeated_cancel_and_close_does_not_leak_session_threads():
    manager, _client, _player = make_manager()
    try:
        for index in range(30):
            stream = manager.start_stream(f"resp-{index}", "conv-1")
            stream.feed(_TEXT)
            if index % 2:
                manager.stop()
            else:
                stream.finish()
                assert wait_for(
                    lambda: manager.playback_outcome(stream.generation)
                    in ("finished", "cancelled"),
                    timeout=5.0,
                )
        manager.stop()
        assert wait_for(
            lambda: not [
                thread
                for thread in threading.enumerate()
                if thread.name in ("ava-tts-sender", "ava-tts-receiver")
                and thread.is_alive()
            ],
            timeout=5.0,
        )
    finally:
        manager.shutdown()
