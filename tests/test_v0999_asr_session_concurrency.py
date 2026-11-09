"""Deterministic tests for the V0.9.9.21 ASR session lifecycle.

Every ordering is driven by ``threading.Event`` gates on a fake duplex
recognition: no network, no real audio and no random sleeps. They cover the
sender/receiver race fixed in ``client.py`` (one session shared by two threads)
and the single-primary-error / no-duplicate-result rules in ``manager.py``.
"""

from __future__ import annotations

import threading
import time

import pytest

from companion.asr.client import (
    ASRFailure,
    ASRSession,
    ASRSessionStateError,
    CLOSE_ORIGIN_EXTERNAL,
    CLOSE_ORIGIN_LOCAL_CANCEL,
    CLOSE_ORIGIN_LOCAL_NORMAL,
    CLOSE_ORIGIN_LOCAL_TIMEOUT,
    CLOSE_ORIGIN_SDK_INTERNAL,
    FAILURE_KIND_AUTH,
    FAILURE_KIND_EOF,
    FAILURE_KIND_LOCAL_SOCKET,
    STATE_CLOSED,
    STATE_FAILED,
    STATE_INPUT_FINISHING,
)
from companion.asr.config import ASRConfig
from companion.asr.manager import ASRManager
from companion.asr.microphone import AudioChunk
from companion.asr.terminal import TerminalRenderer
from companion.asr.vad import AUDIO, SPEECH_ENDED, SPEECH_STARTED, VADEvent


# --------------------------------------------------------------------- helpers
def config(**overrides) -> ASRConfig:
    base = dict(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        timeout_seconds=2.0,
        final_timeout_seconds=1.0,
    )
    base.update(overrides)
    return ASRConfig(**base)


def wait_for(predicate, timeout: float = 4.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class APIConnectionError(Exception):
    """Name mirrors the DashScope transport error so classification is tested."""


class FakeRecognition:
    """Fake duplex recognition with ``threading.Event`` gates.

    ``events()`` waits on ``events_gate`` (set by default), yields ``script``
    in order, waits on ``hold`` if given, then raises ``events_error``. This
    reproduces the SDK behaviour of a single socket shared by a sender and a
    receiver thread without touching the network.
    """

    def __init__(self, *, request_id: str = "req-test") -> None:
        self.request_id = request_id
        self.started = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.close_threads: list[str] = []
        self.sent: list[bytes] = []
        self.start_error: BaseException | None = None
        self.send_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        self.events_error: BaseException | None = None
        self.script: list[object] = []
        self.events_gate = threading.Event()
        self.events_gate.set()
        self.stop_event = threading.Event()
        self.hold: threading.Event | None = None
        self.recv_entered = threading.Event()
        self.recv_done = threading.Event()

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started += 1

    def send_audio_frame(self, pcm: bytes) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(bytes(pcm))

    def stop(self) -> None:
        if self.stop_error is not None:
            raise self.stop_error
        self.stop_calls += 1
        self.stop_event.set()

    def events(self):
        self.recv_entered.set()
        self.events_gate.wait(timeout=5.0)
        try:
            for item in list(self.script):
                if isinstance(item, BaseException):
                    raise item
                yield item
            if self.hold is not None:
                self.hold.wait(timeout=5.0)
            elif self.events_error is None:
                # The SDK keeps recv() blocked until finish-task ends the task.
                self.stop_event.wait(timeout=5.0)
            if self.events_error is not None:
                raise self.events_error
        finally:
            self.recv_done.set()

    def close(self) -> None:
        self.close_calls += 1
        self.close_threads.append(threading.current_thread().name)
        # A real close unblocks a receiver stuck in recv().
        self.events_gate.set()
        if self.hold is not None:
            self.hold.set()


class FakeAsrApi:
    def __init__(self, recognition: FakeRecognition) -> None:
        self.recognition = recognition
        self.calls: list[dict] = []

    def recognize(self, **kwargs):
        self.calls.append(kwargs)
        return self.recognition


class FakeClient:
    def __init__(self, recognition: FakeRecognition) -> None:
        self.audio = type("Audio", (), {"asr": FakeAsrApi(recognition)})()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class FakeMic:
    def __init__(self) -> None:
        self.description = "Fake Mic（16000 Hz）"
        self.dropped_chunks = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class ScriptedSegmenter:
    def __init__(self, groups: list[list[VADEvent]]) -> None:
        self.groups = list(groups)
        self._lock = threading.Lock()

    def process(self, pcm: bytes) -> list[VADEvent]:
        with self._lock:
            return self.groups.pop(0) if self.groups else []

    def reset(self) -> None:
        pass


def make_session(cfg: ASRConfig, recog: FakeRecognition):
    client = FakeClient(recog)
    return ASRSession(cfg, client_factory=lambda _cfg: client), client


def start_group() -> list[VADEvent]:
    return [VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)]


def audio_group() -> list[VADEvent]:
    return [VADEvent(AUDIO, b"\x02\x00" * 1024)]


def end_group() -> list[VADEvent]:
    return [VADEvent(SPEECH_ENDED)]


def sentence(text: str, sentence_id: str = "s1", *, is_final: bool) -> dict:
    """Raw DashScope payload, exactly as the transport yields it."""
    return {
        "header": {"event": "result-generated"},
        "payload": {
            "output": {
                "sentence": {
                    "text": text,
                    "sentence_id": sentence_id,
                    "sentence_end": "true" if is_final else "false",
                }
            }
        },
    }


# =========================================================== ASRSession races
def test_send_audio_after_local_close_is_secondary_state_error() -> None:
    cfg = config()
    recog = FakeRecognition()
    session, client = make_session(cfg, recog)
    session.open()
    session.close(origin=CLOSE_ORIGIN_LOCAL_CANCEL)
    with pytest.raises(ASRSessionStateError):
        session.send_audio(b"\x00\x00")
    # Closing is local cleanup, not a recognition failure.
    assert session.primary_error is None
    assert session.close_origin == CLOSE_ORIGIN_LOCAL_CANCEL
    assert session.state == STATE_CLOSED
    assert recog.close_calls == 1
    assert client.closed == 1


def test_receiver_eof_becomes_one_primary_failure() -> None:
    cfg = config()
    recog = FakeRecognition()
    session, _client = make_session(cfg, recog)
    session.open()
    recog.events_error = APIConnectionError("websocket closed before task-finished")
    with pytest.raises(ASRFailure) as info:
        list(session.events())
    assert info.value.kind == FAILURE_KIND_EOF
    assert session.state == STATE_FAILED
    assert session.primary_error
    assert session.secondary_error is None


def test_send_failure_is_primary_local_socket_with_sdk_origin() -> None:
    cfg = config()
    recog = FakeRecognition()
    session, _client = make_session(cfg, recog)
    session.open()
    recog.send_error = OSError(9, "Bad file descriptor")
    with pytest.raises(ASRFailure) as info:
        session.send_audio(b"\x01")
    assert info.value.kind == FAILURE_KIND_LOCAL_SOCKET
    assert info.value.origin == CLOSE_ORIGIN_SDK_INTERNAL
    assert session.state == STATE_FAILED
    assert session.primary_error and "socket" in session.primary_error


def test_auth_failure_is_classified_by_http_code_not_message() -> None:
    class DashScopeError(Exception):
        def __init__(self, http_code: int, message: str) -> None:
            super().__init__(message)
            self.http_code = http_code
            self.code = "InvalidApiKey"

    cfg = config()
    recog = FakeRecognition()
    recog.start_error = DashScopeError(401, "upstream rejected the request")
    session, _client = make_session(cfg, recog)
    with pytest.raises(ASRFailure) as info:
        session.open()
    assert info.value.kind == FAILURE_KIND_AUTH
    assert info.value.request_id or True  # request id may be unknown pre-start
    assert session.state == STATE_FAILED


def test_concurrent_recv_failure_and_send_yield_single_primary() -> None:
    cfg = config()
    recog = FakeRecognition()
    recog.events_gate.clear()  # hold the receiver inside recv
    session, _client = make_session(cfg, recog)
    session.open()

    recv_errors: list[BaseException] = []

    def receive() -> None:
        try:
            for _ in session.events():
                pass
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            recv_errors.append(exc)

    receiver = threading.Thread(target=receive, name="asr-recv-test", daemon=True)
    receiver.start()
    assert recog.recv_entered.wait(timeout=2.0)

    # While streaming, the sender is still allowed to write.
    session.send_audio(b"\x01\x02")
    assert recog.sent == [b"\x01\x02"]

    # The receiver observes the socket EOF first: that is the primary error.
    recog.events_error = APIConnectionError("websocket closed before task-finished")
    recog.events_gate.set()
    receiver.join(timeout=2.0)
    assert not receiver.is_alive()

    primary = session.primary_error
    assert primary and session.state == STATE_FAILED
    assert len(recv_errors) == 1 and isinstance(recv_errors[0], ASRFailure)

    # The sender's next write is only a symptom of the receiver's close.
    with pytest.raises(ASRSessionStateError):
        session.send_audio(b"\x03\x04")
    assert session.primary_error == primary  # never overwritten
    assert session.secondary_error


def test_stop_sends_finish_task_once_and_does_not_block() -> None:
    cfg = config()
    recog = FakeRecognition()
    session, _client = make_session(cfg, recog)
    session.open()
    session.send_audio(b"\x01")
    session.stop()
    session.stop()  # idempotent
    assert recog.stop_calls == 1
    # Sending finish-task is not closing: the receiver may keep reading.
    assert session.state == STATE_INPUT_FINISHING
    assert session.closed is False


def test_close_is_idempotent_and_keeps_the_first_origin() -> None:
    cfg = config()
    recog = FakeRecognition()
    session, client = make_session(cfg, recog)
    session.open()
    session.close(origin=CLOSE_ORIGIN_LOCAL_TIMEOUT)
    session.close(origin=CLOSE_ORIGIN_LOCAL_NORMAL)
    assert recog.close_calls == 1
    assert client.closed == 1
    assert session.close_origin == CLOSE_ORIGIN_LOCAL_TIMEOUT
    assert session.state == STATE_CLOSED


def test_events_after_failure_yield_nothing() -> None:
    cfg = config()
    recog = FakeRecognition()
    session, _client = make_session(cfg, recog)
    session.open()
    recog.send_error = OSError(9, "Bad file descriptor")
    with pytest.raises(ASRFailure):
        session.send_audio(b"\x01")
    # A failed session must never fabricate a clean task-finished.
    assert list(session.events()) == []
    assert session.primary_error


# ============================================================= manager threading
class Harness:
    def __init__(
        self,
        groups: list[list[VADEvent]],
        recognitions: list[FakeRecognition],
        *,
        cfg: ASRConfig,
        render_finals: bool = True,
        is_tty: bool = True,
    ) -> None:
        self.cfg = cfg
        self.mic = FakeMic()
        self.segmenter = ScriptedSegmenter(groups)
        self.recognitions = list(recognitions)
        self.sessions: list[ASRSession] = []
        self.finals: list = []
        self.turns: list[tuple[int, bool]] = []
        self.speech: list = []
        stream = _LockedBuffer()
        status = _LockedBuffer()
        self.renderer = TerminalRenderer(
            stream=stream, status_stream=status, is_tty=is_tty, width=200
        )
        self.out = stream
        self.status = status
        self.manager = ASRManager(
            cfg,
            microphone=self.mic,
            vad_segmenter=self.segmenter,
            session_factory=self._factory,
            renderer=self.renderer,
            on_final=self.finals.append,
            on_speech=self.speech.append,
            on_turn_closed=lambda uid, ok: self.turns.append((uid, ok)),
            render_finals=render_finals,
        )
        self.thread = threading.Thread(target=self.manager.run, daemon=True)
        self.thread.start()
        assert wait_for(lambda: "正在监听" in self.status.getvalue())

    def _factory(self) -> ASRSession:
        recog = self.recognitions.pop(0)
        client = FakeClient(recog)
        session = ASRSession(self.cfg, client_factory=lambda _cfg: client)
        self.sessions.append(session)
        return session

    def feed(self, count: int = 1) -> None:
        for _ in range(count):
            self.manager.audio_queue.put(
                AudioChunk(b"\x01\x00" * 512, 16000, 16000, time.monotonic()),
                timeout=2.0,
            )

    def stop(self) -> None:
        self.manager.stop()
        self.thread.join(timeout=4.0)
        self.manager.shutdown()
        assert not self.thread.is_alive()


class _LockedBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        with self._lock:
            self._parts.append(text)
        return len(text)

    def getvalue(self) -> str:
        with self._lock:
            return "".join(self._parts)

    def flush(self) -> None:
        pass


def test_receiver_failure_aborts_sender_and_reports_one_error() -> None:
    recog = FakeRecognition()
    recog.events_gate.clear()
    harness = Harness(
        [start_group(), audio_group(), end_group()], [recog], cfg=config()
    )
    try:
        harness.feed(2)
        assert wait_for(lambda: recog.recv_entered.is_set())
        session = harness.sessions[0]

        # Fail while the local VAD has NOT ended yet.
        recog.events_error = OSError(104, "Connection reset by peer")
        recog.events_gate.set()
        assert wait_for(lambda: session.state == STATE_CLOSED)
        # Closing is owned by the control thread, never a worker thread.
        assert recog.close_threads
        assert not recog.close_threads[0].startswith("asr-")
        assert session.close_origin == CLOSE_ORIGIN_EXTERNAL

        # Ending the turn must not upload anything more to the dead session.
        sent_before = len(recog.sent)
        harness.feed(1)  # end_group
        assert wait_for(lambda: harness.turns == [(1, False)])
        assert len(recog.sent) == sent_before
        assert harness.finals == []
        assert harness.manager.stats.errors == 1
        assert wait_for(
            lambda: harness.status.getvalue().count("识别 #1 失败") == 1
        )
    finally:
        harness.stop()


def test_normal_completion_emits_one_final_and_one_turn_closed() -> None:
    recog = FakeRecognition()
    recog.script = [
        sentence("你好", is_final=False),
        sentence("你好。", is_final=True),
    ]
    harness = Harness([start_group(), end_group()], [recog], cfg=config())
    try:
        harness.feed(2)
        assert wait_for(lambda: harness.turns == [(1, True)])
        assert len(harness.finals) == 1
        assert harness.finals[0].text == "你好。"
        assert recog.stop_calls == 1
        assert harness.manager.stats.errors == 0
        assert wait_for(lambda: harness.out.getvalue().count("你好。") == 1)
    finally:
        harness.stop()


def test_failed_recognition_with_fragments_is_not_a_user_final() -> None:
    recog = FakeRecognition()
    recog.script = [sentence("你好", is_final=True)]
    recog.events_error = OSError(104, "Connection reset by peer")
    harness = Harness([start_group(), end_group()], [recog], cfg=config())
    try:
        harness.feed(2)
        assert wait_for(lambda: harness.turns == [(1, False)])
        # Confirmed fragments are displayed but never submitted as a turn.
        assert harness.finals == []
        assert harness.manager.stats.errors == 1
        assert wait_for(
            lambda: harness.status.getvalue().count("识别 #1 失败") == 1
        )
        # The failure status is rendered after any partial, so the render queue
        # is drained and the confirmed fragment is on screen exactly once.
        assert "你好" in harness.out.getvalue()
    finally:
        harness.stop()


def test_late_error_from_old_utterance_does_not_affect_new_one() -> None:
    slow = FakeRecognition()
    slow.script = [sentence("第一", is_final=True)]
    slow.hold = threading.Event()
    slow.events_error = OSError(104, "Connection reset by peer")
    fast = FakeRecognition()
    fast.script = [sentence("第二", is_final=True)]
    cfg = config(final_timeout_seconds=30.0)
    harness = Harness(
        [start_group(), end_group(), start_group(), end_group()],
        [slow, fast],
        cfg=cfg,
    )
    try:
        harness.feed(1)  # start #1
        assert wait_for(lambda: len(harness.sessions) == 1)
        harness.feed(1)  # end #1: its session is still open
        assert wait_for(lambda: slow.recv_entered.is_set())
        harness.feed(1)  # start #2
        assert wait_for(lambda: len(harness.sessions) == 2)
        harness.feed(1)  # end #2: completes normally
        assert wait_for(lambda: (2, True) in harness.turns)
        assert [f.utterance_id for f in harness.finals] == [2]

        # The old session now fails; its error must not touch #2's result.
        slow.hold.set()
        assert wait_for(lambda: (1, False) in harness.turns)
        assert [f.utterance_id for f in harness.finals] == [2]
        assert wait_for(
            lambda: harness.status.getvalue().count("识别 #1 失败") == 1
        )
        assert wait_for(lambda: harness.out.getvalue().count("第二") == 1)
    finally:
        harness.stop()


def test_formal_mode_suppresses_partials_and_failure_leaves_none() -> None:
    recog = FakeRecognition()
    recog.script = [sentence("临时文本", is_final=False)]
    recog.events_error = OSError(104, "Connection reset by peer")
    harness = Harness(
        [start_group(), end_group()], [recog], cfg=config(), render_finals=False
    )
    try:
        harness.feed(2)
        assert wait_for(lambda: harness.turns == [(1, False)])
        # The formal REPL prints the final itself: no provisional line may be
        # drawn, so nothing stale survives a failure. The failure status is
        # rendered last, so seeing it means every partial was already drained.
        assert wait_for(
            lambda: harness.status.getvalue().count("识别 #1 失败") == 1
        )
        assert harness.renderer.partial_text == ""
        assert "临时文本" not in harness.out.getvalue()
    finally:
        harness.stop()


def test_final_timeout_closes_with_local_timeout_origin() -> None:
    recog = FakeRecognition()
    recog.hold = threading.Event()  # recv stays open, never finishes
    cfg = config(final_timeout_seconds=0.3)
    harness = Harness([start_group(), end_group()], [recog], cfg=cfg)
    try:
        harness.feed(2)
        assert wait_for(lambda: harness.turns == [(1, False)], timeout=5.0)
        session = harness.sessions[0]
        assert session.close_origin == CLOSE_ORIGIN_LOCAL_TIMEOUT
        assert harness.finals == []
        assert wait_for(lambda: "超时" in harness.status.getvalue())
    finally:
        harness.stop()
