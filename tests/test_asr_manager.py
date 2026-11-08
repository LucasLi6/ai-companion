"""Offline tests for the ASR manager, client wiring and microphone helpers.

Fake microphone, fake VAD segmenter and fake duplex ASR sessions make every
test deterministic and network-free. The real PyAudio/ONNX/DashScope objects
are never constructed here.
"""

from __future__ import annotations

import io
import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from companion.asr.client import (
    ASRClientError,
    ASREvent,
    ASRSession,
    parse_event,
)
from companion.asr.config import ASRConfig, VAD_WINDOW_SAMPLES
from companion.asr.manager import (
    ASRManager,
    _FinalTimeout,
    _Sentence,
    _SessionDone,
    _Utterance,
)
from companion.asr.microphone import (
    AudioChunk,
    AudioDevice,
    AudioInterrupted,
    AudioNotice,
    MicrophoneCapture,
    MicrophoneError,
    StreamingResampler,
    choose_capture_rate,
    list_input_devices,
    resolve_input_device,
)
from companion.asr.terminal import CLEAR_LINE, TerminalRenderer
from companion.asr.vad import (
    AUDIO,
    SPEECH_ENDED,
    SPEECH_STARTED,
    VADEvent,
)


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


class LockedBuffer(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def write(self, text: str) -> int:  # type: ignore[override]
        with self._lock:
            return super().write(text)

    def getvalue(self) -> str:  # type: ignore[override]
        with self._lock:
            return super().getvalue()


def make_renderer(*, is_tty: bool = False, width: int = 200):
    out = LockedBuffer()
    status = LockedBuffer()
    renderer = TerminalRenderer(
        stream=out,
        status_stream=status,
        is_tty=is_tty,
        width=width,
    )
    return renderer, out, status


class FakeMic:
    def __init__(self) -> None:
        self.description = "Fake Mic（16000 Hz）"
        self.dropped_chunks = 0
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class ScriptedSegmenter:
    """Return one pre-built event group per audio chunk."""

    def __init__(self, groups: list[list[VADEvent]]) -> None:
        self.groups = list(groups)
        self.reset_calls = 0
        self._lock = threading.Lock()

    def process(self, pcm: bytes) -> list[VADEvent]:
        with self._lock:
            if self.groups:
                return self.groups.pop(0)
            return []

    def reset(self) -> None:
        with self._lock:
            self.reset_calls += 1


@dataclass
class SessionScript:
    partials: list[tuple[str, str]] = field(default_factory=list)
    finals: list[tuple[str, str]] = field(default_factory=list)
    fail_on_open: bool = False
    fail_events: bool = False
    finals_before_stop: bool = False
    never_finish: bool = False
    finish_immediately: bool = False
    send_delay: float = 0.0


class FakeSession:
    def __init__(self, script: SessionScript) -> None:
        self.script = script
        self.sent: list[bytes] = []
        self.stop_calls = 0
        self.close_calls = 0
        self.opened = False
        self._stop_event = threading.Event()
        self._release = threading.Event()
        self._lock = threading.Lock()

    def open(self) -> None:
        if self.script.fail_on_open:
            raise ASRClientError("模拟连接失败")
        self.opened = True

    def send_audio(self, pcm: bytes) -> None:
        if self.script.send_delay:
            time.sleep(self.script.send_delay)
        with self._lock:
            self.sent.append(bytes(pcm))

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop_event.set()

    def events(self):
        if self.script.fail_events:
            raise ASRClientError("模拟接收失败")
        for text, sentence_id in self.script.partials:
            yield ASREvent("sentence", text, sentence_id, False)
        if self.script.finish_immediately:
            # The server task ends on its own while the local VAD is still
            # active (no stop() call).
            for text, sentence_id in self.script.finals:
                yield ASREvent("sentence", text, sentence_id, True)
            return
        if self.script.finals_before_stop:
            for text, sentence_id in self.script.finals:
                yield ASREvent("sentence", text, sentence_id, True)
        self._stop_event.wait(timeout=3.0)
        if not self.script.finals_before_stop:
            for text, sentence_id in self.script.finals:
                yield ASREvent("sentence", text, sentence_id, True)
        if self.script.never_finish:
            self._release.wait(timeout=5.0)

    def close(self) -> None:
        self.close_calls += 1
        self._stop_event.set()
        self._release.set()

    @property
    def sent_bytes(self) -> int:
        with self._lock:
            return sum(len(item) for item in self.sent)


class SessionFactory:
    def __init__(self, scripts: list[SessionScript]) -> None:
        self.scripts = list(scripts)
        self.created: list[FakeSession] = []
        self._lock = threading.Lock()

    def __call__(self) -> FakeSession:
        with self._lock:
            script = self.scripts.pop(0) if self.scripts else SessionScript()
            session = FakeSession(script)
            self.created.append(session)
            return session


class RunningManager:
    def __init__(
        self,
        *,
        groups: list[list[VADEvent]],
        scripts: list[SessionScript],
        renderer: TerminalRenderer,
        cfg: ASRConfig | None = None,
    ) -> None:
        self.config = cfg or config()
        self.mic = FakeMic()
        self.segmenter = ScriptedSegmenter(groups)
        self.factory = SessionFactory(scripts)
        self.renderer = renderer
        self.manager = ASRManager(
            self.config,
            microphone=self.mic,
            vad_segmenter=self.segmenter,
            session_factory=self.factory,
            renderer=renderer,
        )
        self.thread = threading.Thread(
            target=self.manager.run, daemon=True
        )
        self.thread.start()

    def feed(self, count: int = 1, pcm: bytes = b"\x01\x00" * 512) -> None:
        for _ in range(count):
            self.manager.audio_queue.put(
                AudioChunk(pcm, 16000, 16000, time.monotonic()),
                timeout=2.0,
            )

    def stop(self) -> None:
        self.manager.stop()
        self.thread.join(timeout=4.0)
        self.manager.shutdown()
        assert not self.thread.is_alive()


def start(
    groups: list[list[VADEvent]],
    scripts: list[SessionScript],
    *,
    renderer=None,
    cfg: ASRConfig | None = None,
    status: LockedBuffer | None = None,
) -> RunningManager:
    if renderer is None:
        renderer, _out, status = make_renderer()
    running = RunningManager(
        groups=groups, scripts=scripts, renderer=renderer, cfg=cfg
    )
    assert status is not None
    assert wait_for(lambda: "正在监听" in status.getvalue())
    return running


def start_group() -> list[VADEvent]:
    return [VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)]


def audio_group() -> list[VADEvent]:
    return [VADEvent(AUDIO, b"\x02\x00" * 1024)]


def end_group() -> list[VADEvent]:
    return [VADEvent(SPEECH_ENDED)]


# --------------------------------------------------------------- manager flows
def test_startup_status_shows_device_model_and_listening() -> None:
    renderer, _out, status = make_renderer()
    running = start([], [], renderer=renderer, status=status)
    try:
        text = status.getvalue()
        assert "Fake Mic" in text
        assert running.config.model in text
        assert "正在监听" in text
    finally:
        running.stop()


def test_partials_replace_each_other_and_final_commits_once() -> None:
    renderer, out, status = make_renderer(is_tty=True, width=200)
    running = start(
        [start_group(), end_group()],
        [
            SessionScript(
                partials=[("你", "s1"), ("你好", "s1"), ("你好世", "s1")],
                finals=[("你好世界", "s1")],
            )
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(1)
        assert wait_for(lambda: renderer.partial_text == "你好世")
        running.feed(1)
        assert wait_for(lambda: renderer.confirmed == ["你好世界"])
        assert out.getvalue().count("你好世界") == 1
        assert renderer.partial_text == ""
    finally:
        running.stop()


def test_final_is_plain_line_on_non_tty() -> None:
    renderer, out, status = make_renderer(is_tty=False)
    running = start(
        [start_group(), end_group()],
        [SessionScript(partials=[("临时", "s1")], finals=[("最终", "s1")])],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(2)
        assert wait_for(lambda: renderer.confirmed == ["最终"])
        assert out.getvalue() == "最终\n"
    finally:
        running.stop()


def test_multiple_utterances_are_ordered_and_not_duplicated() -> None:
    renderer, _out, status = make_renderer()
    running = start(
        [start_group(), end_group(), start_group(), end_group()],
        [
            SessionScript(partials=[("第一", "s1")], finals=[("第一段", "s1")]),
            SessionScript(partials=[("第二", "s1")], finals=[("第二段", "s1")]),
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(4)
        assert wait_for(lambda: renderer.confirmed == ["第一段", "第二段"])
        assert len(running.factory.created) == 2
        assert all(session.stop_calls == 1 for session in running.factory.created)
        assert running.manager.stats.finalized == 2
    finally:
        running.stop()


def test_empty_asr_result_writes_no_blank_transcript_line() -> None:
    renderer, out, status = make_renderer(is_tty=False)
    running = start(
        [start_group(), end_group()],
        [SessionScript(partials=[("临时的", "s1")], finals=[])],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(2)
        assert wait_for(
            lambda: "没有可确认的最终文本" in status.getvalue()
        )
        assert out.getvalue() == ""
        assert renderer.confirmed == []
        assert running.manager.stats.empty == 1
    finally:
        running.stop()


def test_connect_failure_reports_and_keeps_listening() -> None:
    renderer, _out, status = make_renderer()
    running = start(
        [start_group(), end_group(), start_group(), end_group()],
        [
            SessionScript(fail_on_open=True),
            SessionScript(partials=[("恢复", "s1")], finals=[("恢复成功", "s1")]),
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(4)
        assert wait_for(lambda: "识别 #1 失败" in status.getvalue())
        assert wait_for(lambda: renderer.confirmed == ["恢复成功"])
        assert running.manager.stats.errors >= 1
        assert running.manager.stats.finalized == 1
    finally:
        running.stop()


def test_receiver_failure_reports_and_next_utterance_works() -> None:
    renderer, _out, status = make_renderer()
    running = start(
        [start_group(), end_group(), start_group(), end_group()],
        [
            SessionScript(fail_events=True),
            SessionScript(finals=[("下一段", "s1")]),
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(4)
        assert wait_for(lambda: "识别 #1 失败" in status.getvalue())
        assert wait_for(lambda: renderer.confirmed == ["下一段"])
    finally:
        running.stop()


def test_final_timeout_aborts_and_recovers() -> None:
    renderer, _out, status = make_renderer()
    running = start(
        [start_group(), end_group(), start_group(), end_group()],
        [
            SessionScript(
                partials=[("临时", "s1")],
                finals=[("已确认", "s1")],
                finals_before_stop=True,
                never_finish=True,
            ),
            SessionScript(finals=[("第二段", "s1")]),
        ],
        renderer=renderer,
        cfg=config(final_timeout_seconds=0.2),
        status=status,
    )
    try:
        running.feed(2)
        assert wait_for(lambda: "超时" in status.getvalue())
        assert renderer.confirmed == ["已确认"]
        assert running.manager.stats.errors == 1
        running.feed(2)
        assert wait_for(lambda: renderer.confirmed == ["已确认", "第二段"])
    finally:
        running.stop()


def test_microphone_interrupt_aborts_partial_and_recovers() -> None:
    renderer, _out, status = make_renderer(is_tty=True, width=200)
    running = start(
        [start_group(), start_group(), end_group()],
        [
            SessionScript(partials=[("临时", "s1")], finals=[]),
            SessionScript(finals=[("恢复成功", "s1")]),
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(1)
        assert wait_for(lambda: renderer.partial_text == "临时")
        running.manager.audio_queue.put(
            AudioInterrupted("设备断开", time.monotonic()), timeout=2.0
        )
        assert wait_for(lambda: "录音中断" in status.getvalue())
        assert wait_for(lambda: renderer.partial_text == "")
        assert renderer.confirmed == []
        assert running.segmenter.reset_calls >= 1
        running.feed(1)  # start of the next utterance
        running.feed(1)  # end of the next utterance
        assert wait_for(lambda: renderer.confirmed == ["恢复成功"])
    finally:
        running.stop()


def test_asr_sentence_end_does_not_stop_the_vad_utterance() -> None:
    renderer, out, status = make_renderer(is_tty=True, width=200)
    running = start(
        [start_group(), audio_group(), audio_group(), end_group()],
        [
            SessionScript(
                partials=[("先确", "s1")],
                finals=[("先确认", "s1"), ("先确认", "s1")],
                finals_before_stop=True,
            )
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(3)
        assert wait_for(lambda: len(running.factory.created) == 1)
        session = running.factory.created[0]
        assert wait_for(lambda: session.sent_bytes > 0)
        # sentence_end confirms a fragment but neither stops audio nor prints
        # a newline before the local VAD ends.
        assert session.stop_calls == 0
        assert renderer.confirmed == []
        assert out.getvalue().count("\n") == 0
        running.feed(1)
        assert wait_for(lambda: session.stop_calls == 1)
        assert wait_for(lambda: renderer.confirmed == ["先确认"])
        # Duplicate sentence_end events and the partial must not add lines.
        assert out.getvalue().count("先确认\n") == 1
        assert out.getvalue().count("\n") == 1
        assert running.manager.stats.finalized == 1
    finally:
        running.stop()


def test_one_turn_with_two_server_sentences_merges_into_one_line() -> None:
    renderer, out, status = make_renderer(is_tty=False)
    running = start(
        [start_group(), audio_group(), end_group()],
        [
            SessionScript(
                partials=[("你可能", "s1"), ("你可能", "s1")],
                finals=[("你好，", "s1"), ("世界。", "s2")],
                finals_before_stop=True,
            )
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(3)
        assert wait_for(lambda: renderer.confirmed == ["你好，世界。"])
        assert out.getvalue() == "你好，世界。\n"
    finally:
        running.stop()


def test_english_sentences_keep_single_space_when_merged() -> None:
    from companion.asr.manager import _merge_turn_text

    assert _merge_turn_text(["Hello.", "World."]) == "Hello. World."
    assert _merge_turn_text(["你好，", "世界。"]) == "你好，世界。"
    assert _merge_turn_text(["Olá.", "Tudo bem?"]) == "Olá. Tudo bem?"
    assert _merge_turn_text(["你好", "world"]) == "你好world"
    assert _merge_turn_text(["", "  ", None]) == ""
    assert _merge_turn_text(["第一段。", "Second."]) == "第一段。Second."


def test_session_done_before_vad_end_waits_for_the_turn_to_finish() -> None:
    renderer, out, status = make_renderer(is_tty=False)
    running = start(
        [start_group(), audio_group(), end_group()],
        [
            SessionScript(
                partials=[("你好", "s1")],
                finals=[("你好。", "s1")],
                finish_immediately=True,
            )
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(2)
        assert wait_for(lambda: len(running.factory.created) == 1)
        session = running.factory.created[0]
        assert wait_for(lambda: session.close_calls >= 1)
        # The server session is done, but the user is still speaking: no
        # final line may be printed yet.
        time.sleep(0.2)
        assert renderer.confirmed == []
        assert out.getvalue() == ""
        running.feed(1)
        assert wait_for(lambda: renderer.confirmed == ["你好。"])
        assert out.getvalue() == "你好。\n"
    finally:
        running.stop()


def test_two_sessions_finishing_out_of_order_keep_partials_intact() -> None:
    renderer, out, status = make_renderer(is_tty=True, width=200)
    running = start(
        [start_group(), end_group(), start_group(), end_group()],
        [
            SessionScript(
                partials=[("第一临时", "s1")],
                finals=[("第一最终", "s1")],
                finals_before_stop=True,
                never_finish=True,
            ),
            SessionScript(
                partials=[("第二临时", "s1")],
                finals=[("第二最终", "s1")],
                finals_before_stop=True,
            ),
        ],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(1)  # start utterance #1 (slow)
        assert wait_for(lambda: len(running.factory.created) == 1)
        slow = running.factory.created[0]
        running.feed(1)  # end #1: its session stays open for now
        assert wait_for(lambda: slow.stop_calls == 1)
        running.feed(1)  # start utterance #2 while #1 is still open
        assert wait_for(lambda: len(running.factory.created) == 2)
        fast = running.factory.created[1]
        assert wait_for(lambda: renderer.partial_text == "第二临时")
        running.feed(1)  # end #2: it completes first
        assert wait_for(lambda: fast.stop_calls == 1)
        assert wait_for(lambda: renderer.confirmed == ["第二最终"])
        # Utterance #1 is still open; its late completion must not clobber
        # anything already printed for #2.
        slow._release.set()
        assert wait_for(lambda: renderer.confirmed == ["第二最终", "第一最终"])
        assert out.getvalue().count("第二最终\n") == 1
        assert out.getvalue().count("第一最终\n") == 1
    finally:
        running.stop()


def test_final_timeout_starts_only_after_audio_upload_finishes() -> None:
    """A queued or still-uploading session must never trip the final timer.

    The sender is made slow on purpose: the old code armed the timer at local
    VAD end, so the timeout fired while audio was still being uploaded and
    while the second utterance was still queued.
    """
    renderer, _out, status = make_renderer()
    big_audio = b"\x01\x00" * (VAD_WINDOW_SAMPLES * 20)
    groups = [
        [VADEvent(SPEECH_STARTED, big_audio)],
        [VADEvent(SPEECH_ENDED)],
        [VADEvent(SPEECH_STARTED, b"\x01\x00" * VAD_WINDOW_SAMPLES)],
        [VADEvent(SPEECH_ENDED)],
    ]
    running = start(
        groups,
        [
            SessionScript(finals=[("第一最终", "s1")], send_delay=0.05),
            SessionScript(finals=[("第二最终", "s1")], send_delay=0.05),
        ],
        renderer=renderer,
        cfg=config(
            max_concurrent_sessions=1,
            final_timeout_seconds=0.2,
            timeout_seconds=3.0,
        ),
        status=status,
    )
    try:
        running.feed(4)
        assert wait_for(
            lambda: renderer.confirmed == ["第一最终", "第二最终"],
            timeout=10.0,
        )
        assert "超时" not in status.getvalue()
        assert running.manager.stats.errors == 0
        assert running.manager.stats.finalized == 2
        sessions = running.factory.created
        assert len(sessions) == 2
        assert all(session.stop_calls == 1 for session in sessions)
    finally:
        running.stop()


def test_one_thousand_consecutive_turns_clean_up_state() -> None:
    renderer, out, status = make_renderer()
    scripts = [
        SessionScript(finals=[(f"第{index}段", "s1")])
        for index in range(1, 1001)
    ]
    groups: list[list[VADEvent]] = []
    for _ in range(1000):
        groups.append(start_group())
        groups.append(end_group())
    running = start(
        groups,
        scripts,
        renderer=renderer,
        cfg=config(final_timeout_seconds=2.0, max_concurrent_sessions=4),
        status=status,
    )
    try:
        running.feed(2000)
        assert wait_for(
            lambda: len(renderer.confirmed) == 1000, timeout=30.0,
            interval=0.02,
        )
        assert wait_for(
            lambda: not running.manager._utterances
            and not running.manager._open_order
            and not running.manager._threads,
            timeout=10.0,
            interval=0.02,
        )
        assert renderer.confirmed[0] == "第1段"
        assert renderer.confirmed[-1] == "第1000段"
        assert len(set(renderer.confirmed)) == 1000
        assert running.manager.stats.utterances == 1000
        assert running.manager.stats.finalized == 1000
        assert out.getvalue().count("\n") == 1000
    finally:
        running.stop()


def test_late_result_for_finalized_utterance_is_ignored() -> None:
    renderer, out, _status = make_renderer()
    manager = ASRManager(
        config(),
        microphone=FakeMic(),
        vad_segmenter=ScriptedSegmenter([]),
        renderer=renderer,
    )
    utterance = _Utterance(1, 8)
    utterance.finalized = True
    manager._utterances[1] = utterance
    manager._handle(_Sentence(1, "迟到文本", "s1", True, time.monotonic()))
    assert renderer.confirmed == []
    assert out.getvalue() == ""


def test_late_events_after_cleanup_are_safely_ignored() -> None:
    renderer, out, status = make_renderer()
    running = start(
        [start_group(), end_group()],
        [SessionScript(finals=[("完成", "s1")])],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(2)
        assert wait_for(lambda: renderer.confirmed == ["完成"])
        manager = running.manager
        assert wait_for(lambda: manager._utterances == {})
        manager._handle(_Sentence(1, "迟到文本", "s9", True, time.monotonic()))
        manager._handle(_Sentence(1, "迟到部分", "s9", False, time.monotonic()))
        manager._handle(_SessionDone(1, "迟到错误", time.monotonic()))
        manager._handle(_FinalTimeout(1))
        assert renderer.confirmed == ["完成"]
        assert "迟到" not in out.getvalue()
        assert "迟到" not in status.getvalue()
        assert manager._utterances == {}
        assert manager._open_order == []
    finally:
        running.stop()


def test_shutdown_aborts_unfinished_utterance_without_commit() -> None:
    renderer, _out, status = make_renderer(is_tty=True, width=200)
    running = start(
        [start_group()],
        [SessionScript(partials=[("未完成", "s1")], never_finish=True)],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(1)
        assert wait_for(lambda: renderer.partial_text == "未完成")
    finally:
        running.stop()
    assert renderer.confirmed == []
    assert renderer.partial_text == ""
    assert running.manager.stats.aborted == 1
    leftovers = [
        thread for thread in threading.enumerate()
        if thread.name.startswith("asr-")
    ]
    assert leftovers == []


def test_shutdown_leaves_no_asr_threads_running() -> None:
    renderer, _out, status = make_renderer()
    running = start(
        [start_group(), end_group()],
        [SessionScript(finals=[("结束", "s1")])],
        renderer=renderer,
        status=status,
    )
    try:
        running.feed(2)
        assert wait_for(lambda: renderer.confirmed == ["结束"])
    finally:
        running.stop()
    leftovers = [
        thread for thread in threading.enumerate()
        if thread.name.startswith("asr-")
    ]
    assert leftovers == []


# ------------------------------------------------------- client parsing / wiring
def _sentence(text: str, *, sid: str = "", final=None, heartbeat=None, end_time=None):
    sentence = {"text": text, "sentence_id": sid}
    if final is not None:
        sentence["sentence_end"] = final
    if heartbeat is not None:
        sentence["heartbeat"] = heartbeat
    if end_time is not None:
        sentence["end_time"] = end_time
    return {
        "header": {"event": "result-generated"},
        "payload": {"output": {"sentence": sentence}},
    }


def test_parse_event_partial_and_all_final_markers() -> None:
    partial = parse_event(_sentence("你好", sid="s1"))
    assert partial is not None
    assert (partial.text, partial.sentence_id, partial.is_final) == (
        "你好", "s1", False
    )
    explicit = parse_event(_sentence("你好。", sid="s1", final=True))
    assert explicit.is_final is True
    legacy = parse_event(_sentence("你好。", sid="s1", end_time=1234))
    assert legacy.is_final is True
    string_false = parse_event(_sentence("你好", sid="s1", final="false"))
    assert string_false.is_final is False


def test_parse_event_ignores_heartbeats_textless_and_malformed() -> None:
    assert parse_event(_sentence("", sid="s1")) is None
    assert parse_event(_sentence("心跳", heartbeat=True)) is None
    assert parse_event({"header": {"event": "task-finished"}, "payload": {}}) is None
    assert parse_event({"payload": {"output": {"sentence": []}}}) is None
    assert parse_event(None) is None


def test_parse_event_task_failure_is_an_error_event() -> None:
    event = parse_event({"header": {"event": "task-failed"}, "payload": {}})
    assert event is not None
    assert event.kind == "error"
    assert "失败" in event.error


class FakeRecognition:
    def __init__(self, raw_events: list[dict]) -> None:
        self.raw_events = raw_events
        self.started = False
        self.sent: list[bytes] = []
        self.stop_calls = 0
        self.close_calls = 0
        self.request_id = "request-1"

    def start(self):
        self.started = True
        return self

    def send_audio_frame(self, pcm: bytes) -> None:
        self.sent.append(bytes(pcm))

    def stop(self) -> None:
        self.stop_calls += 1

    def events(self):
        yield from self.raw_events

    def close(self) -> None:
        self.close_calls += 1


class FakeAsrApi:
    def __init__(self, recognition: FakeRecognition) -> None:
        self.recognition = recognition
        self.calls: list[dict] = []

    def recognize(self, **kwargs):
        self.calls.append(kwargs)
        return self.recognition

    def recognize_file(self, *args, **kwargs):  # pragma: no cover - must not be used
        raise AssertionError("recognize_file must never be used")


class FakeClient:
    def __init__(self, api: FakeAsrApi) -> None:
        self.audio = SimpleNamespace(asr=api)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_session_uses_streaming_recognize_with_verified_parameters() -> None:
    recognition = FakeRecognition([_sentence("你好", sid="s1")])
    api = FakeAsrApi(recognition)
    client = FakeClient(api)
    session = ASRSession(config(), client_factory=lambda _cfg: client)
    session.open()
    assert recognition.started is True
    assert api.calls, "client.audio.asr.recognize must be called"
    call = api.calls[0]
    assert call["model"] == config().model
    assert call["format"] == "pcm"
    assert call["sample_rate"] == 16000
    assert call["disfluency_removal_enabled"] is False
    assert call["parameters"] == {
        "intermediate_result_enabled": True,
        "max_sentence_silence": 1500,
    }
    session.send_audio(b"\x01\x02")
    assert recognition.sent == [b"\x01\x02"]
    events = list(session.events())
    assert [(e.text, e.is_final) for e in events] == [("你好", False)]
    session.stop()
    assert recognition.stop_calls == 1
    session.close()
    assert recognition.close_calls == 1
    assert client.closed == 1
    assert session.closed is True


def test_session_send_after_close_raises_safe_error() -> None:
    session = ASRSession(config(), client_factory=lambda _cfg: FakeClient(
        FakeAsrApi(FakeRecognition([]))
    ))
    session.close()
    with pytest.raises(ASRClientError):
        session.send_audio(b"\x00\x00")


# ------------------------------------------------------------ microphone helpers
class FakeStream:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.reads = 0
        self.closed = 0
        self.stopped = 0

    def read(self, frames: int, exception_on_overflow: bool = True) -> bytes:
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise OSError("[Errno -9996] Invalid input device")
        time.sleep(0.001)
        return b"\x01\x00" * max(1, frames)

    def stop_stream(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1


class FakePyAudio:
    def __init__(
        self,
        *,
        supported_rates=(16000, 48000),
        rates_by_device: dict[int, set[int]] | None = None,
        streams: list[FakeStream] | None = None,
        default_index: int = 0,
    ) -> None:
        self.supported_rates = set(supported_rates)
        self.rates_by_device = rates_by_device
        self.streams = list(streams or [])
        self.default_index = default_index
        self.opened: list[dict] = []
        self.terminated = 0
        self.devices = {
            0: {
                "index": 0,
                "name": "Built-in Mic",
                "maxInputChannels": 1,
                "maxOutputChannels": 0,
                "defaultSampleRate": 48000.0,
            },
            1: {
                "index": 1,
                "name": "Bluetooth Input",
                "maxInputChannels": 1,
                "maxOutputChannels": 0,
                "defaultSampleRate": 16000.0,
            },
            2: {
                "index": 2,
                "name": "Speakers",
                "maxInputChannels": 0,
                "maxOutputChannels": 2,
                "defaultSampleRate": 48000.0,
            },
        }

    def get_device_count(self) -> int:
        return len(self.devices)

    def get_device_info_by_index(self, index: int) -> dict:
        return self.devices[index]

    def get_default_input_device_info(self) -> dict:
        return self.devices[self.default_index]

    def get_format_from_width(self, width: int) -> int:
        assert width == 2
        return 8

    def is_format_supported(
        self, rate, input_device=None, input_channels=None, input_format=None
    ) -> bool:
        if self.rates_by_device is not None:
            return rate in self.rates_by_device.get(int(input_device), set())
        return rate in self.supported_rates

    def open(self, **kwargs):
        self.opened.append(kwargs)
        if self.streams:
            return self.streams.pop(0)
        return FakeStream()

    def terminate(self) -> None:
        self.terminated += 1


def test_resampler_passthrough_and_48k_to_16k() -> None:
    same = StreamingResampler(16000)
    pcm = b"\x01\x00" * 100
    assert same.process(pcm) == pcm

    resampler = StreamingResampler(48000)
    out = b"".join(
        resampler.process(b"\x01\x00" * 4800)
        for _ in range(1)
    )
    assert abs(len(out) // 2 - 1600) <= 2


def test_pure_python_resampler_is_stateful_across_chunks() -> None:
    resampler = StreamingResampler(48000)
    resampler._audioop = None  # exercise the Python 3.13+ fallback
    pieces = [
        resampler.process(b"\x01\x00" * 1000),
        resampler.process(b"\x01\x00" * 1000),
        resampler.process(b"\x01\x00" * 1000),
    ]
    total_samples = sum(len(piece) // 2 for piece in pieces)
    assert abs(total_samples - 1000) <= 3
    assert all(len(piece) % 2 == 0 for piece in pieces)


def test_choose_capture_rate_prefers_16k_then_falls_back() -> None:
    device = AudioDevice(0, "Mic", 1, 0, 48000, True)
    assert choose_capture_rate(FakePyAudio(supported_rates={16000, 48000}), device) == 16000
    assert choose_capture_rate(FakePyAudio(supported_rates={48000}), device) == 48000


def test_device_resolution_by_index_name_and_default() -> None:
    fake = FakePyAudio()
    devices = list_input_devices(pyaudio_factory=lambda: fake)
    assert [device.index for device in devices] == [0, 1]
    assert devices[0].is_default_input is True
    assert resolve_input_device(fake, "1", devices).name == "Bluetooth Input"
    assert resolve_input_device(fake, "blue", devices).index == 1
    assert resolve_input_device(fake, "", devices).index == 0
    with pytest.raises(MicrophoneError):
        resolve_input_device(fake, "2", devices)


def test_capture_recovers_from_device_disconnect() -> None:
    fake = FakePyAudio(
        supported_rates={48000},
        streams=[FakeStream(fail_after=1), FakeStream()],
    )
    audio_queue: queue.Queue = queue.Queue(maxsize=10000)
    capture = MicrophoneCapture(
        config(debug=True), audio_queue, pyaudio_factory=lambda: fake
    )
    try:
        capture.start()
        assert wait_for(lambda: capture.native_rate == 48000)
        assert capture.resampling is True
        time.sleep(1.2)
        capture.stop()
    finally:
        capture.stop()
    items = []
    while True:
        try:
            items.append(audio_queue.get_nowait())
        except queue.Empty:
            break
    kinds = [type(item).__name__ for item in items]
    assert "AudioInterrupted" in kinds
    assert any(
        isinstance(item, AudioNotice) and "已恢复" in item.text
        for item in items
    )
    interrupt_index = kinds.index("AudioInterrupted")
    assert "AudioChunk" in kinds[interrupt_index + 1:]
    assert fake.opened  # a new stream was opened after the failure


def test_capture_switches_rate_when_default_input_changes() -> None:
    fake = FakePyAudio(
        rates_by_device={0: {48000}, 1: {16000}},
        streams=[FakeStream(), FakeStream()],
    )
    audio_queue: queue.Queue = queue.Queue(maxsize=10000)
    capture = MicrophoneCapture(
        config(debug=True), audio_queue, pyaudio_factory=lambda: fake
    )
    try:
        capture.start()
        assert capture.device is not None and capture.device.index == 0
        assert capture.native_rate == 48000
        assert capture.resampling is True

        fake.default_index = 1  # Bluetooth input, native 16 kHz
        assert wait_for(
            lambda: capture.device is not None and capture.device.index == 1,
            timeout=4.0,
        )
        assert capture.native_rate == 16000
        assert capture.resampling is False
    finally:
        capture.stop()

    items = []
    while True:
        try:
            items.append(audio_queue.get_nowait())
        except queue.Empty:
            break
    assert any(
        isinstance(item, AudioInterrupted) and "默认输入设备" in item.reason
        for item in items
    )
    assert any(
        isinstance(item, AudioNotice) and "默认输入设备已变化" in item.text
        for item in items
    )
