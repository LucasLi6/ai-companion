"""Integration tests: ASR manager ↔ optional speaker observer.

The first group of tests uses a recording observer to lock down the exact
hook call sites (uid allocation, audio buffering, VAD end, abort/cancel,
snapshot at final). The last group wires the real :class:`SpeakerEngine` with
a deterministic fake embedding model, proving that inference runs off the
ASR/VAD threads and that the transcript is never delayed or polluted.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from companion.asr.client import ASREvent
from companion.asr.config import ASRConfig
from companion.asr.manager import ASRManager
from companion.asr.microphone import AudioChunk, AudioInterrupted
from companion.asr.terminal import RenderEvent, TerminalRenderer
from companion.asr.vad import AUDIO, SPEECH_ENDED, SPEECH_STARTED, VADEvent
from companion.speaker_id.config import EMBEDDING_DIM, load_speaker_config
from companion.speaker_id.embedding import pcm_to_float32
from companion.speaker_id.engine import SpeakerEngine
from companion.speaker_id.models import (
    SpeakerResult,
    pending_result,
)
from companion.speaker_id.store import SpeakerStore

SAMPLE_RATE = 16000


# --------------------------------------------------------------------- utils
def wait_for(predicate, *, timeout: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def tone(seconds: float, frequency: float = 220.0) -> bytes:
    n = int(seconds * SAMPLE_RATE)
    samples = 0.2 * np.sin(
        2 * np.pi * frequency * np.arange(n, dtype=np.float32) / SAMPLE_RATE
    )
    return (samples * 32767).astype("<i2").tobytes()


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


VECTORS = {220.0: unit(1, 0), 440.0: unit(0, 1)}


class FakeEmbedding:
    def __init__(self, *, block: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()

    def encode(self, pcm: bytes) -> np.ndarray:
        audio = pcm_to_float32(pcm)
        tail = audio[-SAMPLE_RATE:] if audio.size > SAMPLE_RATE else audio
        spectrum = np.abs(np.fft.rfft(tail))
        frequency = float(np.argmax(spectrum) * SAMPLE_RATE / tail.size)
        key = min(VECTORS, key=lambda candidate: abs(candidate - frequency))
        self.started.set()
        assert self.release.wait(timeout=3.0)
        return VECTORS[key].copy()


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


class FakeMic:
    description = "Fake Mic（16000 Hz）"
    dropped_chunks = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class ScriptedSegmenter:
    def __init__(self, groups: list[list[VADEvent]]) -> None:
        self.groups = list(groups)
        self.reset_calls = 0
        self._lock = threading.Lock()

    def process(self, pcm: bytes) -> list[VADEvent]:
        with self._lock:
            return self.groups.pop(0) if self.groups else []

    def reset(self) -> None:
        with self._lock:
            self.reset_calls += 1


@dataclass
class SessionScript:
    partials: list[tuple[str, str]] = field(default_factory=list)
    finals: list[tuple[str, str]] = field(default_factory=list)
    fail_on_open: bool = False


class FakeSession:
    def __init__(self, script: SessionScript) -> None:
        self.script = script
        self.sent: list[bytes] = []
        self.stop_calls = 0
        self.close_calls = 0
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def open(self) -> None:
        if self.script.fail_on_open:
            from companion.asr.client import ASRClientError

            raise ASRClientError("模拟连接失败")

    def send_audio(self, pcm: bytes) -> None:
        with self._lock:
            self.sent.append(bytes(pcm))

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop_event.set()

    def events(self):
        for text, sentence_id in self.script.partials:
            yield ASREvent("sentence", text, sentence_id, False)
        self._stop_event.wait(timeout=3.0)
        for text, sentence_id in self.script.finals:
            yield ASREvent("sentence", text, sentence_id, True)

    def close(self) -> None:
        self.close_calls += 1
        self._stop_event.set()

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


class RecordingObserver:
    def __init__(self, result: SpeakerResult | None = None) -> None:
        self.result = result
        self.events: list[tuple] = []
        self.snapshot_calls: list[int] = []
        self._lock = threading.Lock()

    def on_start(self, uid: int, pcm: bytes) -> None:
        with self._lock:
            self.events.append(("start", uid, len(pcm)))

    def on_audio(self, uid: int, pcm: bytes) -> None:
        with self._lock:
            self.events.append(("audio", uid, len(pcm)))

    def on_end(self, uid: int) -> None:
        with self._lock:
            self.events.append(("end", uid))

    def on_cancel(self, uid: int) -> None:
        with self._lock:
            self.events.append(("cancel", uid))

    def snapshot(self, uid: int):
        with self._lock:
            self.snapshot_calls.append(uid)
            return self.result


class RaisingObserver:
    def on_start(self, uid, pcm):
        raise RuntimeError("observer start failed")

    def on_audio(self, uid, pcm):
        raise RuntimeError("observer audio failed")

    def on_end(self, uid):
        raise RuntimeError("observer end failed")

    def on_cancel(self, uid):
        raise RuntimeError("observer cancel failed")

    def snapshot(self, uid):
        raise RuntimeError("observer snapshot failed")


def asr_config(**overrides) -> ASRConfig:
    base = dict(
        api_key="unit-test-key",
        workspace_id="unit-test-workspace",
        timeout_seconds=2.0,
        final_timeout_seconds=1.0,
    )
    base.update(overrides)
    return ASRConfig(**base)


class Harness:
    def __init__(
        self,
        *,
        groups: list[list[VADEvent]],
        scripts: list[SessionScript],
        observer=None,
        renderer: TerminalRenderer | None = None,
        out: LockedBuffer | None = None,
        status: LockedBuffer | None = None,
    ) -> None:
        self.out = out if out is not None else LockedBuffer()
        self.status = status if status is not None else LockedBuffer()
        self.renderer = renderer or TerminalRenderer(
            stream=self.out, status_stream=self.status, is_tty=False
        )
        self.segmenter = ScriptedSegmenter(groups)
        self.factory = SessionFactory(scripts)
        self.finals: list = []
        self.manager = ASRManager(
            asr_config(),
            microphone=FakeMic(),
            vad_segmenter=self.segmenter,
            session_factory=self.factory,
            renderer=self.renderer,
            speaker_observer=observer,
        )
        self.manager.set_listeners(on_final=self.finals.append)
        self.thread = threading.Thread(target=self.manager.run, daemon=True)
        self.thread.start()

    def feed(self, count: int = 1, pcm: bytes = b"\x01\x00" * 1024) -> None:
        for _ in range(count):
            self.manager.audio_queue.put(
                AudioChunk(pcm, SAMPLE_RATE, SAMPLE_RATE, time.monotonic()),
                timeout=2.0,
            )

    def wait_ready(self) -> None:
        assert wait_for(lambda: "正在监听" in self.status.getvalue())

    def wait_final(self, *, timeout: float = 4.0):
        assert wait_for(lambda: self.finals, timeout=timeout)
        return self.finals[0]

    def stop(self) -> None:
        self.manager.stop()
        self.thread.join(timeout=4.0)
        self.manager.shutdown()
        assert not self.thread.is_alive()


# ------------------------------------------------------- observer call sites
def test_observer_hooks_follow_the_utterance_lifecycle() -> None:
    snapshot = pending_result(1, model_id="fake")
    observer = RecordingObserver(snapshot)
    harness = Harness(
        groups=[
            [VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)],
            [VADEvent(AUDIO, b"\x02\x00" * 1024)],
            [VADEvent(SPEECH_ENDED)],
        ],
        scripts=[SessionScript(finals=[("你好。", "s1")])],
        observer=observer,
    )
    try:
        harness.wait_ready()
        harness.feed(3)
        final = harness.wait_final()

        assert observer.events[0] == ("start", 1, 2048)
        assert ("audio", 1, 2048) in observer.events
        assert observer.events[-1] == ("end", 1)
        assert observer.snapshot_calls == [1]
        assert final.speaker is snapshot
        assert final.text == "你好。"
    finally:
        harness.stop()


def test_observer_is_called_in_order_start_audio_end() -> None:
    observer = RecordingObserver(pending_result(1))
    harness = Harness(
        groups=[
            [VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)],
            [VADEvent(AUDIO, b"\x02\x00" * 1024)],
            [VADEvent(SPEECH_ENDED)],
        ],
        scripts=[SessionScript(finals=[("好。", "s1")])],
        observer=observer,
    )
    try:
        harness.wait_ready()
        harness.feed(3)
        harness.wait_final()
        kinds = [event[0] for event in observer.events]
        assert kinds == ["start", "audio", "end"]
    finally:
        harness.stop()


def test_abort_calls_on_cancel_and_emits_no_final() -> None:
    observer = RecordingObserver()
    harness = Harness(
        groups=[[VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)]],
        scripts=[SessionScript(partials=[("临时", "s1")])],
        observer=observer,
    )
    try:
        harness.wait_ready()
        harness.feed(1)
        harness.manager.audio_queue.put(
            AudioInterrupted("设备断开", time.monotonic()), timeout=2.0
        )
        assert wait_for(lambda: ("cancel", 1) in observer.events)
        assert harness.finals == []
    finally:
        harness.stop()


def test_observer_exceptions_never_stop_recognition() -> None:
    harness = Harness(
        groups=[
            [VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)],
            [VADEvent(SPEECH_ENDED)],
        ],
        scripts=[SessionScript(finals=[("继续。", "s1")])],
        observer=RaisingObserver(),
    )
    try:
        harness.wait_ready()
        harness.feed(2)
        final = harness.wait_final()
        assert final.text == "继续。"
        assert final.speaker is None  # snapshot failed safely
        assert harness.out.getvalue() == "继续。\n"
    finally:
        harness.stop()


def test_without_observer_behavior_is_unchanged() -> None:
    harness = Harness(
        groups=[
            [VADEvent(SPEECH_STARTED, b"\x01\x00" * 1024)],
            [VADEvent(SPEECH_ENDED)],
        ],
        scripts=[SessionScript(finals=[("原始行为。", "s1")])],
    )
    try:
        harness.wait_ready()
        harness.feed(2)
        final = harness.wait_final()
        assert final.text == "原始行为。"
        assert final.speaker is None
        assert harness.out.getvalue() == "原始行为。\n"
        assert "[speaker]" not in harness.status.getvalue()
    finally:
        harness.stop()


# ---------------------------------------------------------- TTY rendering
def result_for(status: str, speaker_id: str | None, *, quality: str = "HIGH"):
    from companion.speaker_id.models import SpeakerResult

    return SpeakerResult(
        utterance_id=1,
        speaker_id=speaker_id,
        person_id=None,
        status=status,
        reid_quality=quality,
        similarity=0.89 if status == "KNOWN" else None,
        margin=None,
        voiced_seconds=2.4,
        provisional=False,
        model_id="fake",
        inference_ms=10.0,
        completed_at=1.0,
    )


def test_tty_renderer_puts_identity_below_the_partial() -> None:
    out, status = LockedBuffer(), LockedBuffer()
    renderer = TerminalRenderer(
        stream=out, status_stream=status, is_tty=True, width=120
    )
    identity = result_for("NEW_CANDIDATE", "TEMP-0001")

    renderer.render(RenderEvent(kind="partial", utterance_id=1, text="你好"))
    renderer.render(
        RenderEvent(kind="speaker_status", utterance_id=1, speaker_result=identity)
    )
    assert renderer.partial_text == "你好"
    assert renderer.speaker_text == "[ID: TEMP-0001 | NEW | 可复识: HIGH]"

    renderer.render(
        RenderEvent(
            kind="final", utterance_id=1, text="你好。", speaker_result=identity
        )
    )
    assert renderer.confirmed == ["你好。"]
    assert renderer.partial_text == ""
    assert renderer.speaker_text == ""
    assert out.getvalue().count("你好。") == 1


def test_tty_late_identity_update_never_reprints_the_transcript() -> None:
    out, status = LockedBuffer(), LockedBuffer()
    renderer = TerminalRenderer(
        stream=out, status_stream=status, is_tty=True, width=120
    )
    identity = result_for("NEW_CANDIDATE", "TEMP-0001")
    renderer.render(
        RenderEvent(
            kind="final", utterance_id=1, text="第一句。", speaker_result=identity
        )
    )
    # The engine publishes a later refinement for an already confirmed turn.
    renderer.render(
        RenderEvent(kind="speaker_status", utterance_id=1, speaker_result=identity)
    )
    assert out.getvalue().count("第一句。") == 1
    assert "[speaker] #1" in status.getvalue()

    # A delayed identity for the old turn must not replace the new partial.
    renderer.render(RenderEvent(kind="partial", utterance_id=2, text="第二句"))
    before = out.getvalue()
    renderer.render(
        RenderEvent(kind="speaker_status", utterance_id=1, speaker_result=identity)
    )
    assert renderer.partial_text == "第二句"
    assert out.getvalue().startswith(before)
    assert out.getvalue().count("第一句。") == 1


# ------------------------------------------------------- real engine wiring
def test_real_engine_identity_is_attached_and_rendered(tmp_path) -> None:
    embedding = FakeEmbedding()
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    engine = SpeakerEngine(
        load_speaker_config(env={}),
        embedding_model=embedding,
        store=store,
    )
    out, status = LockedBuffer(), LockedBuffer()
    renderer = TerminalRenderer(stream=out, status_stream=status, is_tty=False)
    harness = Harness(
        groups=[
            [VADEvent(SPEECH_STARTED, tone(2.0))],
            [VADEvent(SPEECH_ENDED)],
        ],
        scripts=[SessionScript(finals=[("你好。", "s1")])],
        observer=engine,
        renderer=renderer,
        out=out,
        status=status,
    )
    engine.set_on_result(harness.manager.post_speaker_result)
    engine.start()
    try:
        harness.wait_ready()
        harness.feed(1)
        # The identity is produced while the VAD turn is still open.
        assert wait_for(
            lambda: engine.snapshot(1).status == "NEW_CANDIDATE"
        )
        harness.feed(1)
        final = harness.wait_final()

        assert final.text == "你好。"
        assert final.speaker is not None
        assert final.speaker.speaker_id == "TEMP-0001"
        assert "TEMP-0001" not in final.text
        assert wait_for(
            lambda: "TEMP-0001" in harness.out.getvalue()
            and "TEMP-0001" in harness.status.getvalue()
        )
        assert harness.out.getvalue() == (
            "你好。\n[Speaker: TEMP-0001 | NEW | ReID: HIGH]\n"
        )
        assert "[speaker] #1 [ID: TEMP-0001 | NEW | 可复识: HIGH]" in (
            harness.status.getvalue()
        )
    finally:
        harness.stop()
        engine.shutdown()
        store.close()


def test_asr_upload_continues_while_inference_is_blocked(tmp_path) -> None:
    embedding = FakeEmbedding(block=True)
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    engine = SpeakerEngine(
        load_speaker_config(env={}),
        embedding_model=embedding,
        store=store,
    )
    harness = Harness(
        groups=[
            [VADEvent(SPEECH_STARTED, tone(2.0))],
            [VADEvent(AUDIO, b"\x02\x00" * 1024)],
            [VADEvent(SPEECH_ENDED)],
        ],
        scripts=[SessionScript(finals=[("你好。", "s1")])],
        observer=engine,
    )
    engine.set_on_result(harness.manager.post_speaker_result)
    engine.start()
    try:
        harness.wait_ready()
        harness.feed(1)
        assert embedding.started.wait(timeout=3.0)
        session = harness.factory.created[0]
        # The voice-print worker is still inside encode(); the ASR sender has
        # already uploaded the turn's audio.
        assert wait_for(lambda: session.sent_bytes > 0)
        before = session.sent_bytes
        harness.feed(1)
        assert wait_for(lambda: session.sent_bytes > before)

        embedding.release.set()
        harness.feed(1)
        final = harness.wait_final()
        assert final.text == "你好。"
        assert wait_for(
            lambda: engine.snapshot(1).status == "NEW_CANDIDATE"
        )
    finally:
        embedding.release.set()
        harness.stop()
        engine.shutdown()
        store.close()


def test_final_does_not_wait_for_a_slow_voice_print(tmp_path) -> None:
    embedding = FakeEmbedding(block=True)
    store = SpeakerStore(tmp_path / "speakers.sqlite3")
    store.initialize()
    engine = SpeakerEngine(
        load_speaker_config(env={}),
        embedding_model=embedding,
        store=store,
    )
    out, status = LockedBuffer(), LockedBuffer()
    renderer = TerminalRenderer(stream=out, status_stream=status, is_tty=False)
    harness = Harness(
        groups=[
            [VADEvent(SPEECH_STARTED, tone(2.0))],
            [VADEvent(SPEECH_ENDED)],
        ],
        scripts=[SessionScript(finals=[("你好。", "s1")])],
        observer=engine,
        renderer=renderer,
        out=out,
        status=status,
    )
    engine.set_on_result(harness.manager.post_speaker_result)
    engine.start()
    try:
        harness.wait_ready()
        harness.feed(1)
        assert embedding.started.wait(timeout=3.0)

        harness.feed(1)  # VAD end while inference is still blocked
        final = harness.wait_final(timeout=2.0)
        assert final.text == "你好。"
        assert final.speaker is not None
        assert final.speaker.status == "PENDING"
        assert out.getvalue() == (
            "你好。\n[Speaker: PENDING | ReID: 评估中]\n"
        )
        snapshot_at_final = final.speaker

        embedding.release.set()
        assert wait_for(
            lambda: engine.snapshot(1).status == "NEW_CANDIDATE"
        )
        # The late identity update is printed as a status line only; the
        # confirmed transcript is never reprinted.
        assert wait_for(
            lambda: "[speaker] #1" in status.getvalue()
        )
        assert out.getvalue() == (
            "你好。\n[Speaker: PENDING | ReID: 评估中]\n"
        )
        assert snapshot_at_final.status == "PENDING"  # frozen snapshot
    finally:
        embedding.release.set()
        harness.stop()
        engine.shutdown()
        store.close()
