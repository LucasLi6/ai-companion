"""Thread and utterance coordinator for the independent ASR demo.

Pipeline::

    Capture Worker -> VAD Worker -> ASR Sender <-> ASR Receiver -> Renderer

All queues are bounded. The capture thread never waits for the network. Every
utterance gets its own ``utterance_id`` and DashScope session, so a late result
from an older utterance can never overwrite a newer one. The VAD boundary owns
when input stops; ASR ``sentence_end`` only confirms text.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from companion.asr.client import ASRClientError, ASRSession
from companion.asr.config import ASRConfig
from companion.asr.microphone import (
    AudioChunk,
    AudioInterrupted,
    AudioNotice,
    MicrophoneCapture,
)
from companion.asr.terminal import RenderEvent, TerminalRenderer
from companion.asr.vad import (
    AUDIO,
    SPEECH_ENDED,
    SPEECH_STARTED,
    SileroVAD,
    VADEvent,
    VADSegmenter,
)

# The provider accepts small frames; ~100 ms mono 16-bit PCM is 3200 bytes.
SEND_FRAME_BYTES = 3200
SEND_FRAME_SECONDS = 0.1

_STOP = object()
_ABORT = object()


@dataclass
class ASRStats:
    """Non-secret session counters for the exit summary."""

    utterances: int = 0
    finalized: int = 0
    empty: int = 0
    aborted: int = 0
    errors: int = 0
    deferred: int = 0
    dropped_control_events: int = 0
    final_latencies_ms: list[float] = field(default_factory=list)


# --------------------------------------------------------------- control items
@dataclass(frozen=True)
class _SpeechStarted:
    audio: bytes
    at: float


@dataclass(frozen=True)
class _SpeechAudio:
    audio: bytes
    at: float


@dataclass(frozen=True)
class _SpeechEnded:
    at: float


@dataclass(frozen=True)
class _Interrupted:
    reason: str


@dataclass(frozen=True)
class _Notice:
    text: str
    is_error: bool = False


@dataclass(frozen=True)
class _Sentence:
    uid: int
    text: str
    sentence_id: str
    is_final: bool
    at: float


@dataclass(frozen=True)
class _SessionDone:
    uid: int
    error: Optional[str]
    at: float


@dataclass(frozen=True)
class _FinalTimeout:
    uid: int


class _Utterance:
    """Mutable state for one VAD-bounded utterance."""

    def __init__(self, uid: int, send_queue_frames: int) -> None:
        self.uid = uid
        self.send_queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=max(1, send_queue_frames)
        )
        self.session: Optional[ASRSession] = None
        self.opened = threading.Event()
        self.sender_done = threading.Event()
        self.open_error: Optional[str] = None
        self.started = False
        self.sender: Optional[threading.Thread] = None
        self.receiver: Optional[threading.Thread] = None
        self.vad_end_at: Optional[float] = None
        self.final_sentence_at: Optional[float] = None
        self.finals: list[tuple[str, str]] = []
        self.final_ids: set[str] = set()
        self.partial = ""
        self.partial_id = ""
        self.finalized = False
        self.aborted = False
        self.stop_sent = False
        self.timer: Optional[threading.Timer] = None
        self.final_latency_ms: Optional[float] = None


class ASRManager:
    """Owns the capture/VAD/session/render threads and their state machine."""

    def __init__(
        self,
        config: ASRConfig,
        *,
        microphone: Any = None,
        vad_segmenter: Optional[VADSegmenter] = None,
        session_factory: Optional[Callable[[], ASRSession]] = None,
        renderer: Optional[TerminalRenderer] = None,
        audio_queue: Optional["queue.Queue[Any]"] = None,
    ) -> None:
        self.config = config
        self.audio_queue: "queue.Queue[Any]" = (
            audio_queue
            if audio_queue is not None
            else queue.Queue(maxsize=config.mic_queue_windows)
        )
        self._microphone = (
            microphone
            if microphone is not None
            else MicrophoneCapture(config, self.audio_queue)
        )
        self._vad_segmenter = vad_segmenter
        self._session_factory = session_factory or (lambda: ASRSession(config))
        self._renderer = renderer or TerminalRenderer()
        self.stats = ASRStats()

        self._control: "queue.Queue[Any]" = queue.Queue(maxsize=2048)
        self._render_queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=config.render_queue_events
        )
        self._stop = threading.Event()
        self._started = False
        self._stopped = False
        self._vad_thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._threads: list[threading.Thread] = []
        self._utterances: dict[int, _Utterance] = {}
        self._open_order: list[_Utterance] = []
        self._current: Optional[_Utterance] = None
        self._active_sessions = 0
        self._next_uid = 0

    # ------------------------------------------------------------------ public
    @property
    def microphone(self) -> Any:
        return self._microphone

    @property
    def dropped_audio_chunks(self) -> int:
        return int(getattr(self._microphone, "dropped_chunks", 0) or 0)

    def start(self) -> None:
        if self._started:
            return
        if self._vad_segmenter is None:
            vad = SileroVAD(self.config)
            vad.preload()
            self._vad_segmenter = VADSegmenter(vad.probability, self.config)

        self._render_thread = threading.Thread(
            target=self._render_loop, name="asr-render", daemon=True
        )
        self._render_thread.start()
        self._vad_thread = threading.Thread(
            target=self._vad_loop, name="asr-vad", daemon=True
        )
        self._vad_thread.start()
        try:
            self._microphone.start()
        except Exception:
            self._stop.set()
            self._render_queue.put_nowait(None)
            if self._vad_thread is not None:
                self._vad_thread.join(timeout=1.0)
            if self._render_thread is not None:
                self._render_thread.join(timeout=1.0)
            raise
        self._started = True
        description = getattr(self._microphone, "description", "默认输入设备")
        self._render(
            RenderEvent(
                kind="status",
                text=(
                    f"设备: {description} | 模型: {self.config.model} "
                    "| 正在监听…（Ctrl+C 结束）"
                ),
            )
        )

    def run(self) -> None:
        """Process control events until stopped; must run in the main thread."""
        self.start()
        try:
            while not self._stop.is_set():
                try:
                    event = self._control.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._handle(event)
                except Exception as exc:  # noqa: BLE001 - keep listening
                    self._render_status(
                        f"内部处理错误（{type(exc).__name__}），已跳过该事件。",
                        is_error=True,
                    )
        finally:
            self.shutdown()

    def stop(self) -> None:
        """Signal all workers to stop; safe from a signal handler."""
        self._stop.set()
        try:
            self._microphone.stop()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        try:
            self._microphone.stop()
        except Exception:  # noqa: BLE001 - best effort
            pass
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=2.0)
        for utterance in list(self._utterances.values()):
            if not utterance.finalized:
                self._abort_utterance(
                    utterance,
                    "程序退出，已中止当前识别。",
                    report=False,
                    promote=False,
                )
        for thread in self._threads:
            thread.join(timeout=2.0)
        try:
            self._render_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
        self._renderer.close()

    # ----------------------------------------------------------------- workers
    def _vad_loop(self) -> None:
        segmenter = self._vad_segmenter
        while not self._stop.is_set():
            try:
                item = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, AudioInterrupted):
                try:
                    segmenter.reset()
                except Exception:  # noqa: BLE001 - best effort
                    pass
                self._put_control(_Interrupted(item.reason), critical=True)
            elif isinstance(item, AudioNotice):
                self._put_control(
                    _Notice(item.text, item.is_error), critical=False
                )
            elif isinstance(item, AudioChunk):
                try:
                    events = segmenter.process(item.pcm)
                except Exception as exc:  # noqa: BLE001 - keep listening
                    self._put_control(
                        _Notice(
                            f"VAD 处理失败（{type(exc).__name__}），已跳过该段音频。",
                            True,
                        ),
                        critical=False,
                    )
                    continue
                for event in events:
                    self._dispatch_vad_event(event)

    def _dispatch_vad_event(self, event: VADEvent) -> None:
        if event.kind == SPEECH_STARTED:
            self._put_control(
                _SpeechStarted(event.audio, time.monotonic()), critical=True
            )
        elif event.kind == AUDIO:
            self._put_control(
                _SpeechAudio(event.audio, time.monotonic()), critical=False
            )
        elif event.kind == SPEECH_ENDED:
            self._put_control(
                _SpeechEnded(time.monotonic()), critical=True
            )

    def _render_loop(self) -> None:
        while True:
            event = self._render_queue.get()
            if event is None:
                break
            try:
                self._renderer.render(event)
            except BrokenPipeError:
                self._stop.set()
                break
            except Exception:  # noqa: BLE001 - rendering must not kill ASR
                continue

    # ------------------------------------------------------------- ASR threads
    def _sender_loop(self, utterance: _Utterance) -> None:
        session: Optional[ASRSession] = None
        try:
            session = self._session_factory()
            utterance.session = session
            session.open()
            utterance.opened.set()
            buffer = bytearray()
            first_buffered_at: Optional[float] = None
            while not (self._stop.is_set() or utterance.aborted):
                try:
                    item = utterance.send_queue.get(timeout=0.02)
                except queue.Empty:
                    item = None
                now = time.monotonic()
                if isinstance(item, (bytes, bytearray)):
                    if not buffer:
                        first_buffered_at = now
                    buffer.extend(item)
                    if len(buffer) >= SEND_FRAME_BYTES:
                        self._flush_audio(session, buffer)
                        first_buffered_at = None
                elif item is _STOP:
                    if buffer:
                        self._flush_audio(session, buffer)
                        buffer.clear()
                    session.stop()
                    return
                elif item is _ABORT:
                    return
                if (
                    buffer
                    and first_buffered_at is not None
                    and now - first_buffered_at >= SEND_FRAME_SECONDS
                ):
                    self._flush_audio(session, buffer)
                    first_buffered_at = None
            if buffer and not utterance.aborted:
                self._flush_audio(session, buffer)
        except ASRClientError as exc:
            utterance.open_error = str(exc)
            self._put_control(
                _SessionDone(utterance.uid, str(exc), time.monotonic()),
                critical=True,
            )
        except Exception as exc:  # noqa: BLE001 - sanitized
            message = f"ASR 发送失败（{type(exc).__name__}）。"
            utterance.open_error = message
            self._put_control(
                _SessionDone(utterance.uid, message, time.monotonic()),
                critical=True,
            )
        finally:
            utterance.opened.set()
            utterance.sender_done.set()

    def _receiver_loop(self, utterance: _Utterance) -> None:
        timeout = self.config.timeout_seconds + 2.0
        if not utterance.opened.wait(timeout=timeout):
            self._put_control(
                _SessionDone(
                    utterance.uid, "ASR 连接超时。", time.monotonic()
                ),
                critical=True,
            )
            return
        session = utterance.session
        if session is None or utterance.open_error:
            return
        try:
            for event in session.events():
                now = time.monotonic()
                if event.kind == "error":
                    self._put_control(
                        _SessionDone(utterance.uid, event.error, now),
                        critical=True,
                    )
                    return
                if event.kind == "sentence":
                    self._put_control(
                        _Sentence(
                            utterance.uid,
                            event.text,
                            event.sentence_id,
                            event.is_final,
                            now,
                        ),
                        critical=True,
                    )
            self._put_control(
                _SessionDone(utterance.uid, None, time.monotonic()),
                critical=True,
            )
        except ASRClientError as exc:
            self._put_control(
                _SessionDone(utterance.uid, str(exc), time.monotonic()),
                critical=True,
            )
        except Exception as exc:  # noqa: BLE001 - sanitized
            self._put_control(
                _SessionDone(
                    utterance.uid,
                    f"ASR 接收失败（{type(exc).__name__}）。",
                    time.monotonic(),
                ),
                critical=True,
            )
        finally:
            session.close()

    def _flush_audio(
        self, session: ASRSession, buffer: bytearray
    ) -> None:
        while buffer:
            chunk = bytes(buffer[:SEND_FRAME_BYTES])
            del buffer[:SEND_FRAME_BYTES]
            session.send_audio(chunk)

    # ------------------------------------------------------------ event handlers
    def _handle(self, event: Any) -> None:
        if isinstance(event, _SpeechStarted):
            self._handle_speech_started(event)
        elif isinstance(event, _SpeechAudio):
            self._handle_speech_audio(event)
        elif isinstance(event, _SpeechEnded):
            self._handle_speech_ended(event)
        elif isinstance(event, _Sentence):
            self._handle_sentence(event)
        elif isinstance(event, _SessionDone):
            self._handle_session_done(event)
        elif isinstance(event, _FinalTimeout):
            self._handle_final_timeout(event)
        elif isinstance(event, _Interrupted):
            self._handle_interrupted(event)
        elif isinstance(event, _Notice):
            self._render_status(event.text, is_error=event.is_error)

    def _handle_speech_started(self, event: _SpeechStarted) -> None:
        self._next_uid += 1
        utterance = _Utterance(
            self._next_uid, self.config.asr_send_queue_frames
        )
        self._utterances[utterance.uid] = utterance
        self._open_order.append(utterance)
        self._current = utterance
        self.stats.utterances += 1
        if not self._enqueue_audio(utterance, event.audio):
            return
        if utterance.finalized:
            return
        if self._active_sessions < self.config.max_concurrent_sessions:
            self._start_session(utterance)
        else:
            self.stats.deferred += 1
            if self.config.debug:
                self._render_status(
                    f"#{utterance.uid} 等待上一个会话结束，音频已缓冲。"
                )
        if self.config.debug:
            self._render_status(f"检测到语音 #{utterance.uid}，开始识别。")

    def _handle_speech_audio(self, event: _SpeechAudio) -> None:
        utterance = self._current
        if utterance is None or utterance.finalized:
            return
        self._enqueue_audio(utterance, event.audio)

    def _handle_speech_ended(self, event: _SpeechEnded) -> None:
        utterance = self._current
        self._current = None
        if utterance is None or utterance.finalized:
            return
        utterance.vad_end_at = event.at
        if utterance.stop_sent:
            return
        utterance.stop_sent = True
        if not self._enqueue_sentinel(utterance, _STOP):
            self._abort_utterance(
                utterance,
                "ASR 发送队列已满，网络可能过慢；已中止当前识别。",
            )
            return
        timeout = max(0.1, self.config.final_timeout_seconds)
        timer = threading.Timer(
            timeout, self._enqueue_final_timeout, args=(utterance.uid,)
        )
        timer.daemon = True
        utterance.timer = timer
        timer.start()
        if self.config.debug:
            self._render_status(f"#{utterance.uid} 语音结束，等待最终识别…")

    def _handle_sentence(self, event: _Sentence) -> None:
        utterance = self._utterances.get(event.uid)
        if (
            utterance is None
            or utterance.finalized
            or utterance.aborted
        ):
            return
        if event.is_final:
            if (
                utterance.vad_end_at is not None
                and utterance.final_sentence_at is None
            ):
                utterance.final_sentence_at = event.at
            sentence_id = event.sentence_id
            if sentence_id and sentence_id in utterance.final_ids:
                return
            if (
                not sentence_id
                and utterance.finals
                and event.text == utterance.finals[-1][1]
            ):
                return
            if sentence_id:
                utterance.final_ids.add(sentence_id)
            key = sentence_id or f"#{len(utterance.finals)}"
            utterance.finals.append((key, event.text))
            replace = (
                not utterance.partial
                or utterance.partial_id == sentence_id
                or (not sentence_id and utterance.partial == event.text)
            )
            if replace:
                utterance.partial = ""
                utterance.partial_id = ""
            self._render(
                RenderEvent(
                    kind="final",
                    utterance_id=event.uid,
                    text=event.text,
                    replace_partial=replace,
                )
            )
            return

        if (
            event.sentence_id
            and event.sentence_id in utterance.final_ids
        ):
            return
        if event.text == utterance.partial:
            return
        utterance.partial = event.text
        utterance.partial_id = event.sentence_id
        self._render(
            RenderEvent(
                kind="partial",
                utterance_id=event.uid,
                text=event.text,
            )
        )

    def _handle_session_done(self, event: _SessionDone) -> None:
        utterance = self._utterances.get(event.uid)
        if utterance is None or utterance.finalized:
            return
        self._finish(utterance, event.error, event.at)

    def _handle_final_timeout(self, event: _FinalTimeout) -> None:
        utterance = self._utterances.get(event.uid)
        if utterance is None or utterance.finalized:
            return
        self._close_session(utterance)
        self._finish(
            utterance,
            f"等待最终识别结果超时（超过 "
            f"{self.config.final_timeout_seconds:.0f} 秒）。",
            time.monotonic(),
        )

    def _handle_interrupted(self, event: _Interrupted) -> None:
        self._render_status(event.reason, is_error=True)
        utterance = self._current
        if utterance is not None and not utterance.finalized:
            self._abort_utterance(
                utterance,
                "录音中断，已中止当前识别；临时文本不会作为最终结果。",
            )

    # ------------------------------------------------------------- transitions
    def _start_session(self, utterance: _Utterance) -> None:
        if utterance.started or utterance.finalized:
            return
        utterance.started = True
        self._active_sessions += 1
        utterance.sender = threading.Thread(
            target=self._sender_loop,
            args=(utterance,),
            name=f"asr-send-{utterance.uid}",
            daemon=True,
        )
        utterance.receiver = threading.Thread(
            target=self._receiver_loop,
            args=(utterance,),
            name=f"asr-recv-{utterance.uid}",
            daemon=True,
        )
        self._threads.extend([utterance.sender, utterance.receiver])
        utterance.sender.start()
        utterance.receiver.start()

    def _finish(
        self,
        utterance: _Utterance,
        error: Optional[str],
        at: float,
    ) -> None:
        if utterance.finalized:
            return
        utterance.finalized = True
        if utterance.timer is not None:
            utterance.timer.cancel()
        if utterance.started:
            self._active_sessions = max(0, self._active_sessions - 1)
        self._close_session(utterance)

        if utterance.partial:
            utterance.partial = ""
            utterance.partial_id = ""
            self._render(
                RenderEvent(kind="clear_partial", utterance_id=utterance.uid)
            )

        if error:
            self.stats.errors += 1
            self._render_status(
                f"识别 #{utterance.uid} 失败：{error}", is_error=True
            )

        if utterance.finals:
            self.stats.finalized += 1
            if utterance.vad_end_at is not None:
                base = utterance.final_sentence_at or at
                latency = max(0.0, (base - utterance.vad_end_at) * 1000.0)
                utterance.final_latency_ms = latency
                self.stats.final_latencies_ms.append(latency)
                if self.config.debug:
                    self._render_status(
                        f"#{utterance.uid} VAD 结束 → 最终结果 "
                        f"{latency:.0f} ms"
                    )
        else:
            self.stats.empty += 1
            if not error:
                self._render_status(
                    f"#{utterance.uid} 没有可确认的最终文本，临时文本已丢弃。"
                )
        self._promote_pending()

    def _abort_utterance(
        self,
        utterance: _Utterance,
        reason: str,
        *,
        report: bool = True,
        promote: bool = True,
    ) -> None:
        if utterance.finalized:
            return
        utterance.aborted = True
        utterance.finalized = True
        if utterance.timer is not None:
            utterance.timer.cancel()
        if utterance.started:
            self._active_sessions = max(0, self._active_sessions - 1)
        self._close_session(utterance)
        self._enqueue_sentinel(utterance, _ABORT)
        if utterance.partial:
            utterance.partial = ""
            utterance.partial_id = ""
            self._render(
                RenderEvent(kind="clear_partial", utterance_id=utterance.uid)
            )
        if not utterance.finals:
            self.stats.aborted += 1
        if report:
            self._render_status(reason, is_error=True)
        if self._current is utterance:
            self._current = None
        if promote:
            self._promote_pending()

    def _promote_pending(self) -> None:
        for utterance in self._open_order:
            if self._active_sessions >= self.config.max_concurrent_sessions:
                break
            if utterance.finalized or utterance.started:
                continue
            self._start_session(utterance)

    # ----------------------------------------------------------------- helpers
    def _enqueue_audio(self, utterance: _Utterance, data: bytes) -> bool:
        if not data or utterance.finalized:
            return False
        try:
            utterance.send_queue.put_nowait(bytes(data))
            return True
        except queue.Full:
            self._abort_utterance(
                utterance,
                "ASR 发送队列已满，网络可能过慢；已中止当前识别。",
            )
            return False

    def _enqueue_sentinel(self, utterance: _Utterance, item: Any) -> bool:
        try:
            utterance.send_queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _enqueue_final_timeout(self, uid: int) -> None:
        self._put_control(
            _FinalTimeout(uid), critical=False, timeout=0.5
        )

    def _close_session(self, utterance: _Utterance) -> None:
        session = utterance.session
        if session is None:
            return
        try:
            session.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def _put_control(
        self,
        event: Any,
        *,
        critical: bool,
        timeout: float = 0.2,
    ) -> bool:
        while not self._stop.is_set():
            try:
                self._control.put(event, timeout=timeout)
                return True
            except queue.Full:
                if not critical:
                    self.stats.dropped_control_events += 1
                    return False
        return False

    def _render(self, event: RenderEvent) -> None:
        try:
            if event.kind == "partial":
                self._render_queue.put_nowait(event)
            else:
                self._render_queue.put(event, timeout=1.0)
        except queue.Full:
            pass

    def _render_status(self, text: str, *, is_error: bool = False) -> None:
        self._render(
            RenderEvent(
                kind="error" if is_error else "status", text=text
            )
        )


__all__ = ["ASRManager", "ASRStats"]
