"""Independent TTS manager: newest-reply-wins synthesis and playback.

Threading model
---------------
``submit`` is called on the chat main thread and only touches bounded queues
under a short lock, so it can never delay text generation or input.

* one TTS worker thread dequeues a :class:`SpeechJob`, checks cancellation and
  calls the TTS client;
* one player thread consumes a separate bounded audio queue and writes PCM to
  the output device in order;
* a monotonically increasing *generation* token identifies the only reply that
  is allowed to play. ``stop()`` and a newer ``submit`` both bump it, so stale
  synthesis and stale queued audio are dropped. This never touches DeepSeek
  text generation.

No ``print()`` is called from these threads; errors are stored for
``/tts status``.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from companion.tts.client import TTSClient
from companion.tts.config import (
    DEFAULT_FLUSH_STRATEGY,
    DEFAULT_SPEECH_RATE,
    TTSConfig,
    describe_tts_config,
    load_tts_config,
)
from companion.tts.errors import TTSError
from companion.tts.player import PyAudioPlayer
from companion.tts.segmenter import TextSegmenter

_STOP = object()
_END_OF_AUDIO = None


@dataclass(frozen=True)
class SpeechJob:
    """One complete reply prepared for speech."""

    response_id: str
    conversation_id: str
    text: str
    generation: int = 0
    # Monotonic timestamp taken when the text reply finished rendering.
    reply_completed_at: Optional[float] = None


@dataclass(frozen=True)
class TTSStatus:
    enabled: bool
    state: str
    model: str
    voice: str
    region: str
    sample_rate: int
    waiting_jobs: int
    queued_audio_chunks: int
    generation: int
    last_error: Optional[str]
    speech_rate: float = DEFAULT_SPEECH_RATE
    flush_strategy: str = DEFAULT_FLUSH_STRATEGY
    output_device: str = "system default"

    def lines(self) -> list[str]:
        """Display lines for ``/tts status``; contains no credentials."""
        return [
            f"tts: {'on' if self.enabled else 'off'} ({self.state})",
            f"model: {self.model}  voice: {self.voice}"
            f"  rate: {self.speech_rate}  flush: {self.flush_strategy}"
            f"  region: {self.region}",
            f"output: {self.output_device}"
            f"  sample_rate: {self.sample_rate} Hz"
            f"  waiting_jobs: {self.waiting_jobs}"
            f"  queued_audio_chunks: {self.queued_audio_chunks}",
            f"last_error: {self.last_error or '-'}",
        ]


def _ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 1)


class TTSStream:
    """One bidirectional TTS session for one streaming reply.

    Text deltas are buffered non-blockingly by :meth:`feed` (called on the chat
    thread), segmented by a dedicated sender thread, and sent through
    ``session.send_text``. A separate receiver thread drains
    ``session.audio_chunks`` into the manager's shared bounded audio queue, so
    sending and receiving are truly concurrent and playback can start before
    the reply ends.
    """

    def __init__(
        self,
        manager: "TTSManager",
        config: TTSConfig,
        client: Any,
        generation: int,
        response_id: str,
        conversation_id: str,
    ) -> None:
        self._manager = manager
        self._config = config
        self._client = client
        self.generation = generation
        self.response_id = response_id
        self.conversation_id = conversation_id

        self._segmenter = TextSegmenter(
            config.segment_min_chars,
            config.segment_max_chars,
            long_fallback_chars=config.long_fallback_chars,
        )
        self._condition = threading.Condition()
        self._pending: deque[str] = deque()
        self._pending_chars = 0
        self._input_closed = False
        self._cancelled = False
        self._failed = threading.Event()
        self._done = threading.Event()
        self._session: Any = None
        self._sender: Optional[threading.Thread] = None
        self._receiver: Optional[threading.Thread] = None
        self._closer: Optional[threading.Thread] = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def start(self) -> None:
        self._sender = threading.Thread(
            target=self._send_loop,
            name="ava-tts-sender",
            daemon=True,
        )
        self._sender.start()

    def feed(self, display_delta: str) -> None:
        """Accept already-validated display text without blocking the caller."""
        if not display_delta:
            return
        with self._condition:
            if self._cancelled or self._input_closed:
                return
            if (
                self._pending_chars + len(display_delta)
                > self._config.max_text_chars
            ):
                # The provider is far enough behind that the backlog would
                # exceed one reply's allowance. Stop accepting new text and
                # let the sender speak what it already buffered; memory stays
                # bounded and the displayed reply is unaffected.
                self._input_closed = True
                self._condition.notify_all()
                return
            self._pending.append(display_delta)
            self._pending_chars += len(display_delta)
            self._condition.notify_all()
        self._manager._record(self.generation, "first_display_delta_at")

    def note_first_text(self) -> None:
        """Record the first *visible* reply text (tags do not count)."""
        self._manager._record(self.generation, "first_text_at")

    def finish(self) -> None:
        """No more text for this reply; the sender flushes the tail."""
        with self._condition:
            if self._cancelled:
                return
            self._input_closed = True
            self._condition.notify_all()
        self._manager._record(self.generation, "response_completed_at")

    def cancel(self) -> None:
        """Invalidate this reply and close its session without blocking."""
        with self._condition:
            if self._cancelled:
                return
            self._cancelled = True
            self._pending.clear()
            self._pending_chars = 0
            self._condition.notify_all()
            session = self._session
        self._failed.set()
        if session is not None:
            self._closer = threading.Thread(
                target=self._close_session,
                args=(session,),
                name="ava-tts-closer",
                daemon=True,
            )
            self._closer.start()
        elif self._receiver is None:
            self._done.set()

    def join(self, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        for thread in (self._sender, self._receiver, self._closer):
            if thread is None or not thread.is_alive():
                continue
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    # --------------------------------------------------------------- sender
    def _send_loop(self) -> None:
        session = None
        try:
            self._manager._record(
                self.generation, "tts_connect_started_at"
            )
            session = self._client.open_stream()
            self._manager._record(
                self.generation, "tts_connect_finished_at"
            )
            with self._condition:
                self._session = session
                cancelled = self._cancelled
            if cancelled:
                self._close_session(session)
                return
            self._manager._record(
                self.generation, "tts_session_started_at"
            )
            self._receiver = threading.Thread(
                target=self._receive_loop,
                args=(session,),
                name="ava-tts-receiver",
                daemon=True,
            )
            self._receiver.start()

            while True:
                with self._condition:
                    while (
                        not self._pending
                        and not self._input_closed
                        and not self._cancelled
                    ):
                        self._condition.wait(0.1)
                    if self._cancelled:
                        break
                    if not self._pending and self._input_closed:
                        break
                    delta = self._pending.popleft()
                    self._pending_chars -= len(delta)
                if not self._manager._is_current(self.generation):
                    break
                for segment in self._segmenter.feed(delta):
                    if not self._send_segment(session, segment):
                        return

            if (
                not self._cancelled
                and not self._failed.is_set()
                and self._manager._is_current(self.generation)
            ):
                for segment in self._segmenter.finish():
                    if not self._send_segment(session, segment):
                        return
                self._stop_session(session)
        except Exception as exc:  # noqa: BLE001 - TTS-isolated
            if not self._cancelled:
                self._manager._fail(self.generation, exc)
            self._failed.set()
            if session is not None:
                self._close_session(session)
        finally:
            if self._receiver is None:
                self._done.set()

    def _send_segment(self, session: Any, segment: str) -> bool:
        if not segment or not self._manager._is_current(self.generation):
            return False
        self._manager._record(
            self.generation, "first_segment_ready_at"
        )
        session.send_text(segment)
        self._manager._record(
            self.generation, "first_segment_sent_at"
        )
        if self._config.flush_strategy == "sentence_flush":
            # Comparison strategy only: flush at a complete sentence (or a
            # long-fallback boundary), never after a plain comma.
            flush = getattr(session, "flush", None)
            if callable(flush):
                flush()
        # ``auto`` (default): no flush — the server detects the sentence end
        # and starts synthesis on its own.
        return True

    # ------------------------------------------------------------- receiver
    def _receive_loop(self, session: Any) -> None:
        try:
            for chunk in session.audio_chunks():
                if (
                    self._cancelled
                    or self._failed.is_set()
                    or not self._manager._is_current(self.generation)
                ):
                    break
                data = bytes(chunk) if isinstance(
                    chunk, (bytes, bytearray, memoryview)
                ) else b""
                if not data:
                    continue
                self._manager._record(
                    self.generation, "first_audio_received_at"
                )
                if not self._manager._put_audio(
                    self.generation, data
                ):
                    break
        except Exception as exc:  # noqa: BLE001 - TTS-isolated
            if not self._cancelled:
                self._manager._fail(self.generation, exc)
            self._failed.set()
        finally:
            if (
                not self._cancelled
                and not self._failed.is_set()
                and self._manager._is_current(self.generation)
            ):
                self._manager._put_audio(
                    self.generation, _END_OF_AUDIO
                )
            self._close_session(session)
            self._done.set()

    # --------------------------------------------------------------- session
    def _stop_session(self, session: Any) -> None:
        stop = getattr(session, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001 - best effort
                pass

    def _close_session(self, session: Any) -> None:
        for name in ("stop", "close"):
            method = getattr(session, name, None)
            if callable(method):
                try:
                    method()
                except Exception:  # noqa: BLE001 - cleanup
                    pass


class TTSManager:
    """Owns the TTS worker, audio queue, player and generation state."""

    def __init__(
        self,
        config_loader: Callable[[], TTSConfig] = load_tts_config,
        client_factory: Optional[Callable[[TTSConfig], Any]] = None,
        player_factory: Optional[Callable[[TTSConfig], Any]] = None,
        audio_queue_size: int = 16,
        join_timeout: float = 3.0,
        status_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._config_loader = config_loader
        self._client_factory = client_factory or (lambda cfg: TTSClient(cfg))
        self._player_factory = player_factory or (lambda cfg: PyAudioPlayer(cfg))
        self._audio_queue_size = max(1, int(audio_queue_size))
        self._join_timeout = join_timeout
        self._status_env = status_env

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._text_queue: queue.Queue = queue.Queue(maxsize=1)
        self._audio_queue: queue.Queue = queue.Queue(
            maxsize=self._audio_queue_size
        )
        self._worker: Optional[threading.Thread] = None
        self._player_thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._player: Any = None
        self._config: Optional[TTSConfig] = None

        self._enabled = False
        self._started = False
        self._generation = 0
        self._queued_generation: Optional[int] = None
        self._status = "disabled"
        self._last_error: Optional[str] = None
        self._timings: dict[int, dict[str, float]] = {}
        self._last_finished_generation: Optional[int] = None
        self._active_stream: Optional[TTSStream] = None
        # Every stream that has not finished yet. Superseded streams are
        # cancelled without joining on the chat path; this set lets explicit
        # teardown join them within one bounded deadline.
        self._streams: set[TTSStream] = set()

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def generation(self) -> int:
        """Current generation; new submissions invalidate older ones."""
        with self._lock:
            return self._generation

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Validate config, open the audio device and start both threads."""
        with self._lock:
            if self._enabled:
                return
        config = self._config_loader()
        client = self._client_factory(config)
        player = self._player_factory(config)
        try:
            player.open()
        except Exception:
            self._close_quietly(client)
            self._close_quietly(player)
            raise

        with self._lock:
            if self._enabled:  # lost a race with another start()
                self._close_quietly(client)
                self._close_quietly(player)
                return
            self._config = config
            self._client = client
            self._player = player
            self._text_queue = queue.Queue(maxsize=1)
            self._audio_queue = queue.Queue(
                maxsize=self._audio_queue_size
            )
            self._stop_event.clear()
            self._enabled = True
            self._started = True
            self._status = "idle"
            self._last_error = None
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="ava-tts-worker",
                daemon=True,
            )
            self._player_thread = threading.Thread(
                target=self._player_loop,
                name="ava-tts-player",
                daemon=True,
            )
            worker, player_thread = self._worker, self._player_thread
        worker.start()
        player_thread.start()

    def enable(self) -> tuple[bool, str]:
        """``/tts on``: enable only after successful initialization."""
        try:
            self.start()
        except TTSError as exc:
            message = str(exc)
            with self._lock:
                self._last_error = message
            return False, message
        except Exception as exc:  # noqa: BLE001 - sanitized
            message = f"TTS 初始化失败: {type(exc).__name__}"
            with self._lock:
                self._last_error = message
            return False, message
        described = self._description()
        return True, (
            "tts: on  "
            f"model={described['model']}  voice={described['voice']}  "
            f"rate={described['rate']}x  region={described['region']}  "
            f"output={described['output_device']}  "
            f"{described['sample_rate']} Hz"
        )

    def stop(self, *, wait: bool = False) -> None:
        """Stop current playback and drop queued speech; stay enabled.

        Invalidates the running synthesis task as well as queued audio. Text
        generation and the REPL are never touched. By default this never waits
        for provider I/O: it is called on the chat thread for every new user
        input. ``wait=True`` is for explicit teardown (``/tts off``/exit).
        """
        with self._lock:
            self._generation += 1
            self._queued_generation = None
            player = self._player
            stream = self._active_stream
            self._active_stream = None
            if self._enabled:
                self._status = "idle"
            self._last_finished_generation = None
        if stream is not None:
            stream.cancel()
            if wait:
                stream.join(min(self._join_timeout, 1.0))
        self._drain(self._text_queue)
        self._drain(self._audio_queue)
        interrupt = getattr(player, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except Exception:  # noqa: BLE001 - best effort
                pass

    def disable(self) -> None:
        """``/tts off``: stop playback, release threads and the audio device."""
        self.stop(wait=True)
        self._teardown_threads()
        with self._lock:
            self._enabled = False
            self._status = "disabled"

    def shutdown(self) -> None:
        """Release threads, audio device and network objects for process exit."""
        self.stop(wait=True)
        self._teardown_threads()
        with self._lock:
            self._enabled = False
            self._started = False
            self._status = "disabled"

    # -------------------------------------------------------------- streaming
    def start_stream(
        self,
        response_id: str,
        conversation_id: str,
        *,
        submitted_at: Optional[float] = None,
        router_completed_at: Optional[float] = None,
    ) -> Optional[TTSStream]:
        """Begin one streaming TTS session for a reply (Phase B).

        Called on the chat thread when ``response_started`` arrives. Never
        performs network I/O here; the sender thread opens the session, so a
        slow provider cannot delay text generation. Returns ``None`` when TTS
        is disabled or not initialized.
        """
        with self._lock:
            if not self._enabled or not self._started or self._config is None:
                return None
            previous = self._active_stream
            self._active_stream = None
            self._generation += 1
            generation = self._generation
            self._queued_generation = None
            self._status = "streaming"
            now = time.monotonic()
            timings = {
                "response_started_at": now,
                "tts_queued_at": now,
            }
            if submitted_at is not None:
                timings["user_submitted_at"] = submitted_at
            if router_completed_at is not None:
                timings["router_completed_at"] = router_completed_at
            self._timings[generation] = timings
            self._prune_timings()
            config = self._config
            client = self._client
            player = self._player

        # Bluetooth/speaker switch: re-open the audio device for the new reply
        # (before any of its audio exists) instead of staying on the device
        # that was default when TTS was enabled. Never blocks on the network.
        refresh = getattr(player, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:  # noqa: BLE001 - playback is best effort
                pass

        if previous is not None:
            # No join: this runs on the chat thread at response_started.
            # cancel() is non-blocking and hands session teardown to a
            # closer/sender thread; a stale reply can never mix audio into
            # the new generation because audio is generation-tagged.
            previous.cancel()
        # Any queued chunk belongs to an older reply: the new generation has
        # not produced any audio yet, so nothing valid can be discarded.
        self._drain(self._audio_queue)

        stream = TTSStream(
            self, config, client, generation, response_id, conversation_id
        )
        with self._lock:
            if generation != self._generation or not self._started:
                return None
            self._streams = {item for item in self._streams if not item.done}
            self._active_stream = stream
            self._streams.add(stream)
        stream.start()
        return stream

    def _record(self, generation: int, name: str) -> None:
        """Record the first occurrence of one latency mark."""
        with self._lock:
            if generation != self._generation:
                return
            timings = self._timings.get(generation)
            if timings is None:
                return
            timings.setdefault(name, time.monotonic())

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return (
                self._started
                and not self._stop_event.is_set()
                and generation == self._generation
            )

    def cancel_stream(self, stream: Optional[TTSStream]) -> None:
        """Cancel one streaming reply without waiting for provider I/O.

        Safe to call from the chat event loop: it only flips state, drains
        queued audio for that reply and interrupts the device. A stream that
        was already superseded is closed locally without touching the newer
        reply.
        """
        if stream is None:
            return
        with self._lock:
            active = stream is self._active_stream
            if active:
                self._active_stream = None
                self._generation += 1
                self._queued_generation = None
                if self._enabled:
                    self._status = "idle"
            player = self._player
        stream.cancel()
        if not active:
            return
        self._drain(self._audio_queue)
        interrupt = getattr(player, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except Exception:  # noqa: BLE001 - best effort
                pass

    # ---------------------------------------------------------------- submit
    def submit(self, job: SpeechJob) -> bool:
        """Queue one complete reply. Non-blocking; the newest job wins."""
        text = (job.text or "").strip()
        with self._lock:
            if not self._enabled or not self._started:
                return False
            if not text:
                self._last_error = "TTS_TEXT_EMPTY"
                return False
            max_chars = (
                self._config.max_text_chars
                if self._config is not None
                else 0
            )
            if max_chars and len(text) > max_chars:
                self._last_error = (
                    f"TTS_TEXT_TOO_LONG: {len(text)} > {max_chars}"
                )
                return False
            if _looks_like_protocol_header(text):
                self._last_error = "TTS_PROTOCOL_HEADER"
                return False

            previous = self._active_stream
            self._active_stream = None
            generation = self._generation + 1
            now = time.monotonic()
            queued = dataclasses.replace(
                job,
                text=text,
                generation=generation,
                reply_completed_at=job.reply_completed_at or now,
            )
            if not self._enqueue_newest(queued):
                self._last_error = "TTS_QUEUE_FULL"
                return False
            # Publish only after the job is actually queued: a failed enqueue
            # must never cut off the reply that is currently playing.
            self._generation = generation
            self._queued_generation = generation
            self._timings[generation] = {
                "reply_completed_at": queued.reply_completed_at or now,
                "tts_queued_at": now,
            }
            self._prune_timings()
        if previous is not None:
            # Never join here: submit() is on the reply path. The cancelled
            # stream closes its own session asynchronously.
            previous.cancel()
        return True

    def _enqueue_newest(self, job: SpeechJob) -> bool:
        """Drop an unstarted older job so at most one reply waits."""
        try:
            self._text_queue.put_nowait(job)
            return True
        except queue.Full:
            pass
        try:
            self._text_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._text_queue.put_nowait(job)
            return True
        except queue.Full:
            # The worker may have consumed the old job between the two calls.
            try:
                self._text_queue.get_nowait()
                self._text_queue.put_nowait(job)
                return True
            except (queue.Empty, queue.Full):
                return False

    # ---------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        while True:
            try:
                job = self._text_queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            if job is _STOP:
                return
            with self._lock:
                if self._queued_generation == job.generation:
                    self._queued_generation = None
            self._process(job)

    def _process(self, job: SpeechJob) -> None:
        with self._lock:
            if not self._started or job.generation != self._generation:
                return
            self._status = "synthesizing"
            timings = self._timings.setdefault(job.generation, {})
            timings["tts_request_started_at"] = time.monotonic()
            client = self._client
            player = self._player

        refresh = getattr(player, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:  # noqa: BLE001 - playback is best effort
                pass

        try:
            first_chunk = True
            for chunk in client.stream_pcm(job.text):
                if not chunk:
                    continue
                with self._lock:
                    if job.generation != self._generation:
                        return
                    if first_chunk:
                        timings["first_audio_received_at"] = time.monotonic()
                        first_chunk = False
                if not self._put_audio(job.generation, chunk):
                    return
        except Exception as exc:  # noqa: BLE001 - TTS-isolated
            self._fail(job.generation, exc)
            return

        with self._lock:
            if job.generation != self._generation:
                return
            timings["tts_stream_finished_at"] = time.monotonic()
        # End marker lets the player report playback completion and go idle.
        self._put_audio(job.generation, _END_OF_AUDIO)

    def _put_audio(self, generation: int, data: Any) -> bool:
        """Block the TTS worker (never the chat thread) on the bounded queue."""
        while True:
            with self._lock:
                if (
                    not self._started
                    or self._stop_event.is_set()
                    or generation != self._generation
                ):
                    return False
            try:
                self._audio_queue.put((generation, data), timeout=0.1)
                return True
            except queue.Full:
                continue

    def _fail(self, generation: int, exc: Exception) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._last_error = _safe_error(exc)
            self._status = "error"
            self._invalidate_locked(generation)
        self._discard_audio_generation(generation)

    def _invalidate_locked(self, generation: int) -> None:
        """Invalidate ``generation`` without cancelling a newer queued reply."""
        if generation != self._generation:
            return
        queued = self._queued_generation
        if queued is not None and queued > generation:
            # The next reply is already waiting; keep it valid.
            self._generation = queued
        else:
            self._generation += 1
            self._queued_generation = None

    def _discard_audio_generation(self, generation: int) -> None:
        """Drop queued chunks from one failed reply, keeping a newer reply."""
        kept: list[tuple[int, Any]] = []
        while True:
            try:
                item = self._audio_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] != generation:
                kept.append(item)
        for item in kept:
            while True:
                if self._stop_event.is_set():
                    return
                try:
                    self._audio_queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue

    # ---------------------------------------------------------------- player
    def _player_loop(self) -> None:
        while True:
            try:
                generation, data = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            if data is _STOP:
                return
            with self._lock:
                if (
                    not self._started
                    or generation != self._generation
                ):
                    continue
                player = self._player
                if data is _END_OF_AUDIO:
                    timings = self._timings.setdefault(generation, {})
                    timings["playback_finished_at"] = time.monotonic()
                    self._last_finished_generation = generation
                    self._status = "idle"
                    continue

            with self._lock:
                if generation == self._generation:
                    timings = self._timings.setdefault(generation, {})
                    timings.setdefault(
                        "first_audio_write_started_at", time.monotonic()
                    )
            try:
                player.write(data)
            except Exception as exc:  # noqa: BLE001 - TTS-isolated
                with self._lock:
                    if generation == self._generation:
                        self._last_error = _safe_error(exc)
                        self._status = "error"
                        self._invalidate_locked(generation)
                self._discard_audio_generation(generation)
                continue

            with self._lock:
                if generation == self._generation:
                    timings = self._timings.setdefault(generation, {})
                    timings.setdefault(
                        "first_audio_played_at", time.monotonic()
                    )
                    self._status = "playing"

    # ---------------------------------------------------------------- status
    def status(self) -> TTSStatus:
        described = self._description()
        with self._lock:
            return TTSStatus(
                enabled=self._enabled,
                state=self._status,
                model=described["model"],
                voice=described["voice"],
                region=described["region"],
                sample_rate=described["sample_rate"],
                waiting_jobs=self._text_queue.qsize(),
                queued_audio_chunks=self._audio_queue.qsize(),
                generation=self._generation,
                last_error=self._last_error,
                speech_rate=described["rate"],
                flush_strategy=described["flush_strategy"],
                output_device=described["output_device"],
            )

    def _output_device_name(self) -> str:
        with self._lock:
            player = self._player
        name = getattr(player, "device_name", None)
        return str(name) if name else "system default"

    def _description(self) -> dict:
        with self._lock:
            config = self._config
        if config is not None:
            return {
                "model": config.model,
                "voice": config.voice,
                "region": config.region,
                "sample_rate": config.sample_rate,
                "rate": config.speech_rate,
                "flush_strategy": config.flush_strategy,
                "output_device": self._output_device_name(),
            }
        described = describe_tts_config(self._status_env)
        described["output_device"] = (
            described.get("output_device") or "system default"
        )
        return described

    def metrics(self, generation: Optional[int] = None) -> Optional[dict]:
        """Timing report for one reply, in ms.

        Without an explicit ``generation`` this reports the most recently
        finished reply; the CLI passes the reply's generation so text that
        completed while audio is still pending is never confused with it.
        """
        with self._lock:
            if generation is None:
                generation = self._last_finished_generation
                if generation is None:
                    if not self._timings:
                        return None
                    generation = max(self._timings)
            timings = dict(self._timings.get(generation, {}))
        if not timings:
            return None
        user = timings.get("user_submitted_at")
        return {
            "generation": generation,
            # Unified user-submit-relative marks (Phase B / CLI report).
            "user_to_router_ms": _ms(
                user, timings.get("router_completed_at")
            ),
            "user_to_response_started_ms": _ms(
                user, timings.get("response_started_at")
            ),
            "router_to_response_ms": _ms(
                timings.get("router_completed_at"),
                timings.get("response_started_at"),
            ),
            "user_to_first_text_ms": _ms(
                user, timings.get("first_text_at")
            ),
            "user_to_first_audio_ms": _ms(
                user, timings.get("first_audio_write_started_at")
            ),
            "user_to_text_done_ms": _ms(
                user, timings.get("response_completed_at")
            ),
            "response_to_first_text_ms": _ms(
                timings.get("response_started_at"),
                timings.get("first_text_at"),
            ),
            "first_text_to_first_segment_ms": _ms(
                timings.get("first_text_at"),
                timings.get("first_segment_ready_at"),
            ),
            "tts_connect_ms": _ms(
                timings.get("tts_connect_started_at"),
                timings.get("tts_connect_finished_at"),
            ),
            "first_segment_to_first_pcm_ms": _ms(
                timings.get("first_segment_sent_at"),
                timings.get("first_audio_received_at"),
            ),
            "first_pcm_to_write_ms": _ms(
                timings.get("first_audio_received_at"),
                timings.get("first_audio_write_started_at"),
            ),
            # Phase B: streaming latencies (None until each mark exists).
            "response_to_first_display_ms": _ms(
                timings.get("response_started_at"),
                timings.get("first_display_delta_at"),
            ),
            "response_to_first_segment_ms": _ms(
                timings.get("response_started_at"),
                timings.get("first_segment_ready_at"),
            ),
            "response_to_session_ms": _ms(
                timings.get("response_started_at"),
                timings.get("tts_session_started_at"),
            ),
            "first_segment_to_first_audio_played_ms": _ms(
                timings.get("first_segment_ready_at"),
                timings.get("first_audio_played_at"),
            ),
            "first_segment_sent_to_first_audio_played_ms": _ms(
                timings.get("first_segment_sent_at"),
                timings.get("first_audio_played_at"),
            ),
            "first_display_delta_to_first_audio_played_ms": _ms(
                timings.get("first_display_delta_at"),
                timings.get("first_audio_played_at"),
            ),
            "response_started_to_first_audio_played_ms": _ms(
                timings.get("response_started_at"),
                timings.get("first_audio_played_at"),
            ),
            "response_to_playback_done_ms": _ms(
                timings.get("response_completed_at")
                or timings.get("reply_completed_at"),
                timings.get("playback_finished_at"),
            ),
            # Phase A: complete-reply latencies.
            "reply_to_speech_start_ms": _ms(
                timings.get("reply_completed_at"),
                timings.get("first_audio_played_at"),
            ),
            "reply_to_request_start_ms": _ms(
                timings.get("reply_completed_at"),
                timings.get("tts_request_started_at"),
            ),
            "request_to_first_audio_ms": _ms(
                timings.get("tts_request_started_at"),
                timings.get("first_audio_received_at"),
            ),
            "first_audio_received_to_played_ms": _ms(
                timings.get("first_audio_received_at"),
                timings.get("first_audio_played_at"),
            ),
            "first_audio_to_playback_done_ms": _ms(
                timings.get("first_audio_played_at"),
                timings.get("playback_finished_at"),
            ),
            "reply_to_playback_done_ms": _ms(
                timings.get("reply_completed_at"),
                timings.get("playback_finished_at"),
            ),
            "playback_finished": "playback_finished_at" in timings,
        }

    def _prune_timings(self) -> None:
        if len(self._timings) <= 20:
            return
        for key in sorted(self._timings)[:-20]:
            self._timings.pop(key, None)

    # --------------------------------------------------------------- cleanup
    def _teardown_threads(self) -> None:
        with self._lock:
            self._started = False
            self._stop_event.set()
            self._queued_generation = None
            stream = self._active_stream
            self._active_stream = None
            streams = list(self._streams)
            self._streams.clear()
            worker, player_thread = self._worker, self._player_thread
            player, client = self._player, self._client
            self._worker = None
            self._player_thread = None
            self._player = None
            self._client = None
        if stream is not None:
            stream.cancel()
            stream.join(min(self._join_timeout, 1.0))
        # Superseded streams were cancelled without blocking the chat thread.
        # Process exit still gets one bounded deadline to join them.
        deadline = time.monotonic() + self._join_timeout
        for extra in streams:
            if extra is stream:
                continue
            extra.cancel()
            extra.join(max(0.0, deadline - time.monotonic()))
        self._drain(self._text_queue)
        self._drain(self._audio_queue)
        self._put_control(self._text_queue, _STOP)
        self._put_control(self._audio_queue, (self._generation, _STOP))
        for thread in (worker, player_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=self._join_timeout)
        self._close_quietly(player)
        self._close_quietly(client)

    def _put_control(self, target: queue.Queue, item: Any) -> None:
        for _ in range(2):
            try:
                target.put_nowait(item)
                return
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    pass

    @staticmethod
    def _drain(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _close_quietly(obj: Any) -> None:
        closer = getattr(obj, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001 - cleanup must never raise
                pass


def _looks_like_protocol_header(text: str) -> bool:
    head = text.lstrip()[:8].upper()
    return head.startswith("VOICE=") or head.startswith("TEXT=")


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, TTSError):
        return str(exc)
    return f"TTS 内部错误: {type(exc).__name__}"


__all__ = ["SpeechJob", "TTSManager", "TTSStatus", "TTSStream"]
