"""Half-duplex voice coordinator for the formal REPL.

State machine (mic off by default)::

    DISABLED --/mic on--> LISTENING
    LISTENING --VAD speech start--> RECORDING
    RECORDING --ASR Final--> THINKING (mic paused)
    THINKING --first TTS audio--> SPEAKING
    SPEAKING --that generation's playback end--> LISTENING (after cooldown)
    THINKING --text reply done, no speech--> LISTENING
    any state --/mic off or error--> DISABLED

Only an ASR ``FinalUtterance`` accepted while ``RECORDING`` becomes a user
message. Partials are display-only and never reach this module. While
``THINKING``/``SPEAKING`` the ASR input is paused and its VAD state reset, so
Ava can never recognize her own speech (no AEC, therefore no barge-in).

All state transitions run on the asyncio loop. ASR and TTS callbacks arrive on
their own threads and are marshalled with ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from companion.asr.manager import FinalUtterance, SpeechActivity
from companion.voice.barge_in_config import BargeInConfig
from companion.voice.bridge import (
    BridgeFailure,
    PartialTranscript,
    TurnClosed,
    VoiceBridge,
)

DEFAULT_COOLDOWN_SECONDS = 0.3
MAX_COOLDOWN_SECONDS = 2.0


class VoiceState(str, Enum):
    DISABLED = "DISABLED"
    LISTENING = "LISTENING"
    RECORDING = "RECORDING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass
class VoiceStats:
    """Counters used by tests and ``/mic status``; never contains audio."""

    finals_accepted: int = 0
    stale_finals: int = 0
    turn_errors: int = 0
    playback_events: int = 0


class VoiceCoordinator:
    """Owns mic lifecycle, half-duplex state and input pacing.

    ``manager_factory`` returns an object with the ASR manager contract
    (``start``, ``run``, ``stop``, ``shutdown``, ``set_listeners`` and
    ``set_input_enabled``). It is called only when the user enables the mic,
    so a missing API key or microphone can never break text chat.
    """

    def __init__(
        self,
        *,
        manager_factory: Callable[[], Any],
        tts_manager: Any = None,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_notice: Optional[Callable[[str], None]] = None,
        barge_in: Optional[BargeInConfig] = None,
        on_barge_in: Optional[Callable[[int], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._manager_factory = manager_factory
        self._tts = tts_manager
        self.cooldown_seconds = max(
            0.0, min(MAX_COOLDOWN_SECONDS, float(cooldown_seconds))
        )
        self._on_speech_started = on_speech_started
        self._on_notice = on_notice
        self._barge_config = barge_in
        self._barge_enabled = bool(barge_in is not None and barge_in.enabled)
        self._on_barge_in = on_barge_in
        self._partial_listener: Optional[Callable[[int, str], None]] = None
        # Newest display sequence accepted per utterance. A delayed partial
        # must never overwrite newer text or a later utterance's subtitle.
        self._partial_sequence: dict[int, int] = {}
        self._barge_armed = False
        self._barge_disabled_reason: Optional[str] = None
        self._interrupted_generation: Optional[int] = None
        self._echo_tail_task: Optional[asyncio.Task] = None
        self._echo_tail_epoch = 0
        # Monotonic token for one reply turn. Callbacks from an older turn
        # (a cancelled reply finishing late) are ignored by token mismatch.
        self._turn_token = 0
        self.barge_stats: dict[str, Any] = {
            "confirmed": 0,
            "stop_latency_ms": None,
            "fallbacks": 0,
        }
        self.logger = logger or logging.getLogger("companion.voice")

        self._state = VoiceState.DISABLED
        self._manager: Any = None
        self._bridge: Optional[VoiceBridge] = None
        self._runner: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._consume_task: Optional[asyncio.Task] = None
        self._closed = False

        # One accepted turn at a time. The REPL consumes this queue serially,
        # so a second final can never start an overlapping reply.
        self._finals: "asyncio.Queue[FinalUtterance]" = asyncio.Queue(maxsize=1)
        self._current_uid: Optional[int] = None
        self._expected_generation: Optional[int] = None
        self._playback_terminal_generation: Optional[int] = None
        self._playback_delay = 0.0
        self._turn_finished = False
        self._resume_task: Optional[asyncio.Task] = None
        self._resume_epoch = 0
        self._last_error: Optional[str] = None
        # Set exactly while a proactive message may be delivered or while we
        # are waiting for a proactive reply's real playback to finish. A
        # proactive delivery pauses it through begin_turn().
        self._proactive_allowed = asyncio.Event()
        self._proactive_allowed.set()
        self.stats = VoiceStats()

        self._tts_registered = False
        register = getattr(tts_manager, "add_playback_listener", None)
        if callable(register):
            register(self._on_playback_threadsafe)
            self._tts_registered = True

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def enabled(self) -> bool:
        return self._state != VoiceState.DISABLED

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _set_state(self, state: VoiceState) -> None:
        self._state = state
        if state in (VoiceState.DISABLED, VoiceState.LISTENING):
            self._proactive_allowed.set()
        else:
            self._proactive_allowed.clear()

    def _notice(self, text: str) -> None:
        callback = self._on_notice
        if callback is None:
            return
        try:
            callback(text)
        except Exception:  # noqa: BLE001 - display must not break voice
            pass

    def should_deliver_proactive(self) -> bool:
        """Proactive messages may not preempt a recording or a reply."""
        return self._state in (
            VoiceState.DISABLED,
            VoiceState.LISTENING,
        )

    async def wait_until_proactive_allowed(self) -> None:
        """Wait until no user turn or Ava playback is in progress.

        Used to *defer* (never silently drop) a proactive message. It returns
        as soon as the coordinator is LISTENING/DISABLED again, which happens
        only after the real playback-complete event and the existing cooldown.
        """
        while not self.should_deliver_proactive():
            await self._proactive_allowed.wait()

    def description(self) -> str:
        manager = self._manager
        device = "未启用"
        model = "-"
        if manager is not None:
            microphone = getattr(manager, "microphone", None)
            device = str(
                getattr(microphone, "description", "默认输入设备")
            )
            config = getattr(manager, "config", None)
            model = str(getattr(config, "model", "-"))
        return (
            f"mic: {self._state.value.lower()}  device={device}"
            f"  model={model}"
        )

    def barge_in_status(self) -> dict[str, Any]:
        """Aggregated barge-in/AEC diagnostics for ``/mic status``."""
        result: dict[str, Any] = {
            "enabled": self._barge_enabled,
            "armed": self._barge_armed,
            "fallback_reason": self._barge_disabled_reason,
            **self.barge_stats,
        }
        manager = self._manager
        getter = getattr(manager, "barge_in_stats", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                result["asr"] = getter()
        dropped = getattr(manager, "dropped_audio_chunks", None)
        if dropped is not None:
            result["mic_dropped_chunks"] = dropped
        return result

    # --------------------------------------------------------------- lifecycle
    async def enable(self) -> tuple[bool, str]:
        """Start the ASR pipeline and begin persistent listening."""
        if self._closed:
            return False, "mic: off (会话已结束)"
        if self._state != VoiceState.DISABLED and self._manager is not None:
            return True, "mic: already listening"

        bridge = VoiceBridge(self._loop)
        try:
            manager = self._manager_factory()
        except Exception as exc:  # noqa: BLE001 - config/device failure
            message = _safe_message(exc)
            self._last_error = message
            bridge.close()
            self._set_state(VoiceState.DISABLED)
            return False, f"mic: off ({message})"

        setter = getattr(manager, "set_listeners", None)
        if callable(setter):
            setter(
                on_final=bridge.publish_final,
                on_speech=bridge.publish_speech,
                on_turn_closed=bridge.publish_turn_closed,
                # The formal REPL prints the recognized text once itself;
                # the standalone demo keeps the renderer's final line.
                render_finals=False,
            )
        # Provisional transcripts are delivered to the surface for live
        # display. Attached separately so listener implementations written
        # before partials existed keep their signature.
        partial_setter = getattr(manager, "set_partial_listener", None)
        if callable(partial_setter):
            with contextlib.suppress(Exception):
                partial_setter(bridge.publish_partial)

        if self._barge_config is not None:
            attach = getattr(
                manager, "set_barge_in_unavailable_listener", None
            )
            if callable(attach):
                with contextlib.suppress(Exception):
                    attach(self._on_barge_in_unavailable_threadsafe)
            self._barge_disabled_reason = None
            self._barge_armed = False
            self._interrupted_generation = None

        try:
            await asyncio.to_thread(manager.start)
        except Exception as exc:  # noqa: BLE001 - VAD/device/key failure
            message = _safe_message(exc)
            self._last_error = message
            await asyncio.to_thread(_quiet_shutdown, manager)
            bridge.close()
            self._set_state(VoiceState.DISABLED)
            return False, f"mic: off ({message})"

        self._bridge = bridge
        self._manager = manager
        self._stopping.clear()
        self._runner = threading.Thread(
            target=self._run_manager,
            args=(manager,),
            name="ava-voice-asr",
            daemon=True,
        )
        self._runner.start()
        self._consume_task = asyncio.create_task(
            self._consume(bridge), name="ava-voice-consume"
        )
        self._set_state(VoiceState.LISTENING)
        self._last_error = None
        self._notice("[mic] Listening...")
        microphone = getattr(manager, "microphone", None)
        device = getattr(microphone, "description", "默认输入设备")
        return True, f"mic: on  device={device}"

    async def disable(self) -> str:
        """Stop listening, abort an in-flight recording and release threads."""
        self._cancel_resume()
        self._cancel_echo_tail()
        if self._state == VoiceState.DISABLED and self._manager is None:
            return "mic: off"
        if self._current_uid is not None:
            self._clear_partial(self._current_uid)
        self._set_state(VoiceState.DISABLED)
        self._current_uid = None
        self._expected_generation = None
        self._turn_finished = False
        self._playback_terminal_generation = None
        self._barge_armed = False
        self._interrupted_generation = None
        await self._stop_manager()
        self._notice("[mic] off")
        return "mic: off"

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self.disable()
        if self._tts_registered:
            remove = getattr(self._tts, "remove_playback_listener", None)
            if callable(remove):
                with contextlib.suppress(Exception):
                    remove(self._on_playback_threadsafe)
            self._tts_registered = False

    def _run_manager(self, manager: Any) -> None:
        try:
            manager.run()
        except Exception as exc:  # noqa: BLE001 - worker death is reported
            bridge = self._bridge
            if bridge is not None and not self._stopping.is_set():
                bridge.notify_failure(
                    f"语音输入已停止（{type(exc).__name__}）。"
                )
        finally:
            self._runner = None

    async def _stop_manager(self, *, cancel_consumer: bool = True) -> None:
        self._stopping.set()
        manager, self._manager = self._manager, None
        bridge, self._bridge = self._bridge, None
        runner = self._runner
        if manager is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(manager.stop)
            if runner is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(runner.join, 3.0)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(_quiet_shutdown, manager)
        if bridge is not None:
            bridge.close()
        task = self._consume_task
        self._consume_task = None
        current = asyncio.current_task()
        if task is not None and task is not current and not task.done():
            if cancel_consumer:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # ---------------------------------------------------------- ASR consumers
    async def _consume(self, bridge: VoiceBridge) -> None:
        while True:
            try:
                item = await bridge.next_item()
            except asyncio.CancelledError:
                raise
            if item is None:
                return
            try:
                if isinstance(item, FinalUtterance):
                    self._accept_final(item)
                elif isinstance(item, SpeechActivity):
                    self._accept_speech(item)
                elif isinstance(item, PartialTranscript):
                    self._accept_partial(item)
                elif isinstance(item, TurnClosed):
                    self._accept_turn_closed(item)
                elif isinstance(item, BridgeFailure):
                    await self._handle_failure(item.message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep consuming
                self.logger.warning(
                    "voice_event_failed type=%s", type(exc).__name__
                )

    def set_partial_listener(
        self, listener: Optional[Callable[[int, str], None]]
    ) -> None:
        """Attach a display-only provisional-transcript callback.

        Called on the asyncio loop, never from an ASR thread, so a surface may
        repaint its input line directly. Partials are never persisted, never
        become a reply, and are cleared by the final result.
        """
        self._partial_listener = listener

    def _accept_partial(self, partial: PartialTranscript) -> None:
        # Only the turn currently being recorded owns the live subtitle. An
        # older utterance's delayed partial must never overwrite a newer turn.
        if self._state is not VoiceState.RECORDING:
            return
        if (
            self._current_uid is not None
            and partial.utterance_id != self._current_uid
        ):
            return
        listener = self._partial_listener
        if listener is None:
            return
        previous = self._partial_sequence.get(partial.utterance_id, -1)
        if partial.sequence <= previous:
            # An out-of-order update from an older revision.
            return
        self._partial_sequence[partial.utterance_id] = partial.sequence
        try:
            listener(partial.utterance_id, partial.text)
        except Exception:  # noqa: BLE001 - display must not break voice
            pass

    def _clear_partial(self, utterance_id: int) -> None:
        """Erase the provisional line for one utterance from the surface.

        Sent as an empty text so the display can distinguish "no partial any
        more" from "no partial yet" and repaint its subtitle.
        """
        self._partial_sequence.pop(int(utterance_id), None)
        current = self._current_uid
        if current is not None and int(utterance_id) != current:
            # A late final/close from an older utterance must not erase the
            # subtitle of the turn that is recording now.
            return
        listener = self._partial_listener
        if listener is None:
            return
        try:
            listener(int(utterance_id), "")
        except Exception:  # noqa: BLE001 - display must not break voice
            pass

    def _accept_speech(self, activity: SpeechActivity) -> None:
        if activity.kind == "barge_in":
            self._accept_barge_in(activity)
            return
        if activity.kind != "started":
            return
        if self._state != VoiceState.LISTENING:
            # The "started" that follows a handled barge-in arrives while we
            # are already RECORDING with the same utterance id; it is not a
            # second turn and must not create another one.
            return
        self._current_uid = activity.utterance_id
        self._set_state(VoiceState.RECORDING)
        # Cancel any pending proactive cycle *before* the user produces a
        # message: a delivered proactive line must not preempt this turn.
        if self._on_speech_started is not None:
            try:
                self._on_speech_started()
            except Exception:  # noqa: BLE001 - scheduling is best effort
                pass

    def _accept_barge_in(self, activity: SpeechActivity) -> None:
        """A confirmed user interruption while Ava is speaking.

        The ASR manager already preserved the pre-roll and is recognizing the
        interruption as its own turn (same utterance id). Here we only stop
        playback and abandon the old reply; recognition keeps running.
        """
        if not self._barge_enabled:
            return
        if self._state != VoiceState.SPEAKING:
            return
        generation = (
            int(activity.generation)
            if activity.generation is not None
            else self._expected_generation
        )
        if generation is None:
            return
        if (
            self._expected_generation is not None
            and generation != self._expected_generation
        ):
            # A signal for a reply that is not the one playing now.
            return
        if self._interrupted_generation == generation:
            # Duplicate signal: stop exactly once per interruption.
            return

        self._interrupted_generation = generation
        self._barge_armed = False
        self.barge_stats["confirmed"] += 1
        # Invalidate the old turn before any callback can observe it.
        self._turn_token += 1
        self._cancel_resume()
        self._cancel_echo_tail()

        # 1. Stop generation immediately; never wait for ASR to finish.
        self._stop_tts_now(activity.at)
        # 2. Keep the confirmed user turn continuous after playback stops. The
        #    pipeline uses remaining real references, then bounded silent far
        #    frames, so reference exhaustion can never drop the user's words.
        manager = self._manager
        confirm = getattr(manager, "confirm_barge_in", None)
        if callable(confirm):
            with contextlib.suppress(Exception):
                confirm(generation)
        # 3. Tell the embedder to cancel the old reply task (LLM stream).
        callback = self._on_barge_in
        if callback is not None:
            try:
                callback(generation)
            except Exception:  # noqa: BLE001 - old reply cleanup is best effort
                pass
        # 4. Preserve the user's turn for the ASR already in flight.
        self._current_uid = activity.utterance_id
        self._expected_generation = None
        self._playback_terminal_generation = None
        self._turn_finished = True
        self._set_state(VoiceState.RECORDING)
        self._schedule_echo_tail()
        if self._on_speech_started is not None:
            try:
                self._on_speech_started()
            except Exception:  # noqa: BLE001 - scheduling is best effort
                pass
        manager = self._manager
        end = getattr(manager, "end_barge_in", None)
        if callable(end):
            with contextlib.suppress(Exception):
                # Keep already-received far-end frames for the echo tail,
                # but accept no new playback from the stopped generation.
                end(generation)

    def _stop_tts_now(self, confirmed_at: float) -> None:
        stop = getattr(self._tts, "stop", None)
        if not callable(stop):
            return
        try:
            stop(wait=False)
        except TypeError:
            with contextlib.suppress(Exception):
                stop()
        except Exception:  # noqa: BLE001 - playback is already being stopped
            pass
        # This measures the stop() call (queue drain + device interrupt
        # request). It is not proof that the speaker is already silent; the
        # TTS manager does not expose a device-silence timestamp.
        elapsed = round(
            max(0.0, (time.monotonic() - confirmed_at) * 1000.0), 1
        )
        self.barge_stats["stop_latency_ms"] = elapsed
        self.barge_stats["stop_call_ms"] = elapsed

    def _schedule_echo_tail(self) -> None:
        """Switch the ASR input back to its normal path after the echo tail."""
        self._cancel_echo_tail()
        delay = (
            self._barge_config.echo_tail_seconds
            if self._barge_config is not None
            else 0.3
        )
        self._echo_tail_epoch += 1
        epoch = self._echo_tail_epoch

        async def _tail() -> None:
            if delay > 0:
                await asyncio.sleep(delay)
            if epoch != self._echo_tail_epoch:
                return
            if self._state != VoiceState.RECORDING:
                return
            manager = self._manager
            setter = getattr(manager, "set_input_mode", None)
            if callable(setter):
                with contextlib.suppress(Exception):
                    setter("NORMAL")
            end_tail = getattr(manager, "end_echo_tail", None)
            if callable(end_tail):
                with contextlib.suppress(Exception):
                    end_tail()

        self._echo_tail_task = asyncio.create_task(
            _tail(), name="ava-voice-echo-tail"
        )

    def _cancel_echo_tail(self) -> None:
        self._echo_tail_epoch += 1
        task = self._echo_tail_task
        self._echo_tail_task = None
        if task is not None and not task.done():
            task.cancel()

    def _accept_final(self, final: FinalUtterance) -> None:
        self._clear_partial(final.utterance_id)
        if self._state != VoiceState.RECORDING:
            # A duplicate, late or already-muted result must never start a
            # second turn (Ava's own voice is not recognized because input is
            # paused while speaking).
            self.stats.stale_finals += 1
            return
        self._cancel_resume()
        self._cancel_echo_tail()
        self._set_state(VoiceState.THINKING)
        self._pause_input()
        self._current_uid = None
        self._expected_generation = None
        self._playback_terminal_generation = None
        self._turn_finished = False
        self._interrupted_generation = None
        self.stats.finals_accepted += 1
        try:
            self._finals.put_nowait(final)
        except asyncio.QueueFull:
            self.stats.stale_finals += 1

    def _accept_turn_closed(self, closed: TurnClosed) -> None:
        self._clear_partial(closed.utterance_id)
        if self._state == VoiceState.RECORDING:
            # Recognition failed, timed out, was aborted, or produced no text
            # (its final would have moved us to THINKING). Go back to a safe
            # listening state instead of waiting forever. The resume scheduler
            # only runs from THINKING/SPEAKING, so leave RECORDING first.
            self._current_uid = None
            self._turn_finished = True
            self._set_state(VoiceState.THINKING)
            self._schedule_resume(0.0)

    async def _handle_failure(self, message: str) -> None:
        self.stats.turn_errors += 1
        self._last_error = message
        self._set_state(VoiceState.DISABLED)
        self._notice(f"[mic] error: {message}")
        await self._stop_manager(cancel_consumer=False)

    # --------------------------------------------------------- turn interface
    async def next_final(self) -> FinalUtterance:
        """Wait for the next accepted user turn (consumed serially)."""
        return await self._finals.get()

    def begin_turn(self) -> int:
        """Mark the start of a reply, including a keyboard-submitted one.

        Idempotent: it pauses input and arms the next playback wait. Returns a
        turn token; pass it to :meth:`on_reply_started` and
        :meth:`on_turn_finished` so callbacks from a reply that was interrupted
        (and is unwinding late) cannot restore the newer state. Called by the
        REPL right before every accepted input so keyboard and voice share the
        same half-duplex rules.
        """
        self._turn_token += 1
        token = self._turn_token
        if self._state == VoiceState.DISABLED:
            return token
        self._cancel_resume()
        self._cancel_echo_tail()
        self._pause_input()
        self._set_state(VoiceState.THINKING)
        self._expected_generation = None
        self._playback_terminal_generation = None
        self._turn_finished = False
        self._current_uid = None
        self._interrupted_generation = None
        return token

    def on_reply_started(
        self,
        generation: Optional[int] = None,
        *,
        token: Optional[int] = None,
    ) -> None:
        """Ava's spoken reply started (first TTS generation is now playing)."""
        if generation is None:
            return
        if token is not None and token != self._turn_token:
            return
        if self._state != VoiceState.THINKING:
            return
        self._expected_generation = int(generation)
        self._playback_terminal_generation = None
        self._set_state(VoiceState.SPEAKING)
        self._arm_barge_in(int(generation))

    def _arm_barge_in(self, generation: int) -> None:
        """Enable real-time detection only if the AEC channel is really ready.

        Arming is tied to an actual playback reference generation, not merely
        to ``on_reply_started``: the manager refuses unless the optional
        pipeline is loaded and healthy, and playback frames must actually
        arrive before detection can confirm anything.

        Recovery is per-generation: a failure in one reply must not disable
        barge-in for every later reply, so the sticky failure reason is cleared
        before each new attempt.
        """
        self._barge_disabled_reason = None
        if not self._barge_enabled:
            self._barge_armed = False
            self._pause_input()
            return
        manager = self._manager
        begin = getattr(manager, "begin_barge_in", None)
        if not callable(begin):
            self._barge_armed = False
            self._pause_input()
            return
        try:
            armed = bool(begin(generation))
        except Exception:  # noqa: BLE001 - optional side channel
            armed = False
        self._barge_armed = armed
        if not armed:
            # Half-duplex fallback: keep the mic paused while Ava speaks.
            self._pause_input()

    def _on_barge_in_unavailable_threadsafe(self, reason: str) -> None:
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(
                self._handle_barge_in_unavailable, reason
            )
        except RuntimeError:
            pass

    def _handle_barge_in_unavailable(self, reason: str) -> None:
        """AEC failed mid-reply; fall back without losing a real turn."""
        self.barge_stats["fallbacks"] += 1
        self._barge_armed = False
        if self._barge_disabled_reason is None:
            self._barge_disabled_reason = reason or "aec_unavailable"
            self._notice(
                "[mic] 实时打断不可用，已回退到半双工模式。"
            )
        manager = self._manager
        setter = getattr(manager, "set_input_mode", None)
        if not callable(setter):
            return
        if self._state == VoiceState.RECORDING:
            # A turn is already being recognized; keep it on the ordinary
            # path rather than aborting the user's first words.
            with contextlib.suppress(Exception):
                setter("NORMAL")
        elif self._state in (VoiceState.THINKING, VoiceState.SPEAKING):
            with contextlib.suppress(Exception):
                setter("MUTED")

    def on_turn_finished(
        self,
        delivery: Optional[Mapping[str, Any]],
        *,
        token: Optional[int] = None,
    ) -> None:
        """The text turn is over; resume only when speech is really done."""
        if token is not None and token != self._turn_token:
            # A reply that a barge-in superseded is finishing late.
            return
        if self._state not in (VoiceState.THINKING, VoiceState.SPEAKING):
            return
        self._turn_finished = True
        if self._state == VoiceState.SPEAKING and (
            self._expected_generation is not None
        ):
            if (
                self._playback_terminal_generation
                != self._expected_generation
            ):
                # Still playing: WAIT/IGNORE-as-text never applies here, and
                # a fabricated "idle" is not proof this reply ended.
                return
            self._schedule_resume(self._playback_delay)
            return
        # TTS was off, never started, or the turn produced no spoken reply.
        self._schedule_resume(0.0)

    # ------------------------------------------------------------ TTS playback
    def _on_playback_threadsafe(self, event: Any) -> None:
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._handle_playback, event)
        except RuntimeError:
            pass

    def _handle_playback(self, event: Any) -> None:
        if self._state not in (VoiceState.THINKING, VoiceState.SPEAKING):
            return
        if self._expected_generation is None:
            return
        if int(getattr(event, "generation", -1)) != self._expected_generation:
            # A completion from an older reply must never unmute the mic for
            # the reply that is still speaking.
            return
        self.stats.playback_events += 1
        self._playback_terminal_generation = int(event.generation)
        kind = str(getattr(event, "kind", ""))
        self._playback_delay = (
            self.cooldown_seconds if kind == "finished" else 0.0
        )
        if self._turn_finished:
            self._schedule_resume(self._playback_delay)

    # ------------------------------------------------------------------ resume
    def _pause_input(self) -> None:
        manager = self._manager
        setter = getattr(manager, "set_input_enabled", None)
        if callable(setter):
            with contextlib.suppress(Exception):
                setter(False)

    def _resume_input(self) -> None:
        manager = self._manager
        setter = getattr(manager, "set_input_enabled", None)
        if callable(setter):
            with contextlib.suppress(Exception):
                setter(True)

    def _cancel_resume(self) -> None:
        self._resume_epoch += 1
        task = self._resume_task
        self._resume_task = None
        if task is not None and not task.done():
            task.cancel()

    def _schedule_resume(self, delay: float) -> None:
        self._cancel_resume()
        epoch = self._resume_epoch

        async def _resume() -> None:
            if delay > 0:
                await asyncio.sleep(delay)
            if epoch != self._resume_epoch:
                return
            if self._state not in (VoiceState.THINKING, VoiceState.SPEAKING):
                return
            if not self._turn_finished:
                return
            if (
                self._expected_generation is not None
                and self._playback_terminal_generation
                != self._expected_generation
            ):
                return
            self._finish_resume()

        self._resume_task = asyncio.create_task(
            _resume(), name="ava-voice-resume"
        )

    def _finish_resume(self) -> None:
        self._cancel_echo_tail()
        self._expected_generation = None
        self._playback_terminal_generation = None
        self._playback_delay = 0.0
        self._turn_finished = False
        self._current_uid = None
        self._barge_armed = False
        self._interrupted_generation = None
        self._set_state(VoiceState.LISTENING)
        self._resume_input()
        self._notice("[mic] Listening...")


def _safe_message(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        text = type(exc).__name__
    return text


def _quiet_shutdown(manager: Any) -> None:
    with contextlib.suppress(Exception):
        manager.shutdown()


__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "VoiceCoordinator",
    "VoiceState",
    "VoiceStats",
]
