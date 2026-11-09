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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from companion.asr.manager import FinalUtterance, SpeechActivity
from companion.voice.bridge import (
    BridgeFailure,
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
        if self._state == VoiceState.DISABLED and self._manager is None:
            return "mic: off"
        self._set_state(VoiceState.DISABLED)
        self._current_uid = None
        self._expected_generation = None
        self._turn_finished = False
        self._playback_terminal_generation = None
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

    def _accept_speech(self, activity: SpeechActivity) -> None:
        if activity.kind != "started":
            return
        if self._state != VoiceState.LISTENING:
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

    def _accept_final(self, final: FinalUtterance) -> None:
        if self._state != VoiceState.RECORDING:
            # A duplicate, late or already-muted result must never start a
            # second turn (Ava's own voice is not recognized because input is
            # paused while speaking).
            self.stats.stale_finals += 1
            return
        self._cancel_resume()
        self._set_state(VoiceState.THINKING)
        self._pause_input()
        self._current_uid = None
        self._expected_generation = None
        self._playback_terminal_generation = None
        self._turn_finished = False
        self.stats.finals_accepted += 1
        try:
            self._finals.put_nowait(final)
        except asyncio.QueueFull:
            self.stats.stale_finals += 1

    def _accept_turn_closed(self, closed: TurnClosed) -> None:
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

    def begin_turn(self) -> None:
        """Mark the start of a reply, including a keyboard-submitted one.

        Idempotent: it pauses input and arms the next playback wait. Called by
        the REPL right before every accepted input so keyboard and voice share
        the same half-duplex rules.
        """
        if self._state == VoiceState.DISABLED:
            return
        self._cancel_resume()
        self._pause_input()
        self._set_state(VoiceState.THINKING)
        self._expected_generation = None
        self._playback_terminal_generation = None
        self._turn_finished = False
        self._current_uid = None

    def on_reply_started(self, generation: Optional[int] = None) -> None:
        """Ava's spoken reply started (first TTS generation is now playing)."""
        if generation is None:
            return
        if self._state != VoiceState.THINKING:
            return
        self._expected_generation = int(generation)
        self._playback_terminal_generation = None
        self._set_state(VoiceState.SPEAKING)

    def on_turn_finished(self, delivery: Optional[Mapping[str, Any]]) -> None:
        """The text turn is over; resume only when speech is really done."""
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
        self._expected_generation = None
        self._playback_terminal_generation = None
        self._playback_delay = 0.0
        self._turn_finished = False
        self._current_uid = None
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
