"""Thread and utterance coordinator for the independent ASR demo.

Pipeline::

    Capture Worker -> VAD Worker -> ASR Sender <-> ASR Receiver -> Renderer

All queues are bounded. The capture thread never waits for the network. Every
utterance gets its own ``utterance_id`` and DashScope session, so a late result
from an older utterance can never overwrite a newer one. The VAD boundary owns
when input stops; ASR ``sentence_end`` only confirms text.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import unicodedata
from array import array
from math import sqrt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from companion.asr.client import (
    ASRClientError,
    ASRSession,
    ASRSessionStateError,
    CLOSE_ORIGIN_EXTERNAL,
    CLOSE_ORIGIN_LOCAL_CANCEL,
    CLOSE_ORIGIN_LOCAL_NORMAL,
    CLOSE_ORIGIN_LOCAL_TIMEOUT,
    STATE_TASK_FINISHED,
    classify_failure,
)
from companion.asr.config import ASRConfig, VAD_WINDOW_MS
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
    WAITING_FOR_SPEECH,
    SileroVAD,
    VADEvent,
    VADSegmenter,
)

# The provider accepts small frames; ~100 ms mono 16-bit PCM is 3200 bytes.
SEND_FRAME_BYTES = 3200
SEND_FRAME_SECONDS = 0.1
# AEC processes 20 ms frames; evidence duration is never shorter than one
# frame, which keeps the fused Silero/AEC timeline honest for short chunks.
AEC_FRAME_MS = 20.0


def _pcm_rms(pcm: bytes) -> float:
    """Root-mean-square level of mono int16 PCM (0.0 for silence).

    Used only when a pipeline returns a valid block without reporting its own
    levels. This is a real measurement of the returned PCM, never a
    VAD-only confirmation.
    """
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[:usable])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    return sqrt(total / len(samples))


class InputMode(str, Enum):
    """How the VAD thread treats captured microphone audio.

    ``NORMAL`` is the unchanged recognition path, ``MUTED`` drains and
    discards input (half-duplex while Ava speaks) and ``BARGE_IN`` feeds the
    AEC pipeline's echo-cancelled output to the same Silero segmenter.
    """

    NORMAL = "NORMAL"
    MUTED = "MUTED"
    BARGE_IN = "BARGE_IN"


_STOP = object()
_ABORT = object()

# Joining server sentences into one turn line. No spaces around CJK/full-width
# text or before closing punctuation; a space between Latin sentences/words.
_NO_SPACE_BEFORE = set(",.;:!?%)]}、。，！？；：")
_NO_SPACE_AFTER = set("([{《「“‘")


def _is_wide(char: str) -> bool:
    return unicodedata.east_asian_width(char) in ("W", "F")


def _needs_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        return False
    if _is_wide(left[-1]) or _is_wide(right[0]):
        return False
    if left[-1] in _NO_SPACE_AFTER or right[0] in _NO_SPACE_BEFORE:
        return False
    return True


def _merge_turn_text(texts: Any) -> str:
    """Join confirmed sentence fragments of one speaking turn.

    The server may emit several ``sentence_end`` fragments for one VAD turn;
    the demo prints them as a single line. CJK fragments join directly, Latin
    fragments keep a single space, and closing punctuation never hugs a space.
    """
    parts = [str(text or "").strip() for text in texts]
    parts = [text for text in parts if text]
    if not parts:
        return ""
    merged = parts[0]
    for part in parts[1:]:
        merged += (" " if _needs_space(merged, part) else "") + part
    return merged


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split()) or "-"


@dataclass
class ASRStats:
    """Non-secret session counters for the exit summary."""

    utterances: int = 0
    finalized: int = 0
    empty: int = 0
    aborted: int = 0
    errors: int = 0
    deferred: int = 0
    truncations: int = 0
    unfinished: int = 0
    dropped_control_events: int = 0
    final_latencies_ms: list[float] = field(default_factory=list)
    # Live queue/link diagnostics (updated on the sender/control threads).
    queued_audio_ms: float = 0.0
    oldest_audio_age_ms: Optional[float] = None
    sender_throughput_bps: Optional[float] = None
    connection_open_latency_ms: Optional[float] = None


# --------------------------------------------------------------- control items
@dataclass(frozen=True)
class _SpeechStarted:
    audio: bytes
    at: float
    barge_in: bool = False


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


@dataclass(frozen=True)
class _InputClosed:
    """The sender finished flushing audio and called ``recognize.stop()``.

    Only from this point on may the final-result timeout start: while a
    session is still queued or still uploading audio, waiting is expected and
    must not be reported as a timeout.
    """

    uid: int
    at: float


@dataclass(frozen=True)
class _Mute:
    """Stop feeding microphone audio to the VAD and drop the open turn."""


@dataclass(frozen=True)
class _BargeInUnavailable:
    """The optional AEC channel cannot be trusted for this reply."""

    reason: str


# ------------------------------------------------------- external integrations
class SpeakerObserver(Protocol):
    """Optional, non-blocking voice-print hook owned by the embedding app.

    The ASR demo knows nothing about the model or the identity store. Every
    method must return immediately: ``on_audio`` may only buffer the PCM, and
    ``snapshot`` must be an O(1) read of an already completed result. A slow
    implementation would delay ASR finalization, which is forbidden.
    """

    def on_start(self, utterance_id: int, pcm: bytes) -> None: ...

    def on_audio(self, utterance_id: int, pcm: bytes) -> None: ...

    def on_end(self, utterance_id: int) -> None: ...

    def on_finalized(self, utterance_id: int, success: bool) -> None: ...

    def on_cancel(self, utterance_id: int) -> None: ...

    def snapshot(self, utterance_id: int) -> Any: ...


@dataclass(frozen=True)
class FinalUtterance:
    """One completed user turn delivered to an embedding application.

    Emitted at most once per utterance, only after the local VAD ended *and*
    the ASR session finished successfully with at least one confirmed
    fragment. Empty, cancelled, aborted and failed recognitions never produce
    one. ``speaker`` is an opaque identity snapshot (see
    ``companion.speaker_id.models.SpeakerResult``) or ``None`` when speaker
    identification is disabled; it is deliberately a separate field so the
    transcript text never carries identity state.
    """

    utterance_id: int
    text: str
    completed_at: float
    speaker: Any = None


@dataclass(frozen=True)
class PartialTranscript:
    """A provisional transcript for an utterance still being recognized.

    Display-only: it never becomes a user message, is never persisted and is
    never spoken. ``sequence`` increases per utterance so a surface can discard
    an out-of-order update instead of overwriting newer text with older text.
    """

    utterance_id: int
    text: str
    sequence: int = 0
    is_final: bool = False
    at: float = 0.0


@dataclass(frozen=True)
class SpeechActivity:
    """VAD speech start/end notice for half-duplex orchestration.

    ``kind`` is ``"started"``, ``"ended"`` or ``"barge_in"``. A barge-in
    notice is emitted *before* the matching ``"started"`` notice for the same
    ``utterance_id``; it carries the TTS generation whose playback it
    interrupted. Consumers must not treat this as a transcript.
    """

    kind: str
    utterance_id: int
    at: float
    generation: Optional[int] = None


@dataclass(frozen=True)
class NearEndEvidence:
    """One timestamp-aligned near-end observation for the barge-in gate.

    AEC produces 20 ms frames, Silero VAD makes decisions on 32 ms windows.
    Evidence joins them on the microphone timeline instead of comparing two
    unrelated raw energy values: ``duration_ms`` and ``vad_probability`` come
    from the VAD window(s) overlapping this AEC chunk, while ``near_rms`` and
    ``clean_rms`` are that chunk's measured levels. ``metrics_valid`` is false
    when no trustworthy energy measurement exists; ``reference_valid`` is
    false when the block had no usable far-end reference. Neither may ever
    confirm barge-in.
    """

    generation: Optional[int]
    start_at: float
    duration_ms: float
    vad_probability: Optional[float]
    near_rms: float
    clean_rms: float
    metrics_valid: bool
    reference_valid: bool


class NearEndSpeechGate:
    """Confirm that barge-in speech is the user, not Ava's own leaked echo.

    The AEC pipeline reports, per processed chunk, the pre-cancellation
    ``near_rms`` and the post-cancellation ``clean_rms``. When the near end is
    mostly Ava's echo, cancellation removes it and ``clean_rms`` collapses;
    a real user interjection survives cancellation, so ``clean_rms`` stays
    well above the *adaptive background/residual-echo floor*.

    Confirmation needs both a minimum amount of evidence time (fused with the
    Silero probability where available) and several consecutive qualifying
    windows, so one echo burst can never stop Ava. Silero's probability is
    used to reject clearly non-speech loud windows (background noise/TV) and
    to slightly relax the energy margin for confident speech; it is never
    sufficient by itself.

    Metrics are mandatory. The previous fallback confirmed barge-in on VAD
    alone when the pipeline did not report energy; that is exactly how Ava's
    own voice was re-recognized as the user, so it is gone. A missing metric
    or a missing far-end reference means *no* confirmation.
    """

    def __init__(
        self,
        *,
        confirm_windows: int = 6,
        clean_rms_floor: Optional[float] = None,
        echo_ratio: Optional[float] = None,
        noise_ratio: float = 3.0,
        initial_noise_floor: float = 60.0,
        confirm_duration_ms: Optional[float] = None,
    ) -> None:
        self.confirm_windows = max(1, int(confirm_windows))
        # Legacy knobs are accepted for compatibility but no longer decide
        # anything: a fixed absolute floor and a fixed near/clean ratio cannot
        # adapt to microphone gain or residual echo.
        self.clean_rms_floor = clean_rms_floor
        self.echo_ratio = echo_ratio
        self.noise_ratio = max(1.0, float(noise_ratio))
        self.initial_noise_floor = max(1.0, float(initial_noise_floor))
        self.confirm_duration_ms = (
            max(1.0, float(confirm_duration_ms))
            if confirm_duration_ms is not None
            else self.confirm_windows * 20.0
        )
        self._noise_floor = self.initial_noise_floor
        self._streak = 0
        self._accumulated_ms = 0.0
        self._observations = 0
        self._generation: Optional[int] = None
        self._last_evidence: Optional[NearEndEvidence] = None
        # Diagnostics only; these never confirm anything.
        self.unmetered_rejections = 0
        self.echo_rejections = 0
        self.noise_rejections = 0

    @property
    def confirmed(self) -> bool:
        return (
            self._streak >= self.confirm_windows
            and self._accumulated_ms >= self.confirm_duration_ms
        )

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def observations(self) -> int:
        """Aligned evidence windows seen since the last reset."""
        return self._observations

    def observe_evidence(self, evidence: NearEndEvidence) -> bool:
        """Consume one aligned AEC/VAD observation; True once confirmed."""
        self._last_evidence = evidence
        self._observations += 1
        if (
            evidence.generation is not None
            and self._generation is not None
            and evidence.generation != self._generation
        ):
            # Evidence from different replies must never accumulate into one
            # confirmation, and an old reply's tail cannot poison a new one.
            self.reset(generation=evidence.generation)
        if evidence.generation is not None:
            self._generation = evidence.generation

        if not evidence.metrics_valid or not evidence.reference_valid:
            self.unmetered_rejections += 1
            self._streak = 0
            self._accumulated_ms = 0.0
            return False

        probability = evidence.vad_probability
        if probability is not None:
            try:
                probability = max(0.0, min(1.0, float(probability)))
            except (TypeError, ValueError):
                probability = None

        level = float(evidence.clean_rms or 0.0)
        threshold = self._noise_floor * self.noise_ratio
        if probability is not None:
            if probability < 0.3:
                # Loud but not speech-like (door, TV, Ava's residual echo):
                # learn it as background instead of confirming a user turn.
                self.noise_rejections += 1
                self._streak = 0
                self._accumulated_ms = 0.0
                self._adapt_noise(level)
                return False
            if probability >= 0.6:
                # Confident Silero speech needs a smaller margin, but the
                # energy must still be clearly above the learned floor.
                threshold *= 0.75

        survived = level >= threshold
        if not survived:
            self.echo_rejections += 1
            self._streak = 0
            self._accumulated_ms = 0.0
            self._adapt_noise(level)
            return False

        self._streak += 1
        duration = max(1.0, float(evidence.duration_ms or 0.0))
        self._accumulated_ms += duration
        return self.confirmed

    def observe(
        self,
        *,
        near_rms: float,
        clean_rms: float,
        metrics_available: bool = True,
        vad_probability: Optional[float] = None,
        reference_valid: bool = True,
        duration_ms: Optional[float] = None,
        generation: Optional[int] = None,
        start_at: Optional[float] = None,
    ) -> bool:
        """Compatibility wrapper that builds :class:`NearEndEvidence`.

        Callers that predate the aligned evidence API still work, but a
        missing metric now means "reject", never "confirm on VAD alone".
        """
        evidence = NearEndEvidence(
            generation=generation,
            start_at=time.monotonic() if start_at is None else float(start_at),
            duration_ms=(
                self.confirm_duration_ms / self.confirm_windows
                if duration_ms is None
                else float(duration_ms)
            ),
            vad_probability=vad_probability,
            near_rms=float(near_rms or 0.0),
            clean_rms=float(clean_rms or 0.0),
            metrics_valid=bool(metrics_available),
            reference_valid=bool(reference_valid),
        )
        return self.observe_evidence(evidence)

    def _adapt_noise(self, level: float) -> None:
        """Slowly track background/residual echo, never a speech level.

        Adaptation is only applied to non-qualifying windows, so real speech
        cannot raise the floor above itself. Upward steps are slower than
        downward ones so a sudden echo burst cannot desensitize the gate
        before the user actually speaks.
        """
        if level <= 0.0:
            self._noise_floor = max(1.0, self._noise_floor * 0.98)
            return
        if level > self._noise_floor:
            self._noise_floor += (level - self._noise_floor) * 0.10
        else:
            self._noise_floor += (level - self._noise_floor) * 0.25
        # Stay bounded: a microphone gain change must not make the gate deaf
        # forever, and an absurd floor must not disable barge-in permanently.
        self._noise_floor = min(max(1.0, self._noise_floor), 12000.0)

    def reset(self, generation: Optional[int] = None) -> None:
        self._streak = 0
        self._accumulated_ms = 0.0
        self._observations = 0
        self._last_evidence = None
        if generation is not None:
            self._generation = generation



class _Utterance:
    """Mutable state for one VAD-bounded utterance."""

    def __init__(
        self,
        uid: int,
        send_queue_frames: int,
        max_utterance_seconds: int = 0,
    ) -> None:
        self.uid = uid
        frames = max(1, int(send_queue_frames))
        if max_utterance_seconds > 0:
            # The documented bound is one whole utterance, not the fixed
            # handful of seconds: while the session handshake is still in
            # progress nothing drains this queue, so a legitimate long
            # statement must still fit. Memory stays bounded by that number.
            frames = max(
                frames,
                int(max_utterance_seconds * 1000 / VAD_WINDOW_MS) + 1,
            )
        self.send_queue: "queue.Queue[Any]" = queue.Queue(maxsize=frames)
        self.session: Optional[ASRSession] = None
        self.opened = threading.Event()
        self.sender_done = threading.Event()
        self.receiver_done = threading.Event()
        # Set by the first thread to observe a fatal failure: the sender and
        # receiver cooperate so buffered audio is no longer uploaded and the
        # receiver's wait is not left unbounded.
        self.abort_event = threading.Event()
        self.open_error: Optional[str] = None
        self.started = False
        self.sender: Optional[threading.Thread] = None
        self.receiver: Optional[threading.Thread] = None
        self.vad_end_at: Optional[float] = None
        self.finals: list[tuple[str, str]] = []
        self.final_ids: set[str] = set()
        self.partial = ""
        self.partial_id = ""
        self.partial_seq = 0
        self.session_done = False
        self.session_error: Optional[str] = None
        self.session_done_at: Optional[float] = None
        self.primary_error: Optional[str] = None
        self.secondary_error: Optional[str] = None
        self.finalized = False
        self.aborted = False
        self.stop_sent = False
        # The upload queue hit its bound while the person was still talking.
        # The turn is retained for diagnostics only: it is an *incomplete*
        # turn, never a successful final built from a truncated prefix.
        self.truncated = False
        self.incomplete_reason: Optional[str] = None
        self.stop_requested = threading.Event()
        self.final_timer: Optional[threading.Timer] = None
        self.final_latency_ms: Optional[float] = None
        # Queue/connection diagnostics. These show whether a slow turn is a
        # one-off connection handshake or a sustained upload deficit, instead
        # of hiding it behind a bigger buffer.
        self.queued_bytes = 0
        self.queued_since: Optional[float] = None
        self.sent_bytes = 0
        self.send_started_at: Optional[float] = None
        self.open_latency_ms: Optional[float] = None
        self.max_queued_bytes = 0
        self.last_server_event_at: Optional[float] = None
        self.last_audio_sent_at: Optional[float] = None


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
        on_final: Optional[Callable[[FinalUtterance], None]] = None,
        on_speech: Optional[Callable[[SpeechActivity], None]] = None,
        on_turn_closed: Optional[Callable[[int, bool], None]] = None,
        on_partial: Optional[Callable[[PartialTranscript], None]] = None,
        render_finals: bool = True,
        speaker_observer: Optional[SpeakerObserver] = None,
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
        self.logger = logging.getLogger("companion.asr")

        # Optional embedding hooks. They are called from the thread that runs
        # the control loop (``run``), never from a sender/receiver worker, and
        # must return quickly: a slow callback would delay utterance
        # bookkeeping. No Ava core module is imported here.
        self._on_final = on_final
        self._on_speech = on_speech
        self._on_turn_closed = on_turn_closed
        self._on_partial = on_partial
        self.render_finals = render_finals
        # Optional voice-print observer. Its methods are called from the
        # control thread and must never block; exceptions are swallowed so a
        # speaker-model problem can never stop recognition.
        self._speaker_observer = speaker_observer

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
        self._threads_lock = threading.Lock()
        self._utterances: dict[int, _Utterance] = {}
        self._open_order: list[_Utterance] = []
        self._current: Optional[_Utterance] = None
        self._active_sessions = 0
        self._next_uid = 0
        # Microphone audio is only segmented while input is enabled. The VAD
        # thread resets the segmenter on every transition, so audio captured
        # before a pause can never become an utterance after resume.
        self._input_enabled = threading.Event()
        self._input_enabled.set()
        self._vad_input_enabled = True
        # Explicit input modes keep ``set_input_enabled`` working while adding
        # the optional AEC barge-in path. ``_input_mode`` is a plain reference
        # assignment; the VAD thread applies the matching profile itself.
        self._input_mode = InputMode.NORMAL
        self._vad_mode = InputMode.NORMAL
        self._echo_canceller: Any = None
        self._normal_vad_profile: Optional[dict[str, Any]] = None
        self._barge_in_vad_threshold = 0.65
        self._barge_in_min_speech_ms = 192
        self._barge_in_pre_roll_ms = 400
        self._barge_in_generation: Optional[int] = None
        self._barge_in_detecting = False
        self._barge_in_tail = False
        self._barge_in_candidate_at: Optional[float] = None
        self._barge_gate = NearEndSpeechGate()
        # VAD-thread candidate lifecycle. VAD can fire before the gate has
        # accumulated enough evidence; instead of deleting that audio (and the
        # user's first words with it), buffer the candidate until the gate
        # confirms or fails. Nothing here ever reaches the control thread
        # before confirmation.
        self._barge_candidate_events: list[VADEvent] = []
        self._barge_candidate_bytes = 0
        self._barge_candidate_windows = 0
        self._barge_candidate_started_at: Optional[float] = None
        self._barge_candidate_max_seconds = 2.5
        self._barge_candidate_max_bytes = 16000 * 2 * 3
        # Bounded evidence window: an unconfirmed candidate must fail instead
        # of waiting for VAD to end (echo can make VAD stay in speech forever).
        self._barge_candidate_max_windows = max(
            12, self._barge_gate.confirm_windows * 2
        )
        self._last_vad_probability: Optional[float] = None
        self._barge_in_unavailable_listener: Optional[
            Callable[[str], None]
        ] = None
        self._barge_in_unavailable_reported = False
        self._barge_stats_lock = threading.Lock()
        self._barge_stats: dict[str, Any] = {
            "candidates": 0,
            "confirmed": 0,
            "last_user_speech_to_confirm_ms": None,
        }

    # ------------------------------------------------------------------ public
    @property
    def microphone(self) -> Any:
        return self._microphone

    @property
    def dropped_audio_chunks(self) -> int:
        return int(getattr(self._microphone, "dropped_chunks", 0) or 0)

    def set_listeners(
        self,
        *,
        on_final: Optional[Callable[[FinalUtterance], None]] = None,
        on_speech: Optional[Callable[[SpeechActivity], None]] = None,
        on_turn_closed: Optional[Callable[[int, bool], None]] = None,
        on_partial: Optional[Callable[[PartialTranscript], None]] = None,
        render_finals: Optional[bool] = None,
    ) -> None:
        """Attach or replace the embedding hooks after construction.

        Used by the voice bridge, which is created before the manager. The
        callbacks may be invoked from the ASR control thread and must not
        block; a callback exception is swallowed so recognition keeps running.
        """
        self._on_final = on_final
        self._on_speech = on_speech
        self._on_turn_closed = on_turn_closed
        if on_partial is not None:
            self._on_partial = on_partial
        if render_finals is not None:
            self.render_finals = bool(render_finals)

    def set_speaker_observer(
        self, observer: Optional[SpeakerObserver]
    ) -> None:
        """Attach or replace the optional voice-print observer."""
        self._speaker_observer = observer

    def set_partial_listener(
        self, listener: Optional[Callable[[PartialTranscript], None]]
    ) -> None:
        """Attach a display-only listener for provisional transcripts.

        Separate from :meth:`set_listeners` so an embedder whose listener
        signature predates partials keeps working. The callback runs on the
        ASR control thread and must never block.
        """
        self._on_partial = listener

    def set_input_enabled(self, enabled: bool) -> None:
        """Backward-compatible alias for ``set_input_mode(NORMAL/MUTED)``.

        While muted the VAD thread drains and discards captured chunks and
        resets the segmentation state; the control loop aborts any turn that
        is still open, so no provisional partial can be submitted after the
        application decided the mic is no longer listening.
        """
        self.set_input_mode(
            InputMode.NORMAL if enabled else InputMode.MUTED
        )

    def set_input_mode(self, mode: "InputMode | str") -> bool:
        """Switch the microphone path; safe from any thread.

        ``BARGE_IN`` is accepted only after an echo canceller was attached and
        is currently usable. Returns False when the mode was refused, so the
        caller can fall back to the existing half-duplex behaviour.
        """
        try:
            chosen = InputMode(mode)
        except ValueError:
            return False
        if chosen == InputMode.BARGE_IN and not self.barge_in_available():
            return False
        self._input_mode = chosen
        if chosen == InputMode.MUTED:
            self._input_enabled.clear()
            self._put_control(_Mute(), critical=True)
        else:
            self._input_enabled.set()
        return True

    @property
    def input_mode(self) -> InputMode:
        return self._input_mode

    # ---------------------------------------------------------- barge-in API
    def attach_echo_canceller(
        self,
        pipeline: Any,
        *,
        vad_threshold: float = 0.65,
        min_speech_ms: int = 192,
        pre_roll_ms: int = 400,
    ) -> None:
        """Attach the optional AEC pipeline; no effect on the normal path."""
        self._echo_canceller = pipeline
        self._barge_in_vad_threshold = float(vad_threshold)
        self._barge_in_min_speech_ms = max(1, int(min_speech_ms))
        self._barge_in_pre_roll_ms = max(0, int(pre_roll_ms))
        self._capture_normal_vad_profile()

    def set_barge_in_unavailable_listener(
        self, listener: Optional[Callable[[str], None]]
    ) -> None:
        """Called once from the control thread if AEC becomes untrustworthy."""
        self._barge_in_unavailable_listener = listener

    def barge_in_available(self) -> bool:
        pipeline = self._echo_canceller
        if pipeline is None:
            return False
        try:
            return bool(
                getattr(pipeline, "available", False)
                and getattr(pipeline, "healthy", True)
            )
        except Exception:  # noqa: BLE001 - optional side channel
            return False

    def begin_barge_in(self, generation: int) -> bool:
        """Arm AEC detection for one TTS generation.

        Called on the asyncio loop when a spoken reply starts. The pipeline is
        reset for this generation; no detection decision is possible until the
        player supplies real far-end frames. Recovery is per-generation: the
        pipeline's ``start()`` rebuilds the canceller and clears any
        ``failed``/``degraded`` flag from a previous reply before the
        availability check, so one bad reply can never disable later ones.
        """
        pipeline = self._echo_canceller
        if pipeline is None:
            return False
        try:
            if not bool(pipeline.start(int(generation))):
                return False
        except Exception:  # noqa: BLE001 - optional side channel
            return False
        self._barge_in_generation = int(generation)
        self._barge_in_detecting = True
        self._barge_in_tail = False
        self._barge_in_candidate_at = None
        self._barge_candidate_events = []
        self._barge_candidate_bytes = 0
        self._barge_candidate_windows = 0
        self._barge_candidate_started_at = None
        self._last_vad_probability = None
        self._barge_gate.reset()
        self._barge_in_unavailable_reported = False
        with self._barge_stats_lock:
            self._barge_stats["confirmed"] = 0
            self._barge_stats["candidates"] = 0
        return self.set_input_mode(InputMode.BARGE_IN)

    def end_barge_in(self, generation: Optional[int] = None) -> None:
        """Stop accepting playback for a generation (echo tail is retained)."""
        pipeline = self._echo_canceller
        self._barge_in_detecting = False
        if pipeline is None:
            return
        try:
            pipeline.stop(
                self._barge_in_generation
                if generation is None
                else int(generation)
            )
        except Exception:  # noqa: BLE001 - optional side channel
            pass
        self._barge_in_generation = None

    def confirm_barge_in(self, generation: int) -> bool:
        """Keep the confirmed user turn continuous after TTS is stopped.

        Detection is over (the pre-roll was already preserved by the VAD
        segmenter). The pipeline switches to a bounded echo-tail mode: it keeps
        using real far-end references while they exist, then runs the same AEC
        instance on silent far-end frames so the acknowledged turn can never be
        dropped merely because the playback reference ran out.

        A pipeline without tail support falls back to the ordinary input path
        *without resetting the open utterance*, so the user's audio continues
        to be recognized.
        """
        self._barge_in_detecting = False
        self._barge_in_candidate_at = None
        pipeline = self._echo_canceller
        if pipeline is not None:
            begin_tail = getattr(pipeline, "begin_echo_tail", None)
            if callable(begin_tail):
                try:
                    if bool(begin_tail(int(generation))):
                        self._barge_in_tail = True
                        return True
                except Exception:  # noqa: BLE001 - optional side channel
                    pass
        # No usable tail support: the interruption is already confirmed, so
        # raw microphone audio is safe from here on. Switch without a reset.
        self._barge_in_tail = False
        self.set_input_mode(InputMode.NORMAL)
        return False

    def end_echo_tail(self) -> None:
        """Finish the confirmed-barge-in continuity window."""
        self._barge_in_tail = False
        pipeline = self._echo_canceller
        if pipeline is None:
            return
        end_tail = getattr(pipeline, "end_echo_tail", None)
        if callable(end_tail):
            try:
                end_tail()
            except Exception:  # noqa: BLE001 - optional side channel
                pass

    def barge_in_stats(self) -> dict:
        """Counters for ``/mic status`` and the real-device diagnostic report."""
        with self._barge_stats_lock:
            stats = dict(self._barge_stats)
        stats.update(
            {
                "enabled": self._echo_canceller is not None,
                "available": self.barge_in_available(),
                "mode": self._input_mode.value,
                "generation": self._barge_in_generation,
                "detecting": self._barge_in_detecting,
                "tail": self._barge_in_tail,
            }
        )
        pipeline = self._echo_canceller
        if pipeline is not None:
            try:
                stats["aec"] = pipeline.stats()
            except Exception:  # noqa: BLE001 - diagnostics only
                stats["aec"] = None
        return stats

    def start(self) -> None:
        if self._started:
            return
        if self._vad_segmenter is None:
            vad = SileroVAD(self.config)
            vad.preload()
            self._vad_segmenter = VADSegmenter(vad.probability, self.config)
        self._capture_normal_vad_profile()

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
                    self._check_stream_idle()
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
        self.end_barge_in()
        pipeline = self._echo_canceller
        if pipeline is not None:
            try:
                pipeline.reset()
            except Exception:  # noqa: BLE001 - optional side channel
                pass
        for utterance in list(self._utterances.values()):
            if not utterance.finalized:
                self._abort_utterance(
                    utterance,
                    "程序退出，已中止当前识别。",
                    report=False,
                    promote=False,
                )
        with self._threads_lock:
            threads = list(self._threads)
        for thread in threads:
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
            self._sync_vad_mode(segmenter)
            try:
                item = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._sync_vad_mode(segmenter)
            mode = self._input_mode
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
                if mode == InputMode.MUTED:
                    # Paused: keep draining so the bounded capture queue never
                    # wedges the microphone thread, but never segment.
                    continue
                pcm = item.pcm
                barge_in_audio = False
                if mode == InputMode.BARGE_IN:
                    pcm, accepted, barge_in_audio = (
                        self._process_barge_in_audio(item)
                    )
                    if not accepted:
                        # Never run VAD on raw, un-cancelled microphone audio:
                        # without a valid reference there is no safe barge-in
                        # decision to make. A gap also invalidates the partial
                        # candidate, so audio on either side of the gap can
                        # never be stitched into one confirmed interruption.
                        if self._barge_in_detecting:
                            self._reset_barge_in_candidate(segmenter)
                        continue
                try:
                    events = segmenter.process(pcm)
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
                    if barge_in_audio and self._barge_in_detecting:
                        if self._barge_candidate_events:
                            if not self._barge_gate.confirmed:
                                if event.kind == SPEECH_ENDED:
                                    # The utterance ended before any evidence
                                    # confirmed the user: Ava's own echo.
                                    self._discard_barge_candidate(segmenter)
                                    continue
                                if not self._buffer_barge_candidate(event):
                                    self._discard_barge_candidate(segmenter)
                                continue
                            # The gate confirmed while the VAD segment was in
                            # progress: release the buffered pre-roll/audio so
                            # the user's first words are part of the turn.
                            if not self._flush_barge_candidate():
                                self._discard_barge_candidate(segmenter)
                                continue
                        elif (
                            event.kind == SPEECH_STARTED
                            and not self._barge_gate.confirmed
                        ):
                            # VAD_SPEECH_CANDIDATE: wait for aligned evidence
                            # instead of deleting the segment. Nothing leaves
                            # the VAD thread until the gate confirms.
                            if (
                                self._barge_gate.observations
                                >= self._barge_gate.confirm_windows + 2
                            ):
                                # Evidence was already conclusive against
                                # this segment before VAD fired: Ava's echo.
                                self._discard_barge_candidate(segmenter)
                                continue
                            self._start_barge_candidate(event)
                            continue
                    self._dispatch_vad_event(event, barge_in=barge_in_audio)
                if (
                    barge_in_audio
                    and self._barge_in_detecting
                    and self._barge_candidate_events
                ):
                    if self._barge_gate.confirmed:
                        if not self._flush_barge_candidate():
                            self._discard_barge_candidate(segmenter)
                    elif self._barge_candidate_expired():
                        self._discard_barge_candidate(segmenter)

    def _sync_vad_mode(self, segmenter: Any) -> None:
        mode = self._input_mode
        previous = self._vad_mode
        if mode == previous:
            return
        self._vad_mode = mode
        if previous == InputMode.BARGE_IN and mode != InputMode.BARGE_IN:
            # The confirmed-barge-in continuity window is over; no pipeline
            # tail may linger into the ordinary recognition path.
            self.end_echo_tail()
        if mode == InputMode.MUTED:
            if self._vad_input_enabled:
                self._vad_input_enabled = False
                try:
                    segmenter.reset()
                except Exception:  # noqa: BLE001 - best effort
                    pass
            self._restore_normal_vad_profile()
            return
        if not self._vad_input_enabled:
            self._vad_input_enabled = True
            try:
                segmenter.reset()
            except Exception:  # noqa: BLE001 - best effort
                pass
        if mode == InputMode.BARGE_IN:
            # Arming always starts from a clean waiting state: pre-roll
            # captured before the echo canceller was fed real far-end audio
            # must never become the beginning of the user's utterance.
            if getattr(segmenter, "state", None) == WAITING_FOR_SPEECH:
                try:
                    segmenter.reset()
                except Exception:  # noqa: BLE001 - best effort
                    pass
            self._activate_barge_in_profile()
        else:
            self._restore_normal_vad_profile()

    def _capture_normal_vad_profile(self) -> None:
        segmenter = self._vad_segmenter
        if segmenter is None or self._normal_vad_profile is not None:
            return
        try:
            self._normal_vad_profile = {
                "threshold": float(segmenter.threshold),
                "min_speech_ms": int(segmenter.min_speech_ms),
                "pre_roll_ms": int(segmenter.pre_roll_ms),
            }
        except Exception:  # noqa: BLE001 - fake/legacy segmenter
            self._normal_vad_profile = None

    def _restore_normal_vad_profile(self) -> None:
        segmenter = self._vad_segmenter
        if segmenter is None:
            return
        try:
            segmenter.window_observer = None
            profile = self._normal_vad_profile
            if profile:
                segmenter.configure(**profile)
        except Exception:  # noqa: BLE001 - fake/legacy segmenter
            pass

    def _activate_barge_in_profile(self) -> None:
        segmenter = self._vad_segmenter
        if segmenter is None:
            return
        # Only a waiting segmenter may take new detection parameters; an open
        # utterance (which should not exist while Ava is speaking) is kept.
        if getattr(segmenter, "state", None) != WAITING_FOR_SPEECH:
            return
        try:
            segmenter.configure(
                threshold=self._barge_in_vad_threshold,
                min_speech_ms=self._barge_in_min_speech_ms,
                pre_roll_ms=self._barge_in_pre_roll_ms,
            )
            segmenter.window_observer = self._observe_barge_in_window
        except Exception:  # noqa: BLE001 - fake/legacy segmenter
            return
        self._barge_in_detecting = True
        self._barge_in_candidate_at = None
        self._barge_gate.reset()

    def _observe_barge_in_window(self, probability: float) -> None:
        """Record candidate latency; runs on the VAD thread, must be trivial."""
        if not self._barge_in_detecting:
            return
        with self._barge_stats_lock:
            self._last_vad_probability = float(probability)
        if probability >= self._barge_in_vad_threshold:
            if self._barge_in_candidate_at is None:
                self._barge_in_candidate_at = time.monotonic()
                with self._barge_stats_lock:
                    self._barge_stats["candidates"] += 1
        else:
            self._barge_in_candidate_at = None

    def _process_barge_in_audio(
        self, item: AudioChunk
    ) -> tuple[bytes, bool, bool]:
        """Process one chunk in BARGE_IN mode.

        Returns ``(pcm, accepted, tagged_as_barge_in)``. Only AEC-cancelled
        audio is accepted during detection. After the interruption is
        confirmed, a failing tail switches this turn to NORMAL and accepts its
        raw PCM rather than dropping the user's audio.
        """
        pipeline = self._echo_canceller
        if pipeline is None:
            return b"", False, False
        try:
            result = pipeline.process_microphone(item.pcm, item.timestamp)
        except Exception as exc:  # noqa: BLE001 - optional side channel
            self._notify_barge_in_unavailable(
                f"aec:{type(exc).__name__}"
            )
            if self._barge_in_tail:
                return self._accept_after_tail_failure(item)
            return b"", False, False
        try:
            degraded = bool(getattr(pipeline, "degraded", False))
        except Exception:  # noqa: BLE001 - fake pipeline
            degraded = False
        if degraded:
            reason = getattr(pipeline, "unavailable_reason", "") or (
                "aec_degraded"
            )
            self._notify_barge_in_unavailable(str(reason))
            if self._barge_in_tail:
                return self._accept_after_tail_failure(item)
        if not getattr(result, "valid", False):
            reason = str(getattr(result, "reason", "") or "")
            tailing = False
            try:
                tailing = bool(getattr(pipeline, "tail_active", False))
            except Exception:  # noqa: BLE001 - fake pipeline
                tailing = False
            if self._barge_in_tail and (
                tailing or reason.startswith("tail_") or reason in (
                    "aec_error",
                    "aec_length",
                    "no_reference",
                )
            ):
                # The interruption is already confirmed and TTS was stopped;
                # finish this turn on the ordinary path instead of losing it.
                return self._accept_after_tail_failure(item)
            return b"", False, False
        near_rms = float(getattr(result, "near_rms", 0.0) or 0.0)
        clean_rms = float(getattr(result, "clean_rms", 0.0) or 0.0)
        reported_metrics = bool(
            getattr(result, "metrics_valid", True)
        ) and (near_rms > 0.0 or clean_rms > 0.0)
        if not reported_metrics:
            # The pipeline accepted the block but did not report levels.
            # Measure the actual near/clean PCM instead of trusting VAD
            # alone. (The old ``_unmetered`` path confirmed on VAD only.)
            clean_pcm = bytes(getattr(result, "pcm", b"") or b"")
            near_rms = _pcm_rms(item.pcm)
            clean_rms = _pcm_rms(clean_pcm)
            metrics_valid = near_rms > 0.0 or clean_rms > 0.0
        else:
            metrics_valid = True
        if self._barge_in_detecting:
            sample_rate = int(getattr(item, "sample_rate", 16000) or 16000)
            channels = max(1, int(getattr(item, "channels", 1) or 1))
            duration_ms = max(
                AEC_FRAME_MS,
                len(item.pcm) / (2.0 * channels * max(1, sample_rate)) * 1000.0,
            )
            self._barge_gate.observe_evidence(
                NearEndEvidence(
                    generation=self._barge_in_generation,
                    start_at=float(item.timestamp) - duration_ms / 1000.0,
                    duration_ms=duration_ms,
                    vad_probability=self._last_vad_probability,
                    near_rms=near_rms,
                    clean_rms=clean_rms,
                    metrics_valid=metrics_valid,
                    reference_valid=True,
                )
            )
        return bytes(getattr(result, "pcm", b"") or b""), True, True

    def _accept_after_tail_failure(
        self, item: AudioChunk
    ) -> tuple[bytes, bool, bool]:
        """End a broken echo tail and keep the confirmed turn recording.

        Only reachable after a barge-in was confirmed and TTS was stopped; the
        ordinary recognition path is safe by then and must not drop audio.
        """
        self._barge_in_tail = False
        self.set_input_mode(InputMode.NORMAL)
        pipeline = self._echo_canceller
        end_tail = getattr(pipeline, "end_echo_tail", None)
        if callable(end_tail):
            try:
                end_tail()
            except Exception:  # noqa: BLE001 - optional side channel
                pass
        return bytes(item.pcm), True, False

    # -------------------------------------------------- VAD candidate lifecycle
    def _start_barge_candidate(self, event: VADEvent) -> None:
        """Hold a VAD segment until the near-end gate has enough evidence."""
        self._barge_candidate_events = [event]
        self._barge_candidate_bytes = len(getattr(event, "audio", b"") or b"")
        self._barge_candidate_windows = 1
        self._barge_candidate_started_at = time.monotonic()

    def _buffer_barge_candidate(self, event: VADEvent) -> bool:
        """Accumulate one VAD event; False when the candidate must be dropped.

        Buffering is bounded in both time and bytes, so an unconfirmed
        candidate can never grow without limit or stall recognition.
        """
        if self._barge_candidate_started_at is None:
            self._barge_candidate_started_at = time.monotonic()
        if self._barge_candidate_expired():
            return False
        if self._barge_candidate_windows >= self._barge_candidate_max_windows:
            return False
        size = len(getattr(event, "audio", b"") or b"")
        if self._barge_candidate_bytes + size > self._barge_candidate_max_bytes:
            return False
        self._barge_candidate_events.append(event)
        self._barge_candidate_bytes += size
        self._barge_candidate_windows += 1
        return True

    def _barge_candidate_expired(self) -> bool:
        started = self._barge_candidate_started_at
        if started is None:
            return False
        return (time.monotonic() - started) > self._barge_candidate_max_seconds

    def _flush_barge_candidate(self) -> bool:
        """Release a confirmed candidate to the control thread, in order."""
        events = self._barge_candidate_events
        self._barge_candidate_events = []
        self._barge_candidate_bytes = 0
        self._barge_candidate_windows = 0
        self._barge_candidate_started_at = None
        if not events:
            return False
        for event in events:
            self._dispatch_vad_event(event, barge_in=True)
        return True

    def _discard_barge_candidate(self, segmenter: Any) -> None:
        """FAILED candidate: drop its audio and re-arm from a clean VAD state."""
        self._barge_candidate_events = []
        self._barge_candidate_bytes = 0
        self._barge_candidate_windows = 0
        self._barge_candidate_started_at = None
        self._barge_in_candidate_at = None
        self._barge_gate.reset()
        try:
            segmenter.reset()
        except Exception:  # noqa: BLE001 - best effort
            pass
        with self._barge_stats_lock:
            self._barge_stats["echo_rejected"] = (
                int(self._barge_stats.get("echo_rejected", 0)) + 1
            )

    def _reset_barge_in_candidate(self, segmenter: Any) -> None:
        """Drop an unfinished candidate after a gap in valid AEC audio."""
        self._barge_candidate_events = []
        self._barge_candidate_bytes = 0
        self._barge_candidate_windows = 0
        self._barge_candidate_started_at = None
        self._barge_in_candidate_at = None
        self._barge_gate.reset()
        try:
            segmenter.reset()
        except Exception:  # noqa: BLE001 - best effort
            pass
        with self._barge_stats_lock:
            self._barge_stats["gaps"] = (
                int(self._barge_stats.get("gaps", 0)) + 1
            )

    def _notify_barge_in_unavailable(self, reason: str) -> None:
        with self._barge_stats_lock:
            if self._barge_in_unavailable_reported:
                return
            self._barge_in_unavailable_reported = True
            self._barge_stats.setdefault("errors", 0)
            self._barge_stats["errors"] += 1
        # The control thread picks the fallback mode: until it does, invalid
        # AEC blocks are dropped and VAD never sees raw microphone audio.
        self._barge_in_detecting = False
        self._put_control(
            _BargeInUnavailable(str(reason)), critical=True
        )

    def _dispatch_vad_event(
        self, event: VADEvent, *, barge_in: bool = False
    ) -> None:
        if event.kind == SPEECH_STARTED:
            self._put_control(
                _SpeechStarted(
                    event.audio, time.monotonic(), barge_in=barge_in
                ),
                critical=True,
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
        opened_ok = False
        try:
            session = self._session_factory()
            utterance.session = session
            bind = getattr(session, "bind_utterance", None)
            if callable(bind):
                bind(utterance.uid)
            open_started = time.monotonic()
            session.open()
            opened_ok = True
            open_ms = (time.monotonic() - open_started) * 1000.0
            utterance.open_latency_ms = round(open_ms, 1)
            self.stats.connection_open_latency_ms = round(open_ms, 1)
            utterance.send_started_at = time.monotonic()
            utterance.opened.set()
            buffer = bytearray()
            first_buffered_at: Optional[float] = None
            while not (
                self._stop.is_set()
                or utterance.aborted
                or utterance.abort_event.is_set()
            ):
                try:
                    item = utterance.send_queue.get(timeout=0.02)
                except queue.Empty:
                    item = None
                if isinstance(item, (bytes, bytearray)):
                    utterance.queued_bytes = max(
                        0, utterance.queued_bytes - len(item)
                    )
                    if utterance.send_queue.empty():
                        utterance.queued_since = None
                    self._update_queue_metrics(utterance)
                if item is None and utterance.stop_requested.is_set():
                    # The control thread could not enqueue the stop sentinel
                    # (the queue was full of accepted audio). Everything ahead
                    # of it has now been uploaded, so finish here: no turn is
                    # aborted and no audio is dropped.
                    item = _STOP
                now = time.monotonic()
                if isinstance(item, (bytes, bytearray)):
                    if not buffer:
                        first_buffered_at = now
                    buffer.extend(item)
                    if len(buffer) >= SEND_FRAME_BYTES:
                        if not utterance.abort_event.is_set():
                            self._flush_audio(utterance, session, buffer)
                        first_buffered_at = None
                elif item is _STOP:
                    if buffer and not utterance.abort_event.is_set():
                        self._flush_audio(utterance, session, buffer)
                        buffer.clear()
                    if not utterance.abort_event.is_set():
                        mark_last = getattr(session, "mark_last_audio_sent", None)
                        if callable(mark_last):
                            mark_last()
                        session.stop()
                    self._put_control(
                        _InputClosed(utterance.uid, time.monotonic()),
                        critical=True,
                    )
                    return
                elif item is _ABORT:
                    return
                if (
                    buffer
                    and first_buffered_at is not None
                    and now - first_buffered_at >= SEND_FRAME_SECONDS
                ):
                    if not utterance.abort_event.is_set():
                        self._flush_audio(utterance, session, buffer)
                    first_buffered_at = None
            if (
                buffer
                and not utterance.aborted
                and not utterance.abort_event.is_set()
            ):
                self._flush_audio(utterance, session, buffer)
        except ASRClientError as exc:
            self._handle_sender_error(utterance, exc, opened_ok)
        except Exception as exc:  # noqa: BLE001 - sanitized
            self._handle_sender_error(
                utterance, classify_failure(exc, action="发送"), opened_ok
            )
        finally:
            utterance.opened.set()
            utterance.sender_done.set()
            self._log_lifecycle(utterance, "sender_exited", session)
            self._unregister_thread(threading.current_thread())

    def _receiver_loop(self, utterance: _Utterance) -> None:
        session: Optional[ASRSession] = None
        try:
            timeout = self.config.timeout_seconds + 2.0
            if not utterance.opened.wait(timeout=timeout):
                utterance.abort_event.set()
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
                    utterance.last_server_event_at = now
                    if event.kind == "error":
                        utterance.abort_event.set()
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
                if self._session_terminated_cleanly(session):
                    self._put_control(
                        _SessionDone(utterance.uid, None, time.monotonic()),
                        critical=True,
                    )
            except ASRClientError as exc:
                self._handle_receiver_error(utterance, exc)
            except Exception as exc:  # noqa: BLE001 - sanitized
                self._handle_receiver_error(
                    utterance, classify_failure(exc, action="接收")
                )
        finally:
            utterance.receiver_done.set()
            self._log_lifecycle(utterance, "receiver_exited", session)
            self._unregister_thread(threading.current_thread())

    def _handle_sender_error(
        self,
        utterance: _Utterance,
        exc: ASRClientError,
        opened_ok: bool,
    ) -> None:
        secondary = opened_ok and isinstance(exc, ASRSessionStateError)
        utterance.abort_event.set()
        if secondary:
            utterance.secondary_error = str(exc)
            self._log_lifecycle(utterance, "secondary_error", utterance.session)
            return
        message = str(exc)
        utterance.open_error = message
        utterance.primary_error = message
        self._put_control(
            _SessionDone(utterance.uid, message, time.monotonic()),
            critical=True,
        )

    def _handle_receiver_error(
        self, utterance: _Utterance, exc: ASRClientError
    ) -> None:
        # The session's events() generator raises ASRSessionStateError for a
        # receive failure that is only a symptom of an earlier close; that must
        # never overwrite the primary error.
        secondary = isinstance(exc, ASRSessionStateError)
        utterance.abort_event.set()
        if secondary:
            utterance.secondary_error = str(exc)
            self._log_lifecycle(utterance, "secondary_error", utterance.session)
            return
        message = str(exc)
        utterance.primary_error = message
        self._put_control(
            _SessionDone(utterance.uid, message, time.monotonic()),
            critical=True,
        )

    def _flush_audio(
        self,
        utterance: _Utterance,
        session: ASRSession,
        buffer: bytearray,
    ) -> None:
        while buffer and not utterance.abort_event.is_set():
            chunk = bytes(buffer[:SEND_FRAME_BYTES])
            del buffer[:SEND_FRAME_BYTES]
            session.send_audio(chunk)
            utterance.sent_bytes += len(chunk)
            utterance.last_audio_sent_at = time.monotonic()
        if utterance.send_started_at is not None:
            elapsed = time.monotonic() - utterance.send_started_at
            if elapsed > 0.0:
                self.stats.sender_throughput_bps = round(
                    utterance.sent_bytes / elapsed, 1
                )
        self._update_queue_metrics(utterance)

    def _update_queue_metrics(self, utterance: _Utterance) -> None:
        """Expose queue backpressure without changing the queue budget."""
        now = time.monotonic()
        # 16 kHz mono int16: 32 bytes per millisecond.
        self.stats.queued_audio_ms = round(utterance.queued_bytes / 32.0, 1)
        self.stats.oldest_audio_age_ms = (
            None
            if utterance.queued_since is None
            else round((now - utterance.queued_since) * 1000.0, 1)
        )

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
        elif isinstance(event, _InputClosed):
            self._handle_input_closed(event)
        elif isinstance(event, _FinalTimeout):
            self._handle_final_timeout(event)
        elif isinstance(event, _Mute):
            self._handle_mute()
        elif isinstance(event, _BargeInUnavailable):
            self._handle_barge_in_unavailable(event)
        elif isinstance(event, _Interrupted):
            self._handle_interrupted(event)
        elif isinstance(event, _Notice):
            self._render_status(event.text, is_error=event.is_error)

    def _record_barge_in_confirmed(self) -> None:
        """Record detection latency; called on the control thread."""
        now = time.monotonic()
        candidate_at = self._barge_in_candidate_at
        self._barge_in_candidate_at = None
        self._barge_in_detecting = False
        latency_ms = (
            max(0.0, (now - candidate_at) * 1000.0)
            if candidate_at is not None
            else None
        )
        with self._barge_stats_lock:
            self._barge_stats["confirmed"] += 1
            self._barge_stats["last_user_speech_to_confirm_ms"] = (
                round(latency_ms, 1) if latency_ms is not None else None
            )

    def _handle_speech_started(self, event: _SpeechStarted) -> None:
        if (
            event.barge_in
            and self._barge_in_detecting
            and not self._barge_gate.confirmed
        ):
            # The VAD saw speech during barge-in detection, but the near-end
            # gate did not confirm it survived cancellation: this is Ava's own
            # leaked echo, not a user interruption. Never start a user turn for
            # it (self-recognition fix), and never stop TTS for it.
            self._barge_in_candidate_at = None
            return
        self._next_uid += 1
        utterance = _Utterance(
            self._next_uid,
            self.config.asr_send_queue_frames,
            self.config.max_utterance_seconds,
        )
        self._utterances[utterance.uid] = utterance
        self._open_order.append(utterance)
        self._current = utterance
        self.stats.utterances += 1
        if event.barge_in:
            # One utterance, one id: the interruption notice and the normal
            # lifecycle events all refer to the same ASR turn.
            self._record_barge_in_confirmed()
            self._emit_speech(
                "barge_in",
                utterance.uid,
                event.at,
                generation=self._barge_in_generation,
            )
        self._emit_speech("started", utterance.uid, event.at)
        self._speaker_start(utterance.uid, event.audio)
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
        self._speaker_audio(utterance.uid, event.audio)
        self._enqueue_audio(utterance, event.audio)

    def _handle_speech_ended(self, event: _SpeechEnded) -> None:
        utterance = self._current
        self._current = None
        if utterance is None or utterance.finalized:
            return
        utterance.vad_end_at = event.at
        self._emit_speech("ended", utterance.uid, event.at)
        self._speaker_end(utterance.uid)
        self._log_lifecycle(utterance, "vad_ended")
        if not utterance.stop_sent:
            utterance.stop_sent = True
            if not self._enqueue_sentinel(utterance, _STOP):
                self._abort_utterance(
                    utterance,
                    "ASR 发送队列已满，网络可能过慢；已中止当前识别。",
                )
                return
            if self.config.debug:
                self._render_status(
                    f"#{utterance.uid} 语音结束，等待发送完成后收尾…"
                )
        # The session may already have ended while the user was still
        # speaking; only now are both conditions of a turn complete.
        self._maybe_finish(utterance)

    def _handle_sentence(self, event: _Sentence) -> None:
        utterance = self._utterances.get(event.uid)
        if (
            utterance is None
            or utterance.finalized
            or utterance.aborted
        ):
            return
        text = str(event.text or "").strip()
        if not text:
            return
        if event.is_final:
            sentence_id = event.sentence_id
            if sentence_id and sentence_id in utterance.final_ids:
                return
            if (
                not sentence_id
                and utterance.finals
                and text == utterance.finals[-1][1]
            ):
                return
            if sentence_id:
                utterance.final_ids.add(sentence_id)
            key = sentence_id or f"#{len(utterance.finals)}"
            utterance.finals.append((key, text))
            # A confirmed fragment is only accumulated. It is never printed
            # as its own line; the whole turn is merged and rendered once by
            # ``_finish`` after VAD end and session completion.
            if utterance.partial and (
                utterance.partial_id == sentence_id
                or (not sentence_id and utterance.partial == text)
            ):
                utterance.partial = ""
                utterance.partial_id = ""
            return

        if (
            event.sentence_id
            and event.sentence_id in utterance.final_ids
        ):
            return
        if text == utterance.partial:
            return
        utterance.partial = text
        utterance.partial_id = event.sentence_id
        preview = _merge_turn_text(
            [fragment for _, fragment in utterance.finals]
            + ([utterance.partial] if utterance.partial else [])
        )
        # Provisional text is shown to the embedder as soon as it exists, so
        # the formal REPL can display what ASR heard while the user is still
        # speaking. It never becomes a message and is never persisted.
        if preview:
            self._emit_partial(utterance, preview)
        # Partials are part of the standalone transcript renderer. In formal
        # voice mode (render_finals=False) the embedding app prints the final
        # itself, so provisional lines are fully suppressed to avoid a stale
        # partial leaking into the REPL after a failure/cancel/timeout.
        if preview and self.render_finals:
            self._render(
                RenderEvent(
                    kind="partial",
                    utterance_id=event.uid,
                    text=preview,
                )
            )

    def _handle_session_done(self, event: _SessionDone) -> None:
        utterance = self._utterances.get(event.uid)
        if utterance is None or utterance.finalized:
            return
        if utterance.session_done:
            # A second worker reported after the primary outcome. Never emit a
            # duplicate failure or a duplicate Final/TurnClosed.
            if event.error is not None:
                utterance.secondary_error = event.error
                self._log_lifecycle(utterance, "secondary_error")
            return
        utterance.session_done = True
        utterance.session_error = event.error
        utterance.session_done_at = event.at
        if event.error is not None:
            utterance.primary_error = event.error
            if utterance.vad_end_at is None:
                # The session failed while the user was still speaking: stop
                # uploading to the dead connection immediately. The turn is
                # still finalized only after the VAD ends, so a failed
                # recognition never becomes a normal final.
                utterance.abort_event.set()
        # The session will produce no more events, so the control thread (the
        # single owner of the socket) releases it now. For a normal
        # completion this happens right after task-finished; for a failure it
        # prevents the sender from writing into a dead socket.
        self._close_session(
            utterance,
            origin=(
                CLOSE_ORIGIN_EXTERNAL
                if event.error is not None
                else CLOSE_ORIGIN_LOCAL_NORMAL
            ),
        )
        self._maybe_finish(utterance)

    def _check_stream_idle(self) -> None:
        """Fail an upload stream the server stopped answering.

        A slow *connection handshake* is a queue/age diagnostic
        (``connection_open_latency_ms``/``oldest_audio_age_ms``), not this
        timeout: this only triggers when the sender has been uploading and
        the server produced no event at all for the configured window.
        """
        timeout = float(
            getattr(self.config, "stream_idle_timeout_seconds", 0.0)
        )
        if timeout <= 0.0:
            return
        now = time.monotonic()
        for utterance in list(self._utterances.values()):
            if (
                utterance.finalized
                or not utterance.started
                or utterance.session_done
                or utterance.abort_event.is_set()
            ):
                continue
            anchors = [
                value
                for value in (
                    utterance.send_started_at,
                    utterance.last_audio_sent_at,
                    utterance.last_server_event_at,
                )
                if value is not None
            ]
            if not anchors:
                continue
            if now - max(anchors) <= timeout:
                continue
            message = (
                f"识别连接长时间无响应（超过 {timeout:.0f} 秒），"
                "本次语音未完整识别。"
            )
            utterance.abort_event.set()
            self._close_session(
                utterance, origin=CLOSE_ORIGIN_LOCAL_TIMEOUT
            )
            self._finish(utterance, message, now)

    def _handle_input_closed(self, event: _InputClosed) -> None:
        """All audio was sent and ``stop()`` was called: start final timing."""
        utterance = self._utterances.get(event.uid)
        if (
            utterance is None
            or utterance.finalized
            or utterance.session_done
        ):
            return
        if utterance.final_timer is not None:
            return
        timeout = max(0.1, self.config.final_timeout_seconds)
        timer = threading.Timer(
            timeout, self._enqueue_final_timeout, args=(utterance.uid,)
        )
        timer.daemon = True
        utterance.final_timer = timer
        timer.start()
        if self.config.debug:
            self._render_status(
                f"#{utterance.uid} 音频发送完成，等待最终识别…"
            )

    def _handle_final_timeout(self, event: _FinalTimeout) -> None:
        utterance = self._utterances.get(event.uid)
        if utterance is None or utterance.finalized:
            return
        utterance.abort_event.set()
        self._close_session(utterance, origin=CLOSE_ORIGIN_LOCAL_TIMEOUT)
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

    def _handle_barge_in_unavailable(self, event: _BargeInUnavailable) -> None:
        """The AEC side channel failed; fall back without losing a real turn.

        Before the user was confirmed, the safe fallback is the existing
        half-duplex behaviour (pause the mic while Ava speaks). After a turn
        has already started, aborting it would risk losing the user's first
        words; keep the turn and switch it to the ordinary raw-mic path.
        """
        self._barge_in_detecting = False
        self._barge_in_tail = False
        utterance = self._current
        if utterance is not None and not utterance.finalized:
            self._input_mode = InputMode.NORMAL
            self._input_enabled.set()
        else:
            self._input_mode = InputMode.MUTED
            self._input_enabled.clear()
        listener = self._barge_in_unavailable_listener
        if listener is not None:
            try:
                listener(event.reason)
            except Exception:  # noqa: BLE001 - optional integration hook
                pass

    def _handle_mute(self) -> None:
        """Drop every turn still open when the application stops listening.

        The audio pipeline is already discarding chunks; this removes turns
        whose control events were queued before the pause took effect. Their
        provisional partials are never submitted as user messages.
        """
        for utterance in list(self._utterances.values()):
            if not utterance.finalized:
                self._abort_utterance(
                    utterance,
                    "已暂停监听，当前录音已丢弃。",
                    report=False,
                    promote=False,
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
        self._register_thread(utterance.sender)
        self._register_thread(utterance.receiver)
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
        # O(1) snapshot. If the recognition worker has not finished, the
        # latest provisional result (or PENDING) is attached — the final is
        # never delayed waiting for voice-print inference.
        speaker = self._speaker_snapshot(utterance.uid)
        if error and utterance.primary_error is None:
            utterance.primary_error = error
        if utterance.final_timer is not None:
            utterance.final_timer.cancel()
        if utterance.started:
            self._active_sessions = max(0, self._active_sessions - 1)
        self._close_session(utterance, origin=CLOSE_ORIGIN_LOCAL_NORMAL)
        # The session is over; late events for this id are now safely ignored
        # and no longer need to be kept in the open/known collections. The
        # session threads unregister themselves when they exit.
        self._remove_utterance(utterance)

        if not utterance.finals and utterance.partial:
            utterance.partial = ""
            utterance.partial_id = ""
            self._render(
                RenderEvent(kind="clear_partial", utterance_id=utterance.uid)
            )
        else:
            utterance.partial = ""
            utterance.partial_id = ""

        if error:
            self.stats.errors += 1
            self._render_status(
                f"识别 #{utterance.uid} 失败：{error}", is_error=True
            )

        merged = _merge_turn_text(
            [fragment for _, fragment in utterance.finals]
        )
        # A truncated upload, a timeout or a session that closed without a
        # confirmed final is an *incomplete* turn. Its recognized prefix is
        # never submitted as a user message, never answered, never stored in
        # memory and never used for speaker learning.
        incomplete = utterance.incomplete_reason or ""
        ok = merged != "" and not error and not incomplete
        # Tell the optional voice-print hook whether this utterance may ever
        # contribute a sample. This is separate from VAD end: the speaker side
        # may already have analysed the audio, but learning waits for this
        # final outcome. The call is non-blocking.
        self._speaker_finalized(utterance.uid, ok)
        if incomplete:
            self.stats.unfinished += 1
            self._render_status(
                f"[asr] {incomplete}", is_error=True
            )
            if merged and self.render_finals:
                # Visible for reference, explicitly not a user message.
                self._render_status(
                    f"[asr] #{utterance.uid} 已识别部分（未提交）：{merged}"
                )
        elif merged:
            self.stats.finalized += 1
            if self.render_finals:
                self._render(
                    RenderEvent(
                        kind="final",
                        utterance_id=utterance.uid,
                        text=merged,
                        speaker_result=speaker,
                    )
                )
            if utterance.vad_end_at is not None:
                latency = max(
                    0.0, (at - utterance.vad_end_at) * 1000.0
                )
                utterance.final_latency_ms = latency
                self.stats.final_latencies_ms.append(latency)
                if self.config.debug:
                    self._render_status(
                        f"#{utterance.uid} VAD 结束 → 最终结果 "
                        f"{latency:.0f} ms"
                    )
            if not error:
                # A failed recognition with confirmed fragments is never a
                # normal user message: the application must not answer it.
                self._emit_final(
                    FinalUtterance(
                        utterance_id=utterance.uid,
                        text=merged,
                        completed_at=time.monotonic(),
                        speaker=speaker,
                    )
                )
        else:
            self.stats.empty += 1
            if not error:
                self._render_status(
                    f"#{utterance.uid} 没有可确认的最终文本，临时文本已丢弃。"
                )
        self._emit_turn_closed(utterance.uid, ok)
        self._promote_pending()
        self._log_lifecycle(utterance, "utterance_finalized")

    def _maybe_finish(self, utterance: _Utterance) -> None:
        """Finish only once BOTH local VAD ended and the ASR session ended."""
        if utterance.finalized or not utterance.session_done:
            return
        if utterance.vad_end_at is None:
            return
        self._finish(
            utterance,
            utterance.session_error,
            utterance.session_done_at or time.monotonic(),
        )

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
        utterance.abort_event.set()
        utterance.finalized = True
        self._speaker_cancel(utterance.uid)
        if utterance.final_timer is not None:
            utterance.final_timer.cancel()
        if utterance.started:
            self._active_sessions = max(0, self._active_sessions - 1)
        self._close_session(utterance, origin=CLOSE_ORIGIN_LOCAL_CANCEL)
        self._enqueue_sentinel(utterance, _ABORT)
        self._remove_utterance(utterance)
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
        self._emit_turn_closed(utterance.uid, False)
        if promote:
            self._promote_pending()
        self._log_lifecycle(utterance, "utterance_aborted")

    def _promote_pending(self) -> None:
        for utterance in self._open_order:
            if self._active_sessions >= self.config.max_concurrent_sessions:
                break
            if utterance.finalized or utterance.started:
                continue
            self._start_session(utterance)

    def _remove_utterance(self, utterance: _Utterance) -> None:
        """Drop all manager-side references to a finished utterance."""
        if self._utterances.get(utterance.uid) is utterance:
            self._utterances.pop(utterance.uid, None)
        try:
            self._open_order.remove(utterance)
        except ValueError:
            pass
        if self._current is utterance:
            self._current = None

    def _register_thread(self, thread: threading.Thread) -> None:
        with self._threads_lock:
            self._threads.append(thread)

    def _unregister_thread(self, thread: threading.Thread) -> None:
        with self._threads_lock:
            try:
                self._threads.remove(thread)
            except ValueError:
                pass

    # ----------------------------------------------------------------- helpers
    def _enqueue_audio(self, utterance: _Utterance, data: bytes) -> bool:
        if not data or utterance.finalized:
            return False
        if utterance.truncated:
            # Past the upload budget: keep draining the microphone but never
            # pretend the partial prefix is the whole utterance. The turn is
            # finalized as incomplete (no Final, no Router, no memory).
            return False
        try:
            data = bytes(data)
            utterance.send_queue.put_nowait(data)
            utterance.queued_bytes += len(data)
            utterance.max_queued_bytes = max(
                utterance.max_queued_bytes, utterance.queued_bytes
            )
            if utterance.queued_since is None:
                utterance.queued_since = time.monotonic()
            return True
        except queue.Full:
            # The link cannot keep up (typically the session handshake is
            # still in progress). This is not recoverable by dropping the
            # tail: everything after this point is lost, so the turn is
            # explicitly incomplete instead of silently truncated.
            first = not utterance.truncated
            utterance.truncated = True
            if first:
                utterance.incomplete_reason = (
                    "本次语音未完整识别，后续音频丢失；已识别部分仅供参考。"
                )
                self.stats.truncations += 1
                self._log_lifecycle(utterance, "truncated")
                if self.config.debug:
                    self._render_status(
                        f"[asr] 本次语音未完整识别，后续音频丢失；"
                        f"已识别部分仅供参考。(#{utterance.uid})",
                        is_error=True,
                    )
            return False

    def _enqueue_sentinel(self, utterance: _Utterance, item: Any) -> bool:
        try:
            utterance.send_queue.put_nowait(item)
            return True
        except queue.Full:
            # The queue is full of accepted audio that still has to be
            # uploaded. The sender honours ``stop_requested`` as soon as it has
            # drained those frames, so no slot is needed and nothing is lost.
            utterance.stop_requested.set()
            return True

    def _enqueue_final_timeout(self, uid: int) -> None:
        self._put_control(
            _FinalTimeout(uid), critical=False, timeout=0.5
        )

    def _close_session(
        self,
        utterance: _Utterance,
        origin: str = CLOSE_ORIGIN_LOCAL_NORMAL,
    ) -> None:
        """Idempotently release the session; the sole socket-closing owner.

        Only the control thread finalizes/aborts a turn and closes the socket,
        so close ordering is a single serialized point instead of a race
        between the sender and receiver workers.
        """
        session = utterance.session
        if session is None:
            return
        close = getattr(session, "close", None)
        if not callable(close):
            return
        try:
            close(origin=origin)
        except TypeError:
            # A fake/legacy session's close() takes no keyword argument.
            try:
                close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        except Exception:  # noqa: BLE001 - best effort
            pass

    def _session_state(self, session: Any) -> Optional[str]:
        state = getattr(session, "state", None)
        if callable(state):
            try:
                state = state()
            except Exception:  # noqa: BLE001 - fake/legacy session
                return None
        return state if isinstance(state, str) else None

    def _session_terminated_cleanly(self, session: Any) -> bool:
        state = self._session_state(session)
        if state is None:
            return True
        return state == STATE_TASK_FINISHED

    def _log_lifecycle(
        self,
        utterance: _Utterance,
        event_type: str,
        session: Any = None,
    ) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        if session is None:
            session = utterance.session
        request_id = getattr(session, "request_id", None)
        state = self._session_state(session)
        origin = getattr(session, "close_origin", None)
        primary = utterance.primary_error or getattr(
            session, "primary_error", None
        )
        secondary = utterance.secondary_error or getattr(
            session, "secondary_error", None
        )
        last_event = getattr(session, "last_server_event", None)
        failure = getattr(session, "primary_failure", None)
        self.logger.debug(
            "asr_lifecycle utterance_id=%s request_id=%s thread_name=%s"
            " timestamp_monotonic=%.3f session_state=%s event_type=%s"
            " last_server_event=%s close_origin=%s primary_error=%s"
            " secondary_error=%s error_type=%s error_code=%s",
            utterance.uid,
            request_id or "-",
            threading.current_thread().name,
            time.monotonic(),
            state or "-",
            event_type,
            last_event or "-",
            origin or "-",
            _one_line(primary),
            _one_line(secondary),
            getattr(failure, "exc_type", "") or "-",
            getattr(failure, "code", "")
            if getattr(failure, "code", None) is not None
            else "-",
        )

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

    # ---------------------------------------------------------- external hooks
    def _emit_final(self, final: FinalUtterance) -> None:
        callback = self._on_final
        if callback is None:
            return
        try:
            callback(final)
        except Exception:  # noqa: BLE001 - recognition must keep running
            pass

    def _emit_partial(self, utterance: _Utterance, text: str) -> None:
        callback = self._on_partial
        if callback is None:
            return
        utterance.partial_seq += 1
        event = PartialTranscript(
            utterance_id=utterance.uid,
            text=str(text),
            sequence=utterance.partial_seq,
            is_final=False,
            at=time.monotonic(),
        )
        try:
            callback(event)
        except Exception:  # noqa: BLE001 - recognition must keep running
            pass

    def _emit_speech(
        self,
        kind: str,
        uid: int,
        at: float,
        *,
        generation: Optional[int] = None,
    ) -> None:
        callback = self._on_speech
        if callback is None:
            return
        try:
            callback(SpeechActivity(kind, uid, at, generation=generation))
        except Exception:  # noqa: BLE001 - recognition must keep running
            pass

    def _emit_turn_closed(self, uid: int, ok: bool) -> None:
        callback = self._on_turn_closed
        if callback is None:
            return
        try:
            callback(uid, ok)
        except Exception:  # noqa: BLE001 - recognition must keep running
            pass

    # ---------------------------------------------------------- speaker hooks
    def post_speaker_result(self, result: Any) -> None:
        """Render an identity snapshot published by the observer's worker.

        Called from the voice-print thread; it only enqueues a render event,
        so it never touches the terminal from another thread.
        """
        if result is None:
            return
        uid = int(getattr(result, "utterance_id", 0) or 0)
        self._render(
            RenderEvent(
                kind="speaker_status",
                utterance_id=uid,
                speaker_result=result,
            )
        )

    def _speaker_start(self, uid: int, pcm: bytes) -> None:
        observer = self._speaker_observer
        if observer is None or not pcm:
            return
        try:
            observer.on_start(uid, pcm)
        except Exception:  # noqa: BLE001 - voice-print must not stop ASR
            self.logger.debug("speaker_on_start_failed uid=%s", uid)

    def _speaker_audio(self, uid: int, pcm: bytes) -> None:
        observer = self._speaker_observer
        if observer is None or not pcm:
            return
        try:
            observer.on_audio(uid, pcm)
        except Exception:  # noqa: BLE001 - voice-print must not stop ASR
            self.logger.debug("speaker_on_audio_failed uid=%s", uid)

    def _speaker_end(self, uid: int) -> None:
        observer = self._speaker_observer
        if observer is None:
            return
        try:
            observer.on_end(uid)
        except Exception:  # noqa: BLE001 - voice-print must not stop ASR
            self.logger.debug("speaker_on_end_failed uid=%s", uid)

    def _speaker_cancel(self, uid: int) -> None:
        observer = self._speaker_observer
        if observer is None:
            return
        try:
            observer.on_cancel(uid)
        except Exception:  # noqa: BLE001 - voice-print must not stop ASR
            self.logger.debug("speaker_on_cancel_failed uid=%s", uid)

    def _speaker_finalized(self, uid: int, success: bool) -> None:
        observer = self._speaker_observer
        callback = getattr(observer, "on_finalized", None) if observer else None
        if callback is None:
            return
        try:
            callback(uid, bool(success))
        except Exception:  # noqa: BLE001 - voice-print must not stop ASR
            self.logger.debug("speaker_on_finalized_failed uid=%s", uid)

    def _speaker_snapshot(self, uid: int) -> Any:
        observer = self._speaker_observer
        if observer is None:
            return None
        try:
            return observer.snapshot(uid)
        except Exception:  # noqa: BLE001 - never block or fail the final
            return None


__all__ = [
    "ASRManager",
    "ASRStats",
    "FinalUtterance",
    "NearEndEvidence",
    "NearEndSpeechGate",
    "PartialTranscript",
    "SpeechActivity",
]
