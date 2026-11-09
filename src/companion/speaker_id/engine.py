"""Real-time speaker-identification worker.

Design rules (V0.9.9.22):

* the engine is the only place a voice-print model runs, and it runs on one
  long-lived worker thread — never on the microphone, VAD, ASR sender or ASR
  receiver thread;
* ``on_start``/``on_audio``/``on_end``/``on_cancel`` only touch a bounded
  per-utterance buffer and a small bounded wake-up queue, so audio upload and
  VAD boundaries are never delayed by inference;
* analysis starts while the user is still speaking (after ~1.5 s of audio),
  refines at ~2.7 s, then refreshes roughly every second on a bounded recent
  window;
* ``snapshot`` is an O(1) read of the latest immutable result and never waits
  for inference;
* an assignment is locked per utterance, so one noisy window cannot create a
  second temporary identity.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from companion.speaker_id.config import SpeakerConfig
from companion.speaker_id.embedding import validate_embedding
from companion.speaker_id.matcher import SpeakerMatcher
from companion.speaker_id.models import (
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    QUALITY_RANK,
    SpeakerResult,
    pending_result,
)
from companion.speaker_id.quality import VoiceQuality, evaluate_voice_quality

_STOP = object()

_MAX_RETAINED_RESULTS = 512
_MAX_CANCELLED_IDS = 2048
_MAX_LATENCIES = 256

# The worker wakes at most this often even with no new event, so a due
# analysis and an ended utterance are never left waiting behind a slow queue.
_POLL_SECONDS = 0.02

# A final analysis is only repeated at VAD end when at least this much extra
# audio arrived since the last one; otherwise the existing snapshot stands.
_MIN_FINAL_GAP_SECONDS = 0.3


@dataclass
class _UtteranceState:
    uid: int
    started_at: float
    buffer: bytearray = field(default_factory=bytearray)
    total_bytes: int = 0
    ended: bool = False
    cancelled: bool = False
    analysis_count: int = 0
    next_analysis_seconds: float = 0.0
    last_analysis_seconds: float = 0.0
    assigned_speaker_id: Optional[str] = None
    # Identity created by this utterance; its own provisional sample must not
    # be used to "match" the utterance against itself.
    created_speaker_id: Optional[str] = None
    enrolled: bool = False
    # One independent sample per utterance, added when the utterance ends so
    # later analyses of the same utterance never compare against it.
    pending_enrollment: Optional[tuple] = None
    first_identity_at: Optional[float] = None


class SpeakerEngine:
    """Recognition worker plus the ``SpeakerObserver``-shaped public API."""

    def __init__(
        self,
        config: SpeakerConfig,
        *,
        embedding_model: Any,
        store: Any,
        on_result: Optional[Callable[[SpeakerResult], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._embedding = embedding_model
        self._store = store
        self._matcher = SpeakerMatcher(config)
        self._on_result = on_result
        self._clock = clock
        self.sample_rate = config.sample_rate
        self.model_key = config.model_key

        self._lock = threading.RLock()
        self._states: dict[int, _UtteranceState] = {}
        self._results: dict[int, SpeakerResult] = {}
        self._cancelled: set[int] = set()
        self._cancelled_order: deque[int] = deque(maxlen=_MAX_CANCELLED_IDS)
        self._wakeup: "queue.Queue[Any]" = queue.Queue(
            maxsize=config.command_queue_size
        )
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        self.logger = logging.getLogger("companion.speaker")

        # Counters for the demo summary / benchmark. Guarded by ``_lock``.
        self.analyses = 0
        self.errors = 0
        self.dropped_wakeups = 0
        self.first_identity_latencies_ms: list[float] = []

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._worker = threading.Thread(
            target=self._worker_loop, name="speaker-id", daemon=True
        )
        self._worker.start()

    def shutdown(self) -> None:
        """Stop the worker and join it; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._clear_queue()
        try:
            self._wakeup.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=self.config.shutdown_timeout_seconds)
        with self._lock:
            self._states.clear()

    def set_on_result(
        self, callback: Optional[Callable[[SpeakerResult], None]]
    ) -> None:
        self._on_result = callback

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # ------------------------------------------------- SpeakerObserver hooks
    def on_start(self, utterance_id: int, pcm: bytes) -> None:
        """Begin one utterance. Non-blocking: no model work happens here."""
        now = self._clock()
        with self._lock:
            state = _UtteranceState(
                uid=int(utterance_id),
                started_at=now,
                next_analysis_seconds=self.config.first_analysis_seconds,
            )
            self._states[state.uid] = state
            self._cancelled.discard(state.uid)
            self._results.pop(state.uid, None)
            self._append_locked(state, pcm)
            self._trim_states_locked()
            pending = pending_result(
                state.uid, model_id=self.model_key, at=now
            )
        self._notify(pending)
        self._wake()

    def on_audio(self, utterance_id: int, pcm: bytes) -> None:
        """Append audio for a live utterance. Never runs inference."""
        with self._lock:
            state = self._states.get(int(utterance_id))
            if state is None or state.ended or state.cancelled:
                return
            self._append_locked(state, pcm)

    def on_end(self, utterance_id: int) -> None:
        """No more audio will arrive for this utterance."""
        uid = int(utterance_id)
        with self._lock:
            state = self._states.get(uid)
            if state is not None:
                state.ended = True
        self._wake()

    def on_cancel(self, utterance_id: int) -> None:
        """Discard the utterance: failed ASR sessions leave no identity work."""
        uid = int(utterance_id)
        with self._lock:
            self._states.pop(uid, None)
            self._results.pop(uid, None)
            if uid not in self._cancelled:
                if len(self._cancelled_order) == self._cancelled_order.maxlen:
                    self._cancelled.discard(self._cancelled_order[0])
                self._cancelled.add(uid)
                self._cancelled_order.append(uid)
        self._wake()

    def snapshot(self, utterance_id: int) -> SpeakerResult:
        """O(1) latest snapshot; never waits for analysis."""
        uid = int(utterance_id)
        with self._lock:
            result = self._results.get(uid)
            if result is not None:
                return result
            state = self._states.get(uid)
            voiced = (
                state.total_bytes / (2.0 * self.sample_rate)
                if state is not None
                else 0.0
            )
        return pending_result(
            uid,
            model_id=self.model_key,
            voiced_seconds=voiced,
            at=self._clock(),
        )

    # Spec §9 name; identical O(1) read.
    get_latest_result = snapshot

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "analyses": self.analyses,
                "errors": self.errors,
                "dropped_wakeups": self.dropped_wakeups,
                "first_identity_latencies_ms": list(
                    self.first_identity_latencies_ms
                ),
            }

    # --------------------------------------------------------------- buffers
    def _append_locked(self, state: _UtteranceState, pcm: bytes) -> None:
        if not pcm:
            return
        state.buffer.extend(pcm)
        state.total_bytes += len(pcm)
        limit = self.config.window_bytes
        if len(state.buffer) > limit:
            del state.buffer[: len(state.buffer) - limit]

    def _trim_states_locked(self) -> None:
        limit = self.config.max_active_utterances
        while len(self._states) > limit:
            finished = [
                state
                for state in self._states.values()
                if state.ended or state.cancelled
            ]
            candidates = finished or list(self._states.values())
            victim = min(candidates, key=lambda state: state.started_at)
            self._states.pop(victim.uid, None)

    # --------------------------------------------------------------- wakeups
    def _wake(self) -> None:
        try:
            self._wakeup.put_nowait("wake")
        except queue.Full:
            with self._lock:
                self.dropped_wakeups += 1

    def _clear_queue(self) -> None:
        while True:
            try:
                self._wakeup.get_nowait()
            except queue.Empty:
                return

    # ------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._wakeup.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            try:
                self._scan()
            except Exception as exc:  # noqa: BLE001 - worker must survive
                self.logger.debug(
                    "speaker_scan_failed error_type=%s", type(exc).__name__
                )
            self._prune()
        self._prune(force=True)

    def _scan(self) -> None:
        """Process at most one action per call so the lock is never held long."""
        with self._lock:
            ended_uid = None
            due_uid = None
            for uid, state in self._states.items():
                if state.cancelled:
                    ended_uid = uid
                    break
                if state.ended:
                    ended_uid = uid
                    break
                seconds = state.total_bytes / (2.0 * self.sample_rate)
                if seconds + 1e-6 >= state.next_analysis_seconds:
                    due_uid = uid
                    break
        if ended_uid is not None:
            with self._lock:
                state = self._states.get(ended_uid)
                is_ended = state is not None and state.ended
                is_cancelled = state is not None and state.cancelled
            if is_cancelled or state is None:
                self._process_cancel(ended_uid)
            elif is_ended:
                self._process_end(ended_uid)
            return
        if due_uid is not None:
            self._process_analyze(due_uid)

    def _process_analyze(self, uid: int) -> None:
        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled or state.ended:
                return
            total_seconds = state.total_bytes / (2.0 * self.sample_rate)
            if total_seconds + 1e-6 < state.next_analysis_seconds:
                return
            window = bytes(state.buffer)
        self._analyze(uid, window, total_seconds, final=False)

    def _process_end(self, uid: int) -> None:
        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled:
                return
            state.ended = True
            total_seconds = state.total_bytes / (2.0 * self.sample_rate)
            window = bytes(state.buffer)
            already = state.analysis_count > 0
            gap = total_seconds - state.last_analysis_seconds
        try:
            if (
                total_seconds >= self.config.min_analysis_seconds
                and (not already or gap >= _MIN_FINAL_GAP_SECONDS)
            ):
                self._analyze(uid, window, total_seconds, final=True)
            elif not already:
                quality = self._evaluate(window)
                if quality is None:
                    self._publish_error(uid, "QualityError")
                else:
                    self._publish(
                        uid,
                        self._build_result(
                            uid,
                            status=INSUFFICIENT_AUDIO,
                            speaker_id=None,
                            quality=quality,
                            similarity=None,
                            margin=None,
                            inference_ms=None,
                            final=True,
                        ),
                    )
                    self._record_observation(
                        uid, None, INSUFFICIENT_AUDIO, None, None
                    )
        finally:
            with self._lock:
                state = self._states.get(uid)
                pending = None
                if state is not None and not state.enrolled:
                    pending = state.pending_enrollment
                self._states.pop(uid, None)
            if pending is not None:
                # The last analysis already matched an existing identity but
                # VAD ended before another one was due. Enrol it now, once,
                # after this utterance can no longer be compared to it.
                speaker_id, vector, quality = pending
                self._enroll(speaker_id, vector, quality)

    def _process_cancel(self, uid: int) -> None:
        with self._lock:
            self._states.pop(uid, None)
            self._results.pop(uid, None)

    # ------------------------------------------------------------ analysis
    def _analyze(
        self,
        uid: int,
        window: bytes,
        total_seconds: float,
        *,
        final: bool,
    ) -> None:
        quality = self._evaluate(window)
        if quality is None:
            self._publish_error(uid, "QualityError")
            return
        if not quality.usable:
            self._publish(
                uid,
                self._build_result(
                    uid,
                    status=INSUFFICIENT_AUDIO,
                    speaker_id=None,
                    quality=quality,
                    similarity=None,
                    margin=None,
                    inference_ms=None,
                    final=final,
                ),
            )
            self._record_observation(
                uid, None, INSUFFICIENT_AUDIO, None, None
            )
            self._advance(uid, total_seconds)
            return

        started = self._clock()
        try:
            vector = validate_embedding(
                self._embedding.encode(window),
                self.config.embedding_dim,
            )
        except Exception as exc:  # noqa: BLE001 - ASR keeps running
            self._publish_error(uid, type(exc).__name__)
            self._advance(uid, total_seconds)
            return
        inference_ms = max(0.0, (self._clock() - started) * 1000.0)

        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled:
                return
            locked_id = state.assigned_speaker_id
            created_here = state.created_speaker_id

        try:
            candidates, incompatible = self._store.load_candidates(
                model_id=self._model_label,
                embedding_version=self.config.embedding_version,
            )
        except Exception as exc:  # noqa: BLE001 - storage failure is non-fatal
            self._publish_error(uid, type(exc).__name__)
            self._advance(uid, total_seconds)
            return

        if incompatible and not candidates and locked_id is None:
            # Refuse to compare vectors from different model versions. This is
            # an explicit error, not an "unknown speaker".
            self._publish(
                uid,
                self._build_result(
                    uid,
                    status=ERROR,
                    speaker_id=None,
                    quality=quality,
                    similarity=None,
                    margin=None,
                    inference_ms=inference_ms,
                    final=final,
                ),
            )
            self._advance(uid, total_seconds)
            return

        decision = self._matcher.match(
            vector,
            candidates,
            quality=quality,
            locked_speaker_id=locked_id,
        )

        speaker_id = decision.speaker_id
        assigned = locked_id
        created_now = False
        if decision.status == NEW_CANDIDATE:
            locked_still_exists = locked_id is not None and any(
                candidate.speaker_id == locked_id for candidate in candidates
            )
            if locked_still_exists:
                # The assignment survives a noisy window, but only while the
                # identity still exists (it may have been deleted/merged).
                speaker_id = locked_id
            else:
                speaker_id = self._create_candidate(uid)
                if speaker_id is None:
                    self._advance(uid, total_seconds)
                    return
                assigned = speaker_id
                created_now = True
        elif decision.status in (CANDIDATE_MATCH, KNOWN):
            speaker_id = decision.speaker_id
            assigned = speaker_id

        created_by_this_utterance = created_here is not None or created_now
        pending = None
        enroll_now = False
        if speaker_id and created_now:
            # The identity must exist immediately so the next analysis of this
            # same utterance can lock onto it. This provisional sample is the
            # utterance's own audio, so it is never shown as a match score.
            enroll_now = True
        elif (
            speaker_id
            and not created_by_this_utterance
            and decision.status in (CANDIDATE_MATCH, KNOWN)
        ):
            if final:
                enroll_now = True
            else:
                # Defer the sample until the utterance ends: matching for the
                # rest of this utterance must only use older recordings.
                pending = (speaker_id, vector, quality)
        if enroll_now:
            self._enroll(speaker_id, vector, quality)

        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled:
                return
            if assigned:
                state.assigned_speaker_id = assigned
            if created_now:
                state.created_speaker_id = speaker_id
            if enroll_now:
                state.enrolled = True
            elif pending is not None:
                state.pending_enrollment = pending

        # An identity first heard in this utterance stays "new" for the whole
        # utterance: the rising self-similarity of later windows is not
        # evidence of a known person.
        status = NEW_CANDIDATE if created_by_this_utterance else decision.status
        similarity = None if created_by_this_utterance else decision.similarity
        margin = None if created_by_this_utterance else decision.margin

        result = self._build_result(
            uid,
            status=status,
            speaker_id=speaker_id,
            quality=quality,
            similarity=similarity,
            margin=margin,
            inference_ms=inference_ms,
            final=final,
        )
        self._publish(uid, result)
        self._record_observation(
            uid,
            speaker_id,
            status,
            similarity,
            margin,
        )
        self._advance(uid, total_seconds)
        if assigned:
            self._note_first_identity(uid)

    def _create_candidate(self, uid: int) -> Optional[str]:
        try:
            return self._store.create_candidate()
        except Exception as exc:  # noqa: BLE001 - non-fatal
            self._publish_error(uid, type(exc).__name__)
            return None

    def _enroll(self, speaker_id: str, vector, quality: VoiceQuality) -> None:
        required = QUALITY_RANK.get(
            self.config.minimum_enrollment_quality, 2
        )
        if quality.rank < required:
            return
        try:
            self._store.add_embedding(
                speaker_id,
                vector,
                quality=quality.quality,
                voiced_seconds=quality.voiced_seconds,
                model_id=self._model_label,
                embedding_version=self.config.embedding_version,
            )
        except Exception:  # noqa: BLE001 - a lost sample must not stop ASR
            self.logger.debug(
                "speaker_enroll_failed speaker_id=%s", speaker_id
            )

    def _record_observation(
        self,
        uid: int,
        speaker_id: Optional[str],
        status: str,
        similarity: Optional[float],
        margin: Optional[float],
    ) -> None:
        try:
            self._store.record_observation(
                uid,
                speaker_id=speaker_id,
                match_status=status,
                similarity=similarity,
                margin=margin,
            )
        except Exception:  # noqa: BLE001 - diagnostics only
            self.logger.debug("speaker_observation_failed uid=%s", uid)

    def _advance(self, uid: int, total_seconds: float) -> None:
        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled:
                return
            state.analysis_count += 1
            state.last_analysis_seconds = total_seconds
            if state.analysis_count == 1:
                following = self.config.second_analysis_seconds
            else:
                following = (
                    total_seconds + self.config.repeat_analysis_seconds
                )
            state.next_analysis_seconds = max(
                following, total_seconds + 0.2
            )
            self.analyses += 1

    def _evaluate(self, window: bytes) -> Optional[VoiceQuality]:
        try:
            return evaluate_voice_quality(
                window,
                sample_rate=self.sample_rate,
                min_voiced_seconds=self.config.min_voiced_seconds,
            )
        except Exception:  # noqa: BLE001 - never kill the worker
            return None

    def _build_result(
        self,
        uid: int,
        *,
        status: str,
        speaker_id: Optional[str],
        quality: VoiceQuality,
        similarity: Optional[float],
        margin: Optional[float],
        inference_ms: Optional[float],
        final: bool,
    ) -> SpeakerResult:
        return SpeakerResult(
            utterance_id=int(uid),
            speaker_id=speaker_id,
            person_id=None,
            status=status,
            reid_quality=quality.quality,
            similarity=similarity,
            margin=margin,
            voiced_seconds=quality.voiced_seconds,
            provisional=not final,
            model_id=self.model_key,
            inference_ms=inference_ms,
            completed_at=self._clock(),
        )

    @property
    def _model_label(self) -> str:
        return f"{self.config.model_id}@{self.config.model_revision}"

    # -------------------------------------------------------------- publish
    def _publish(self, uid: int, result: SpeakerResult) -> None:
        with self._lock:
            if uid in self._cancelled:
                return
            self._results[uid] = result
            self._prune_results_locked()
        self._notify(result)

    def _publish_error(self, uid: int, error_type: str) -> None:
        quality = VoiceQuality(False, 0.0, "LOW", "声纹识别异常")
        with self._lock:
            if uid in self._cancelled:
                return
            self.errors += 1
        self.logger.debug(
            "speaker_error uid=%s error_type=%s", uid, error_type
        )
        self._publish(
            uid,
            self._build_result(
                uid,
                status=ERROR,
                speaker_id=None,
                quality=quality,
                similarity=None,
                margin=None,
                inference_ms=None,
                final=True,
            ),
        )

    def _notify(self, result: SpeakerResult) -> None:
        callback = self._on_result
        if callback is None:
            return
        try:
            callback(result)
        except Exception:  # noqa: BLE001 - rendering must not kill ASR
            pass

    def _note_first_identity(self, uid: int) -> None:
        with self._lock:
            state = self._states.get(uid)
            if state is None or state.first_identity_at is not None:
                return
            state.first_identity_at = self._clock()
            latency = max(
                0.0, (state.first_identity_at - state.started_at) * 1000.0
            )
            if len(self.first_identity_latencies_ms) >= _MAX_LATENCIES:
                self.first_identity_latencies_ms.pop(0)
            self.first_identity_latencies_ms.append(latency)

    # ---------------------------------------------------------------- prune
    def _prune(self, *, force: bool = False) -> None:
        with self._lock:
            self._prune_results_locked()
            if force:
                self._states.clear()

    def _prune_results_locked(self) -> None:
        now = self._clock()
        cutoff = now - self.config.result_retention_seconds
        for uid, result in list(self._results.items()):
            if result.completed_at < cutoff:
                self._results.pop(uid, None)
        if len(self._results) > _MAX_RETAINED_RESULTS:
            oldest = sorted(
                self._results.items(),
                key=lambda item: item[1].completed_at,
            )
            for uid, _ in oldest[: len(self._results) - _MAX_RETAINED_RESULTS]:
                self._results.pop(uid, None)


__all__ = ["SpeakerEngine"]
