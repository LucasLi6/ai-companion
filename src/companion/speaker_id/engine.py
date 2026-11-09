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
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from companion.speaker_id.config import SpeakerConfig
from companion.speaker_id.embedding import validate_embedding
from companion.speaker_id.matcher import MatchDecision, SpeakerMatcher
from companion.speaker_id.models import (
    AMBIGUOUS,
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
_MAX_SOURCE_HISTORY = 256
_MAX_PENDING_REVOCATIONS = 256

# The worker wakes at most this often even with no new event, so a due
# analysis and an ended utterance are never left waiting behind a slow queue.
_POLL_SECONDS = 0.02

# A final analysis is only repeated at VAD end when at least this much extra
# audio arrived since the last one; otherwise the existing snapshot stands.
_MIN_FINAL_GAP_SECONDS = 0.3

# Explicit outcomes of the per-utterance learning decision. A successful match
# is not a successful write: only ``ADDED`` means a new reference vector was
# kept for the identity.
LEARN_ADDED = "ADDED"
LEARN_DUPLICATE = "DUPLICATE"
LEARN_LOW_QUALITY = "LOW_QUALITY"
LEARN_LOW_SIMILARITY = "LOW_SIMILARITY"
LEARN_LOW_MARGIN = "LOW_MARGIN"
LEARN_IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
LEARN_ASR_FAILED = "ASR_FAILED"
LEARN_STORE_ERROR = "STORE_ERROR"

_LEARNING_KEYS = {
    LEARN_ADDED: "added",
    LEARN_DUPLICATE: "duplicate",
    LEARN_LOW_QUALITY: "low_quality",
    LEARN_LOW_SIMILARITY: "low_similarity",
    LEARN_LOW_MARGIN: "low_margin",
    LEARN_IDENTITY_CONFLICT: "identity_conflict",
    LEARN_ASR_FAILED: "asr_failed",
    LEARN_STORE_ERROR: "store_error",
}


def _timing_summary(samples) -> dict[str, Any]:
    """Summarise bounded timing samples without inventing missing data."""
    values = sorted(float(value) for value in samples)
    if not values:
        return {"count": 0, "avg_ms": None, "p50_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values),
        "p50_ms": values[len(values) // 2],
        "max_ms": values[-1],
    }


@dataclass(frozen=True)
class _PendingEnrollment:
    """A verified match that may become a permanent sample at finalisation.

    The decision is carried unchanged so the write can re-check the same
    thresholds after ASR has finished; nothing is learnt while the utterance
    could still be cancelled or fail.
    """

    speaker_id: str
    vector: Any
    quality: VoiceQuality
    decision: MatchDecision
    speaker_status: str


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
    # A later window contradicted the earlier identity evidence: no sample
    # from this utterance may be committed, even if an earlier window passed.
    identity_conflict: bool = False
    # Why this utterance is pending: a verified match from a window, waiting
    # for VAD end + ASR final result before it may be written.
    pending_enrollment: Optional[_PendingEnrollment] = None
    # Why the last analysed window could not become a learning sample even
    # though it matched an existing identity (for statistics only).
    enrollment_block_reason: Optional[str] = None
    # Outcome of the provisional first sample created by this utterance.
    bootstrap_learning: Optional[str] = None
    # The final analysis for this utterance has run (successfully or not).
    analysis_done: bool = False
    # ``None`` until ASR reports the final outcome; only ``True`` may learn.
    asr_success: Optional[bool] = None
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

        # Utterance ids restart at 0 with every ASR manager, so observations
        # and samples are scoped by this process-lifetime session id.
        self.session_id = uuid.uuid4().hex
        # Samples committed for an utterance, kept briefly so a cancellation
        # that arrives after the commit can still revoke them. Bounded.
        self._source_history: "deque[tuple[int, str]]" = deque(
            maxlen=_MAX_SOURCE_HISTORY
        )
        self._source_speakers: dict[int, str] = {}
        # Revocations are queued by the ASR thread (which must not do SQLite
        # work) and drained by the worker thread.
        self._revocations: "deque[tuple[str, str, int]]" = deque(
            maxlen=_MAX_PENDING_REVOCATIONS
        )

        # Counters for the demo summary / benchmark. Guarded by ``_lock``.
        self.analyses = 0
        self.errors = 0
        self.dropped_wakeups = 0
        self.first_identity_latencies_ms: list[float] = []
        # Per-utterance learning outcomes. ``added`` counts only samples the
        # store actually kept; a reliable match never implies a write.
        self._learning: dict[str, int] = {
            key: 0 for key in _LEARNING_KEYS.values()
        }
        # Bounded rolling timing samples. Every value is measured with
        # ``perf_counter``; unmeasured behaviour is reported as absent, never
        # as a fabricated score.
        self._timings: dict[str, deque[float]] = {
            "quality_check_ms": deque(maxlen=_MAX_LATENCIES),
            "embedding_ms": deque(maxlen=_MAX_LATENCIES),
            "candidate_load_ms": deque(maxlen=_MAX_LATENCIES),
            "matching_ms": deque(maxlen=_MAX_LATENCIES),
            "enrollment_ms": deque(maxlen=_MAX_LATENCIES),
        }
        self._perf = time.perf_counter

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

    def on_finalized(self, utterance_id: int, success: bool) -> None:
        """ASR produced its final result (``success``) or failed.

        This is a distinct event from VAD end: the engine may already have run
        its final analysis, and only a successful final text may turn a
        pending match into a permanent sample. O(1), never waits for storage.
        """
        uid = int(utterance_id)
        with self._lock:
            state = self._states.get(uid)
            if state is None:
                if not success:
                    # ASR failed after the state was already committed/removed:
                    # revoke any sample that carries this utterance's source.
                    self._queue_revocation_locked(uid)
                return
            state.asr_success = bool(success)
            if not success:
                state.pending_enrollment = None
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
            self._queue_revocation_locked(uid)
        self._wake()

    def _queue_revocation_locked(self, uid: int) -> None:
        speaker_id = self._source_speakers.get(uid)
        if speaker_id is None:
            return
        self._revocations.append((speaker_id, self.session_id, uid))

    def _remember_source_locked(self, uid: int, speaker_id: str) -> None:
        self._source_speakers[uid] = speaker_id
        self._source_history.append((uid, speaker_id))
        while len(self._source_speakers) > _MAX_SOURCE_HISTORY:
            oldest_uid, oldest_speaker = self._source_history[0]
            # Only evict if the history entry still maps to this speaker (a
            # uid cannot be reused inside one session, but stay defensive).
            if self._source_speakers.get(oldest_uid) == oldest_speaker:
                self._source_speakers.pop(oldest_uid, None)
            self._source_history.popleft()

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

    def is_finished(self, utterance_id: int) -> bool:
        """True once the utterance's final analysis and learning commit ended.

        The state is removed only after ``_enroll``/revocation returned, so a
        benchmark or test can wait on this instead of guessing from the match
        result (which does not imply that learning has run).
        """
        with self._lock:
            return int(utterance_id) not in self._states

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "analyses": self.analyses,
                "errors": self.errors,
                "dropped_wakeups": self.dropped_wakeups,
                "first_identity_latencies_ms": list(
                    self.first_identity_latencies_ms
                ),
                "learning": dict(self._learning),
                "timings_ms": {
                    name: _timing_summary(samples)
                    for name, samples in self._timings.items()
                },
                "session_id": self.session_id,
            }

    def _record_learning(self, outcome: Optional[str]) -> None:
        """Count one real learning outcome; ``None`` means nothing was tried."""
        if outcome is None:
            return
        key = _LEARNING_KEYS.get(outcome)
        if key is None:
            return
        with self._lock:
            self._learning[key] += 1

    def _record_timing(self, name: str, milliseconds: float) -> None:
        samples = self._timings.get(name)
        if samples is None:
            return
        with self._lock:
            samples.append(max(0.0, float(milliseconds)))

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
            if self._revocations:
                speaker_id, session_id, uid = self._revocations.popleft()
                self._source_speakers.pop(uid, None)
            else:
                speaker_id = session_id = uid = None

            cancelled_uid = None
            ready_uid = None
            ended_uid = None
            due_uid = None
            for current_uid, state in self._states.items():
                if state.cancelled:
                    cancelled_uid = current_uid
                    break
                if state.ended and state.analysis_done:
                    if state.asr_success is not None:
                        ready_uid = current_uid
                        break
                    # Final analysis is done but ASR has not reported yet:
                    # wait; never re-run the final analysis.
                    continue
                if state.ended:
                    ended_uid = current_uid
                    break
                seconds = state.total_bytes / (2.0 * self.sample_rate)
                if seconds + 1e-6 >= state.next_analysis_seconds:
                    due_uid = current_uid
                    break

        if speaker_id is not None:
            self._revoke(speaker_id, session_id, uid)
            return
        if cancelled_uid is not None:
            self._process_cancel(cancelled_uid)
            return
        if ready_uid is not None:
            self._finalize(ready_uid)
            return
        if ended_uid is not None:
            self._process_end(ended_uid)
            return
        if due_uid is not None:
            self._process_analyze(due_uid)

    def _revoke(self, speaker_id: str, session_id: str, uid: int) -> None:
        revoke = getattr(self._store, "revoke_embedding", None)
        if revoke is None:
            return
        try:
            revoke(
                speaker_id,
                source_session_id=session_id,
                source_utterance_id=uid,
            )
        except Exception:  # noqa: BLE001 - storage failure is non-fatal
            self.logger.debug("speaker_revoke_failed uid=%s", uid)

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
        """Run the final analysis for an ended utterance; never learn here.

        learning only happens in :meth:`_finalize`, after ASR has reported its
        final result. Re-entering this method for an utterance whose final
        analysis already ran is a no-op, so a slow ASR result cannot cause the
        model to be invoked twice.
        """
        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled or state.analysis_done:
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
            # Otherwise the last (non-final) analysis already stands and any
            # eligible match is still held in ``pending_enrollment``.
        finally:
            with self._lock:
                state = self._states.get(uid)
                if state is not None:
                    state.analysis_done = True

    def _finalize(self, uid: int) -> None:
        """Commit or discard the utterance's pending sample, then drop state.

        Called only with VAD ended, the final analysis done and the ASR result
        known. A failed/cancelled ASR session or a self-contradicting identity
        discards the pending sample and revokes the provisional bootstrap
        sample, so one bad utterance cannot pollute a long-lived voice print.
        """
        with self._lock:
            state = self._states.get(uid)
            if state is None:
                return
            if not (
                state.ended
                and state.analysis_done
                and state.asr_success is not None
            ):
                return
            if state.cancelled:
                self._states.pop(uid, None)
                return
            pending = state.pending_enrollment
            conflict = state.identity_conflict
            success = state.asr_success is True
            created_speaker = state.created_speaker_id
            provisional = state.enrolled
            block_reason = state.enrollment_block_reason
            bootstrap_learning = state.bootstrap_learning

        # Process (write/revoke) before removing the state so a caller that
        # observes the utterance as finished can also observe its effect.
        # ``_enroll`` registers the source first, so a cancellation racing
        # this write still queues a revocation that runs afterwards.
        outcome: Optional[str]
        if not success:
            if provisional and created_speaker:
                self._revoke(created_speaker, self.session_id, uid)
            outcome = LEARN_ASR_FAILED
        elif conflict:
            if provisional and created_speaker:
                self._revoke(created_speaker, self.session_id, uid)
            outcome = LEARN_IDENTITY_CONFLICT
        elif pending is not None:
            failure = self._enrollment_failure_reason(
                pending.decision,
                pending.quality,
                pending.speaker_status,
                identity_conflict=False,
            )
            if failure is None:
                outcome = self._enroll(
                    pending.speaker_id,
                    pending.vector,
                    pending.quality,
                    source_utterance_id=uid,
                )
            else:
                outcome = failure
        elif provisional:
            # The provisional first sample was written during analysis; it is
            # a real sample only if this utterance succeeded without conflict.
            outcome = bootstrap_learning or LEARN_ADDED
        else:
            # A window that matched an existing identity but failed one of the
            # enrollment gates: report the gate, not a successful learn.
            outcome = block_reason

        self._record_learning(outcome)

        with self._lock:
            self._states.pop(uid, None)

    def _process_cancel(self, uid: int) -> None:
        with self._lock:
            state = self._states.pop(uid, None)
            self._results.pop(uid, None)
            provisional = (
                state is not None and state.enrolled
                and state.created_speaker_id is not None
            )
            created_speaker = (
                state.created_speaker_id if state is not None else None
            )
        if provisional and created_speaker:
            self._revoke(created_speaker, self.session_id, uid)


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

        started = self._perf()
        try:
            vector = validate_embedding(
                self._embedding.encode(window),
                self.config.embedding_dim,
            )
        except Exception as exc:  # noqa: BLE001 - ASR keeps running
            self._publish_error(uid, type(exc).__name__)
            self._advance(uid, total_seconds)
            return
        inference_ms = max(0.0, (self._perf() - started) * 1000.0)
        self._record_timing("embedding_ms", inference_ms)

        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled:
                return
            locked_id = state.assigned_speaker_id
            created_here = state.created_speaker_id
            previous_pending = state.pending_enrollment

        try:
            load_started = self._perf()
            candidates, incompatible = self._store.load_candidates(
                model_id=self._model_label,
                embedding_version=self.config.embedding_version,
            )
            self._record_timing(
                "candidate_load_ms",
                max(0.0, (self._perf() - load_started) * 1000.0),
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

        match_started = self._perf()
        decision = self._matcher.match(
            vector,
            candidates,
            quality=quality,
            locked_speaker_id=locked_id,
        )
        self._record_timing(
            "matching_ms",
            max(0.0, (self._perf() - match_started) * 1000.0),
        )
        status_by_id = {
            candidate.speaker_id: candidate.status for candidate in candidates
        }

        speaker_id = decision.speaker_id
        assigned = locked_id
        created_now = False
        created_by_this_utterance = created_here is not None
        if decision.status == NEW_CANDIDATE:
            if locked_id is not None and locked_id in status_by_id:
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
                created_by_this_utterance = True
        elif decision.status in (CANDIDATE_MATCH, KNOWN):
            speaker_id = decision.speaker_id
            assigned = speaker_id

        # A contradiction between windows of the same utterance invalidates
        # every sample from it, even an earlier high-scoring one.
        conflict = decision.status == AMBIGUOUS
        if (
            previous_pending is not None
            and decision.status in (CANDIDATE_MATCH, KNOWN)
            and decision.speaker_id != previous_pending.speaker_id
        ):
            conflict = True
        if (
            created_here is not None
            and decision.status in (CANDIDATE_MATCH, KNOWN)
            and decision.speaker_id != created_here
        ):
            conflict = True

        pending: Optional[_PendingEnrollment] = None
        enroll_bootstrap = False
        block_reason: Optional[str] = None
        if speaker_id and created_now:
            # The identity must exist immediately so the next analysis of this
            # same utterance can lock onto it. This provisional sample is the
            # utterance's own audio; it is unverified, carries its source key,
            # and is revoked if the utterance later fails or contradicts
            # itself. It is never the proof for adding a further sample.
            enroll_bootstrap = True
        elif (
            speaker_id
            and not created_by_this_utterance
            and decision.status in (CANDIDATE_MATCH, KNOWN)
        ):
            speaker_status = status_by_id.get(speaker_id, "")
            failure = self._enrollment_failure_reason(
                decision,
                quality,
                speaker_status,
                identity_conflict=conflict,
            )
            if failure is None:
                pending = _PendingEnrollment(
                    speaker_id=speaker_id,
                    vector=vector,
                    quality=quality,
                    decision=decision,
                    speaker_status=speaker_status,
                )
            else:
                block_reason = failure

        bootstrap_learning: Optional[str] = None
        if enroll_bootstrap:
            bootstrap_learning = self._enroll(
                speaker_id, vector, quality, source_utterance_id=uid
            )

        with self._lock:
            state = self._states.get(uid)
            if state is None or state.cancelled:
                return
            if assigned:
                state.assigned_speaker_id = assigned
            if created_now:
                state.created_speaker_id = speaker_id
            if enroll_bootstrap:
                state.enrolled = True
                state.bootstrap_learning = bootstrap_learning
            if conflict:
                state.identity_conflict = True
                state.pending_enrollment = None
                state.enrollment_block_reason = None
            elif pending is not None:
                state.pending_enrollment = pending
                state.enrollment_block_reason = None
            elif block_reason is not None and state.pending_enrollment is None:
                state.enrollment_block_reason = block_reason

        # An identity first heard in this utterance stays "new" for the whole
        # utterance: the rising self-similarity of later windows is not
        # evidence of a known person. A contradictory window is still reported
        # as AMBIGUOUS instead of pretending the new identity matched.
        self_match = created_by_this_utterance and decision.status in (
            CANDIDATE_MATCH,
            KNOWN,
        )
        if self_match or decision.status == NEW_CANDIDATE:
            status = NEW_CANDIDATE
        else:
            status = decision.status
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

    def _eligible_for_enrollment(
        self,
        decision: MatchDecision,
        quality: Optional[VoiceQuality],
        speaker_status: str,
        *,
        identity_conflict: bool,
    ) -> bool:
        """Stricter proof for a permanent sample than for re-identification."""
        return (
            self._enrollment_failure_reason(
                decision,
                quality,
                speaker_status,
                identity_conflict=identity_conflict,
            )
            is None
        )

    def _enrollment_failure_reason(
        self,
        decision: MatchDecision,
        quality: Optional[VoiceQuality],
        speaker_status: str,
        *,
        identity_conflict: bool,
    ) -> Optional[str]:
        """Return why a window must not become a sample, or ``None``.

        Reusing an identity only needs the match threshold; writing a new
        reference changes every future match for that identity, so it requires
        usable audio, enrollment-grade quality, a positive match, a score above
        the independent enrollment threshold, competition protection, no
        identity conflict and the same model version (candidates are loaded
        only for this version).
        """
        if identity_conflict:
            return LEARN_IDENTITY_CONFLICT
        if decision is None or quality is None or not quality.usable:
            return LEARN_LOW_QUALITY
        required = QUALITY_RANK.get(
            self.config.minimum_enrollment_quality, 2
        )
        if quality.rank < required:
            return LEARN_LOW_QUALITY
        if decision.status not in (CANDIDATE_MATCH, KNOWN):
            if decision.status == AMBIGUOUS:
                return LEARN_IDENTITY_CONFLICT
            return LEARN_LOW_SIMILARITY
        if speaker_status not in ("KNOWN", "CANDIDATE"):
            if speaker_status == AMBIGUOUS:
                return LEARN_IDENTITY_CONFLICT
            return LEARN_LOW_SIMILARITY
        if decision.similarity is None:
            return LEARN_LOW_SIMILARITY
        if decision.similarity < self.config.enrollment_match_threshold:
            return LEARN_LOW_SIMILARITY
        if (
            decision.margin is not None
            and decision.margin < self.config.enrollment_margin
        ):
            return LEARN_LOW_MARGIN
        return None


    def _create_candidate(self, uid: int) -> Optional[str]:
        try:
            return self._store.create_candidate()
        except Exception as exc:  # noqa: BLE001 - non-fatal
            self._publish_error(uid, type(exc).__name__)
            return None

    def _enroll(
        self,
        speaker_id: str,
        vector,
        quality: VoiceQuality,
        *,
        source_utterance_id: int,
    ) -> str:
        """Write at most one sample and report exactly what the store did."""
        required = QUALITY_RANK.get(
            self.config.minimum_enrollment_quality, 2
        )
        if quality.rank < required:
            return LEARN_LOW_QUALITY
        # Register the source *before* the write so a cancellation racing the
        # commit is still able to revoke the sample.
        with self._lock:
            self._remember_source_locked(int(source_utterance_id), speaker_id)
        started = self._perf()
        try:
            outcome = self._store.add_embedding(
                speaker_id,
                vector,
                quality=quality.quality,
                voiced_seconds=quality.voiced_seconds,
                model_id=self._model_label,
                embedding_version=self.config.embedding_version,
                source_session_id=self.session_id,
                source_utterance_id=int(source_utterance_id),
                is_verified_reference=False,
            )
        except Exception:  # noqa: BLE001 - a lost sample must not stop ASR
            self.logger.debug(
                "speaker_enroll_failed speaker_id=%s", speaker_id
            )
            return LEARN_STORE_ERROR
        finally:
            self._record_timing(
                "enrollment_ms",
                max(0.0, (self._perf() - started) * 1000.0),
            )
        return LEARN_ADDED if outcome == LEARN_ADDED else LEARN_DUPLICATE

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
                session_id=self.session_id,
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
        started = self._perf()
        try:
            return evaluate_voice_quality(
                window,
                sample_rate=self.sample_rate,
                min_voiced_seconds=self.config.min_voiced_seconds,
            )
        except Exception:  # noqa: BLE001 - never kill the worker
            return None
        finally:
            self._record_timing(
                "quality_check_ms",
                max(0.0, (self._perf() - started) * 1000.0),
            )

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
