"""Deferred voice-print startup so the model never delays ASR.

The voice-print model needs several seconds to load and warm up. Doing that
before ``ASRManager.run()`` would delay the microphone and the first
transcript, and a model/network problem would stop voice input entirely. This
adapter owns the model, the local store and the :class:`SpeakerEngine`, builds
them on a background thread, and exposes the same non-blocking
``SpeakerObserver`` interface to the ASR manager::

    observer = DeferredSpeakerObserver(config)
    manager = ASRManager(config, speaker_observer=observer)
    observer.start_loading()
    manager.run()

Rules:

* before the background thread reports READY every callback returns
  immediately; ``snapshot`` answers PENDING (ERROR-quality UNKNOWN after a
  failed load, with :attr:`error` explaining why);
* an utterance that began before READY is skipped for its whole lifetime, even
  if READY arrives while it is still being spoken — a partially loaded model
  must never see a mutilated window;
* after READY the callbacks forward to the warmed engine, which buffers and
  analyses on its own worker thread;
* a load/warmup/store failure is recorded and the ASR flow continues without
  voice identity; no exception escapes into the audio path;
* ``shutdown`` stops the engine and store and makes late callbacks no-ops.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from companion.speaker_id.config import SpeakerConfig
from companion.speaker_id.engine import SpeakerEngine
from companion.speaker_id.models import (
    ERROR,
    QUALITY_UNKNOWN,
    PENDING,
    SpeakerResult,
    pending_result,
)

IDLE = "IDLE"
LOADING = "LOADING"
READY = "READY"
UNAVAILABLE = "UNAVAILABLE"
CLOSED = "CLOSED"

_MAX_SKIPPED = 2048


class DeferredSpeakerObserver:
    """Owns the model/store/engine while keeping startup off the ASR path."""

    def __init__(
        self,
        config: SpeakerConfig,
        *,
        embedding_model: Any = None,
        store: Any = None,
        on_result: Optional[Callable[[SpeakerResult], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._embedding_model = embedding_model
        self._store = store
        self._on_result = on_result
        self._clock = clock
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._engine: Optional[SpeakerEngine] = None
        self._started = False
        self._closed = False
        self._error: Optional[str] = None
        self._status = IDLE
        self._load_ms: Optional[float] = None
        self._warmup_ms: Optional[float] = None
        self._ready_at: Optional[float] = None
        # Utterances that were opened before READY are skipped entirely.
        self._skipped: set[int] = set()
        self._skipped_order: deque[int] = deque(maxlen=_MAX_SKIPPED)
        self.logger = logging.getLogger("companion.speaker")

    # ---------------------------------------------------------------- status
    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def engine(self) -> Optional[SpeakerEngine]:
        with self._lock:
            return self._engine

    @property
    def timings(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "model_load_ms": self._load_ms,
                "warmup_ms": self._warmup_ms,
                "ready_at": self._ready_at,
                "error": self._error,
            }

    def set_on_result(
        self, callback: Optional[Callable[[SpeakerResult], None]]
    ) -> None:
        """Forward results to the embedding application (may be None)."""
        self._on_result = callback

    # ------------------------------------------------------------- lifecycle
    def start_loading(self) -> None:
        """Start the background load; returns immediately."""
        with self._lock:
            if self._started or self._closed:
                return
            self._started = True
            self._status = LOADING
            self._thread = threading.Thread(
                target=self._load,
                name="speaker-bootstrap",
                daemon=True,
            )
            self._thread.start()

    def _load(self) -> None:
        model = None
        store = None
        engine = None
        started_at = self._clock()
        try:
            model = self._embedding_model or self._build_model()
            load = getattr(model, "load", None)
            if callable(load):
                load()
            loaded_at = self._clock()
            warmup = getattr(model, "warmup", None)
            if callable(warmup):
                warmup()
            warmed_at = self._clock()

            store = self._store or self._build_store()
            store.initialize()

            engine = SpeakerEngine(
                self.config,
                embedding_model=model,
                store=store,
                on_result=self._forward,
                clock=self._clock,
            )
            engine.start()

            with self._lock:
                if self._closed:
                    engine.shutdown()
                    self._close_owned(model, store)
                    return
                self._engine = engine
                self._load_ms = max(0.0, (loaded_at - started_at) * 1000.0)
                self._warmup_ms = max(0.0, (warmed_at - loaded_at) * 1000.0)
                self._ready_at = self._clock()
                self._status = READY
                self._ready.set()
        except Exception as exc:  # noqa: BLE001 - ASR must keep running
            message = "声纹功能不可用（" + type(exc).__name__ + "）。"
            with self._lock:
                self._error = message
                self._status = UNAVAILABLE
                self._engine = None
            self.logger.debug(
                "speaker_bootstrap_failed error_type=%s", type(exc).__name__
            )
            if engine is not None:
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            if not self._closed:
                self._close_owned(model, store)

    def _build_model(self) -> Any:
        from companion.speaker_id.embedding import SpeakerEmbeddingModel

        return SpeakerEmbeddingModel(
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            model_path=self.config.model_path,
            device=self.config.device,
            sample_rate=self.config.sample_rate,
            dim=self.config.embedding_dim,
        )

    def _build_store(self) -> Any:
        from companion.speaker_id.store import SpeakerStore

        return SpeakerStore(
            self.config.db_path,
            max_embeddings_per_speaker=self.config.max_embeddings_per_speaker,
            duplicate_sample_similarity=(
                self.config.duplicate_sample_similarity
            ),
            minimum_enrollment_quality=(
                self.config.minimum_enrollment_quality
            ),
        )

    def _close_owned(self, model: Any, store: Any) -> None:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        if store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    def shutdown(self) -> None:
        """Stop the engine/store and make every later callback a no-op."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._status = CLOSED
            engine = self._engine
            self._engine = None
            self._ready.clear()
            thread = self._thread
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.config.shutdown_timeout_seconds)
        # If the loader is still running it owns the model/store and will
        # close them when it observes ``_closed``; closing underneath it
        # would race with initialisation.
        loader_alive = thread is not None and thread.is_alive()
        with self._lock:
            if not loader_alive:
                self._close_owned(
                    self._embedding_model if engine is not None else None,
                    self._store,
                )

    # ------------------------------------------------- SpeakerObserver hooks
    def _forward_target(self, uid: int) -> Optional[SpeakerEngine]:
        if self._closed:
            return None
        engine = self._engine
        if engine is None or not self._ready.is_set():
            return None
        if uid in self._skipped:
            return None
        return engine

    def on_start(self, utterance_id: int, pcm: bytes) -> None:
        uid = int(utterance_id)
        if self._closed:
            return
        if not self._ready.is_set():
            # Opened before the model was ready: skip it for its whole
            # lifetime, even if READY arrives while it is still spoken.
            with self._lock:
                if len(self._skipped_order) == self._skipped_order.maxlen:
                    self._skipped.discard(self._skipped_order[0])
                self._skipped.add(uid)
                self._skipped_order.append(uid)
            return
        engine = self._forward_target(uid)
        if engine is not None:
            engine.on_start(uid, pcm)

    def on_audio(self, utterance_id: int, pcm: bytes) -> None:
        engine = self._forward_target(int(utterance_id))
        if engine is not None:
            engine.on_audio(int(utterance_id), pcm)

    def on_end(self, utterance_id: int) -> None:
        uid = int(utterance_id)
        engine = self._forward_target(uid)
        if engine is not None:
            engine.on_end(uid)
        self._forget(uid)

    def on_finalized(self, utterance_id: int, success: bool) -> None:
        uid = int(utterance_id)
        engine = self._forward_target(uid)
        if engine is not None:
            engine.on_finalized(uid, success)
        self._forget(uid)

    def on_cancel(self, utterance_id: int) -> None:
        uid = int(utterance_id)
        engine = self._forward_target(uid)
        if engine is not None:
            engine.on_cancel(uid)
        self._forget(uid)

    def _forget(self, uid: int) -> None:
        if not self._skipped:
            return
        with self._lock:
            self._skipped.discard(uid)

    def snapshot(self, utterance_id: int) -> SpeakerResult:
        uid = int(utterance_id)
        engine = self._forward_target(uid)
        if engine is not None:
            return engine.snapshot(uid)
        with self._lock:
            error = self._error
            status = self._status
            ready_at = self._ready_at
        if status == UNAVAILABLE or error is not None:
            return SpeakerResult(
                utterance_id=uid,
                speaker_id=None,
                person_id=None,
                status=ERROR,
                reid_quality=QUALITY_UNKNOWN,
                similarity=None,
                margin=None,
                voiced_seconds=0.0,
                provisional=False,
                model_id=self.config.model_key,
                inference_ms=None,
                completed_at=self._clock(),
            )
        return pending_result(
            uid,
            model_id=self.config.model_key,
            at=ready_at if ready_at is not None else self._clock(),
        )

    def _forward(self, result: SpeakerResult) -> None:
        if self._closed:
            return
        callback = self._on_result
        if callback is None:
            return
        try:
            callback(result)
        except Exception:  # noqa: BLE001 - rendering must not kill ASR
            pass


__all__ = [
    "CLOSED",
    "DeferredSpeakerObserver",
    "IDLE",
    "LOADING",
    "READY",
    "UNAVAILABLE",
]
