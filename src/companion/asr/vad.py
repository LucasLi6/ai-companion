"""Local Silero VAD segmentation for the independent ASR demo.

The model runs on CPU through ONNX Runtime. NumPy, ONNX Runtime and the model
file are imported/loaded lazily so the offline test-suite can exercise the
segmentation state machine with a fake probability function and no optional
dependencies.

The VAD owns the utterance boundary. Qwen ASR ``sentence_end`` events may
confirm text, but they never end the local utterance or stop audio delivery.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from companion.asr.config import ASRConfig, SAMPLE_RATE, VAD_WINDOW_SAMPLES

WAITING_FOR_SPEECH = "WAITING_FOR_SPEECH"
SPEECH_ACTIVE = "SPEECH_ACTIVE"
WAITING_FOR_END = "WAITING_FOR_END"

SPEECH_STARTED = "speech_started"
AUDIO = "audio"
SPEECH_ENDED = "speech_ended"

VAD_MODEL_ENV = "AVA_ASR_VAD_MODEL"


class VADError(Exception):
    """VAD dependency, model or inference failure (display-safe)."""


@dataclass(frozen=True)
class VADEvent:
    """One segmentation output consumed by the manager."""

    kind: str
    audio: bytes = b""


def _window_ms(config: ASRConfig) -> int:
    return max(1, int(round(1000 * VAD_WINDOW_SAMPLES / SAMPLE_RATE)))


def _ceil_windows(milliseconds: int, window_ms: int) -> int:
    if milliseconds <= 0:
        return 0
    return max(1, int(math.ceil(milliseconds / window_ms)))


def resolve_vad_model_path(explicit: str = "") -> Path:
    """Find a local ``silero_vad.onnx`` without downloading anything.

    Search order: explicit argument, ``AVA_ASR_VAD_MODEL``, the demo cache,
    then the ``silero-vad`` package's bundled model data.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get(VAD_MODEL_ENV, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.home() / ".cache" / "ava-asr" / "silero_vad.onnx")

    try:
        import importlib.util

        spec = importlib.util.find_spec("silero_vad")
    except Exception:  # noqa: BLE001 - optional package, broken install
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            base = Path(root)
            candidates.extend(
                [base / "data" / "silero_vad.onnx", base / "silero_vad.onnx"]
            )

    for path in candidates:
        if path.is_file():
            return path
    raise VADError(
        "找不到 Silero VAD ONNX 模型。请执行 pip install 'ai-companion[asr]'，"
        f"或设置 {VAD_MODEL_ENV} 指向 silero_vad.onnx。"
    )


class SileroVAD:
    """Minimal stateful Silero VAD ONNX wrapper (v4 and v5 layouts)."""

    # Silero v5 carries 64 samples of the previous 16 kHz window as context and
    # the ONNX graph consumes context+window (576 samples). Without it the
    # model returns near-zero speech probability even on clear speech.
    CONTEXT_SAMPLES = 64

    def __init__(
        self,
        config: ASRConfig,
        *,
        model_path: str = "",
        session_factory: Optional[Callable[[Path], Any]] = None,
        threads: int = 1,
    ) -> None:
        self._config = config
        self._model_path = model_path or config.vad_model_path
        self._session_factory = session_factory
        self._threads = max(1, threads)
        self._session: Any = None
        self._np: Any = None
        self._input_names: dict[str, str] = {}
        self._output_names: dict[str, str] = {}
        self._state: Any = None
        self._context: Any = None
        self._use_context = False
        self._h: Any = None
        self._c: Any = None
        self._input_shapes: dict[str, tuple[int, ...]] = {}
        self._preloaded = False

    def preload(self) -> None:
        """Load dependencies and the model now so setup errors are immediate."""
        if self._preloaded:
            return
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VADError(
                "numpy 未安装：请执行 pip install 'ai-companion[asr]'"
            ) from exc
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VADError(
                "onnxruntime 未安装：请执行 pip install 'ai-companion[asr]'"
            ) from exc

        path = resolve_vad_model_path(self._model_path)
        try:
            if self._session_factory is not None:
                session = self._session_factory(path)
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = self._threads
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    str(path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise VADError(
                f"无法加载 Silero VAD ONNX 模型（{type(exc).__name__}）。"
            ) from exc

        self._np = np
        self._session = session
        self._input_names = {
            item.name.lower(): item.name for item in session.get_inputs()
        }
        self._output_names = {
            item.name.lower(): item.name for item in session.get_outputs()
        }
        self._input_shapes = {
            item.name.lower(): tuple(
                int(dim) if isinstance(dim, int) else 1 for dim in item.shape
            )
            for item in session.get_inputs()
        }
        self._use_context = "staten" in self._output_names
        self._reset_state()
        self._preloaded = True

    def _reset_state(self) -> None:
        np = self._np
        if np is None:
            return
        state_shape = self._input_shapes.get("state") or (2, 1, 128)
        h_shape = self._input_shapes.get("h") or (2, 1, 64)
        c_shape = self._input_shapes.get("c") or (2, 1, 64)
        self._state = np.zeros(state_shape, dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)
        self._h = np.zeros(h_shape, dtype=np.float32)
        self._c = np.zeros(c_shape, dtype=np.float32)

    def reset(self) -> None:
        if self._preloaded:
            self._reset_state()

    def probability(self, window: bytes) -> float:
        """Return speech probability for exactly 512 mono int16 samples."""
        if not self._preloaded:
            self.preload()
        np = self._np
        expected = VAD_WINDOW_SAMPLES * 2
        if len(window) != expected:
            raise VADError(
                f"Silero VAD 需要 {VAD_WINDOW_SAMPLES} 个采样（{expected} 字节）。"
            )
        audio = np.frombuffer(window, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
        audio = audio.reshape(1, -1)
        model_input = audio
        if self._use_context:
            if (
                self._context is None
                or self._context.shape != (1, self.CONTEXT_SAMPLES)
            ):
                self._context = np.zeros(
                    (1, self.CONTEXT_SAMPLES), dtype=np.float32
                )
            model_input = np.concatenate([self._context, audio], axis=1)
        feed: dict[str, Any] = {
            self._input_names.get("input", "input"): model_input
        }
        if "state" in self._input_names:
            feed[self._input_names["state"]] = self._state
        elif "h" in self._input_names and "c" in self._input_names:
            feed[self._input_names["h"]] = self._h
            feed[self._input_names["c"]] = self._c
        if "sr" in self._input_names:
            feed[self._input_names["sr"]] = np.array(
                SAMPLE_RATE, dtype=np.int64
            )
        try:
            outputs = self._session.run(None, feed)
        except Exception as exc:  # noqa: BLE001 - sanitized
            raise VADError(
                f"Silero VAD 推理失败（{type(exc).__name__}）。"
            ) from exc

        names = list(self._output_names)
        result: dict[str, Any] = {}
        for index, name in enumerate(names):
            if index < len(outputs):
                result[name] = outputs[index]
        probability = result.get("output")
        if probability is None and outputs:
            probability = outputs[0]
        # ``result`` is keyed by the lower-cased ONNX output names, so the v5
        # recurrent state output ``stateN`` is stored as ``"staten"``. Looking
        # up the original casing silently missed it and froze the state.
        if "staten" in result:
            self._state = result["staten"]
        elif "hn" in result:
            self._h = result["hn"]
            if "cn" in result:
                self._c = result["cn"]
        if self._use_context:
            self._context = audio[:, -self.CONTEXT_SAMPLES:].copy()
        value = float(np.asarray(probability).reshape(-1)[0])
        return max(0.0, min(1.0, value))


class VADSegmenter:
    """Window-level Silero state machine with pre-roll and silence handling."""

    def __init__(
        self,
        probability_fn: Callable[[bytes], float],
        config: ASRConfig,
        *,
        window_samples: int = VAD_WINDOW_SAMPLES,
    ) -> None:
        self._probability = probability_fn
        self.window_samples = int(window_samples)
        self.window_bytes = self.window_samples * 2
        self.window_ms = max(
            1, int(round(1000 * self.window_samples / SAMPLE_RATE))
        )
        self.threshold = float(config.vad_threshold)
        self.min_speech_windows = _ceil_windows(
            config.min_speech_ms, self.window_ms
        )
        self.end_silence_windows = _ceil_windows(
            config.end_silence_ms, self.window_ms
        )
        self.pre_roll_windows = _ceil_windows(
            config.pre_roll_ms, self.window_ms
        )
        self.min_speech_ms = self.min_speech_windows * self.window_ms
        self.end_silence_ms = self.end_silence_windows * self.window_ms
        self.pre_roll_ms = self.pre_roll_windows * self.window_ms
        self._buffer = bytearray()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._buffer.clear()
        self.state = WAITING_FOR_SPEECH
        self._pre_roll: deque[bytes] = deque(maxlen=max(1, self.pre_roll_windows))
        self._tail: deque[bytes] = deque(maxlen=max(1, self.pre_roll_windows))
        self._candidate: Optional[list[bytes]] = None
        self._candidate_speech_ms = 0
        self._candidate_silence_ms = 0
        self._silence_ms = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    # ---------------------------------------------------------------- process
    def process(self, pcm: bytes) -> list[VADEvent]:
        """Feed arbitrary-length PCM; return zero or more segmentation events."""
        if not pcm:
            return []
        self._buffer.extend(pcm)
        events: list[VADEvent] = []
        while len(self._buffer) >= self.window_bytes:
            window = bytes(self._buffer[: self.window_bytes])
            del self._buffer[: self.window_bytes]
            events.extend(self._process_window(window))
        return events

    def flush(self) -> list[VADEvent]:
        """Close an active utterance (used on shutdown)."""
        events: list[VADEvent] = []
        if (
            self._candidate is not None
            and self._candidate_speech_ms >= self.min_speech_ms
        ):
            self._activate(events)
        if self.state in (SPEECH_ACTIVE, WAITING_FOR_END):
            events.append(VADEvent(SPEECH_ENDED))
            self._reset_after_end()
        self._candidate = None
        self._candidate_speech_ms = 0
        self._candidate_silence_ms = 0
        self._buffer.clear()
        return events

    def _process_window(self, window: bytes) -> list[VADEvent]:
        probability = float(self._probability(window))
        events: list[VADEvent] = []

        if self.state == WAITING_FOR_SPEECH:
            if self._candidate is None:
                if probability >= self.threshold:
                    # Snapshot only the preceding windows, then the current
                    # speech window; a full pre-roll therefore survives the
                    # first syllable instead of being displaced by it.
                    self._candidate = list(self._pre_roll) + [window]
                    self._candidate_speech_ms = self.window_ms
                    self._candidate_silence_ms = 0
                    if self._candidate_speech_ms >= self.min_speech_ms:
                        self._activate(events)
                else:
                    self._pre_roll.append(window)
                return events

            self._candidate.append(window)
            if probability >= self.threshold:
                self._candidate_speech_ms += self.window_ms
                self._candidate_silence_ms = 0
            else:
                self._candidate_silence_ms += self.window_ms
            if self._candidate_silence_ms >= self.end_silence_ms:
                # A short noise burst never reaches min speech; keep its tail
                # as pre-roll and go back to waiting.
                self._pre_roll.extend(
                    self._candidate[-max(1, self.pre_roll_windows):]
                )
                self._candidate = None
                self._candidate_speech_ms = 0
                self._candidate_silence_ms = 0
            elif self._candidate_speech_ms >= self.min_speech_ms:
                self._activate(events)
            return events

        if self.state == SPEECH_ACTIVE:
            events.append(VADEvent(AUDIO, window))
            self._tail.append(window)
            if probability < self.threshold:
                self.state = WAITING_FOR_END
                self._silence_ms = self.window_ms
            return events

        # WAITING_FOR_END: a short pause resumes speech; a long silence ends
        # the local utterance exactly once.
        events.append(VADEvent(AUDIO, window))
        self._tail.append(window)
        if probability >= self.threshold:
            self.state = SPEECH_ACTIVE
            self._silence_ms = 0
            return events
        self._silence_ms += self.window_ms
        if self._silence_ms >= self.end_silence_ms:
            events.append(VADEvent(SPEECH_ENDED))
            self._reset_after_end()
        return events

    def _activate(self, events: list[VADEvent]) -> None:
        audio = b"".join(self._candidate or [])
        self._candidate = None
        self._candidate_speech_ms = 0
        self._candidate_silence_ms = 0
        self._pre_roll.clear()
        self._tail.clear()
        self.state = SPEECH_ACTIVE
        events.append(VADEvent(SPEECH_STARTED, audio))

    def _reset_after_end(self) -> None:
        self.state = WAITING_FOR_SPEECH
        self._pre_roll = deque(self._tail, maxlen=max(1, self.pre_roll_windows))
        self._tail = deque(maxlen=max(1, self.pre_roll_windows))
        self._silence_ms = 0


__all__ = [
    "AUDIO",
    "SPEECH_ACTIVE",
    "SPEECH_ENDED",
    "SPEECH_STARTED",
    "SileroVAD",
    "VADError",
    "VADEvent",
    "VADSegmenter",
    "WAITING_FOR_END",
    "WAITING_FOR_SPEECH",
    "resolve_vad_model_path",
]
