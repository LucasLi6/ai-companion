"""Reproducible benchmark for the local speaker-identification engine.

This script intentionally runs the *real* pinned ERes2NetV2 model through the
same ``SpeakerEmbeddingModel``/``SpeakerEngine`` used by the demo. It never
substitutes fixed vectors or simulated scores for recognition results.

What it measures (all with the real model, CPU by default):

  load_ms                 model construction + weight load (one time)
  warmup_ms               one warm inference (one time)
  inference_ms            1.5 / 2 / 3 / 5 s windows, P50 and P95
  first_identity_ms       speech start -> first non-PENDING identity result
  audio_at_first_id_s     how much voiced audio was needed for that result
  reid_success_ratio      same person re-identified to the same temporary id
  false_new_ratio         a known person received a new temporary id
  false_match_ratio       a person matched a different enrolled id
  ambiguous_ratio         score fell inside the uncertainty band
  device / voiced length / sample counts / P50 / P95

With ``--continuous-learning`` the script also runs each stage from a fresh
store and reports recognition on a clip that never participated in learning,
alongside the engine's real ``learning`` counters (added / duplicate /
low_similarity / low_margin / low_quality / identity_conflict / asr_failed /
store_error).

Accuracy numbers need a labelled, consented dataset in this layout::

    DATASET/
      alice/01.wav  alice/02.wav ...
      bob/01.wav    bob/02.wav ...

Each speaker's first file is enrolled, every later file is a test. Without
``--dataset`` the script reports accuracy as *untested* — it must not invent a
number. Speech-start -> ASR-final latency needs a live DashScope session and is
also reported as untested here; use the ASR demo for that measurement.

Usage::

    .venv/bin/python scripts/benchmark_speaker.py
    .venv/bin/python scripts/benchmark_speaker.py --device mps --repeats 5
    .venv/bin/python scripts/benchmark_speaker.py --dataset /path/to/clips \
        --continuous-learning --learning-stages 1,3,5,10,20 --json /tmp/spk.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from companion.speaker_id.config import load_speaker_config
from companion.speaker_id.embedding import (
    SpeakerEmbeddingError,
    SpeakerEmbeddingModel,
)
from companion.speaker_id.engine import SpeakerEngine
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    SpeakerResult,
    pending_result,
)
from companion.speaker_id.store import SpeakerStore

SAMPLE_RATE = 16000
DEFAULT_DURATIONS = (1.5, 2.0, 3.0, 5.0)
DEFAULT_LEARNING_STAGES = (1, 3, 5, 10, 20)
# Learning commits happen on the engine worker after ASR is reported. Every
# wait is bounded so a stuck worker fails the benchmark instead of hanging it.
FINALIZE_TIMEOUT_SECONDS = 30.0


# ------------------------------------------------------------------- audio io
def synthetic_audio(seconds: float) -> bytes:
    """Deterministic speech-like multi-tone; timing only, not identity."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    signal = (
        0.10 * np.sin(2 * np.pi * 180 * t)
        + 0.06 * np.sin(2 * np.pi * 340 * t)
        + 0.04 * np.sin(2 * np.pi * 900 * t)
    )
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    return (np.clip(signal * envelope, -1, 1) * 32767).astype("<i2").tobytes()


def read_wav(path: Path) -> bytes:
    try:
        import soundfile as sf  # type: ignore

        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = audio[:, 0]
    except ImportError:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise SystemExit(f"{path}: 只支持 16-bit PCM WAV")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        mono = samples.reshape(-1, channels)[:, 0]
    if rate != SAMPLE_RATE:
        target = int(len(mono) * SAMPLE_RATE / rate)
        mono = np.interp(
            np.linspace(0, len(mono) - 1, target),
            np.arange(len(mono)),
            mono,
        ).astype(np.float32)
    return (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()


def trim(pcm: bytes, seconds: float) -> bytes:
    wanted = int(seconds * SAMPLE_RATE) * 2
    return pcm[:wanted] if len(pcm) >= wanted else pcm


# --------------------------------------------------------------- statistics
def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


# ------------------------------------------------------------------- engine
class ResultSink:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.results: dict[int, SpeakerResult] = {}
        self.first_results: dict[int, SpeakerResult] = {}
        self.first_at: dict[int, float] = {}

    def __call__(self, result: SpeakerResult) -> None:
        with self._condition:
            self.results[result.utterance_id] = result
            if (
                result.status != "PENDING"
                and result.utterance_id not in self.first_at
            ):
                self.first_results[result.utterance_id] = result
                self.first_at[result.utterance_id] = time.perf_counter()
            self._condition.notify_all()

    def wait(self, uid: int, *, timeout: float = 60.0) -> SpeakerResult:
        with self._condition:
            self._condition.wait_for(
                lambda: self.results.get(uid) is not None
                and self.results[uid].status != "PENDING",
                timeout=timeout,
            )
            return self.results.get(uid) or pending_result(uid)

    def first(self, uid: int) -> Optional[SpeakerResult]:
        """First non-PENDING result for an utterance, or None."""
        with self._condition:
            return self.first_results.get(uid)


class TempEngine:
    """One engine on a temporary database; never touches Ava's person DB."""

    def __init__(self, config, model, directory: Path) -> None:
        self.store = SpeakerStore(directory / "benchmark.sqlite3")
        self.store.initialize()
        self.sink = ResultSink()
        self.engine = SpeakerEngine(
            config, embedding_model=model, store=self.store,
            on_result=self.sink,
        )
        self.engine.start()

    def run(self, uid: int, pcm: bytes, *, learn: bool = True) -> SpeakerResult:
        """One whole utterance, including its learning commit.

        A match result is not a finished utterance: the engine only learns
        after ``on_finalized`` and the worker has processed it. Waiting for
        ``is_finished`` guarantees the next utterance starts from the committed
        sample set (and that a rejected sample is fully rejected).
        """
        self.engine.on_start(uid, pcm)
        self.engine.on_end(uid)
        result = self.sink.wait(uid, timeout=FINALIZE_TIMEOUT_SECONDS)
        self.engine.on_finalized(uid, learn)
        if not self._wait_finished(uid):
            raise TimeoutError(f"utterance {uid} did not finish in time")
        return result

    def identify(self, uid: int, pcm: bytes) -> SpeakerResult:
        """Recognise without learning (ASR is reported as failed)."""
        return self.run(uid, pcm, learn=False)

    def _wait_finished(self, uid: int) -> bool:
        deadline = time.monotonic() + FINALIZE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.engine.is_finished(uid):
                return True
            time.sleep(0.01)
        return False

    def run_realtime(self, uid: int, pcm: bytes):
        """Feed 20 ms frames in wall-clock time, like a microphone.

        Returns ``(first_result, first_identity_ms, before_vad_end)``. The
        whole clip is fed, then the final analysis and learning commit run so
        a following utterance observes the updated sample set.
        """
        chunk = int(0.02 * SAMPLE_RATE) * 2
        started = time.perf_counter()
        deadline = started + 0.02
        self.engine.on_start(uid, pcm[:chunk])
        for offset in range(chunk, len(pcm), chunk):
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            deadline += 0.02
            self.engine.on_audio(uid, pcm[offset:offset + chunk])
        feed_end = time.perf_counter()
        self.engine.on_end(uid)
        first = self.sink.first(uid)
        if first is None:
            first = self.sink.wait(uid, timeout=FINALIZE_TIMEOUT_SECONDS)
        first_at = self.sink.first_at.get(uid, feed_end)
        self.engine.on_finalized(uid, True)
        if not self._wait_finished(uid):
            raise TimeoutError(f"utterance {uid} did not finish in time")
        return first, (first_at - started) * 1000.0, first_at <= feed_end

    def close(self) -> None:
        self.engine.shutdown()
        self.store.close()


# ------------------------------------------------------------------ benchmarks
def benchmark_model(model, audio_by_duration, repeats: int) -> dict:
    results = {}
    for seconds, pcm in audio_by_duration.items():
        times = []
        embeddings = []
        for _ in range(repeats):
            started = time.perf_counter()
            vector = model.encode(pcm)
            times.append((time.perf_counter() - started) * 1000.0)
            embeddings.append(float(np.linalg.norm(vector)))
        results[f"{seconds:g}s"] = {
            **summary(times),
            "embedding_norm": statistics.fmean(embeddings),
            "dims": int(len(vector)),
        }
    return results


def benchmark_first_identity(config, model, audio_by_duration) -> dict:
    results = {}
    for index, (seconds, pcm) in enumerate(
        audio_by_duration.items(), start=1
    ):
        # A fresh store per duration means every measurement is a genuine
        # first-ever identity, with no sample from a shorter clip to match.
        with tempfile.TemporaryDirectory(prefix="ava-speaker-bench-") as tmp:
            engine = TempEngine(config, model, Path(tmp))
            try:
                result, elapsed, before_end = engine.run_realtime(index, pcm)
            finally:
                engine.close()
        results[f"{seconds:g}s"] = {
            # Wall clock from the first 20 ms frame to the first non-PENDING
            # identity, including the 1.5 s of speech the engine waits for
            # before its first analysis.
            "first_identity_ms": elapsed,
            "before_vad_end": before_end,
            "status": result.status,
            "speaker_id": result.speaker_id,
            "audio_seconds_at_first_id": result.voiced_seconds,
            "inference_ms": result.inference_ms,
            "reid_quality": result.reid_quality,
        }
    return results


def load_dataset(root: Path) -> dict[str, list[Path]]:
    dataset: dict[str, list[Path]] = {}
    for speaker_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            path for path in speaker_dir.iterdir()
            if path.suffix.lower() in (".wav", ".flac")
        )
        if files:
            dataset[speaker_dir.name] = files
    return dataset


def benchmark_accuracy(config, model, dataset: dict[str, list[Path]]) -> dict:
    enrolled: dict[str, Optional[str]] = {}
    counts = {
        "tests": 0,
        "reid_success": 0,
        "false_new": 0,
        "false_match": 0,
        "ambiguous": 0,
        "unusable_or_error": 0,
    }
    details = []
    with tempfile.TemporaryDirectory(prefix="ava-speaker-bench-") as tmp:
        engine = TempEngine(config, model, Path(tmp))
        uid = 0
        try:
            for label, files in dataset.items():
                uid += 1
                first = engine.run(uid, read_wav(files[0]))
                enrolled[label] = first.speaker_id
                details.append(
                    {
                        "speaker": label,
                        "file": files[0].name,
                        "phase": "enroll",
                        "status": first.status,
                        "speaker_id": first.speaker_id,
                        "similarity": first.similarity,
                        "quality": first.reid_quality,
                    }
                )
                for path in files[1:]:
                    uid += 1
                    result = engine.run(uid, read_wav(path))
                    counts["tests"] += 1
                    expected = enrolled[label]
                    if expected is None:
                        # This speaker's enrolment clip was unusable; its test
                        # clips cannot be scored as a false match.
                        counts["unusable_or_error"] += 1
                    elif result.status in (KNOWN, CANDIDATE_MATCH):
                        if result.speaker_id == expected:
                            counts["reid_success"] += 1
                        else:
                            counts["false_match"] += 1
                    elif result.status == NEW_CANDIDATE:
                        counts["false_new"] += 1
                    elif result.status == AMBIGUOUS:
                        counts["ambiguous"] += 1
                    elif result.status in (INSUFFICIENT_AUDIO, ERROR):
                        counts["unusable_or_error"] += 1
                    details.append(
                        {
                            "speaker": label,
                            "file": path.name,
                            "phase": "test",
                            "status": result.status,
                            "speaker_id": result.speaker_id,
                            "expected_speaker_id": expected,
                            "similarity": result.similarity,
                            "margin": result.margin,
                            "quality": result.reid_quality,
                        }
                    )
        finally:
            engine.close()

    tests = counts["tests"]
    ratio = lambda value: (value / tests) if tests else None  # noqa: E731
    return {
        **counts,
        "reid_success_ratio": ratio(counts["reid_success"]),
        "false_new_ratio": ratio(counts["false_new"]),
        "false_match_ratio": ratio(counts["false_match"]),
        "ambiguous_ratio": ratio(counts["ambiguous"]),
        "details": details,
    }


def benchmark_continuous_learning(
    config,
    model,
    dataset: dict[str, list[Path]],
    stages=DEFAULT_LEARNING_STAGES,
) -> dict:
    """Recognition after N successful learning utterances per speaker.

    Each stage starts from a fresh temporary store. The first clip enrolls a
    speaker; the reserved last clip is *never* used for learning and is the
    only clip scored. Learning clips are cycled from the remaining files (or
    the enrollment clip when a speaker has no files between), so the store's
    duplicate/diversity rules decide what is actually kept.

    Accuracy is only "measured" for a labelled, consented dataset. With no
    usable dataset the caller reports ``untested`` instead of inventing it.
    """
    usable = {
        label: files
        for label, files in dataset.items()
        if len(files) >= 2
    }
    if len(usable) < 2:
        return {
            "status": "untested",
            "reason": (
                "至少需要两位说话人，且每位至少两段音频"
                "（一段登记、一段从不参与学习的测试）。"
            ),
        }

    stage_results = []
    for stage in stages:
        with tempfile.TemporaryDirectory(prefix="ava-speaker-learn-") as tmp:
            engine = TempEngine(config, model, Path(tmp))
            try:
                enrolled: dict[str, Optional[str]] = {}
                uid = 0
                for label, files in usable.items():
                    uid += 1
                    first = engine.run(uid, read_wav(files[0]))
                    enrolled[label] = first.speaker_id

                attempts = 0
                for round_index in range(int(stage)):
                    for label, files in usable.items():
                        pool = files[1:-1] or [files[0]]
                        clip = pool[round_index % len(pool)]
                        uid += 1
                        engine.run(uid, read_wav(clip))
                        attempts += 1

                # Snapshot learning statistics before recognition tests, which
                # are reported to the engine as ASR failures and must not count.
                learning = dict(engine.engine.stats()["learning"])
                stored = sum(
                    int(item.get("samples") or 0)
                    for item in engine.store.list_speakers()
                )

                tests = 0
                reid = 0
                false_match = 0
                unknown = 0
                for label, files in usable.items():
                    uid += 1
                    result = engine.identify(uid, read_wav(files[-1]))
                    tests += 1
                    expected = enrolled[label]
                    if expected is None:
                        unknown += 1
                    elif result.status in (KNOWN, CANDIDATE_MATCH):
                        if result.speaker_id == expected:
                            reid += 1
                        else:
                            false_match += 1
                    else:
                        unknown += 1
            finally:
                engine.close()

        ratio = lambda value: (value / tests) if tests else None  # noqa: E731
        added = int(learning.get("added", 0))
        stage_results.append(
            {
                "learning_count": attempts,
                "stored_embedding_count": stored,
                "recognition_accuracy": ratio(reid),
                "false_match_rate": ratio(false_match),
                "unknown_or_ambiguous_rate": ratio(unknown),
                "added_samples": added,
                "rejected_samples_by_reason": {
                    key: value
                    for key, value in learning.items()
                    if key != "added"
                },
                "test_count": tests,
            }
        )

    return {
        "status": "measured",
        "speakers": sorted(usable),
        "stages": stage_results,
        "note": (
            "每个阶段从空库开始；最后一段录音从不参与学习。"
            "准确率只来自该标注数据集，不适用于其他设备或噪声条件。"
        ),
    }


def benchmark_calibration(model, dataset: dict[str, list[Path]]) -> dict:
    """Pairwise score distributions plus conservative threshold suggestions.

    This is a *proposal* tool, not a certification: a handful of clips per
    speaker is far too little to claim accuracy. It only tells you whether the
    enrolled clips of different people are separated at all.
    """
    embeddings: dict[str, list[np.ndarray]] = {}
    for label, files in dataset.items():
        vectors = []
        for path in files:
            try:
                vectors.append(model.encode(read_wav(path)))
            except SpeakerEmbeddingError:
                continue
        if vectors:
            embeddings[label] = vectors
    if len(embeddings) < 2:
        return {
            "status": "insufficient",
            "reason": "至少需要两位说话人各一段可用音频。",
        }

    same: list[float] = []
    different: list[float] = []
    for label, vectors in embeddings.items():
        for index, vector in enumerate(vectors):
            for other_index, other in enumerate(vectors):
                if index != other_index:
                    same.append(float(vector @ other))
            for other_label, other_vectors in embeddings.items():
                if other_label == label:
                    continue
                for other in other_vectors:
                    different.append(float(vector @ other))
    if not same or not different:
        return {
            "status": "insufficient",
            "reason": "每位说话人至少需要两段音频才能估计同人分布。",
        }

    p95_different = float(np.percentile(different, 95))
    p99_different = float(np.percentile(different, 99))
    p5_same = float(np.percentile(same, 5))
    p50_same = float(np.percentile(same, 50))
    separable = p5_same > p95_different
    suggestion = None
    if separable:
        candidate = round((p95_different + p5_same) / 2.0, 3)
        known = round(max(candidate + 0.05, (p99_different + p50_same) / 2.0), 3)
        # Creating an identity requires being clearly below the different-speaker
        # range. Using p95 - 0.05 would suggest a gate below this model's
        # different-speaker scores (mean ~0.12), which would make every new
        # person ambiguous; p99 + headroom keeps the false-new risk bounded.
        new_candidate = round(max(0.0, p99_different + 0.05), 3)
        suggestion = {
            "new_candidate_threshold": new_candidate,
            "candidate_match_threshold": candidate,
            "known_match_threshold": min(0.99, known),
            "minimum_match_margin": 0.06,
            "note": (
                "仅基于当前小样本；上线前请用更多说话人/设备/噪声条件复测。"
                "分数重叠时不要降低阈值。"
            ),
        }
    return {
        "pairs_same": len(same),
        "pairs_different": len(different),
        "same": {
            "p5": p5_same,
            "p50": p50_same,
            "p95": float(np.percentile(same, 95)),
        },
        "different": {
            "p5": float(np.percentile(different, 5)),
            "p50": float(np.percentile(different, 50)),
            "p95": p95_different,
            "p99": p99_different,
        },
        "separable": separable,
        "suggestion": suggestion,
    }


# ---------------------------------------------------------------------- main
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the local ERes2NetV2 speaker engine (real model)."
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps", "auto"), default="cpu"
    )
    parser.add_argument(
        "--durations", default="1.5,2,3,5",
        help="comma-separated window lengths in seconds",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dataset", default=None,
        help="labelled directory: DATASET/<speaker>/<clip>.wav",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="with --dataset, also print pairwise score distributions and a "
        "conservative threshold suggestion",
    )
    parser.add_argument(
        "--continuous-learning", action="store_true",
        help="with --dataset, measure recognition after 1/3/5/10/20 successful "
        "learning utterances per speaker",
    )
    parser.add_argument(
        "--learning-stages", default="1,3,5,10,20",
        help="comma-separated cumulative learning counts per speaker",
    )
    parser.add_argument(
        "--audio", default=None,
        help="optional 16 kHz WAV used instead of synthetic timing audio",
    )
    parser.add_argument("--json", default=None, help="write the report as JSON")
    parser.add_argument(
        "--skip-identity", action="store_true",
        help="only measure model load/warmup/inference",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    durations = tuple(
        float(part) for part in args.durations.split(",") if part.strip()
    )
    if not durations:
        print("错误: --durations 不能为空。", file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("错误: --repeats 必须大于等于 1。", file=sys.stderr)
        return 2
    try:
        learning_stages = tuple(
            int(part) for part in args.learning_stages.split(",") if part.strip()
        )
    except ValueError:
        print("错误: --learning-stages 必须是逗号分隔的整数。", file=sys.stderr)
        return 2
    if not learning_stages or any(stage < 1 for stage in learning_stages):
        print("错误: --learning-stages 每项必须大于等于 1。", file=sys.stderr)
        return 2

    config = load_speaker_config(device=args.device)
    model = SpeakerEmbeddingModel(
        model_id=config.model_id,
        model_revision=config.model_revision,
        model_path=config.model_path,
        device=config.device,
    )

    print(
        f"模型: {config.model_id}@{config.model_revision} "
        f"| 设备: {config.device}",
        file=sys.stderr,
    )
    started = time.perf_counter()
    try:
        model.load()
    except SpeakerEmbeddingError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    load_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    try:
        model.warmup()
    except SpeakerEmbeddingError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        model.close()
        return 2
    warmup_ms = (time.perf_counter() - started) * 1000.0

    source = read_wav(Path(args.audio)) if args.audio else None
    audio_by_duration = {}
    for seconds in durations:
        if source is not None:
            pcm = trim(source, seconds)
            if len(pcm) < int(seconds * SAMPLE_RATE) * 2:
                continue
        else:
            pcm = synthetic_audio(seconds)
        audio_by_duration[seconds] = pcm
    if not audio_by_duration:
        print("错误: --audio 太短，无法覆盖任何时长。", file=sys.stderr)
        model.close()
        return 2

    report: dict = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "embedding_version": config.embedding_version,
        "device": model.active_device,
        "thresholds": {
            "known_match": config.known_match_threshold,
            "candidate_match": config.candidate_match_threshold,
            "new_candidate": config.new_candidate_threshold,
            "minimum_margin": config.minimum_match_margin,
            "minimum_enrollment_quality": (
                config.minimum_enrollment_quality
            ),
        },
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "inference": benchmark_model(model, audio_by_duration, args.repeats),
    }

    if not args.skip_identity:
        report["first_identity"] = benchmark_first_identity(
            config, model, audio_by_duration
        )

    dataset: dict[str, list[Path]] = {}
    if args.dataset:
        root = Path(args.dataset).expanduser()
        dataset = load_dataset(root) if root.is_dir() else {}
        if not dataset:
            print(
                f"错误: {root} 下没有找到 <speaker>/<clip>.wav。",
                file=sys.stderr,
            )
            model.close()
            return 2
        report["accuracy"] = benchmark_accuracy(config, model, dataset)
        if args.calibrate:
            report["calibration"] = benchmark_calibration(model, dataset)
    elif args.calibrate:
        print("错误: --calibrate 需要同时提供 --dataset。", file=sys.stderr)
        model.close()
        return 2
    else:
        report["accuracy"] = {
            "status": "untested",
            "reason": "未提供带说话人标签且已获同意的数据集（--dataset）。",
        }

    if args.continuous_learning:
        if dataset:
            report["continuous_learning"] = benchmark_continuous_learning(
                config, model, dataset, learning_stages
            )
        else:
            report["continuous_learning"] = {
                "status": "untested",
                "reason": (
                    "持续学习准确率需要带说话人标签且已获同意的数据集"
                    "（--dataset），未使用随机向量代替。"
                ),
            }

    report["asr_final_latency"] = {
        "status": "untested",
        "reason": "实时 ASR 最终延迟需要真实 DashScope 会话，请用 ASR Demo 测量。",
    }
    model.close()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已写入 {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
