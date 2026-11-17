"""Configuration for the optional local speaker identification.

Every value is a conservative starting point, not a calibrated constant. The
match thresholds are only meaningful after the offline calibration described
in ``scripts/benchmark_speaker.py`` has been run on labelled, consented
multi-speaker recordings; until then the engine prefers ambiguous/unknown
results over creating new identities.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from companion.asr.config import load_local_env

DEFAULT_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
DEFAULT_MODEL_REVISION = "v1.0.1"
DEFAULT_EMBEDDING_VERSION = "eres2netv2-192-v1"
EMBEDDING_DIM = 192
SAMPLE_RATE = 16000

# Default store. Kept out of the Ava database on purpose; the demo never
# touches ``persons`` or ``memories``.
DEFAULT_DB_PATH = Path.home() / ".cache" / "ava-speaker" / "speakers.sqlite3"

DEVICES = ("cpu", "mps", "auto")

# ---------------------------------------------------------------- scheduling
# Analysis starts while the user is still speaking. The first snapshot is
# published once roughly this much audio exists; the second refines it; later
# windows refresh it every ~1 s. These are experiment values, not a minimum
# audio requirement of the model.
DEFAULT_FIRST_ANALYSIS_SECONDS = 1.5
DEFAULT_SECOND_ANALYSIS_SECONDS = 2.7
DEFAULT_REPEAT_ANALYSIS_SECONDS = 1.0

# Only a bounded recent window is ever copied for analysis: a long monologue
# must not grow memory or inference time without bound.
DEFAULT_MAX_WINDOW_SECONDS = 4.5
DEFAULT_MIN_ANALYSIS_SECONDS = 0.9
DEFAULT_MIN_VOICED_SECONDS = 0.9

# Bounded command queue; audio chunks are dropped (never blocking the VAD or
# ASR thread) if the worker cannot keep up.
DEFAULT_COMMAND_QUEUE_SIZE = 256
DEFAULT_MAX_ACTIVE_UTTERANCES = 8
DEFAULT_RESULT_RETENTION_SECONDS = 120.0

# ---------------------------------------------------------------- matching
# Conservative defaults derived from real measurements with the pinned model
# on the model package's labelled example clips, scored on the same
# quality-gated 1.5 s windows the engine actually analyses:
#   * same speaker across recordings: cosine p5 0.480 / p50 0.563 / p95 0.637;
#   * different speakers: p95 0.175 / p99 0.206 / max 0.207 (n=96 pairs).
# The thresholds sit in that gap rather than on it:
#   * below 0.30 the closest identity is no closer than a different person,
#     so a temporary identity may be created;
#   * 0.30 .. 0.43 is the uncertainty band (AMBIGUOUS, no new identity);
#   * >= 0.43 reuses the existing identity; >= 0.48 may report a promoted
#     KNOWN identity. These are still small-sample values: run
#     ``scripts/benchmark_speaker.py --calibrate`` on consented labelled data
#     before relying on identity claims.
DEFAULT_KNOWN_MATCH_THRESHOLD = 0.48
DEFAULT_CANDIDATE_MATCH_THRESHOLD = 0.43
DEFAULT_NEW_CANDIDATE_THRESHOLD = 0.30
DEFAULT_MINIMUM_MATCH_MARGIN = 0.06
DEFAULT_MINIMUM_ENROLLMENT_QUALITY = "MEDIUM"

# Enrolment is a *write* into the long-lived store, so it gets its own,
# stricter proof than recognition. The default is deliberately not below the
# known-match threshold; it is a conservative starting point, not a calibrated
# value. Matching (read-only) may reuse an identity at 0.43, but only a
# clearly stronger score is allowed to add a permanent reference sample.
DEFAULT_ENROLLMENT_MATCH_THRESHOLD = 0.55
DEFAULT_ENROLLMENT_MARGIN = 0.10

DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER = 8
DEFAULT_MAX_CENTERS_PER_SPEAKER = 2

# A new sample is considered redundant when it is this close to an existing
# vector of the same identity. Duplicates are only kept if they are better
# quality, and then replace the closest non-protected sample.
DEFAULT_DUPLICATE_SAMPLE_SIMILARITY = 0.995

# Bounded gain preprocessing is OFF by default: it changes the waveform that
# reaches the model and therefore the embedding. Enabling it must go through
# the versioned model key, so old and new vectors are never compared.
DEFAULT_GAIN_PREPROCESSING = False
GAIN_PREPROCESSING_VERSION = "gain-v1"
DEFAULT_MAX_GAIN_DB = 12.0

# ------------------------------------------------------------ voice profiles
# V2 modes. ``legacy`` is the default so existing installations behave exactly
# as before; ``shadow`` computes the new score next to the old one and returns
# the old verdict; ``variants`` uses clustered sample centroids and the
# cross-utterance tracker.
PROFILE_MODES = ("legacy", "shadow", "variants")
DEFAULT_PROFILE_MODE = "legacy"

MATCH_STRATEGIES = ("legacy_max_sample", "variant_centroid")
DEFAULT_MATCH_STRATEGY = "legacy_max_sample"
PROFILE_ALGORITHM_VERSION = "voice-profile-v2"

# A cluster merge threshold is a *starting point*, not a calibrated constant:
# it sits at the enrollment similarity floor. Run the benchmark on labelled
# data before treating it as optimal. It is deliberately not Speakr's 0.72.
DEFAULT_VARIANT_MAX_COUNT = 4
DEFAULT_VARIANT_MERGE_THRESHOLD = 0.55
DEFAULT_SINGLETON_PENALTY = 0.0

DEFAULT_TRACKING_ENABLED = True
DEFAULT_TRACK_EXPIRY_SECONDS = 15.0
DEFAULT_TRACK_RECENT_TURNS = 4
DEFAULT_NEW_ID_MIN_TURNS = 2
DEFAULT_NEW_ID_MIN_VOICED_SECONDS = 3.0
DEFAULT_PENDING_MAX_COUNT = 12
DEFAULT_AUTO_LEARN_ENABLED = True


class SpeakerConfigError(Exception):
    """Missing or invalid speaker configuration (display-safe)."""


@dataclass(frozen=True)
class SpeakerConfig:
    db_path: Path = DEFAULT_DB_PATH
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    model_path: str = ""
    embedding_version: str = DEFAULT_EMBEDDING_VERSION
    embedding_dim: int = EMBEDDING_DIM
    sample_rate: int = SAMPLE_RATE
    device: str = "cpu"

    first_analysis_seconds: float = DEFAULT_FIRST_ANALYSIS_SECONDS
    second_analysis_seconds: float = DEFAULT_SECOND_ANALYSIS_SECONDS
    repeat_analysis_seconds: float = DEFAULT_REPEAT_ANALYSIS_SECONDS
    max_window_seconds: float = DEFAULT_MAX_WINDOW_SECONDS
    min_analysis_seconds: float = DEFAULT_MIN_ANALYSIS_SECONDS
    min_voiced_seconds: float = DEFAULT_MIN_VOICED_SECONDS

    command_queue_size: int = DEFAULT_COMMAND_QUEUE_SIZE
    max_active_utterances: int = DEFAULT_MAX_ACTIVE_UTTERANCES
    result_retention_seconds: float = DEFAULT_RESULT_RETENTION_SECONDS
    shutdown_timeout_seconds: float = 3.0

    known_match_threshold: float = DEFAULT_KNOWN_MATCH_THRESHOLD
    candidate_match_threshold: float = DEFAULT_CANDIDATE_MATCH_THRESHOLD
    new_candidate_threshold: float = DEFAULT_NEW_CANDIDATE_THRESHOLD
    minimum_match_margin: float = DEFAULT_MINIMUM_MATCH_MARGIN
    minimum_enrollment_quality: str = DEFAULT_MINIMUM_ENROLLMENT_QUALITY
    enrollment_match_threshold: float = DEFAULT_ENROLLMENT_MATCH_THRESHOLD
    enrollment_margin: float = DEFAULT_ENROLLMENT_MARGIN
    duplicate_sample_similarity: float = DEFAULT_DUPLICATE_SAMPLE_SIMILARITY
    gain_preprocessing: bool = DEFAULT_GAIN_PREPROCESSING
    max_gain_db: float = DEFAULT_MAX_GAIN_DB

    max_embeddings_per_speaker: int = DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER
    max_centers_per_speaker: int = DEFAULT_MAX_CENTERS_PER_SPEAKER

    # ---------------------------------------------------------- V2 profiles
    profile_mode: str = DEFAULT_PROFILE_MODE
    match_strategy: str = DEFAULT_MATCH_STRATEGY
    variant_max_count: int = DEFAULT_VARIANT_MAX_COUNT
    variant_merge_threshold: float = DEFAULT_VARIANT_MERGE_THRESHOLD
    singleton_penalty: float = DEFAULT_SINGLETON_PENALTY
    tracking_enabled: bool = DEFAULT_TRACKING_ENABLED
    track_expiry_seconds: float = DEFAULT_TRACK_EXPIRY_SECONDS
    track_recent_turns: int = DEFAULT_TRACK_RECENT_TURNS
    new_id_min_turns: int = DEFAULT_NEW_ID_MIN_TURNS
    new_id_min_voiced_seconds: float = DEFAULT_NEW_ID_MIN_VOICED_SECONDS
    pending_max_count: int = DEFAULT_PENDING_MAX_COUNT
    auto_learn_enabled: bool = DEFAULT_AUTO_LEARN_ENABLED
    profile_algorithm_version: str = PROFILE_ALGORITHM_VERSION

    debug: bool = False

    @property
    def model_key(self) -> str:
        """Version key stored next to every embedding.

        Vectors produced by a different model or preprocessing version are
        never compared; the key is the single source of that decision. Gain
        preprocessing (off by default) is part of the key precisely because it
        changes the waveform the model sees.
        """
        version = self.embedding_version
        if self.gain_preprocessing:
            version = f"{version}+{GAIN_PREPROCESSING_VERSION}"
        return (
            f"{self.model_id}@{self.model_revision}"
            f"/{version}"
        )

    @property
    def window_bytes(self) -> int:
        return int(self.max_window_seconds * self.sample_rate) * 2


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    return str(env.get(name, default) or "").strip() or default


def _env_float(
    env: Mapping[str, str], name: str, default: float, *, minimum: float
) -> float:
    raw = _env_str(env, name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SpeakerConfigError(f"{name} 必须是数字。") from exc
    if value < minimum:
        raise SpeakerConfigError(f"{name} 必须大于等于 {minimum}。")
    return value


def _env_int(
    env: Mapping[str, str], name: str, default: int, *, minimum: int
) -> int:
    raw = _env_str(env, name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SpeakerConfigError(f"{name} 必须是整数。") from exc
    if value < minimum:
        raise SpeakerConfigError(f"{name} 必须大于等于 {minimum}。")
    return value


def _env_bool(
    env: Mapping[str, str], name: str, default: bool
) -> bool:
    raw = _env_str(env, name, "1" if default else "0").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise SpeakerConfigError(f"{name} 必须是 true/false。")


def _env_float_tolerant(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a V2 tuning value without ever breaking voice input.

    A malformed or out-of-range *new* profile setting falls back to the
    conservative default instead of raising: a typo must not disable the ASR
    demo. Existing (pre-V2) settings keep their strict behaviour.
    """
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, float(minimum)), float(maximum))


def _env_int_tolerant(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return min(max(value, int(minimum)), int(maximum))


def _env_choice(
    env: Mapping[str, str],
    name: str,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    raw = str(env.get(name, "") or "").strip().lower()
    return raw if raw in allowed else default


def load_speaker_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    db_path: Optional[str | Path] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    debug: bool = False,
) -> SpeakerConfig:
    """Build and validate the speaker settings.

    Explicit arguments win over ``AVA_SPEAKER_*`` environment variables, which
    win over the conservative defaults. No credential or audio is read here.
    """
    if env is None:
        load_local_env()
        env = os.environ

    model_id = _env_str(env, "AVA_SPEAKER_MODEL_ID", DEFAULT_MODEL_ID)
    model_revision = _env_str(
        env, "AVA_SPEAKER_MODEL_REVISION", DEFAULT_MODEL_REVISION
    )
    embedding_version = _env_str(
        env, "AVA_SPEAKER_EMBEDDING_VERSION", DEFAULT_EMBEDDING_VERSION
    )

    chosen_model_path = (
        str(model_path).strip()
        if model_path is not None
        else _env_str(env, "AVA_SPEAKER_MODEL_PATH", "")
    )

    chosen_device = (
        str(device).strip()
        if device is not None
        else _env_str(env, "AVA_SPEAKER_DEVICE", "cpu")
    ).lower()
    if chosen_device not in DEVICES:
        raise SpeakerConfigError(
            "AVA_SPEAKER_DEVICE 必须是 " + " / ".join(DEVICES) + " 之一。"
        )

    if db_path is not None:
        chosen_db = Path(db_path).expanduser()
    else:
        chosen_db = Path(
            _env_str(env, "AVA_SPEAKER_DB", str(DEFAULT_DB_PATH))
        ).expanduser()

    first = _env_float(
        env, "AVA_SPEAKER_FIRST_ANALYSIS_SECONDS",
        DEFAULT_FIRST_ANALYSIS_SECONDS, minimum=0.2,
    )
    second = _env_float(
        env, "AVA_SPEAKER_SECOND_ANALYSIS_SECONDS",
        DEFAULT_SECOND_ANALYSIS_SECONDS, minimum=first,
    )
    repeat = _env_float(
        env, "AVA_SPEAKER_REPEAT_ANALYSIS_SECONDS",
        DEFAULT_REPEAT_ANALYSIS_SECONDS, minimum=0.2,
    )
    max_window = _env_float(
        env, "AVA_SPEAKER_MAX_WINDOW_SECONDS",
        DEFAULT_MAX_WINDOW_SECONDS, minimum=first,
    )
    min_analysis = _env_float(
        env, "AVA_SPEAKER_MIN_ANALYSIS_SECONDS",
        DEFAULT_MIN_ANALYSIS_SECONDS, minimum=0.2,
    )
    min_voiced = _env_float(
        env, "AVA_SPEAKER_MIN_VOICED_SECONDS",
        DEFAULT_MIN_VOICED_SECONDS, minimum=0.2,
    )

    known = _env_float(
        env, "AVA_SPEAKER_KNOWN_THRESHOLD",
        DEFAULT_KNOWN_MATCH_THRESHOLD, minimum=0.0,
    )
    candidate = _env_float(
        env, "AVA_SPEAKER_CANDIDATE_THRESHOLD",
        DEFAULT_CANDIDATE_MATCH_THRESHOLD, minimum=0.0,
    )
    new_candidate = _env_float(
        env, "AVA_SPEAKER_NEW_CANDIDATE_THRESHOLD",
        DEFAULT_NEW_CANDIDATE_THRESHOLD, minimum=0.0,
    )
    margin = _env_float(
        env, "AVA_SPEAKER_MINIMUM_MARGIN",
        DEFAULT_MINIMUM_MATCH_MARGIN, minimum=0.0,
    )
    if not 0.0 <= new_candidate <= candidate <= known <= 1.0:
        raise SpeakerConfigError(
            "阈值必须满足 0 <= new_candidate <= candidate <= known <= 1。"
        )

    enrollment_match = _env_float(
        env, "AVA_SPEAKER_ENROLLMENT_THRESHOLD",
        max(DEFAULT_ENROLLMENT_MATCH_THRESHOLD, known), minimum=0.0,
    )
    if enrollment_match < known:
        raise SpeakerConfigError(
            "AVA_SPEAKER_ENROLLMENT_THRESHOLD 不得低于 known 阈值。"
        )
    enrollment_margin = _env_float(
        env, "AVA_SPEAKER_ENROLLMENT_MARGIN",
        DEFAULT_ENROLLMENT_MARGIN, minimum=0.0,
    )
    duplicate_similarity = _env_float(
        env, "AVA_SPEAKER_DUPLICATE_SIMILARITY",
        DEFAULT_DUPLICATE_SAMPLE_SIMILARITY, minimum=0.0,
    )
    max_gain_db = _env_float(
        env, "AVA_SPEAKER_MAX_GAIN_DB",
        DEFAULT_MAX_GAIN_DB, minimum=0.0,
    )
    gain_preprocessing = _env_bool(
        env, "AVA_SPEAKER_GAIN_PREPROCESSING", DEFAULT_GAIN_PREPROCESSING
    )

    enrollment = _env_str(
        env, "AVA_SPEAKER_MINIMUM_ENROLLMENT_QUALITY",
        DEFAULT_MINIMUM_ENROLLMENT_QUALITY,
    ).upper()
    if enrollment not in ("HIGH", "MEDIUM", "LOW"):
        raise SpeakerConfigError(
            "AVA_SPEAKER_MINIMUM_ENROLLMENT_QUALITY 必须是 HIGH / MEDIUM / LOW。"
        )

    max_embeddings = _env_int(
        env, "AVA_SPEAKER_MAX_EMBEDDINGS",
        DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER, minimum=1,
    )
    max_centers = _env_int(
        env, "AVA_SPEAKER_MAX_CENTERS",
        DEFAULT_MAX_CENTERS_PER_SPEAKER, minimum=0,
    )

    profile_mode = _env_choice(
        env, "AVA_SPEAKER_PROFILE_MODE", DEFAULT_PROFILE_MODE, PROFILE_MODES
    )
    if str(env.get("AVA_SPEAKER_MATCH_STRATEGY", "") or "").strip():
        match_strategy = _env_choice(
            env, "AVA_SPEAKER_MATCH_STRATEGY",
            DEFAULT_MATCH_STRATEGY, MATCH_STRATEGIES,
        )
    elif profile_mode == "variants":
        # Turning on variants mode alone must actually use variants; the
        # scoring strategy stays overridable for the A/B experiment.
        match_strategy = "variant_centroid"
    else:
        match_strategy = DEFAULT_MATCH_STRATEGY
    variant_max_count = _env_int_tolerant(
        env, "AVA_SPEAKER_VARIANT_MAX_COUNT",
        DEFAULT_VARIANT_MAX_COUNT, minimum=1, maximum=8,
    )
    variant_merge_threshold = _env_float_tolerant(
        env, "AVA_SPEAKER_VARIANT_MERGE_THRESHOLD",
        DEFAULT_VARIANT_MERGE_THRESHOLD, minimum=0.0, maximum=1.0,
    )
    singleton_penalty = _env_float_tolerant(
        env, "AVA_SPEAKER_SINGLETON_PENALTY",
        DEFAULT_SINGLETON_PENALTY, minimum=0.0, maximum=0.5,
    )
    tracking_enabled = _env_bool(
        env, "AVA_SPEAKER_TRACK_ENABLED", DEFAULT_TRACKING_ENABLED
    )
    track_expiry_seconds = _env_float_tolerant(
        env, "AVA_SPEAKER_TRACK_EXPIRY_SECONDS",
        DEFAULT_TRACK_EXPIRY_SECONDS, minimum=1.0, maximum=600.0,
    )
    track_recent_turns = _env_int_tolerant(
        env, "AVA_SPEAKER_TRACK_RECENT_TURNS",
        DEFAULT_TRACK_RECENT_TURNS, minimum=1, maximum=20,
    )
    new_id_min_turns = _env_int_tolerant(
        env, "AVA_SPEAKER_NEW_ID_MIN_TURNS",
        DEFAULT_NEW_ID_MIN_TURNS, minimum=1, maximum=10,
    )
    new_id_min_voiced_seconds = _env_float_tolerant(
        env, "AVA_SPEAKER_NEW_ID_MIN_VOICED_SECONDS",
        DEFAULT_NEW_ID_MIN_VOICED_SECONDS, minimum=0.0, maximum=60.0,
    )
    pending_max_count = _env_int_tolerant(
        env, "AVA_SPEAKER_PENDING_MAX_COUNT",
        DEFAULT_PENDING_MAX_COUNT, minimum=1, maximum=64,
    )
    auto_learn_enabled = _env_bool(
        env, "AVA_SPEAKER_AUTO_LEARN_ENABLED", DEFAULT_AUTO_LEARN_ENABLED
    )

    return SpeakerConfig(
        db_path=chosen_db,
        model_id=model_id,
        model_revision=model_revision,
        model_path=chosen_model_path,
        embedding_version=embedding_version,
        device=chosen_device,
        first_analysis_seconds=first,
        second_analysis_seconds=second,
        repeat_analysis_seconds=repeat,
        max_window_seconds=max_window,
        min_analysis_seconds=min_analysis,
        min_voiced_seconds=min_voiced,
        known_match_threshold=known,
        candidate_match_threshold=candidate,
        new_candidate_threshold=new_candidate,
        minimum_match_margin=margin,
        minimum_enrollment_quality=enrollment,
        enrollment_match_threshold=enrollment_match,
        enrollment_margin=enrollment_margin,
        duplicate_sample_similarity=duplicate_similarity,
        gain_preprocessing=gain_preprocessing,
        max_gain_db=max_gain_db,
        max_embeddings_per_speaker=max_embeddings,
        max_centers_per_speaker=max_centers,
        profile_mode=profile_mode,
        match_strategy=match_strategy,
        variant_max_count=variant_max_count,
        variant_merge_threshold=variant_merge_threshold,
        singleton_penalty=singleton_penalty,
        tracking_enabled=tracking_enabled,
        track_expiry_seconds=track_expiry_seconds,
        track_recent_turns=track_recent_turns,
        new_id_min_turns=new_id_min_turns,
        new_id_min_voiced_seconds=new_id_min_voiced_seconds,
        pending_max_count=pending_max_count,
        auto_learn_enabled=auto_learn_enabled,
        debug=debug,
    )


__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_DUPLICATE_SAMPLE_SIMILARITY",
    "DEFAULT_EMBEDDING_VERSION",
    "DEFAULT_ENROLLMENT_MARGIN",
    "DEFAULT_ENROLLMENT_MATCH_THRESHOLD",
    "DEFAULT_GAIN_PREPROCESSING",
    "DEFAULT_MAX_GAIN_DB",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEVICES",
    "EMBEDDING_DIM",
    "GAIN_PREPROCESSING_VERSION",
    "SAMPLE_RATE",
    "SpeakerConfig",
    "SpeakerConfigError",
    "load_speaker_config",
]
