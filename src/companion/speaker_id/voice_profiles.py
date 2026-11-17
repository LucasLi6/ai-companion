"""Voice-profile mathematics: samples -> variants -> person-level scores.

This module is deliberately pure NumPy. It never touches a microphone, ASR,
the model, TTS, SQLite or Ava's memory: the engine hands it already-normalized
embeddings and gets numbers back. That keeps the V2 profile logic testable
offline and keeps the real-time worker's extra work to a few dot products.

Design notes
------------
* A *sample* is one stored utterance embedding. A *variant* is a group of
  samples that are close enough to summarise as one centroid. A *profile*
  belongs to one person and owns both.
* Clustering is deterministic: samples are ordered by ``(created_at,
  sample_id)`` and pairs of clusters are merged by the highest centroid
  cosine, with the lowest cluster indexes breaking ties. The same samples
  always produce the same variants, on every run and every platform.
* Two scoring strategies are supported and must never be silently mixed:
  ``legacy_max_sample`` reproduces the pre-V2 max-over-samples score, while
  ``variant_centroid`` averages the samples of each cluster before comparing.
  Averaging suppresses an isolated outlier sample instead of letting it
  dominate every future match.
* A variant backed by a single sample can optionally pay a small score
  penalty. The default penalty is zero (no behaviour change); it exists so an
  evaluation can measure the effect instead of assuming it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

STRATEGY_LEGACY_MAX_SAMPLE = "legacy_max_sample"
STRATEGY_VARIANT_CENTROID = "variant_centroid"
STRATEGIES = (STRATEGY_LEGACY_MAX_SAMPLE, STRATEGY_VARIANT_CENTROID)

ALGORITHM_VERSION = "voice-profile-v2"

ORIGIN_LEGACY = "legacy"
ORIGIN_BOOTSTRAP = "bootstrap"
ORIGIN_AUTO = "auto"
ORIGIN_CONFIRMED = "confirmed"
ORIGINS = (
    ORIGIN_LEGACY,
    ORIGIN_BOOTSTRAP,
    ORIGIN_AUTO,
    ORIGIN_CONFIRMED,
)

# A variant centroid is the L2-normalized mean of its samples. A cluster with
# one sample simply keeps that sample. Both are unit vectors.
_MIN_NORM = 1e-6


class VoiceProfileError(ValueError):
    """Invalid embedding or profile geometry (display-safe)."""


@dataclass(frozen=True, eq=False)
class VoiceSample:
    """One stored embedding and the metadata that justifies keeping it."""

    sample_id: int
    vector: np.ndarray
    created_at: float = 0.0
    quality: str = "MEDIUM"
    origin: str = ORIGIN_LEGACY
    quarantined: bool = False

    @property
    def usable(self) -> bool:
        return not self.quarantined


@dataclass(frozen=True, eq=False)
class VoiceVariant:
    """One acoustic cluster of samples, represented by its centroid."""

    index: int
    centroid: np.ndarray
    sample_ids: tuple[int, ...]
    support: int
    created_at: float

    @property
    def is_singleton(self) -> bool:
        return self.support <= 1


@dataclass(frozen=True, eq=False)
class VoiceProfile:
    """A person's samples plus the deterministic variant structure."""

    speaker_id: str
    samples: tuple[VoiceSample, ...]
    variants: tuple[VoiceVariant, ...] = ()
    status: str = "CANDIDATE"
    algorithm_version: str = ALGORITHM_VERSION

    @property
    def vectors(self) -> tuple[np.ndarray, ...]:
        """Legacy view: every usable sample vector, in deterministic order."""
        return tuple(sample.vector for sample in self.samples if sample.usable)


@dataclass(frozen=True)
class ProfileScore:
    """All evidence for one query against one person."""

    speaker_id: str
    raw_max_similarity: float
    variant_max_similarity: float
    effective_similarity: float
    best_variant_index: Optional[int]
    best_sample_id: Optional[int]
    best_variant_support: int
    singleton_penalty_applied: float
    strategy: str


def normalize_embedding(
    vector, expected_dim: Optional[int] = None
) -> np.ndarray:
    """Validate one embedding and return a fresh L2-normalized float32 copy.

    Rejecting bad vectors here (rather than deep inside a dot product) keeps
    NaN/Inf/zero vectors out of the profile and out of SQLite.
    """
    try:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise VoiceProfileError("声纹向量格式无效。") from exc
    if expected_dim is not None and array.shape != (int(expected_dim),):
        raise VoiceProfileError(
            f"声纹向量维度异常：期望 {int(expected_dim)}，"
            f"得到 {array.shape[0]}。"
        )
    if array.size == 0:
        raise VoiceProfileError("声纹向量为空。")
    if not np.all(np.isfinite(array)):
        raise VoiceProfileError("声纹向量包含 NaN 或 Inf。")
    norm = float(np.linalg.norm(array))
    if norm < _MIN_NORM:
        raise VoiceProfileError("声纹向量为零向量。")
    return (array / norm).astype(np.float32)


def cosine(left, right) -> float:
    """Cosine of two unit vectors, clamped for numerical noise."""
    value = float(np.asarray(left, dtype=np.float32) @ np.asarray(
        right, dtype=np.float32
    ))
    return max(-1.0, min(1.0, value))


def _centroid(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise VoiceProfileError("空聚类无法计算质心。")
    mean = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < _MIN_NORM:
        # Opposing vectors cancelled out. Falling back to the first vector is
        # deterministic and keeps a usable unit vector; this cannot happen for
        # samples that passed the enrollment similarity gates.
        return np.asarray(vectors[0], dtype=np.float32).copy()
    return (mean / norm).astype(np.float32)


def build_variants(
    samples: Iterable[VoiceSample],
    *,
    merge_threshold: float = 0.55,
    max_variants: int = 4,
) -> tuple[VoiceVariant, ...]:
    """Group samples into at most ``max_variants`` deterministic clusters.

    Agglomerative, not order-of-arrival based: every pair of clusters is
    considered and the most similar pair is merged first. Merging stops when
    no pair reaches ``merge_threshold`` *and* the cluster count is within
    ``max_variants``; if too many clusters remain, the closest pair is merged
    anyway so the bound is always honoured.
    """
    ordered = sorted(
        (sample for sample in samples if sample.usable),
        key=lambda sample: (float(sample.created_at), int(sample.sample_id)),
    )
    if not ordered:
        return ()
    limit = max(1, int(max_variants))
    threshold = float(merge_threshold)
    clusters: list[list[VoiceSample]] = [[sample] for sample in ordered]
    centroids: list[np.ndarray] = [
        normalize_embedding(sample.vector) for sample in ordered
    ]

    while len(clusters) > 1:
        best_similarity = -2.0
        best_pair: Optional[tuple[int, int]] = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                similarity = cosine(centroids[left], centroids[right])
                if (
                    best_pair is None
                    or similarity > best_similarity + 1e-12
                    or (
                        abs(similarity - best_similarity) <= 1e-12
                        and (left, right) < best_pair
                    )
                ):
                    best_similarity = similarity
                    best_pair = (left, right)
        if best_pair is None:
            break
        if best_similarity < threshold and len(clusters) <= limit:
            break
        left, right = best_pair
        clusters[left].extend(clusters.pop(right))
        centroids[left] = _centroid(
            [normalize_embedding(sample.vector) for sample in clusters[left]]
        )
        centroids.pop(right)

    variants: list[VoiceVariant] = []
    for cluster in clusters:
        sample_ids = tuple(sorted(int(sample.sample_id) for sample in cluster))
        variants.append(
            VoiceVariant(
                index=-1,
                centroid=_centroid(
                    [normalize_embedding(sample.vector) for sample in cluster]
                ),
                sample_ids=sample_ids,
                support=len(sample_ids),
                created_at=min(float(sample.created_at) for sample in cluster),
            )
        )
    variants.sort(
        key=lambda variant: (variant.created_at, variant.sample_ids[0])
    )
    return tuple(
        VoiceVariant(
            index=index,
            centroid=variant.centroid,
            sample_ids=variant.sample_ids,
            support=variant.support,
            created_at=variant.created_at,
        )
        for index, variant in enumerate(variants)
    )


def make_profile(
    speaker_id: str,
    samples: Iterable[VoiceSample],
    *,
    status: str = "CANDIDATE",
    merge_threshold: float = 0.55,
    max_variants: int = 4,
) -> VoiceProfile:
    """Build a profile from raw samples (normalizing and filtering as needed)."""
    cleaned: list[VoiceSample] = []
    for sample in samples:
        if not sample.usable:
            continue
        try:
            vector = normalize_embedding(sample.vector)
        except VoiceProfileError:
            continue
        cleaned.append(
            VoiceSample(
                sample_id=int(sample.sample_id),
                vector=vector,
                created_at=float(sample.created_at),
                quality=str(sample.quality),
                origin=str(sample.origin),
            )
        )
    cleaned.sort(key=lambda sample: (sample.created_at, sample.sample_id))
    return VoiceProfile(
        speaker_id=str(speaker_id),
        samples=tuple(cleaned),
        variants=build_variants(
            cleaned,
            merge_threshold=merge_threshold,
            max_variants=max_variants,
        ),
        status=str(status),
    )


def score_profile(
    query_vector,
    profile: VoiceProfile,
    *,
    strategy: str = STRATEGY_VARIANT_CENTROID,
    singleton_penalty: float = 0.0,
) -> ProfileScore:
    """Score one query against one profile under the chosen strategy.

    ``raw_max_similarity`` is always the legacy max-over-samples score so a
    shadow evaluation can compare strategies without recomputing embeddings.
    ``effective_similarity`` is what the matcher uses: raw max for
    ``legacy_max_sample``, penalised variant maximum for
    ``variant_centroid``.
    """
    if strategy not in STRATEGIES:
        raise VoiceProfileError(f"未知的声纹匹配策略：{strategy}")

    query = normalize_embedding(query_vector)
    raw_max = -1.0
    best_sample_id: Optional[int] = None
    for sample in profile.samples:
        if not sample.usable:
            continue
        value = cosine(query, sample.vector)
        if value > raw_max or best_sample_id is None:
            raw_max = value
            best_sample_id = sample.sample_id
    if best_sample_id is None:
        raw_max = -1.0

    variant_max = -1.0
    best_variant_index: Optional[int] = None
    best_variant_support = 0
    penalty_applied = 0.0
    penalty = max(0.0, float(singleton_penalty))
    for variant in profile.variants:
        value = cosine(query, variant.centroid)
        applied = penalty if variant.is_singleton else 0.0
        adjusted = value - applied
        if best_variant_index is None or adjusted > variant_max:
            variant_max = adjusted
            best_variant_index = variant.index
            best_variant_support = variant.support
            penalty_applied = applied
    if best_variant_index is None:
        variant_max = -1.0

    effective = (
        raw_max if strategy == STRATEGY_LEGACY_MAX_SAMPLE else variant_max
    )
    return ProfileScore(
        speaker_id=profile.speaker_id,
        raw_max_similarity=float(raw_max),
        variant_max_similarity=float(variant_max),
        effective_similarity=float(effective),
        best_variant_index=best_variant_index,
        best_sample_id=best_sample_id,
        best_variant_support=int(best_variant_support),
        singleton_penalty_applied=float(penalty_applied),
        strategy=strategy,
    )


class VoiceProfileIndex:
    """Bounded, invalidatable cache from ``(speaker, samples)`` to profile.

    The engine reloads candidates on every analysis (~1 Hz). Rebuilding the
    variant structure each time would add avoidable work, so the profile is
    cached by a fingerprint of the sample metadata. A new sample, a removal or
    a quality change changes the fingerprint; explicit invalidation is also
    supported for store-wide changes (clear/merge/delete).
    """

    def __init__(
        self,
        *,
        merge_threshold: float = 0.55,
        max_variants: int = 4,
        max_entries: int = 256,
    ) -> None:
        self.merge_threshold = float(merge_threshold)
        self.max_variants = max(1, int(max_variants))
        self.max_entries = max(1, int(max_entries))
        self._cache: "OrderedDict[tuple, VoiceProfile]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _fingerprint(
        speaker_id: str, samples: Sequence[VoiceSample], status: str
    ) -> tuple:
        return (
            str(speaker_id),
            str(status),
            tuple(
                (
                    int(sample.sample_id),
                    float(sample.created_at),
                    str(sample.quality),
                    str(sample.origin),
                    bool(sample.quarantined),
                )
                for sample in samples
            ),
        )

    def profile_for(
        self,
        speaker_id: str,
        samples: Iterable[VoiceSample],
        *,
        status: str = "CANDIDATE",
    ) -> VoiceProfile:
        sample_tuple = tuple(samples)
        key = self._fingerprint(speaker_id, sample_tuple, status)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return cached
        profile = make_profile(
            speaker_id,
            sample_tuple,
            status=status,
            merge_threshold=self.merge_threshold,
            max_variants=self.max_variants,
        )
        with self._lock:
            self._cache[key] = profile
            self._cache.move_to_end(key)
            self.misses += 1
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
                self.evictions += 1
        return profile

    def invalidate(self, speaker_id: Optional[str] = None) -> None:
        with self._lock:
            if speaker_id is None:
                self._cache.clear()
                return
            for key in [
                key for key in self._cache if key[0] == str(speaker_id)
            ]:
                self._cache.pop(key, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "algorithm_version": ALGORITHM_VERSION,
            }


__all__ = [
    "ALGORITHM_VERSION",
    "ORIGINS",
    "ORIGIN_AUTO",
    "ORIGIN_BOOTSTRAP",
    "ORIGIN_CONFIRMED",
    "ORIGIN_LEGACY",
    "ProfileScore",
    "STRATEGIES",
    "STRATEGY_LEGACY_MAX_SAMPLE",
    "STRATEGY_VARIANT_CENTROID",
    "VoiceProfile",
    "VoiceProfileError",
    "VoiceProfileIndex",
    "VoiceSample",
    "VoiceVariant",
    "build_variants",
    "cosine",
    "make_profile",
    "normalize_embedding",
    "score_profile",
]
