"""Embedding providers and pure-Python vector primitives.

V0.3 hybrid retrieval stores one float32 vector per memory per model, all in
the same SQLite file. This module deliberately avoids any native dependency
(numpy is *not* required): vector arithmetic uses ``math`` and ``struct`` so
the default install and the offline test suite work unchanged.

A real ``sentence-transformers`` provider is available lazily — it is imported
only inside its constructor, never at module load, and never downloads weights
on its own. The deterministic fake provider is for tests and offline demos
only; production embedding is opt-in and never silently faked.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class VectorValidationError(ValueError):
    """Raised when a vector is the wrong length or contains non-finite values."""


class EmbeddingBackendUnavailable(RuntimeError):
    """Raised when a real embedding backend is requested but not installed."""


# -- vector primitives -------------------------------------------------------


def pack_vector(vector: list[float]) -> bytes:
    """Pack a float32 vector into a validated little-endian BLOB."""
    return struct.pack(f"<{len(vector)}f", *[float(x) for x in vector])


def unpack_vector(blob: bytes) -> list[float]:
    """Unpack a float32 BLOB back into a list of floats."""
    if len(blob) % 4 != 0:
        raise VectorValidationError("vector BLOB length is not a multiple of 4")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def validate_vector(vector: list[float], dimensions: int) -> list[float]:
    """Validate a vector against an expected dimension and finiteness."""
    values = [float(x) for x in vector]
    if len(values) != dimensions:
        raise VectorValidationError(
            f"vector dimension {len(values)} != expected {dimensions}"
        )
    if not all(math.isfinite(x) for x in values):
        raise VectorValidationError("vector contains non-finite values")
    return values


def normalize_l2(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [x / norm for x in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity without numpy. Returns 0.0 for an empty vector."""
    if len(a) != len(b):
        raise VectorValidationError("cannot compare vectors of different lengths")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# -- model identity ----------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """The static, hashable configuration that identifies one embedding model.

    ``model_id`` is deterministic in this configuration, so a backfill run in a
    fresh process lands on the same model row as the one that indexed the
    memories — no auto-generated id that would drift between restarts.
    """

    provider: str
    model_name: str
    dimensions: int
    model_revision: str | None = None
    normalization: str = "l2"  # "l2" | "none"
    query_prefix: str = ""
    document_prefix: str = ""
    distance_metric: str = "cosine"  # "cosine" | "l2" | "dot"
    max_input_tokens: int | None = None

    def configuration_hash(self) -> str:
        payload = {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "normalization": self.normalization,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "distance_metric": self.distance_metric,
            "max_input_tokens": self.max_input_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def model_id(self) -> str:
        return "emb_" + self.configuration_hash()[:24]

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "normalization": self.normalization,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "distance_metric": self.distance_metric,
            "max_input_tokens": self.max_input_tokens,
            "configuration_hash": self.configuration_hash(),
        }


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A pluggable text -> float32 vector provider.

    ``model_id`` and ``dimensions`` are static and deterministic; ``spec``
    carries the full configuration for the ``embedding_models`` table.
    """

    model_id: str
    dimensions: int
    spec: EmbeddingModelSpec

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


# -- deterministic fake provider (tests / offline) ---------------------------


_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _tokenize(text: str) -> list[str]:
    """Tokenize for the deterministic provider: word tokens plus CJK bigrams."""
    if not text:
        return []
    lowered = text.lower()
    tokens = [t for t in re.split(r"[^\w]+", lowered, flags=re.UNICODE) if t]
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    # CJK bigram fallback so "机器学习" and "机器 学习" still overlap.
    for token in tokens:
        if _CJK_RE.search(token) and 2 <= len(token) <= 32:
            for i in range(len(token) - 1):
                bigram = token[i : i + 2]
                if bigram not in seen:
                    seen.add(bigram)
                    out.append(bigram)
    return out


class FakeEmbeddingProvider:
    """Deterministic, offline bag-of-token-hashing embeddings.

    Shared tokens (including CJK bigrams) produce overlapping non-zero buckets,
    so two paraphrases with common content get a positive cosine similarity.
    This is enough to exercise the hybrid pipeline and RRF plumbing in tests —
    it is *not* a semantic model and must never be presented as one.
    """

    def __init__(self, dimensions: int = 64, seed: int = 0) -> None:
        self.dimensions = dimensions
        self.seed = seed
        self.spec = EmbeddingModelSpec(
            provider="fake",
            model_name="fake-deterministic",
            model_revision="v1",
            dimensions=dimensions,
            normalization="l2",
            query_prefix="query: ",
            document_prefix="passage: ",
            distance_metric="cosine",
            max_input_tokens=512,
        )
        self.model_id = self.spec.model_id()

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(f"{self.seed}:{token}".encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "little")
            idx = h % self.dimensions
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vector[idx] += sign
        return normalize_l2(vector)

    def embed_query(self, text: str) -> list[float]:
        return self._embed((self.spec.query_prefix + text).strip())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed((self.spec.document_prefix + t).strip()) for t in texts]


# -- real local provider (optional ``[embedding]`` extra) --------------------


class LocalSentenceTransformerProvider:
    """Lazy sentence-transformers adapter for a local, real embedding model.

    Constructing it requires the ``[embedding]`` extra. It is the *only* real
    provider: it loads weights from the local cache (or downloads them once),
    so it is explicitly opt-in and never runs in CI or the default suite.
    """

    def __init__(
        self,
        *,
        model_name: str = "intfloat/multilingual-e5-small",
        model_revision: str | None = None,
        dimensions: int = 384,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.dimensions = dimensions
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - requires optional extra
            raise EmbeddingBackendUnavailable(
                "the [embedding] extra is not installed; run "
                "`pip install -e '.[embedding]'` to enable local embeddings"
            ) from exc

        self._model = SentenceTransformer(model_name, revision=model_revision)
        self.spec = EmbeddingModelSpec(
            provider="sentence-transformers",
            model_name=model_name,
            model_revision=model_revision,
            dimensions=dimensions,
            normalization="l2",
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            distance_metric="cosine",
            max_input_tokens=512,
        )
        self.model_id = self.spec.model_id()

    def embed_query(self, text: str) -> list[float]:
        return self._normalize(self._model.encode(self.spec.query_prefix + text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            self._normalize(self._model.encode(self.spec.document_prefix + t))
            for t in texts
        ]

    def _normalize(self, vector) -> list[float]:
        values = [float(x) for x in vector]
        if len(values) != self.dimensions:
            raise VectorValidationError(
                f"model produced {len(values)} dims, expected {self.dimensions}"
            )
        return normalize_l2(values)
