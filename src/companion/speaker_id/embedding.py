"""Adapter for the 3D-Speaker ERes2NetV2 speaker-verification model.

The model *implementation* comes from the installed ModelScope package (the
same code as ``speakerlab/models/eres2net/ERes2NetV2.py`` in the 3D-Speaker
repository). Nothing is fetched from GitHub at runtime: a pinned package
version supplies the architecture and the pinned model revision supplies the
weights, which ModelScope caches locally after the first download.

Only this module needs PyTorch/ModelScope; it is imported lazily by the demo.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from companion.speaker_id.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    EMBEDDING_DIM,
    SAMPLE_RATE,
)

MIN_EMBEDDING_NORM = 1e-6

# The published model repository ships a configuration.json with only
# ``sample_rate`` and ``embed_dim``; ModelScope's ERes2NetV2 class additionally
# requires the architecture keys below. They are fixed by the checkpoint, so
# pinning them here is safer than trusting whatever a cache happens to hold.
# Sizes that do not match the checkpoint would fail ``load_state_dict`` anyway.
MODEL_ARCHITECTURE = {
    "sample_rate": SAMPLE_RATE,
    "embed_dim": EMBEDDING_DIM,
    "baseWidth": 26,
    "scale": 2,
    "expansion": 2,
}


class SpeakerEmbeddingError(Exception):
    """Model dependency, loading or inference failure (display-safe)."""


def pcm_to_float32(pcm: bytes, sample_rate: int = SAMPLE_RATE):
    """Convert mono 16-bit little-endian PCM to ``[-1, 1)`` float32.

    A trailing odd byte is ignored rather than interpreted as a sample.
    """
    import numpy as np

    usable = len(pcm) - (len(pcm) % 2)
    samples = np.frombuffer(pcm[:usable], dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def validate_embedding(vector: Any, dim: int = EMBEDDING_DIM):
    """L2-normalize and validate an embedding before it is used or stored."""
    import numpy as np

    try:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise SpeakerEmbeddingError("声纹向量格式无效。") from exc
    if array.shape != (dim,):
        raise SpeakerEmbeddingError(
            f"声纹向量维度异常：期望 {dim}，得到 {array.shape[0]}。"
        )
    if not np.all(np.isfinite(array)):
        raise SpeakerEmbeddingError("声纹向量包含 NaN 或 Inf。")
    norm = float(np.linalg.norm(array))
    if norm < MIN_EMBEDDING_NORM:
        raise SpeakerEmbeddingError("声纹向量为零向量。")
    return (array / norm).astype(np.float32)


class SpeakerEmbeddingModel:
    """One long-lived model instance; ``load``/``warmup`` run exactly once."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str = DEFAULT_MODEL_REVISION,
        model_path: str = "",
        device: str = "cpu",
        sample_rate: int = SAMPLE_RATE,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_path = model_path
        self.device = (device or "cpu").lower()
        self.sample_rate = sample_rate
        self.dim = dim
        self._pipeline: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------ state
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def active_device(self) -> str:
        return self.device

    # ------------------------------------------------------------------- load
    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # noqa: F401 - optional dependency guard
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SpeakerEmbeddingError(
                "PyTorch 未安装：请执行 pip install -e '.[speaker]'"
            ) from exc
        try:
            import torchaudio  # noqa: F401 - required by the model's FBank
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SpeakerEmbeddingError(
                "torchaudio 未安装：请执行 pip install -e '.[speaker]'"
            ) from exc
        try:
            from modelscope.pipelines import pipeline
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SpeakerEmbeddingError(
                "ModelScope 未安装：请执行 pip install -e '.[speaker]'"
            ) from exc
        self._torch = torch

        source = self.model_path or self.model_id
        kwargs: dict[str, Any] = {
            "task": "speaker-verification",
            "model": source,
            "model_config": dict(MODEL_ARCHITECTURE),
        }
        if not self.model_path:
            kwargs["model_revision"] = self.model_revision

        try:
            self._pipeline = self._create_pipeline(pipeline, kwargs)
        except Exception as exc:  # noqa: BLE001 - normalized for display
            raise SpeakerEmbeddingError(
                "声纹模型加载失败（" + type(exc).__name__ + "）。"
                "请确认已安装 speaker 依赖且模型可访问。"
            ) from exc
        self._loaded = True

    def _create_pipeline(self, pipeline_factory, kwargs: dict) -> Any:
        requested = self.device
        if requested == "auto":
            requested = "mps" if self._mps_available() else "cpu"
        if requested == "cpu":
            return pipeline_factory(**kwargs)
        try:
            return pipeline_factory(device=requested, **kwargs)
        except Exception:  # noqa: BLE001 - accelerate backend may be missing
            # A CPU fallback is mandatory: MPS support depends on the model's
            # operators, and an unsupported device must never stop the demo.
            self.device = "cpu"
            return pipeline_factory(device="cpu", **kwargs)

    def _mps_available(self) -> bool:
        try:  # pragma: no cover - depends on the host
            return bool(self._torch is not None and self._torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            return False

    def warmup(self) -> None:
        """Run one inference so the first real utterance is not the cold path."""
        if not self._loaded:
            raise SpeakerEmbeddingError("声纹模型尚未加载。")
        import numpy as np

        rng = np.random.default_rng(0)
        samples = (rng.normal(0.0, 0.01, self.sample_rate // 2) * 32767)
        pcm = np.clip(samples, -32768, 32767).astype("<i2").tobytes()
        try:
            self.encode(pcm)
        except SpeakerEmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SpeakerEmbeddingError(
                "声纹模型预热失败（" + type(exc).__name__ + "）。"
            ) from exc

    # ----------------------------------------------------------------- encode
    def encode(self, pcm: bytes):
        """Encode one mono 16-bit PCM buffer into a normalized 192-d vector."""
        if not self._loaded or self._pipeline is None:
            raise SpeakerEmbeddingError("声纹模型尚未加载。")
        import numpy as np

        audio = pcm_to_float32(pcm, self.sample_rate)
        if audio.size == 0:
            raise SpeakerEmbeddingError("没有可用于声纹识别的音频。")
        with self._lock:
            try:
                with self._torch.inference_mode():
                    result = self._pipeline([audio], output_emb=True)
            except Exception as exc:  # noqa: BLE001 - sanitized
                raise SpeakerEmbeddingError(
                    "声纹推理失败（" + type(exc).__name__ + "）。"
                ) from exc
        vector = self._extract_embedding(result, np)
        return validate_embedding(vector, self.dim)

    @staticmethod
    def _extract_embedding(result: Any, np):
        embeddings = None
        if isinstance(result, dict):
            embeddings = result.get("embs")
            if embeddings is None:
                embeddings = result.get("embeddings")
        if embeddings is None:
            raise SpeakerEmbeddingError("声纹模型没有返回 embedding。")
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        try:
            array = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise SpeakerEmbeddingError("声纹模型返回的 embedding 无法解析。") from exc
        if array.ndim == 0:
            raise SpeakerEmbeddingError("声纹模型返回的 embedding 为空。")
        first = array.reshape(-1, array.shape[-1])[0]
        return first

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        self._pipeline = None
        self._loaded = False


__all__ = [
    "MIN_EMBEDDING_NORM",
    "MODEL_ARCHITECTURE",
    "SpeakerEmbeddingError",
    "SpeakerEmbeddingModel",
    "pcm_to_float32",
    "validate_embedding",
]
