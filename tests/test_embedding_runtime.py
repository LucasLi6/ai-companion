"""V0.3.1 embedding enablement: settings, lazy provider, hybrid integration."""

from __future__ import annotations

import pytest

from companion.cli import _build_foreground_runtime
from companion.domain.enums import MemoryKind
from companion.memory.embeddings import (
    EmbeddingBackendUnavailable,
    FakeEmbeddingProvider,
    LocalSentenceTransformerProvider,
    SETTING_EMBEDDING_BACKEND,
    SETTING_EMBEDDING_ENABLED,
    SETTING_EMBEDDING_MODEL,
    build_embedding_provider_from_settings,
)
from companion.memory.embedding_repository import EmbeddingRepository
from companion.memory.vector_index import VectorIndex
from companion.memory.models import AccessContext
from companion.providers.mock import MockCognitiveProvider
from companion.runtime import build_runtime

ADMIN = AccessContext("admin", is_admin=True)


async def test_embedding_disabled_by_default(rt):
    assert await build_embedding_provider_from_settings(rt.repos.settings) is None
    runtime = build_runtime(rt.db)
    assert runtime.embedding_service is None
    assert runtime.hybrid_retriever is None


async def test_embedding_backend_requires_explicit_enable(rt):
    await rt.repos.settings.set(SETTING_EMBEDDING_BACKEND, "fake")
    # Backend alone is not enough: the master switch stays off.
    assert await build_embedding_provider_from_settings(rt.repos.settings) is None

    await rt.repos.settings.set(SETTING_EMBEDDING_ENABLED, "true")
    provider = await build_embedding_provider_from_settings(rt.repos.settings)
    assert isinstance(provider, FakeEmbeddingProvider)


async def test_unknown_embedding_backend_is_an_error(rt):
    await rt.repos.settings.set(SETTING_EMBEDDING_ENABLED, "true")
    await rt.repos.settings.set(SETTING_EMBEDDING_BACKEND, "magic")
    with pytest.raises(EmbeddingBackendUnavailable):
        await build_embedding_provider_from_settings(rt.repos.settings)


async def test_local_provider_is_lazy_and_uses_configured_model(rt):
    # Constructing it must not import sentence-transformers or load weights.
    provider = LocalSentenceTransformerProvider(model_name="some/model")
    assert provider._model is None
    assert provider.spec.model_name == "some/model"

    await rt.repos.settings.set(SETTING_EMBEDDING_ENABLED, "true")
    await rt.repos.settings.set(SETTING_EMBEDDING_BACKEND, "local")
    await rt.repos.settings.set(SETTING_EMBEDDING_MODEL, "some/other-model")
    built = await build_embedding_provider_from_settings(rt.repos.settings)
    assert built is not None
    assert built.spec.model_name == "some/other-model"
    assert built._model is None


async def test_cli_foreground_runtime_honours_settings(rt):
    # Default: disabled, no retriever.
    runtime = await _build_foreground_runtime(rt.db, "mock")
    assert runtime is not None and runtime.hybrid_retriever is None

    await rt.repos.settings.set(SETTING_EMBEDDING_ENABLED, "true")
    await rt.repos.settings.set(SETTING_EMBEDDING_BACKEND, "fake")
    runtime = await _build_foreground_runtime(rt.db, "mock")
    assert runtime is not None
    assert runtime.embedding_service is not None
    assert runtime.hybrid_retriever is not None

    # A new process (fresh runtime over the same file) keeps the setting.
    again = await _build_foreground_runtime(rt.db, "mock")
    assert again is not None and again.hybrid_retriever is not None


class _SemanticOnlyProvider(FakeEmbeddingProvider):
    """Vectors that relate the query to the memory with no token overlap."""

    def _vec(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        if any(word in text for word in ("咖啡", "提神", "清醒")):
            vector[0] = 1.0
        else:
            vector[1] = 1.0
        return vector

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class _CaptureCognitive:
    def __init__(self):
        self.memories: list[dict] | None = None
        self._inner = MockCognitiveProvider()

    async def analyze(self, event, context, memories):
        self.memories = list(memories)
        return await self._inner.analyze(event, context, memories)


async def test_say_reaches_semantically_matched_memory(rt, register, say):
    person = await register("Alice")
    runtime = build_runtime(rt.db, embedding_provider=_SemanticOnlyProvider())
    assert runtime.hybrid_retriever is not None

    item = await runtime.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="我每天早上喝咖啡",
        access=ADMIN,
        subjects=[("PERSON", person.person_id)],
    )
    processed = await runtime.worker.run_until_idle()
    assert processed >= 1

    spy = _CaptureCognitive()
    runtime.orchestrator.cognitive = spy

    from companion.domain.enums import EventSource
    from companion.domain.models import make_utterance_event

    event = make_utterance_event(
        conversation_id="semantic",
        text="什么东西能让人清醒提神？",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await runtime.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    assert spy.memories is not None
    reached = {m["memory_id"] for m in spy.memories}
    assert item.memory_id in reached
    matched = next(m for m in spy.memories if m["memory_id"] == item.memory_id)
    assert matched["vector_rank"] is not None


async def test_backend_status_is_honest(rt):
    await rt.repos.settings.set(SETTING_EMBEDDING_ENABLED, "true")
    await rt.repos.settings.set(SETTING_EMBEDDING_BACKEND, "fake")
    provider = await build_embedding_provider_from_settings(rt.repos.settings)
    runtime = build_runtime(rt.db, embedding_provider=provider)
    status = await runtime.embedding_service.status()
    for key in (
        "sqlite_vec_installed",
        "sqlite_vec_loadable",
        "sqlite_vec_enabled",
        "active_backend",
    ):
        assert key in status
    assert status["sqlite_vec_enabled"] is False
    assert status["active_backend"] == "python"
    direct = VectorIndex(EmbeddingRepository(rt.db)).backend_status()
    assert direct["sqlite_vec_enabled"] is False
    assert direct["active_backend"] == "python"
