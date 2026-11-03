"""V0.3 vector memory retrieval: embeddings, idempotent lifecycle, hybrid RRF.

Everything here runs offline: the deterministic fake provider and the
pure-Python vector primitives (no numpy, no sentence-transformers, no network).
The real ``intfloat/multilingual-e5-small`` provider is exercised only when the
optional ``[embedding]`` extra is installed, which is out of scope for this
suite.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from companion.domain.enums import MemoryKind, MemoryVisibility
from companion.memory.embeddings import (
    EmbeddingModelSpec,
    FakeEmbeddingProvider,
    VectorValidationError,
    cosine_similarity,
    dot_product,
    normalize_l2,
    pack_vector,
    unpack_vector,
    validate_vector,
)
from companion.memory.embedding_repository import EmbeddingRepository
from companion.memory.embedding_service import EmbeddingService, content_hash
from companion.memory.hybrid_retriever import HybridRetriever
from companion.memory.models import AccessContext
from companion.runtime import build_runtime

ADMIN = AccessContext("admin", is_admin=True)


@pytest_asyncio.fixture
async def emb(db):
    """An EmbeddingService over the shared test db with the fake provider."""
    return EmbeddingService(EmbeddingRepository(db), FakeEmbeddingProvider())


@pytest_asyncio.fixture
async def rt_embed(db):
    """A runtime with the fake embedding provider wired in (hybrid retrieval)."""
    return build_runtime(db, embedding_provider=FakeEmbeddingProvider())


# -- vector primitives -------------------------------------------------------


def test_pack_unpack_roundtrip():
    vector = [0.0, 1.0, -2.5, 3.75]
    assert unpack_vector(pack_vector(vector)) == vector


def test_unpack_rejects_bad_blob():
    with pytest.raises(VectorValidationError):
        unpack_vector(b"123")


def test_validate_vector_dimension():
    with pytest.raises(VectorValidationError):
        validate_vector([1.0] * 5, 3)


def test_validate_vector_non_finite():
    with pytest.raises(VectorValidationError):
        validate_vector([float("nan"), 1.0, 2.0], 3)
    with pytest.raises(VectorValidationError):
        validate_vector([float("inf"), 1.0, 2.0], 3)


def test_normalize_l2_unit_norm():
    v = normalize_l2([3.0, 4.0])
    assert abs(v[0] - 0.6) < 1e-9
    assert abs(v[1] - 0.8) < 1e-9


def test_normalize_l2_zero_vector():
    assert normalize_l2([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_similarity_identical():
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0


def test_content_hash_deterministic_and_sensitive():
    assert content_hash("PERSONAL_FACT", "a") == content_hash("PERSONAL_FACT", "a")
    assert content_hash("PERSONAL_FACT", "a") != content_hash("EPISODE", "a")
    assert content_hash("PERSONAL_FACT", "a") != content_hash("PERSONAL_FACT", "b")


# -- fake provider -----------------------------------------------------------


def test_fake_provider_deterministic_and_dimension():
    p = FakeEmbeddingProvider(dimensions=64)
    a = p.embed_documents(["机器学习"])[0]
    b = p.embed_documents(["机器学习"])[0]
    assert a == b
    assert len(a) == 64


def test_fake_query_vs_document_prefixes_differ():
    p = FakeEmbeddingProvider()
    q = p.embed_query("机器学习")
    d = p.embed_documents(["机器学习"])[0]
    # The query/document prefixes make the two embeddings different for the
    # same raw text, so the pipeline exercises the prefix contract.
    assert q != d


def test_model_spec_hash_stable():
    s1 = EmbeddingModelSpec(
        provider="fake", model_name="fake-deterministic", dimensions=64
    )
    s2 = EmbeddingModelSpec(
        provider="fake", model_name="fake-deterministic", dimensions=64
    )
    assert s1.configuration_hash() == s2.configuration_hash()
    assert s1.model_id() == s2.model_id()
    assert s1.model_id().startswith("emb_")


# -- embedding lifecycle -----------------------------------------------------


async def test_embed_memory_idempotent(emb, rt):
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="hello", access=ADMIN
    )
    first = await emb.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="hello", memory_version=1
    )
    second = await emb.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="hello", memory_version=1
    )
    assert first["status"] == "READY"
    assert second["status"] == "UNCHANGED"


async def test_embed_memory_stale_guard(emb, rt):
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="newer content", access=ADMIN
    )
    await emb.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="newer content", memory_version=2
    )
    # A stale job carries an older version *and* older content; it must never
    # overwrite the newer vector.
    stale = await emb.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="older content", memory_version=1
    )
    assert stale["status"] == "STALE_SKIPPED"


async def test_embed_failure_does_not_destroy_memory(emb, rt, register):
    person = await register("Alice")
    item = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="一个不会被破坏的记忆",
        access=ADMIN,
        subjects=[("PERSON", person.person_id)],
    )

    class _BadDimProvider(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            return [[0.0] * (self.dimensions + 1) for _ in texts]

    bad = EmbeddingService(EmbeddingRepository(emb.repo.db), _BadDimProvider())
    result = await bad.embed_memory(
        item.memory_id, kind="PERSONAL_FACT", content="一个不会被破坏的记忆", memory_version=1
    )
    assert result["status"] == "FAILED"

    # The memory row is untouched and still retrievable by keyword.
    results = await rt.memory.retrieve("记忆", person.person_id)
    assert item.memory_id in {r["memory_id"] for r in results}


async def test_backfill_only_active(emb, rt):
    active = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="活跃记忆",
        access=ADMIN,
        status="ACTIVE",
    )
    await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="已撤回记忆",
        access=ADMIN,
        status="RETRACTED",
    )
    results = await emb.backfill()
    statuses = {r["memory_id"]: r["status"] for r in results}
    assert statuses[active.memory_id] == "READY"
    assert len(statuses) == 1  # only the ACTIVE memory was embedded


async def test_memory_write_enqueues_embedding_job(rt_embed):
    """A normal ACTIVE write automatically queues vector indexing; the worker
    then produces a READY vector without any manual backfill."""
    item = await rt_embed.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="自动索引的记忆", access=ADMIN
    )
    cur = await rt_embed.db.execute(
        "SELECT job_key, payload FROM processing_jobs WHERE job_type = 'EMBEDDING'"
    )
    rows = [dict(r) for r in await cur.fetchall()]
    assert len(rows) == 1
    assert rows[0]["job_key"] == (
        f"embed:{item.memory_id}:{rt_embed.embedding_service.model_id}:1"
    )

    processed = await rt_embed.worker.run_until_idle()
    assert processed >= 1
    vector = await rt_embed.embedding_service.repo.get_embedding(
        item.memory_id, rt_embed.embedding_service.model_id
    )
    assert vector is not None
    assert vector["embedding_status"] == "READY"
    assert vector["memory_version"] == 1


async def test_update_enqueues_new_version_and_retract_does_not(rt_embed):
    item = await rt_embed.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content="v1", access=ADMIN
    )
    await rt_embed.memory_service.update(
        item.memory_id, access=ADMIN, expected_version=1, content="v2"
    )
    await rt_embed.memory_service.retract(item.memory_id, access=ADMIN)
    cur = await rt_embed.db.execute(
        "SELECT job_key FROM processing_jobs WHERE job_type = 'EMBEDDING'"
        " ORDER BY created_at"
    )
    keys = [r["job_key"] for r in await cur.fetchall()]
    # One job per ACTIVE version; the retraction adds none.
    assert keys == [
        f"embed:{item.memory_id}:{rt_embed.embedding_service.model_id}:1",
        f"embed:{item.memory_id}:{rt_embed.embedding_service.model_id}:2",
    ]


async def test_candidate_write_does_not_enqueue(rt_embed):
    await rt_embed.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="待确认候选",
        access=ADMIN,
        status="CANDIDATE",
    )
    cur = await rt_embed.db.execute(
        "SELECT COUNT(*) AS n FROM processing_jobs WHERE job_type = 'EMBEDDING'"
    )
    assert (await cur.fetchone())["n"] == 0


# -- hybrid retrieval --------------------------------------------------------


def _memory(mid, status="ACTIVE"):
    return {
        "memory_id": mid,
        "kind": "PERSONAL_FACT",
        "content": f"content {mid}",
        "status": status,
        "visibility": "PRIVATE",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


class _StubEmbedding:
    model_id = "emb_test"
    spec = EmbeddingModelSpec(
        provider="fake", model_name="t", dimensions=4, distance_metric="cosine"
    )

    def __init__(self, ranking):
        self._ranking = ranking

    async def embed_query(self, text):
        return [0.0] * 4

    @property
    def index(self):
        return self

    async def search(self, model_id, vector, limit, metric):
        return list(self._ranking)


async def test_hybrid_rrf_fuses_and_identity_leads():
    async def load(memory_id, access):
        return _memory(memory_id)

    hybrid = HybridRetriever(
        _StubEmbedding([("c", 0.9), ("b", 0.8)]),
        load_memory=load,
    )
    results = await hybrid.retrieve(
        "q",
        keyword_results=[_memory("a"), _memory("b")],
        access=ADMIN,
        identity_ids={"a"},
        limit=10,
    )
    ids = [r["memory_id"] for r in results]
    # Identity match "a" leads; "b" (in both lists) fuses higher than "c".
    assert ids[0] == "a"
    assert "b" in ids
    assert "c" in ids
    b = next(r for r in results if r["memory_id"] == "b")
    c = next(r for r in results if r["memory_id"] == "c")
    assert b["fusion_score"] > c["fusion_score"]
    # Non-identity results carry diagnostics; the identity match is pristine.
    a = next(r for r in results if r["memory_id"] == "a")
    assert "fusion_score" not in a
    assert "index_model" in b


async def test_hybrid_vector_visibility_recheck():
    async def load(memory_id, access):
        # "secret" is denied to this requester.
        if memory_id == "secret":
            return None
        return _memory(memory_id)

    hybrid = HybridRetriever(
        _StubEmbedding([("secret", 0.99), ("visible", 0.5)]),
        load_memory=load,
    )
    results = await hybrid.retrieve(
        "q",
        keyword_results=[_memory("visible")],
        access=AccessContext("person_x", is_admin=False),
        identity_ids=set(),
        limit=10,
    )
    ids = {r["memory_id"] for r in results}
    assert "secret" not in ids
    assert "visible" in ids
