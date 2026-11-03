"""Phase 4 — keyword memory retrieval and referenced-id validation."""

from __future__ import annotations

import pytest

from companion.domain.enums import MemoryKind, MemoryVisibility
from companion.memory.models import AccessContext
from companion.memory.store import extract_keywords, validate_referenced_memory_ids

ADMIN = AccessContext("admin", is_admin=True)


async def _create(
    rt,
    person_id,
    content,
    *,
    status="ACTIVE",
    visibility=MemoryVisibility.PRIVATE,
    subjects=None,
):
    return await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content=content,
        access=ADMIN,
        status=status,
        visibility=visibility,
        subjects=[("PERSON", person_id)] if subjects is None else subjects,
    )


# -- keyword extraction -----------------------------------------------------


def test_extract_keywords_splits_on_punctuation():
    kws = extract_keywords("你好，今天怎么样？")
    assert "你好" in kws
    assert "今天怎么样" in kws


def test_extract_keywords_empty():
    assert extract_keywords("") == []
    assert extract_keywords(None) == []


def test_extract_keywords_dedupes_and_caps():
    kws = extract_keywords("aa bb aa cc dd ee", max_keywords=3)
    assert kws == ["aa", "bb", "cc"]


def test_extract_keywords_cjk_bigrams():
    # A long unpunctuated CJK clause yields bigram fallbacks so substring
    # matching can still find "机器学习" inside it.
    kws = extract_keywords("机器学习是什么")
    assert "机器" in kws
    assert "学习" in kws


# -- retrieval --------------------------------------------------------------


async def test_retrieve_matches_any_keyword(rt, register):
    person = await register("Alice")
    python = await _create(rt, person.person_id, "今天学习了Python编程")
    cat = await _create(rt, person.person_id, "我家猫很可爱")

    results = await rt.memory.retrieve("今天，怎么样？", person.person_id)
    ids = {r["memory_id"] for r in results}
    assert python.memory_id in ids
    assert cat.memory_id not in ids


async def test_retrieve_cjk_bigram_matches(rt, register):
    person = await register("Alice")
    ml = await _create(rt, person.person_id, "我喜欢机器学习")
    results = await rt.memory.retrieve("机器学习是什么？", person.person_id)
    assert ml.memory_id in {r["memory_id"] for r in results}


async def test_retrieve_only_active(rt, register):
    person = await register("Alice")
    active = await _create(rt, person.person_id, "测试关键词活跃")
    for status in ("CANDIDATE", "NEEDS_REVIEW", "CONFLICTED", "OUTDATED", "RETRACTED"):
        await _create(rt, person.person_id, f"测试关键词{status}", status=status)

    results = await rt.memory.retrieve("测试关键词", person.person_id)
    ids = {r["memory_id"] for r in results}
    assert ids == {active.memory_id}


async def test_retrieve_respects_person_visibility(rt, register):
    alice = await register("Alice")
    bob = await register("Bob")
    await _create(rt, bob.person_id, "鲍勃的私密信息")
    public = await _create(
        rt,
        bob.person_id,
        "公开的世界知识",
        visibility=MemoryVisibility.PUBLIC,
        subjects=[],
    )

    results = await rt.memory.retrieve("私密 世界", alice.person_id)
    ids = {r["memory_id"] for r in results}
    assert public.memory_id in ids  # PUBLIC is visible to anyone
    assert all("鲍勃" not in r["content"] for r in results)


async def test_retrieve_unregistered_returns_empty(rt):
    assert await rt.memory.retrieve("任何内容", None) == []


# -- referenced-id validation ----------------------------------------------


def test_validate_referenced_ids_drops_fabricated():
    retrieved = [{"memory_id": "mem_1"}, {"memory_id": "mem_2"}]
    assert validate_referenced_memory_ids(
        ["mem_1", "mem_fake", "mem_2", "mem_1"], retrieved
    ) == ["mem_1", "mem_2"]


def test_validate_referenced_ids_empty():
    assert validate_referenced_memory_ids([], [{"memory_id": "mem_1"}]) == []
