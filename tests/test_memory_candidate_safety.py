"""Phase 8 — memory-candidate safety.

Candidates are inert proposals from an untrusted model. This module pins:

- duplicate/blank candidates are collapsed before they reach the database;
- the database itself is idempotent per (source event, content);
- an uncertain candidate becomes a durable CANDIDATE, never an ACTIVE fact;
- re-running consolidation has no further side effects;
- nothing is persisted when memory learning is not authorised.
"""

from __future__ import annotations

import json
import logging

from companion.domain.enums import (
    EventSource,
    MemoryCandidateStatus,
    MemoryOperation,
)
from companion.domain.models import MemoryCandidate, make_utterance_event
from companion.jobs.handlers import memory_consolidation
from companion.memory.candidates import dedupe_candidates, normalize_candidate_content
from companion.memory.models import AccessContext
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime


class CandidateClient:
    def __init__(self, candidates, response_text="好的，记住了。"):
        self.analysis = json.dumps(
            {
                "meaning": "用户在介绍自己",
                "action": "RESPOND",
                "reason_code": "direct_agent_address",
                "referenced_memory_ids": [],
                "memory_candidates": candidates,
                "response_style": "warm",
            },
            ensure_ascii=False,
        )
        self.response_text = response_text
        self.calls: list[str] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        content = self.analysis if request_type == "analysis" else self.response_text
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _person(rt, *, learning=True):
    return await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=learning,
        allow_external_ai=True,
    )


def _event(person_id, text="我喜欢川菜"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


# -- the pure helper ---------------------------------------------------------


def test_normalize_collapses_whitespace_and_case():
    assert normalize_candidate_content("  Alice   LIKES  tea ") == "alice likes tea"


def test_dedupe_preserves_first_seen_order_and_drops_blanks():
    def cand(content, cid):
        return MemoryCandidate(
            candidate_id=cid,
            source_event_id="evt",
            content=content,
            operation=MemoryOperation.ADD,
        )

    unique = dedupe_candidates(
        [
            cand(" Alice 喜欢川菜 ", "1"),
            cand("   ", "2"),
            cand("alice 喜欢川菜", "3"),
            cand("Alice 住在杭州", "4"),
            cand("ALICE 喜欢川菜", "5"),
        ]
    )
    assert [c.candidate_id for c in unique] == ["1", "4"]


# -- end-to-end through the real provider ------------------------------------


async def test_duplicate_model_candidates_are_persisted_once(db):
    client = CandidateClient(
        [
            {"content": "Alice 喜欢川菜"},
            {"content": "  Alice   喜欢川菜  "},
            {"content": "Alice 喜欢川菜"},
            {"content": "Alice 住在杭州"},
        ]
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)
    event = _event(person.person_id)

    await rt.orchestrator.process_utterance(event)

    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert [c.content for c in candidates] == ["Alice 喜欢川菜", "Alice 住在杭州"]


async def test_uncertain_candidate_is_stored_as_candidate_not_active(db):
    # The model supplies no assertion/certainty/evidence, so the trusted layer
    # cannot treat this as a clear self-report: it is durably staged as a
    # CANDIDATE, never presented as an ACTIVE fact.
    client = CandidateClient([{"content": "Alice 喜欢川菜"}])
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)
    event = _event(person.person_id)

    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()

    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert len(candidates) == 1
    assert candidates[0].status == MemoryCandidateStatus.CONSOLIDATED
    assert candidates[0].subject_person_id == person.person_id

    write = await rt.repos.analysis.get_candidate_write(candidates[0].candidate_id)
    assert write["outcome"] == "SAVED"
    assert write["memory_id"] is not None

    cur = await db.execute(
        "SELECT status, self_reported, model_inferred FROM memory_items"
        " WHERE memory_id = ?",
        (write["memory_id"],),
    )
    row = await cur.fetchone()
    assert row["status"] == "CANDIDATE"
    assert row["self_reported"] == 0
    assert row["model_inferred"] == 1
    # Nothing is retrievable as a fact yet.
    assert await rt.memory.retrieve("川菜", person.person_id) == []


async def test_consolidation_rerun_changes_nothing(db):
    client = CandidateClient([{"content": "Alice 喜欢川菜"}])
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt)
    event = _event(person.person_id)

    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()
    first = await rt.repos.analysis.get_candidates(event.event_id)

    await memory_consolidation(
        {"event_id": event.event_id}, rt.repos, logging.getLogger("test")
    )
    again = await rt.repos.analysis.get_candidates(event.event_id)

    assert [(c.candidate_id, c.status) for c in again] == [
        (c.candidate_id, c.status) for c in first
    ]
    assert len(again) == 1


async def test_no_candidates_persisted_without_learning_consent(db):
    client = CandidateClient([{"content": "Alice 喜欢川菜"}])
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    person = await _person(rt, learning=False)
    event = _event(person.person_id)

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.response_text == "好的，记住了。"  # cognition still happened
    assert await rt.repos.analysis.get_candidates(event.event_id) == []


# -- trusted-side database idempotency ---------------------------------------


async def test_save_candidate_is_idempotent_at_the_database_level(rt, register, say):
    person = await register("Alice")
    report = await say(person.person_id, "agent", "你好")
    existing = (await rt.repos.analysis.get_candidates(report.event_id))[0]

    duplicate = MemoryCandidate(
        candidate_id="memc_duplicate",
        source_event_id=report.event_id,
        subject_person_id=person.person_id,
        content=existing.content,
        operation=MemoryOperation.ADD,
    )
    await rt.repos.analysis.save_candidate(duplicate)

    candidates = await rt.repos.analysis.get_candidates(report.event_id)
    assert len(candidates) == 1
    assert candidates[0].candidate_id == existing.candidate_id


async def test_third_party_candidate_is_stored_with_attribution(rt, register, say):
    speaker = await register("Alice")
    subject = await register("Bob")
    report = await say(speaker.person_id, "agent", "Bob 喜欢川菜")

    third_party = MemoryCandidate(
        candidate_id="memc_third_party",
        source_event_id=report.event_id,
        subject_person_id=subject.person_id,
        content="Bob 喜欢川菜",
        operation=MemoryOperation.ADD,
    )
    await rt.repos.analysis.save_candidate(third_party)

    await memory_consolidation(
        {"event_id": report.event_id}, rt.repos, logging.getLogger("test")
    )

    candidates = {
        c.candidate_id: c
        for c in await rt.repos.analysis.get_candidates(report.event_id)
    }
    assert candidates["memc_third_party"].status == MemoryCandidateStatus.CONSOLIDATED

    write = await rt.repos.analysis.get_candidate_write("memc_third_party")
    assert write["outcome"] == "SAVED"

    cur = await rt.db.execute(
        "SELECT status, third_party_reported, self_reported FROM memory_items"
        " WHERE memory_id = ?",
        (write["memory_id"],),
    )
    row = await cur.fetchone()
    # Preserved as an attributed, non-authoritative report -- never ACTIVE.
    assert row["status"] == "CANDIDATE"
    assert row["third_party_reported"] == 1
    assert row["self_reported"] == 0

    provenance = await rt.memory_service.repo.get_provenance(write["memory_id"])
    assert provenance["speaker_person_id"] == speaker.person_id
    assert provenance["subject_person_id"] is None
    # The subject is not silently enrolled in someone else's memory.
    assert await rt.memory_service.list_memory_assignments(
        write["memory_id"], AccessContext(actor_id="admin", is_admin=True)
    ) == []
