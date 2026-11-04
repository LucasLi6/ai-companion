"""V0.2A — durable memory write-through and truthful outcomes.

Every provider here is a fake: no network, no key, no cost. The module pins the
behaviour the V0.2A upgrade promised:

- A direct, clear self-report becomes an ACTIVE, self-reported memory with
  full provenance -- never ``human_confirmed``.
- An explicit "remember this" completes the durable write *before* the reply
  that acknowledges it, and the outcome passed to the response layer is the
  one actually committed.
- An uncertain/third-party/hypothetical candidate is stored truthfully: a
  durable CANDIDATE with attribution, or no memory at all.
- A correction updates the current-state memory in place (old content kept in
  revisions); a duplicate is an ALREADY_EXISTS mapping, not a second memory.
- Writes are idempotent across retries, restarts and lost mappings.
- Revoked learning consent blocks new writes.
- Assignments associate a memory with a person without granting access.
"""

from __future__ import annotations

import pytest

from companion.cognition.models import CognitiveContext
from companion.domain.enums import (
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    EventSource,
    MemoryCandidateStatus,
    MemoryOperation,
    MemoryStatus,
    ReasonCode,
    ReplyNecessity,
)
from companion.domain.models import (
    CognitiveResult,
    Consent,
    MemoryCandidate,
    make_utterance_event,
    new_id,
    utcnow,
)
from companion.memory.models import AccessContext, MemoryPermissionDeniedError
from companion.prompts.response import build_response_messages
from companion.runtime import build_runtime
from companion.storage.database import Database


# -- fakes -------------------------------------------------------------------


class ScriptedCognitive:
    """Returns whatever ``build(event)`` produces and records seen memories."""

    def __init__(self, build):
        self.build = build
        self.seen_memories: list[list[dict]] = []

    async def analyze(self, event, context, memories) -> CognitiveResult:
        self.seen_memories.append(list(memories))
        return self.build(event)


class RecordingResponse:
    def __init__(self, db, text="好的，我记住了。"):
        self.db = db
        self.text = text
        self.calls: list[str] = []
        self.memory_counts_at_generation: list[int] = []

    async def generate(self, event, analysis, context) -> str:
        self.calls.append(event.event_id)
        cur = await self.db.execute("SELECT COUNT(*) AS n FROM memory_items")
        self.memory_counts_at_generation.append((await cur.fetchone())["n"])
        return self.text


def candidate(
    event,
    *,
    content: str,
    subject_id: str | None = None,
    assertion: AssertionType = AssertionType.SELF_REPORT,
    certainty: CertaintyCategory = CertaintyCategory.DIRECT,
    topic: str | None = None,
    span: str | None = None,
    operation: MemoryOperation = MemoryOperation.ADD,
    explicit_remember: bool = False,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=new_id("memc"),
        source_event_id=event.event_id,
        subject_person_id=subject_id,
        content=content,
        operation=operation,
        assertion_type=assertion,
        certainty_category=certainty,
        topic=topic,
        source_evidence_span=span,
        explicit_remember_request=explicit_remember,
    )


def result(
    event,
    candidates=(),
    *,
    action: CognitiveAction = CognitiveAction.RESPOND,
    reason: ReasonCode = ReasonCode.DIRECT_AGENT_ADDRESS,
    necessity: ReplyNecessity = ReplyNecessity.REQUIRED,
    addressed_to: str | None = "agent",
) -> CognitiveResult:
    return CognitiveResult(
        event_id=event.event_id,
        meaning="fake analysis",
        action=action,
        reason_code=reason,
        referenced_memory_ids=[],
        memory_candidates=list(candidates),
        response_style="plain",
        provider="fake",
        intent="statement",
        addressed_to=addressed_to,
        silence_requested=False,
        reply_necessity=necessity,
        conversation_state="ongoing",
    )


async def _person(rt, name="李悦", *, learning=True):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=learning,
        allow_external_ai=False,
    )


def _event(person_id, text, *, conversation="c1", to="agent"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def _only_candidate(rt, event_id):
    candidates = await rt.repos.analysis.get_candidates(event_id)
    assert len(candidates) == 1
    return candidates[0]


# -- A. a clear self-report becomes a durable ACTIVE memory ------------------


async def test_a_clear_self_report_becomes_active_memory(db):
    def build(event):
        return result(
            event,
            [
                candidate(
                    event,
                    content="我叫李悦",
                    topic="name",
                    span="我叫李悦",
                )
            ],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    person = await _person(rt)
    event = _event(person.person_id, "我叫李悦。")

    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    await rt.worker.run_once()

    cand = await _only_candidate(rt, event.event_id)
    assert cand.status == MemoryCandidateStatus.CONSOLIDATED
    write = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
    assert write["outcome"] == "SAVED"

    item = await rt.memory_service.repo.get(write["memory_id"])
    assert item is not None
    assert item.status == MemoryStatus.ACTIVE
    assert item.self_reported is True
    assert item.third_party_reported is False
    assert item.human_confirmed is False  # a model-extracted fact is never auto-confirmed

    prov = await rt.memory_service.repo.get_provenance(item.memory_id)
    assert prov["source_event_id"] == event.event_id
    assert prov["candidate_id"] == cand.candidate_id
    assert prov["speaker_person_id"] == person.person_id
    assert prov["subject_person_id"] == person.person_id
    assert prov["topic"] == "name"
    assert prov["assertion_type"] == AssertionType.SELF_REPORT.value
    assert prov["source_type"] == "utterance"

    assigned = await rt.memory_service.list_memory_assignments(
        item.memory_id, AccessContext(actor_id="admin", is_admin=True)
    )
    assert [a["person_id"] for a in assigned] == [person.person_id]

    # And it is retrievable as a fact for its subject.
    retrieved = await rt.memory.retrieve("我叫什么名字？", person.person_id)
    assert any("李悦" in m["content"] for m in retrieved)


# -- B. explicit remember writes durably before acknowledgement --------------


async def test_b_explicit_remember_is_durable_before_the_response(db):
    def build(event):
        # The model does not mark it explicit; the trusted detector does.
        return result(
            event,
            [
                candidate(
                    event,
                    content="我叫李悦",
                    topic="name",
                    span="我叫李悦",
                )
            ],
        )

    response = RecordingResponse(db)
    rt = build_runtime(db, cognitive=ScriptedCognitive(build), response=response)
    person = await _person(rt)
    event = _event(person.person_id, "请记住：我叫李悦。")

    report = await rt.orchestrator.process_utterance(event)

    # The response was generated after the durable write, and the committed
    # outcome was passed to the response layer.
    assert response.calls == [event.event_id]
    assert response.memory_counts_at_generation == [1]
    assert [o["outcome"] for o in report.memory_write_outcomes] == ["SAVED"]

    # The background job contributes nothing new: the mapping is the anchor.
    await rt.worker.run_once()
    await rt.worker.run_once()
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1

    cand = await _only_candidate(rt, event.event_id)
    assert cand.explicit_remember_request is True
    assert cand.status == MemoryCandidateStatus.CONSOLIDATED
    write = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
    assert write["outcome"] == "SAVED"


def test_b_response_prompt_reports_only_committed_outcomes():
    context = CognitiveContext(
        conversation_id="c1",
        current_event_id="evt",
        current_speaker_label="Alice",
        current_speaker_person_id="p1",
        current_text="请记住我叫李悦",
        memory_write_outcomes=[
            {"candidate_id": "c1", "outcome": "SAVED"},
            {"candidate_id": "c2", "outcome": "FAILED"},
            {"candidate_id": "c3", "outcome": "PENDING_REVIEW"},
            {"candidate_id": "c4", "outcome": "NO_MEMORY"},
        ],
    )
    analysis = CognitiveResult(
        event_id="evt",
        meaning="用户在自我介绍",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        response_style="plain",
    )
    prompt = build_response_messages(context, analysis)[0]["content"]
    assert "已保存" in prompt
    assert "保存失败" in prompt
    assert "待人工确认" in prompt
    assert "无需保存" not in prompt  # NO_MEMORY is never mentioned as saved


# -- C. restart + new conversation: identity facts survive -------------------


async def test_c_self_report_survives_restart_and_answers_identity_question(tmp_path):
    path = tmp_path / "restart.db"

    def write_build(event):
        return result(
            event,
            [candidate(event, content="我叫李悦", topic="name", span="我叫李悦")],
        )

    db = await Database(path).connect()
    await db.initialize()
    rt1 = build_runtime(db, cognitive=ScriptedCognitive(write_build))
    person = await _person(rt1)
    event1 = _event(person.person_id, "我叫李悦。")
    await rt1.orchestrator.process_utterance(event1)
    await rt1.worker.run_once()
    await db.close()

    probe = ScriptedCognitive(
        lambda event: result(event, [], necessity=ReplyNecessity.REQUIRED)
    )
    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2, cognitive=probe)
    event2 = _event(person.person_id, "我叫什么名字？", conversation="c2")
    report = await rt2.orchestrator.process_utterance(event2)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    seen = probe.seen_memories[0]
    assert any("李悦" in m["content"] for m in seen)
    await db2.close()


# -- D/E. duplicates ----------------------------------------------------------


async def test_d_duplicate_event_produces_one_memory_and_one_write(db):
    def build(event):
        return result(
            event,
            [candidate(event, content="我叫李悦", topic="name", span="我叫李悦")],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    person = await _person(rt)
    event = _event(person.person_id, "我叫李悦。")

    first = await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()
    second = await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()

    assert first.status == "COMPLETED"
    assert second.status == "DUPLICATE"
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_candidate_writes")
    assert (await cur.fetchone())["n"] == 1


async def test_e_duplicate_content_is_already_exists_not_a_second_memory(db):
    builder = ScriptedCognitive(
        lambda event: result(
            event,
            [candidate(event, content="我叫李悦", topic="name", span="我叫李悦")],
        )
    )
    rt = build_runtime(db, cognitive=builder)
    person = await _person(rt)

    first = _event(person.person_id, "我叫李悦。")
    second = _event(person.person_id, "顺便说一句，我叫李悦。", conversation="c2")
    await rt.orchestrator.process_utterance(first)
    await rt.worker.run_once()
    await rt.orchestrator.process_utterance(second)
    await rt.worker.run_once()

    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1

    cand2 = await _only_candidate(rt, second.event_id)
    write2 = await rt.repos.analysis.get_candidate_write(cand2.candidate_id)
    assert write2["outcome"] == "ALREADY_EXISTS"
    assert cand2.status == MemoryCandidateStatus.CONSOLIDATED


# -- F. correction ------------------------------------------------------------


async def test_f_correction_updates_in_place_and_keeps_history(db):
    def build(event):
        if "刚才说错了" in event.text:
            return result(
                event,
                [
                    candidate(
                        event,
                        content="我的名字是李悦",
                        topic="name",
                        span="我的名字是李悦",
                        assertion=AssertionType.CORRECTION,
                    )
                ],
            )
        return result(
            event,
            [
                candidate(
                    event,
                    content="我的名字是李约",
                    topic="name",
                    span="我的名字是李约",
                )
            ],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    person = await _person(rt)

    first = _event(person.person_id, "我的名字是李约。")
    await rt.orchestrator.process_utterance(first)
    await rt.worker.run_once()
    cur = await db.execute("SELECT memory_id FROM memory_items")
    memory_id = (await cur.fetchone())["memory_id"]

    second = _event(person.person_id, "我刚才说错了，我的名字是李悦。", conversation="c2")
    await rt.orchestrator.process_utterance(second)
    await rt.worker.run_once()

    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1
    item = await rt.memory_service.repo.get(memory_id)
    assert item.content == "我的名字是李悦"
    assert item.status == MemoryStatus.ACTIVE
    assert item.version >= 2

    cand2 = await _only_candidate(rt, second.event_id)
    write2 = await rt.repos.analysis.get_candidate_write(cand2.candidate_id)
    assert write2["outcome"] == "UPDATED"
    assert write2["memory_id"] == memory_id

    cur = await db.execute(
        "SELECT content FROM memory_revisions WHERE memory_id = ? ORDER BY version",
        (memory_id,),
    )
    contents = [r["content"] for r in await cur.fetchall()]
    assert contents[0] == "我的名字是李约"  # the old value is preserved
    assert contents[-1] == "我的名字是李悦"


# -- G. third-party report ----------------------------------------------------


async def test_g_third_party_report_is_attributed_candidate_not_fact(db):
    speaker = None
    subject = None

    def build(event):
        return result(
            event,
            [
                candidate(
                    event,
                    content="Bob 喜欢川菜",
                    subject_id=subject.person_id,
                    assertion=AssertionType.THIRD_PARTY_REPORT,
                    certainty=CertaintyCategory.DIRECT,
                    span="Bob 喜欢川菜",
                )
            ],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    speaker = await _person(rt, "Alice")
    subject = await _person(rt, "Bob")
    event = _event(speaker.person_id, "Bob 喜欢川菜。")

    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()

    cand = await _only_candidate(rt, event.event_id)
    write = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
    assert write["outcome"] == "SAVED"

    item = await rt.memory_service.repo.get(write["memory_id"])
    assert item.status == MemoryStatus.CANDIDATE  # never ACTIVE
    assert item.third_party_reported is True
    assert item.self_reported is False

    prov = await rt.memory_service.repo.get_provenance(item.memory_id)
    assert prov["speaker_person_id"] == speaker.person_id
    assert prov["subject_person_id"] is None
    assert prov["third_party_reported"] == 1

    # Neither the subject nor the reporter sees it as an established fact.
    assert await rt.memory.retrieve("川菜", subject.person_id) == []
    assert await rt.memory.retrieve("川菜", speaker.person_id) == []
    assert await rt.memory_service.repo.get_subjects(item.memory_id) == []


async def test_g_model_named_subject_can_never_become_a_self_report(db):
    speaker = None
    bob = None

    def build(event):
        hinted = "person_does_not_exist" if "第一句" in event.text else bob.person_id
        return result(
            event,
            [
                candidate(
                    event,
                    content="我叫李悦",
                    subject_id=hinted,
                    assertion=AssertionType.SELF_REPORT,
                    span="我叫李悦",
                )
            ],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    speaker = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")

    for text, conversation in (
        ("第一句：我叫李悦。", "c1"),
        ("第二句：我叫李悦。", "c2"),
    ):
        event = _event(speaker.person_id, text, conversation=conversation)
        await rt.orchestrator.process_utterance(event)
        await rt.worker.run_once()

        cand = await _only_candidate(rt, event.event_id)
        assert cand.assertion_type == AssertionType.THIRD_PARTY_REPORT
        write = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
        item = await rt.memory_service.repo.get(write["memory_id"])
        assert item.status == MemoryStatus.CANDIDATE  # still not a fact about Alice
        assert item.third_party_reported is True
        if conversation == "c1":
            assert cand.subject_person_id is None  # unknown id discarded
        else:
            assert cand.subject_person_id == bob.person_id


# -- H/I. no durable write ----------------------------------------------------


async def test_h_hypothetical_candidate_writes_no_memory(db):
    def build(event):
        return result(
            event,
            [
                candidate(
                    event,
                    content="如果我搬到火星",
                    assertion=AssertionType.HYPOTHETICAL,
                    span="如果我搬到火星",
                )
            ],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    person = await _person(rt)
    event = _event(person.person_id, "如果我搬到火星，你会来吗？")

    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()

    cand = await _only_candidate(rt, event.event_id)
    write = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
    assert write["outcome"] == "NO_MEMORY"
    assert write["memory_id"] is None
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0


async def test_i_revoked_learning_blocks_new_writes(db):
    def build(event):
        return result(
            event,
            [candidate(event, content="我叫李悦", topic="name", span="我叫李悦")],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    person = await _person(rt)
    event = _event(person.person_id, "我叫李悦。")
    await rt.orchestrator.process_utterance(event)

    # Revoke learning after the job was queued but before it runs.
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    await rt.worker.run_once()

    cand = await _only_candidate(rt, event.event_id)
    assert cand.status == MemoryCandidateStatus.SKIPPED
    write = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
    assert write["outcome"] == "DENIED"
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0


# -- O. assignments -----------------------------------------------------------


async def test_o_assignments_respect_privacy_and_persist(tmp_path):
    path = tmp_path / "assign.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db)
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")

    admin = AccessContext(actor_id="admin", is_admin=True)
    alice_ctx = AccessContext(actor_id=alice.person_id, is_admin=False, role="ui")
    bob_ctx = AccessContext(actor_id=bob.person_id, is_admin=False, role="ui")

    item = await rt.memory_service.create(
        kind="PERSONAL_FACT",
        content="Alice 喜欢川菜",
        access=admin,
        status=MemoryStatus.ACTIVE,
        visibility="PRIVATE",
        subjects=[("PERSON", alice.person_id)],
        self_reported=True,
    )

    # A non-admin may associate a visible memory with themselves only.
    assert await rt.memory_service.assign_memory(item.memory_id, alice.person_id, alice_ctx)
    assert (
        await rt.memory_service.assign_memory(item.memory_id, alice.person_id, alice_ctx)
        is False
    )
    with pytest.raises(MemoryPermissionDeniedError):
        await rt.memory_service.assign_memory(item.memory_id, bob.person_id, alice_ctx)
    with pytest.raises(MemoryPermissionDeniedError):
        await rt.memory_service.list_assigned_memories(alice.person_id, bob_ctx)

    # An admin may associate it with another person, but that is not an
    # access grant: the private memory stays invisible to Bob.
    assert await rt.memory_service.assign_memory(item.memory_id, bob.person_id, admin)
    assert await rt.memory_service.get_item(item.memory_id, bob_ctx) is None
    assert await rt.memory_service.list_assigned_memories(bob.person_id, bob_ctx) == []

    assignments = await rt.memory_service.list_memory_assignments(item.memory_id, admin)
    assert {a["person_id"] for a in assignments} == {alice.person_id, bob.person_id}

    # Removing an association never touches the memory itself.
    assert await rt.memory_service.unassign_memory(item.memory_id, bob.person_id, admin)
    assert (
        await rt.memory_service.unassign_memory(item.memory_id, bob.person_id, admin)
        is False
    )
    assert (await rt.memory_service.repo.get(item.memory_id)).content == "Alice 喜欢川菜"

    # Persist across a restart: same file, same assignment.
    await db.close()
    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    remaining = await rt2.memory_service.list_memory_assignments(item.memory_id, admin)
    assert [a["person_id"] for a in remaining] == [alice.person_id]
    await db2.close()


# -- P. crash recovery / failure truthfulness ---------------------------------


async def test_p_lost_mapping_is_recovered_without_a_duplicate(db):
    import logging

    from companion.jobs.handlers import memory_consolidation

    def build(event):
        return result(
            event,
            [candidate(event, content="我叫李悦", topic="name", span="我叫李悦")],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    person = await _person(rt)
    event = _event(person.person_id, "我叫李悦。")
    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()

    cand = await _only_candidate(rt, event.event_id)
    original = await rt.repos.analysis.get_candidate_write(cand.candidate_id)

    # Simulate a crash between the memory write and the mapping/candidate
    # status commit: the memory + provenance exist, the mapping does not.
    await db.execute(
        "DELETE FROM memory_candidate_writes WHERE candidate_id = ?",
        (cand.candidate_id,),
    )
    await rt.repos.analysis.update_candidate_status(
        cand.candidate_id, MemoryCandidateStatus.PENDING
    )

    await memory_consolidation(
        {"event_id": event.event_id}, rt.repos, logging.getLogger("test")
    )

    recovered = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
    assert recovered["outcome"] == "SAVED"
    assert recovered["detail"] == "recovered_from_provenance"
    assert recovered["memory_id"] == original["memory_id"]
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1

    refreshed = await _only_candidate(rt, event.event_id)
    assert refreshed.status == MemoryCandidateStatus.CONSOLIDATED


async def test_p_database_failure_is_never_recorded_as_saved(db, monkeypatch):
    from companion.memory.service import MemoryService

    def build(event):
        return result(
            event,
            [candidate(event, content="我叫李悦", topic="name", span="我叫李悦")],
        )

    rt = build_runtime(db, cognitive=ScriptedCognitive(build))
    person = await _person(rt)
    event = _event(person.person_id, "我叫李悦。")
    await rt.orchestrator.process_utterance(event)

    async def boom(self, *args, **kwargs):
        raise RuntimeError("simulated disk failure")

    # The handler constructs its own service instance, so patch the class.
    monkeypatch.setattr(MemoryService, "create", boom)
    await rt.worker.run_once()

    cand = await _only_candidate(rt, event.event_id)
    assert cand.status == MemoryCandidateStatus.FAILED
    assert await rt.repos.analysis.get_candidate_write(cand.candidate_id) is None
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0

    # The retry (job requeued, not marked success) completes truthfully.
    monkeypatch.undo()
    await rt.worker.run_once()
    write = await rt.repos.analysis.get_candidate_write(cand.candidate_id)
    assert write["outcome"] == "SAVED"
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1
