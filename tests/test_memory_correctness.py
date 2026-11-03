"""Memory correctness regressions 8.1-8.5 and 8.7.

These are end-to-end through the orchestrator + background worker, with a
scripted cognitive provider (no network). They pin the behaviours the CMF
upgrade promised: unrelated attributes never overwrite each other, unsupported
claims are not promoted by a verbatim quote, retrieval carries real subject
associations, assignments never grant access, and corrections are append-only
provenance events.
"""

from __future__ import annotations

import pytest

from companion.cognition.silence import (
    detect_explicit_remember,
    detect_explicit_silence,
)
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
    MemoryCandidate,
    make_utterance_event,
    new_id,
)
from companion.memory.models import AccessContext
from companion.runtime import build_runtime

ADMIN = AccessContext("admin", is_admin=True)


class ScriptedCognitive:
    def __init__(self, build):
        self.build = build
        self.seen_memories: list[list[dict]] = []

    async def analyze(self, event, context, memories) -> CognitiveResult:
        self.seen_memories.append(list(memories))
        return self.build(event)


def _candidate(
    event,
    content,
    *,
    topic=None,
    span=None,
    assertion=AssertionType.SELF_REPORT,
    certainty=CertaintyCategory.DIRECT,
    operation=MemoryOperation.ADD,
    explicit_remember=False,
    subject_id=None,
):
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


def _result(event, candidates=()):
    return CognitiveResult(
        event_id=event.event_id,
        meaning="fake analysis",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        referenced_memory_ids=[],
        memory_candidates=list(candidates),
        response_style="plain",
        provider="fake",
        intent="statement",
        addressed_to="agent",
        silence_requested=False,
        reply_necessity=ReplyNecessity.REQUIRED,
        conversation_state="ongoing",
    )


def _event(person_id, text, *, conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def _person(rt, name="李悦"):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=False,
    )


async def _say(rt, person_id, text, *, candidate_builder, conversation="c1"):
    event = _event(person_id, text, conversation=conversation)
    builder = candidate_builder

    class OneShot:
        async def analyze(self, ev, context, memories):
            return _result(ev, [builder(ev)])

    rt.orchestrator.cognitive = OneShot()
    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()
    return event


async def _active(rt, person_id):
    rows = await rt.memory_service.repo.list_for_subject(
        person_id, statuses=[MemoryStatus.ACTIVE], limit=50
    )
    return [r for r in rows]


async def test_8_1_unrelated_preferences_coexist(db):
    rt = build_runtime(db)
    person = await _person(rt)
    await _say(
        rt,
        person.person_id,
        "我喜欢爵士乐。",
        candidate_builder=lambda ev: _candidate(
            ev, "我喜欢爵士乐。", topic="preference", span="我喜欢爵士乐。"
        ),
        conversation="c1",
    )
    await _say(
        rt,
        person.person_id,
        "我喜欢编程。",
        candidate_builder=lambda ev: _candidate(
            ev, "我喜欢编程。", topic="preference", span="我喜欢编程。"
        ),
        conversation="c2",
    )
    rows = await _active(rt, person.person_id)
    contents = {r["content"] for r in rows}
    assert contents == {"我喜欢爵士乐。", "我喜欢编程。"}


async def test_8_1_unrelated_projects_coexist(db):
    rt = build_runtime(db)
    person = await _person(rt)
    for text, conversation in (
        ("我在开发四足机器人。", "c1"),
        ("我在开发奶茶机。", "c2"),
    ):
        await _say(
            rt,
            person.person_id,
            text,
            candidate_builder=lambda ev, t=text: _candidate(
                ev, t, topic="project", span=t
            ),
            conversation=conversation,
        )
    rows = await _active(rt, person.person_id)
    assert {r["content"] for r in rows} == {"我在开发四足机器人。", "我在开发奶茶机。"}
    # Neither became a review item.
    cur = await db.execute(
        "SELECT status FROM memory_items WHERE content LIKE '%机器人%' OR content LIKE '%奶茶%'"
    )
    assert {r["status"] for r in await cur.fetchall()} == {"ACTIVE"}


async def test_8_2_verbatim_span_does_not_ground_a_new_claim(db):
    rt = build_runtime(db)
    person = await _person(rt)
    event = _event(person.person_id, "我今天写了代码。")

    class OneShot:
        async def analyze(self, ev, context, memories):
            return _result(
                ev,
                [
                    _candidate(
                        ev,
                        "我特别喜欢编程。",
                        topic="preference",
                        span="我今天写了代码。",
                    )
                ],
            )

    rt.orchestrator.cognitive = OneShot()
    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()

    cur = await db.execute("SELECT * FROM memory_items")
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "CANDIDATE"
    assert rows[0]["content"] == "我特别喜欢编程。"


async def test_8_2_explicit_remember_does_not_accept_unsupported_claim(db):
    rt = build_runtime(db)
    person = await _person(rt)
    event = _event(person.person_id, "请记住：我今天写了代码。")

    class OneShot:
        async def analyze(self, ev, context, memories):
            return _result(
                ev,
                [
                    _candidate(
                        ev,
                        "我特别喜欢编程。",
                        topic="preference",
                        span="我今天写了代码。",
                        explicit_remember=True,
                    )
                ],
            )

    rt.orchestrator.cognitive = OneShot()
    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_once()

    cur = await db.execute("SELECT status FROM memory_items")
    rows = await cur.fetchall()
    assert [r["status"] for r in rows] == ["CANDIDATE"]


async def test_8_3_identity_retrieval_uses_real_subjects(db):
    rt = build_runtime(db)
    person = await _person(rt)
    await _say(
        rt,
        person.person_id,
        "我叫李悦。",
        candidate_builder=lambda ev: _candidate(
            ev, "我叫李悦。", topic="name", span="我叫李悦。"
        ),
    )
    memories = await rt.memory.retrieve("我叫什么名字？", person.person_id)
    assert memories
    first = memories[0]
    assert any("李悦" in m["content"] for m in memories)
    assert first["subjects"], "identity retrieval fabricated no subjects"
    assert all(s["subject_id"] != "PERSON:speaker" for s in first["subjects"])
    assert any(
        s["subject_type"] == "PERSON" and s["subject_id"] == person.person_id
        for s in first["subjects"]
    )


async def test_8_4_assignment_associates_but_does_not_grant_access(db):
    rt = build_runtime(db)
    owner = await _person(rt, "李悦")
    other = await _person(rt, "王强")
    owner_access = AccessContext(owner.person_id, is_admin=False, role="friend")
    item = await rt.memory_service.create(
        kind="PERSONAL_FACT",
        content="我的秘密爱好是观鸟。",
        access=owner_access,
        visibility="PRIVATE",
        subjects=[("PERSON", owner.person_id)],
        created_by=owner.person_id,
    )
    await rt.memory_service.assign_memory(item.memory_id, other.person_id, ADMIN)

    assert await rt.memory.retrieve("观鸟", owner.person_id)
    assert await rt.memory.retrieve("观鸟", other.person_id) == []

    listed = await rt.memory_service.list_assigned_memories(
        other.person_id, AccessContext(other.person_id, is_admin=False, role="friend")
    )
    assert listed == []


async def test_8_5_corrections_append_provenance_events(db):
    rt = build_runtime(db)
    person = await _person(rt)
    event1 = await _say(
        rt,
        person.person_id,
        "我叫李悦。",
        candidate_builder=lambda ev: _candidate(
            ev, "我叫李悦。", topic="name", span="我叫李悦。"
        ),
        conversation="c1",
    )
    write = await rt.repos.analysis.get_candidate_write(
        (await rt.repos.analysis.get_candidates(event1.event_id))[0].candidate_id
    )
    memory_id = write["memory_id"]

    for text, conversation in (("我叫李娜。", "c2"), ("我叫王芳。", "c3")):
        await _say(
            rt,
            person.person_id,
            text,
            candidate_builder=lambda ev, t=text: _candidate(
                ev,
                t,
                topic="name",
                span=t,
                assertion=AssertionType.CORRECTION,
            ),
            conversation=conversation,
        )

    item = await rt.memory_service.repo.get(memory_id)
    assert item.content == "我叫王芳。"
    events = await rt.memory_service.repo.list_provenance_events(memory_id)
    types = [e["event_type"] for e in events]
    assert types[0] == "created"
    assert types.count("corrected") == 2
    assert events[-1]["old_version"] < events[-1]["new_version"]
    assert events[-1]["payload"]["entity_key"] == "self"
    assert events[-1]["payload"]["attribute_key"] == "identity:name"
    # The originals are retained as revisions, not discarded.
    revisions = await rt.memory_service.repo.list_revisions(memory_id)
    assert {r.content for r in revisions} >= {"我叫李悦。", "我叫李娜。"}


@pytest.mark.parametrize(
    "text",
    [
        "别回我。",
        "免回。",
        "勿回。",
        "不必回复。",
        "先别说话。",
        "安静一下。",
        "leave me alone",
        "stay quiet",
        "hold your reply",
    ],
)
def test_8_7_silence_expressions(text):
    assert detect_explicit_silence(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "我的朋友刚才说：不要回答我。",
        "如果我说“别回我”，你会怎么做？",
        "他问我为什么 AI 有时候不回答。",
        "请记住：我喜欢爵士乐。",
    ],
)
def test_8_7_quoted_reported_and_hypothetical_are_not_requests(text):
    assert detect_explicit_silence(text) is False


def test_8_7_remember_detection_unchanged():
    assert detect_explicit_remember("请记住：我叫李悦") is True
    assert detect_explicit_remember("不要记住这个") is False
