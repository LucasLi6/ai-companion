"""V0.4 — local scene assembly and privacy invalidation.

The manager is the trusted side of the scene boundary. These tests exercise
topic stacks, evidence revalidation, bounded open loops, environment expiry and
consent-driven invalidation without any model or network call.
"""

from __future__ import annotations

from types import SimpleNamespace

from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.cognition.scene_manager import SceneManager
from companion.cognition.scene_models import (
    ENVIRONMENT_TTL_TURNS,
    MAX_OPEN_LOOPS,
    MAX_TOPIC_STACK,
    SceneState,
)
from companion.domain.enums import CognitiveAction, EventSource
from companion.domain.models import Consent, make_utterance_event, utcnow


async def _person(rt, *, external_ai=False):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
    )
    await rt.conversations.get_or_create("c1")
    return person.person_id


async def _say(rt, person_id, text):
    event = make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    return event


def _proposal(
    *,
    change="SHIFT",
    topic="旅行",
    evidence=None,
    tone="WARM",
    environment=None,
    open_loop=None,
    resolve_loop_topics=None,
):
    return {
        "topic_change": change,
        "topic": topic,
        "mode": "CONVERSATION",
        "discourse_phase": "EXPLORING",
        "tone": tone,
        "confidence": 0.9,
        "evidence_span": evidence if evidence is not None else "我去了日本",
        "environment": environment,
        "open_loop": open_loop,
        "resolve_loop_topics": resolve_loop_topics or [],
    }


def _context(*turns):
    return CognitiveContext(
        conversation_id="c1",
        current_event_id="evt_current",
        current_speaker_id="person_alice",
        current_speaker_label="Alice",
        current_text="我去了日本",
        turns=list(turns),
    )


async def _persist(manager, previous, event, proposal, *, context=None, pending=False):
    candidate = manager.preview(
        previous,
        proposal,
        event,
        context=context,
        pending_followup=pending,
    )
    candidate = manager.finalize(
        candidate, event=event, action=CognitiveAction.RESPOND
    )
    expected = previous.revision if previous is not None else 0
    await manager.scene_repo.apply_transition(
        source_event_id=event.event_id,
        conversation_id=event.conversation_id,
        person_id=event.speaker_person_id,
        expected_revision=expected,
        state=candidate,
        transition={"action": "RESPOND", "topic": candidate.topic},
    )
    return await manager.load(event.conversation_id, event.speaker_person_id)


async def _manager(rt, *, require_external_ai=False):
    return SceneManager(repos=rt.repos, registry=rt.registry, require_external_ai=require_external_ai)


async def test_topic_shift_pushes_stack_and_return_pops(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)

    first = await _say(rt, person, "我去了日本")
    prev = await _persist(manager, None, first, _proposal(topic="旅行"))
    assert prev.state.topic == "旅行"
    assert prev.state.topic_stack == []

    second = await _say(rt, person, "今天聊聊工作")
    prev = await _persist(
        manager,
        prev,
        second,
        _proposal(topic="工作", evidence="今天聊聊工作"),
    )
    assert prev.state.topic == "工作"
    assert prev.state.topic_stack == ["旅行"]

    third = await _say(rt, person, "还是说回日本吧")
    prev = await _persist(
        manager,
        prev,
        third,
        _proposal(change="RETURN", topic="旅行", evidence="还是说回日本吧"),
    )
    assert prev.state.topic == "旅行"
    assert prev.state.topic_stack == []


async def test_return_to_unseen_topic_is_treated_as_shift(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    event = await _say(rt, person, "先聊工作")
    prev = await _persist(manager, None, event, _proposal(topic="工作", evidence="先聊工作"))

    event = await _say(rt, person, "回到旅行")
    prev = await _persist(
        manager,
        prev,
        event,
        _proposal(change="RETURN", topic="旅行", evidence="回到旅行"),
    )
    assert prev.state.topic == "旅行"
    assert prev.state.topic_stack == ["工作"]


async def test_continue_fills_unknown_topic_but_never_invents_one(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)

    unknown = _proposal(topic=None, change="CONTINUE", evidence=None, tone="NEUTRAL")
    event = await _say(rt, person, "嗯")
    prev = await _persist(manager, None, event, unknown)
    assert prev.state.topic == "UNKNOWN"

    event = await _say(rt, person, "最近在学日语")
    prev = await _persist(
        manager,
        prev,
        event,
        _proposal(change="CONTINUE", topic="日语学习", evidence="最近在学日语"),
    )
    assert prev.state.topic == "日语学习"


async def test_unverbatim_topic_does_not_replace_existing_topic(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    event = await _say(rt, person, "我去了日本")
    prev = await _persist(manager, None, event, _proposal(topic="旅行"))

    event = await _say(rt, person, "今天天气不错")
    prev = await _persist(
        manager,
        prev,
        event,
        _proposal(topic="被编造的话题", evidence="原文没有这句话"),
    )
    assert prev.state.topic == "旅行"


async def test_other_persons_text_cannot_authorize_this_topic(db, rt):
    manager = await _manager(rt)
    alice = await _person(rt)
    event = await _say(rt, alice, "你好")
    other_turn = CognitiveTurn(
        event_id="evt_bob",
        role="human",
        speaker_person_id="person_bob",
        speaker_label="Bob",
        text="我在医院",
    )
    candidate = manager.preview(
        None,
        _proposal(topic="医院", evidence="我在医院"),
        event,
        context=_context(other_turn),
    )
    assert candidate.topic == "UNKNOWN"


async def test_unregistered_history_cannot_authorize_this_topic(db, rt):
    manager = await _manager(rt)
    alice = await _person(rt)
    event = await _say(rt, alice, "你好")
    unregistered_turn = CognitiveTurn(
        event_id="evt_unknown_speaker",
        role="human",
        speaker_person_id=None,
        speaker_label="未知说话人",
        text="我在医院",
    )
    candidate = manager.preview(
        None,
        _proposal(topic="医院", evidence="我在医院"),
        event,
        context=_context(unregistered_turn),
    )
    assert candidate.topic == "UNKNOWN"


async def test_open_loop_requires_local_pending_followup(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    loop = {
        "topic": "工作",
        "summary": "工作的事还没讲完",
        "evidence_span": "工作的事等会儿说",
    }
    event = await _say(rt, person, "工作的事等会儿说")

    without_pending = await _persist(
        manager, None, event, _proposal(topic="工作", evidence="工作的事等会儿说", open_loop=loop),
        pending=False,
    )
    assert without_pending.state.open_loops == []

    event = await _say(rt, person, "工作的事等会儿说，先聊别的")
    with_pending = await _persist(
        manager,
        without_pending,
        event,
        _proposal(topic="工作", evidence="工作的事等会儿说", open_loop=loop),
        pending=True,
    )
    assert len(with_pending.state.open_loops) == 1
    assert with_pending.state.open_loops[0].summary == "工作的事还没讲完"


async def test_resolved_open_loop_is_removed(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    loop = {"topic": "工作", "summary": "还没讲完", "evidence_span": "工作等会儿说"}
    event = await _say(rt, person, "工作等会儿说")
    prev = await _persist(
        manager, None, event,
        _proposal(topic="工作", evidence="工作等会儿说", open_loop=loop),
        pending=True,
    )
    assert len(prev.state.open_loops) == 1

    event = await _say(rt, person, "工作的事说完了")
    prev = await _persist(
        manager,
        prev,
        event,
        _proposal(
            change="CONTINUE",
            topic="工作",
            evidence="工作的事说完了",
            resolve_loop_topics=["工作"],
        ),
    )
    assert prev.state.open_loops == []


async def test_duplicate_loop_is_touched_not_duplicated(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    loop = {"topic": "工作", "summary": "还没讲完", "evidence_span": "工作等会儿说"}
    event = await _say(rt, person, "工作等会儿说")
    prev = await _persist(
        manager, None, event,
        _proposal(topic="工作", evidence="工作等会儿说", open_loop=loop),
        pending=True,
    )

    event = await _say(rt, person, "工作还是等会儿说")
    prev = await _persist(
        manager,
        prev,
        event,
        _proposal(
            change="CONTINUE",
            topic="工作",
            evidence="工作还是等会儿说",
            open_loop=loop,
        ),
        pending=True,
    )
    assert len(prev.state.open_loops) == 1
    assert prev.state.open_loops[0].last_event_id == event.event_id


async def test_open_loops_and_topic_stack_are_bounded(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    prev = None
    for index in range(MAX_OPEN_LOOPS + 3):
        text = f"第{index}件事等会儿说"
        event = await _say(rt, person, text)
        prev = await _persist(
            manager,
            prev,
            event,
            _proposal(
                topic=f"话题{index}",
                evidence=text,
                open_loop={
                    "topic": f"话题{index}",
                    "summary": f"第{index}件事还没讲完",
                    "evidence_span": text,
                },
            ),
            pending=True,
        )
    assert len(prev.state.open_loops) == MAX_OPEN_LOOPS

    prev = await manager.load("c1", person)
    for index in range(MAX_TOPIC_STACK + 3):
        text = f"换到话题{index}"
        event = await _say(rt, person, text)
        prev = await _persist(
            manager, prev, event, _proposal(topic=f"话题{index}", evidence=text)
        )
    assert len(prev.state.topic_stack) <= MAX_TOPIC_STACK
    assert prev.state.topic == f"话题{MAX_TOPIC_STACK + 2}"


async def test_environment_is_explicit_and_expires(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    text = "我一个人在家里"
    event = await _say(rt, person, text)
    prev = await _persist(
        manager,
        None,
        event,
        _proposal(
            topic="在家",
            evidence=text,
            environment={
                "location": "家里",
                "other_people_present": "NO",
                "evidence_span": "我一个人在家里",
            },
        ),
    )
    assert prev.state.environment["status"] == "EXPLICIT"
    assert prev.state.environment["location"] == "家里"
    assert prev.state.prompt_dict()["environment"]["location"] == "家里"

    for index in range(ENVIRONMENT_TTL_TURNS + 1):
        event = await _say(rt, person, f"嗯{index}")
        prev = await _persist(manager, prev, event, None)
    assert prev.state.prompt_dict()["environment"]["location"] == "UNKNOWN"
    assert prev.state.prompt_dict()["environment"]["other_people_present"] == "UNKNOWN"


async def test_withdrawn_persistence_invalidates_snapshot(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    event = await _say(rt, person, "我去了日本")
    assert await _persist(manager, None, event, _proposal()) is not None

    consent = await rt.registry.get_consent(person)
    await rt.registry.set_consent(
        Consent(
            person_id=person,
            allow_processing=consent.allow_processing,
            allow_persistence=False,
            allow_memory_learning=consent.allow_memory_learning,
            allow_external_ai=consent.allow_external_ai,
            updated_at=utcnow(),
        )
    )
    assert await manager.load("c1", person) is None
    assert await manager.scene_repo.load("c1", person) is None


async def test_external_ai_withdrawal_invalidates_when_required(db, rt):
    writer = await _manager(rt)
    person = await _person(rt, external_ai=False)
    event = await _say(rt, person, "我去了日本")
    assert await _persist(writer, None, event, _proposal()) is not None

    reader = await _manager(rt, require_external_ai=True)
    # The write path is allowed; the read path refuses to expose derived state
    # to a provider that no longer has external-AI consent.
    assert await reader.load("c1", person) is None


async def test_missing_source_event_invalidates_snapshot(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    event = await _say(rt, person, "我去了日本")
    assert await _persist(manager, None, event, _proposal()) is not None

    await db.execute("DELETE FROM utterances WHERE event_id = ?", (event.event_id,))
    assert await manager.load("c1", person) is None
    assert await manager.scene_repo.load("c1", person) is None


async def test_finalize_touches_matching_loop_without_inventing_one(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    text = "工作的事等会儿说"
    event = await _say(rt, person, text)
    prev = await _persist(
        manager,
        None,
        event,
        _proposal(
            topic="工作",
            evidence=text,
            open_loop={"topic": "工作", "summary": "还没讲完", "evidence_span": text},
        ),
        pending=True,
    )
    candidate = prev.state.model_copy(deep=True)
    event2 = await _say(rt, person, "继续工作")
    final = manager.finalize(candidate, event=event2, action=CognitiveAction.RESPOND)
    assert final.open_loops[0].last_event_id == event2.event_id
    assert len(final.open_loops) == 1

    empty = SceneState(conversation_id="c1", person_id=person)
    final = manager.finalize(empty, event=event2, action=CognitiveAction.WAIT)
    assert final.open_loops == []


async def test_preview_without_proposal_only_ages_state(db, rt):
    manager = await _manager(rt)
    person = await _person(rt)
    event = await _say(rt, person, "我去了日本")
    prev = await _persist(manager, None, event, _proposal())

    event = await _say(rt, person, "嗯")
    candidate = manager.preview(prev, None, event)
    assert candidate.topic == "旅行"
    assert candidate.mode == prev.state.mode
    assert candidate.revision == prev.revision + 1
    assert candidate.last_committed_event_id == event.event_id
