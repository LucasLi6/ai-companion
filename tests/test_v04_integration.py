"""V0.4 — orchestration integration: atomicity, recovery, flags, privacy.

These tests drive the real orchestrator over the real database with small fake
providers (no network). They pin the commit unit, the scene CAS retry, the
crash/replay recovery paths, and that disabling V0.4 leaves V0.3.2 behaviour.
"""

from __future__ import annotations

import json

from companion.cognition.scene_repository import (
    SceneRepository,
    SceneRevisionConflict,
)
from companion.cognition.scene_models import SceneState
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ProcessingStatus,
    ReasonCode,
)
from companion.domain.models import (
    CognitiveResult,
    SceneProposal,
    make_utterance_event,
)
from companion.runtime import build_runtime


class FakeCognitive:
    def __init__(self, proposal=None, *, action=CognitiveAction.RESPOND):
        self.proposal = proposal
        self.action = action
        self.calls = []

    async def analyze(self, event, context, memories):
        self.calls.append(event.event_id)
        return CognitiveResult(
            event_id=event.event_id,
            meaning="fake analysis",
            action=self.action,
            reason_code=(
                ReasonCode.DIRECT_AGENT_ADDRESS
                if self.action == CognitiveAction.RESPOND
                else ReasonCode.AMBIGUOUS_ADDRESSEE
            ),
            response_style="warm",
            provider="fake",
            scene_proposal=self.proposal,
        )


class FakeResponse:
    def __init__(self, text="好的。"):
        self.text = text
        self.calls = []
        self.contexts = []

    async def generate(self, event, analysis, context):
        self.calls.append(event.event_id)
        self.contexts.append(context)
        return self.text


def _proposal(topic="旅行", evidence="我去了日本"):
    return SceneProposal(
        topic_change="SHIFT",
        topic=topic,
        mode="CONVERSATION",
        discourse_phase="EXPLORING",
        tone="WARM",
        confidence=0.9,
        evidence_span=evidence,
    )


def _event(person_id, text, *, conversation="c1", turn=1):
    from datetime import datetime, timedelta, timezone

    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        occurred_at=datetime(2026, 7, 1, tzinfo=timezone.utc)
        + timedelta(seconds=turn),
    )


async def _person(rt, name="Alice", persistence=True):
    person = await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=persistence,
        allow_memory_learning=True,
        allow_external_ai=False,
    )
    await rt.conversations.get_or_create("c1")
    return person


def _rt(db, proposal=None, *, personality=True):
    cognitive = FakeCognitive(proposal)
    response = FakeResponse()
    rt = build_runtime(
        db,
        cognitive=cognitive,
        response=response,
        scene_enabled=True,
        personality_generation_enabled=personality,
    )
    return rt, cognitive, response


async def _receipt(rt, event_id):
    return await SceneRepository(rt.db).get_receipt(event_id)


async def _snapshot(rt, person_id):
    return await SceneRepository(rt.db).load("c1", person_id)


async def test_commit_is_one_atomic_unit_across_tables(db):
    rt, cognitive, response = _rt(db, _proposal())
    alice = await _person(rt)
    event = _event(alice.person_id, "我去了日本")
    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    assert report.response_text == "好的。"
    assert cognitive.calls == [event.event_id]
    assert response.calls == [event.event_id]

    action = await rt.repos.analysis.get_action(event.event_id)
    utterance = await rt.repos.utterance.get(event.event_id)
    conversation = await rt.repos.conversation.get("c1")
    snapshot = await _snapshot(rt, alice.person_id)
    receipt = await _receipt(rt, event.event_id)
    assert action is not None and action.response_text == "好的。"
    assert utterance.processing_status == ProcessingStatus.COMPLETED
    assert conversation.last_action == "RESPOND"
    assert snapshot.state.topic == "旅行"
    assert snapshot.revision == 0
    assert receipt["result_revision"] == 0

    # The scene proposal round-trips through the analysis row; it is *not* a
    # memory item and is never promoted to a long-term fact.
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis.scene_proposal is not None
    assert analysis.scene_proposal.topic == "旅行"
    cur = await db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0


async def test_stale_scene_read_retries_without_a_second_model_call(db, monkeypatch):
    rt, cognitive, response = _rt(db, _proposal(topic="养猫", evidence="我养了只猫"))
    alice = await _person(rt)

    # An earlier turn is already in the scene. Simulate a stale read: the
    # orchestrator's preparation sees no scene, then the write discovers the
    # real revision and must reload + retry with the same response.
    prior = _event(alice.person_id, "先聊工作", turn=0)
    await rt.repos.utterance.save(prior)
    await rt.scene_manager.scene_repo.apply_transition(
        source_event_id=prior.event_id,
        conversation_id="c1",
        person_id=alice.person_id,
        expected_revision=0,
        state=SceneState(
            conversation_id="c1",
            person_id=alice.person_id,
            revision=0,
            topic="工作",
            source_event_ids=[prior.event_id],
            last_committed_event_id=prior.event_id,
        ),
        transition={"action": "RESPOND", "topic": "工作"},
    )

    real_load = rt.scene_manager.load
    calls = {"n": 0}

    async def stale_then_fresh(conversation_id, person_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return await real_load(conversation_id, person_id)

    monkeypatch.setattr(rt.scene_manager, "load", stale_then_fresh)

    event = _event(alice.person_id, "我养了只猫", turn=1)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    assert cognitive.calls == [event.event_id]
    assert response.calls == [event.event_id]  # not regenerated on retry

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.revision == 1
    assert snapshot.state.topic == "养猫"
    assert snapshot.state.topic_stack == ["工作"]
    receipt = await _receipt(rt, event.event_id)
    assert receipt["result_revision"] == 1


async def test_existing_action_with_advanced_snapshot_backfills_receipt_only(db):
    rt, cognitive, response = _rt(db, _proposal())
    alice = await _person(rt)
    event = _event(alice.person_id, "我去了日本")
    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "COMPLETED"
    before = await _snapshot(rt, alice.person_id)
    assert len(response.calls) == 1

    # Crash after the snapshot committed but before the receipt became visible.
    await db.execute(
        "DELETE FROM scene_transition_receipts WHERE source_event_id = ?",
        (event.event_id,),
    )
    await rt.repos.utterance.update_status(
        event.event_id, ProcessingStatus.FAILED
    )

    replay = await rt.orchestrator.process_utterance(event)
    assert replay.status == "COMPLETED"
    assert replay.response_text == first.response_text
    assert len(cognitive.calls) == 1  # persisted analysis reused
    assert len(response.calls) == 1  # no second response was generated

    after = await _snapshot(rt, alice.person_id)
    assert after.revision == before.revision
    assert after.state.topic == before.state.topic
    receipt = await _receipt(rt, event.event_id)
    transition = json.loads(receipt["transition_json"])
    assert transition["recovered"] is True
    assert transition["reason"] == "snapshot_already_advanced"


async def test_existing_action_with_missing_snapshot_rebuilds_scene(db):
    rt, cognitive, response = _rt(db, _proposal())
    alice = await _person(rt)
    event = _event(alice.person_id, "我去了日本")
    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "COMPLETED"
    assert len(response.calls) == 1

    # Crash before any scene row survived; only the action is durable.
    await db.execute(
        "DELETE FROM scene_transition_receipts WHERE source_event_id = ?",
        (event.event_id,),
    )
    await db.execute(
        "DELETE FROM scene_snapshots WHERE conversation_id = 'c1' AND person_id = ?",
        (alice.person_id,),
    )
    await rt.repos.utterance.update_status(
        event.event_id, ProcessingStatus.FAILED
    )

    replay = await rt.orchestrator.process_utterance(event)
    assert replay.status == "COMPLETED"
    assert replay.response_text == first.response_text
    assert len(cognitive.calls) == 1
    assert len(response.calls) == 1

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "旅行"  # from persisted analysis proposal
    assert snapshot.revision == 0
    receipt = await _receipt(rt, event.event_id)
    assert receipt["result_revision"] == 0


async def test_scene_conflict_exhausts_retries_and_rolls_back_the_action(
    db, monkeypatch
):
    rt, cognitive, response = _rt(db, _proposal())
    alice = await _person(rt)
    event = _event(alice.person_id, "我去了日本")

    async def always_conflict(**kwargs):
        raise SceneRevisionConflict("c1", alice.person_id, 0)

    monkeypatch.setattr(
        rt.scene_manager.scene_repo, "apply_transition", always_conflict
    )

    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "FAILED"
    assert "scene revision conflict" in report.error
    # The response may have been generated, but the action was rolled back and
    # the event is left FAILED for a later retry. No newer scene was overwritten.
    assert await rt.repos.analysis.get_action(event.event_id) is None
    assert await _snapshot(rt, alice.person_id) is None
    assert await _receipt(rt, event.event_id) is None
    utterance = await rt.repos.utterance.get(event.event_id)
    assert utterance.processing_status == ProcessingStatus.FAILED


async def test_scene_flag_off_writes_nothing(db):
    rt, cognitive, response = _rt(db, _proposal())
    alice = await _person(rt)
    await rt.repos.settings.set("dialogue.scene.enabled", "false")
    event = _event(alice.person_id, "我去了日本")
    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    assert await _snapshot(rt, alice.person_id) is None
    assert await _receipt(rt, event.event_id) is None
    assert "[SCENE] Transition committed" not in report.steps


async def test_personality_flag_off_skips_style_compilation(db):
    rt, cognitive, response = _rt(db, _proposal(), personality=False)
    alice = await _person(rt)
    await rt.repos.settings.set("dialogue.personality_generation.enabled", "true")
    event = _event(alice.person_id, "我去了日本")
    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    # With personality disabled at construction there is no compiler, even if
    # the setting is later enabled; the mock context stays lightweight.
    assert rt.personality_compiler is None
    assert "[PERSONALITY] Style compiled" not in report.steps
    assert await _snapshot(rt, alice.person_id) is not None


async def test_non_persistence_writes_no_scene(db):
    rt, cognitive, response = _rt(db, _proposal())
    alice = await _person(rt, persistence=False)
    event = _event(alice.person_id, "我去了日本")
    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    assert await _snapshot(rt, alice.person_id) is None
    assert await _receipt(rt, event.event_id) is None
    assert await rt.repos.analysis.get_analysis(event.event_id) is None


async def test_persistence_withdrawal_deletes_existing_scene_on_nonpersist_turn(db):
    rt, cognitive, response = _rt(db, _proposal())
    alice = await _person(rt)
    first = _event(alice.person_id, "我去了日本", turn=1)
    assert (await rt.orchestrator.process_utterance(first)).status == "COMPLETED"
    assert await _snapshot(rt, alice.person_id) is not None

    consent = await rt.registry.get_consent(alice.person_id)
    await rt.registry.set_consent(
        consent.model_copy(update={"allow_persistence": False})
    )
    second = _event(alice.person_id, "还在吗", turn=2)
    report = await rt.orchestrator.process_utterance(second)

    assert report.status == "COMPLETED"
    assert await _snapshot(rt, alice.person_id) is None
    # The non-persistence turn itself wrote no scene or receipt.
    assert await _receipt(rt, second.event_id) is None
    assert await rt.repos.analysis.get_analysis(second.event_id) is None


async def test_untrusted_scene_proposal_cannot_set_a_topic(db):
    rt, cognitive, response = _rt(
        db, _proposal(topic="医院", evidence="这句话不在输入里")
    )
    alice = await _person(rt)
    event = _event(alice.person_id, "我去了日本")
    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "UNKNOWN"
