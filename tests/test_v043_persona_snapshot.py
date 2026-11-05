"""V0.4.3: one trusted CharacterSnapshot per turn.

The cognitive and response providers must describe the same person at the same
revision. These tests pin the snapshot compiler, stance relevance filtering and
the admin-configurable emotional dispositions, then verify through the real
orchestrator that both provider calls receive the identical snapshot.
"""

from __future__ import annotations

from companion.agent.character import (
    build_character_snapshot,
    parse_emotional_dispositions,
)
from companion.agent.models import (
    DEFAULT_EMOTIONAL_DISPOSITIONS,
    EffectiveAgentContext,
)
from companion.agent.stance import filter_relevant_stances
from companion.domain.enums import CognitiveAction, EventSource, ReasonCode
from companion.domain.models import CognitiveResult, make_utterance_event
from companion.runtime import build_runtime


class _Stance:
    def __init__(self, stance, *, topic="音乐", source="RESPONSE"):
        self.stance = stance
        self.topic = topic
        self.source = source


class RecordingCognitive:
    name = "recording"

    def __init__(self):
        self.contexts = []

    async def analyze(self, event, context, memories):
        self.contexts.append(context)
        return CognitiveResult(
            event_id=event.event_id,
            meaning="对方在说话",
            action=CognitiveAction.RESPOND,
            reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
            provider="recording",
        )


class RecordingResponse:
    name = "recording"

    def __init__(self, text="好。"):
        self.text = text
        self.contexts = []

    async def generate(self, event, analysis, context):
        self.contexts.append(context)
        return self.text


def _effective(**updates):
    payload = {
        "agent_id": "agent_1",
        "display_name": "Ava",
        "pronouns": None,
        "description": "一个具体的人。",
        "person_id": "person_1",
        "familiarity": "CLOSE",
        "traits": {"warmth": 3},
        "overrides": {},
        "persona_core": {
            "revision": 4,
            "character_description": "说话不急，先听完再判断。",
            "core_values": ["诚实"],
            "preferences": {"音乐": "老摇滚"},
            "temperament": {"humor": "dry"},
            "conversational_boundaries": ["不编造经历"],
            "quirks": ["听到含糊说法会追问"],
        },
        "response_conservatism": "balanced",
        "interruption_threshold": "medium",
        "curiosity_level": "low",
        "revision": 4,
    }
    payload.update(updates)
    return EffectiveAgentContext(**payload)


def test_snapshot_carries_the_full_character_and_revision():
    snapshot = build_character_snapshot(
        _effective(),
        query_text="你平时听什么音乐？",
        topic="音乐",
        dispositions={"expressiveness": 0.9},
    )

    assert snapshot.revision == 4
    assert snapshot.display_name == "Ava"
    assert snapshot.relationship_tier == "CLOSE"
    assert snapshot.character_description == "说话不急，先听完再判断。"
    assert snapshot.persona_core["core_values"] == ["诚实"]
    assert snapshot.traits == {"warmth": 3}
    assert snapshot.emotional_dispositions["expressiveness"] == 0.9
    payload = snapshot.as_dict()
    assert payload["character_description"] == "说话不急，先听完再判断。"
    assert payload["traits"] == {"warmth": 3}
    assert payload["persona_core"]["revision"] == 4


def test_unrelated_response_stances_are_filtered_but_admin_stances_stay():
    stances = [
        _Stance("我不喜欢嘈杂的现场。", topic="音乐", source="RESPONSE"),
        _Stance("我不喜欢临时改计划。", topic="计划", source="RESPONSE"),
        _Stance("我更喜欢先听完再评价。", topic="交流", source="ADMIN"),
    ]
    music = filter_relevant_stances(
        stances, topic="音乐", query_text="这个乐队怎么样？"
    )
    texts = [item.stance for item in music]
    assert "我不喜欢嘈杂的现场。" in texts
    assert "我不喜欢临时改计划。" not in texts
    # Curated admin principles are character, not noisy history.
    assert "我更喜欢先听完再评价。" in texts

    debugger = filter_relevant_stances(
        stances, topic=None, query_text="这个 bug 怎么查？"
    )
    debugger_texts = [item.stance for item in debugger]
    assert debugger_texts == ["我更喜欢先听完再评价。"]


def test_stance_filter_is_bounded_and_accepts_mappings():
    stances = [
        {"topic": "音乐", "stance": f"我不喜欢第{i}种音乐风格。", "source": "RESPONSE"}
        for i in range(20)
    ]
    kept = filter_relevant_stances(
        stances, topic="音乐", query_text="音乐风格", limit=50
    )
    assert len(kept) == 8


def test_emotional_dispositions_are_validated_and_bounded():
    assert parse_emotional_dispositions(None) == DEFAULT_EMOTIONAL_DISPOSITIONS

    parsed = parse_emotional_dispositions(
        {
            "tolerance_for_teasing": 2.0,
            "conflict_sensitivity": -1,
            "reconciliation_openness": "0.75",
            "expressiveness": "not-a-number",
            "unknown": 1.0,
        }
    )
    assert parsed["tolerance_for_teasing"] == 1.0
    assert parsed["conflict_sensitivity"] == 0.0
    assert parsed["reconciliation_openness"] == 0.75
    assert parsed["expressiveness"] == 0.5
    assert "unknown" not in parsed

    # JSON strings are accepted; malformed input falls back safely.
    from_json = parse_emotional_dispositions('{"expressiveness": 0.25}')
    assert from_json["expressiveness"] == 0.25
    assert parse_emotional_dispositions("{not json")["expressiveness"] == 0.5


async def _runtime_with_recorders(db):
    cognitive = RecordingCognitive()
    response = RecordingResponse()
    runtime = build_runtime(
        db,
        with_worker=False,
        provider="deepseek",
        deepseek_client=object(),
        cognitive=cognitive,
        response=response,
        allow_mixed_providers=True,
        fast_path_enabled=False,
        router_mode="legacy",
)
    return runtime, cognitive, response


async def _register(rt, *, persistence=True):
    return await rt.registry.register(
        "Ada",
        "owner",
        allow_processing=True,
        allow_persistence=persistence,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _event(person_id, text):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def test_cognitive_and_response_receive_the_same_snapshot(db):
    rt, cognitive, response = await _runtime_with_recorders(db)
    person = await _register(rt)
    await rt.agent_service.set_persona_core(
        {
            "character_description": "对熟人话多，对生人先观察。",
            "core_values": ["诚实"],
        },
        actor="admin",
    )

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "你好")
    )
    assert report.status == "COMPLETED"

    cognitive_snapshot = cognitive.contexts[0].character_snapshot
    response_snapshot = response.contexts[0].character_snapshot
    assert cognitive_snapshot is not None
    assert cognitive_snapshot == response_snapshot
    assert cognitive_snapshot["revision"] == 2
    assert (
        cognitive_snapshot["character_description"]
        == "对熟人话多，对生人先观察。"
    )


async def test_non_persistence_turn_still_gets_a_snapshot(db):
    rt, cognitive, response = await _runtime_with_recorders(db)
    person = await _register(rt, persistence=False)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "只聊这一次")
    )
    assert report.status == "COMPLETED"
    assert cognitive.contexts[0].character_snapshot is not None
    assert cognitive.contexts[0].character_snapshot == (
        response.contexts[0].character_snapshot
    )


async def test_admin_dispositions_are_loaded_from_trusted_settings(db):
    import json

    rt, cognitive, response = await _runtime_with_recorders(db)
    person = await _register(rt)
    await rt.repos.settings.set(
        "persona.emotional_dispositions",
        json.dumps({"tolerance_for_teasing": 0.9, "expressiveness": 0.2}),
    )

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "你好")
    )
    assert report.status == "COMPLETED"
    dispositions = cognitive.contexts[0].character_snapshot[
        "emotional_dispositions"
    ]
    assert dispositions["tolerance_for_teasing"] == 0.9
    assert dispositions["expressiveness"] == 0.2
    assert dispositions == response.contexts[0].character_snapshot[
        "emotional_dispositions"
    ]
