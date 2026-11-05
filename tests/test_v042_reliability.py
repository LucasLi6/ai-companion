"""V0.4.2 Step 6 — reliability, safety floor and memory/emotion separation."""

from __future__ import annotations

from companion.core.safety import (
    DANGER_TO_OTHERS,
    SELF_HARM,
    assess_safety,
    detect_language,
    safety_response,
)
from companion.domain.enums import (
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    EventSource,
    MemoryKind,
    MemoryOperation,
    ReasonCode,
)
from companion.domain.models import (
    CognitiveResult,
    DialogueImpulse,
    MemoryCandidate,
    PragmaticAnalysis,
    make_utterance_event,
    new_id,
)
from companion.memory.consolidation import MemoryWriter
from companion.memory.transience import is_temporary_emotion_statement
from companion.runtime import build_runtime
from companion.storage.schema import SCHEMA_VERSION


# -- schema ---------------------------------------------------------------


async def test_schema_16_carries_pragmatic_and_stance_storage(db):
    assert SCHEMA_VERSION == 24

    cur = await db.execute("PRAGMA table_info(cognitive_analyses)")
    columns = {row["name"] for row in await cur.fetchall()}
    assert {
        "pragmatic_json",
        "dialogue_impulse_json",
        "persona_agent_id",
        "persona_revision",
        "persona_snapshot_json",
    } <= columns

    cur = await db.execute("PRAGMA table_info(agent_persona_core)")
    persona_columns = {row["name"] for row in await cur.fetchall()}
    assert "character_description" in persona_columns

    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_stances'"
    )
    assert await cur.fetchone() is not None


# -- analysis recovery ----------------------------------------------------


async def test_pragmatic_and_impulse_survive_analysis_roundtrip(db):
    from companion.storage.repositories import make_repositories

    repos = make_repositories(db)
    await repos.conversation.get_or_create("c_round")
    event = make_utterance_event(
        conversation_id="c_round",
        text="你又来了",
        addressee_hint="agent",
        source=EventSource.TEST,
        event_id="evt_round",
    )
    await repos.utterance.save(event)
    analysis = CognitiveResult(
        event_id="evt_round",
        meaning="对方在接上一个玩笑",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        response_style="short",
        provider="fake",
        dialogue_impulse=DialogueImpulse(
            interpretation="接梗",
            response_direction="简短接住",
            stance="不解释笑点",
            unresolved_uncertainty="是否继续玩笑",
        ),
        pragmatic=PragmaticAnalysis(
            primary_act="TEASING",
            secondary_acts=["INFORMATION"],
            interaction_frame="PLAYFUL",
            refers_to_previous_turn=True,
            antecedent_event_id="evt_prev",
            setup_payoff_role="PAYOFF",
            literal_interpretation="字面一句吐槽",
            implied_interpretation="在延续上次的玩笑",
            topic_transition="CONTINUE",
            confidence=0.8,
            evidence_span="你又来了",
        ),
    )
    await repos.analysis.save_analysis(analysis, new_id("ana"))
    loaded = await repos.analysis.get_analysis("evt_round")

    assert loaded is not None
    assert loaded.dialogue_impulse == analysis.dialogue_impulse
    assert loaded.pragmatic == analysis.pragmatic


# -- safety floor ---------------------------------------------------------


class _ExplodingResponse:
    """Fails the test if the model response path is used."""

    async def generate(self, *args, **kwargs):
        raise AssertionError("model response must not run for a safety turn")


async def test_high_risk_statement_uses_local_reply_without_model(db):
    rt = build_runtime(db, response=_ExplodingResponse())
    person = await rt.registry.register(
        "Lin", "friend", allow_processing=True, allow_persistence=True
    )

    event = make_utterance_event(
        conversation_id="safe",
        text="我现在不想活了，想结束自己的生命",
        addressee_hint="unknown",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.RESPOND
    assert report.reason_code == ReasonCode.SAFETY_RESPONSE
    assert report.response_text == safety_response(SELF_HARM, "zh")

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None
    assert action.reason_code == ReasonCode.SAFETY_RESPONSE
    assert action.response_text == safety_response(SELF_HARM, "zh")
    # Crisis content is not turned into memory candidates.
    assert await rt.repos.analysis.get_candidates(event.event_id) == []

    # A retry reuses the stored action; it does not generate or duplicate it.
    duplicate = await rt.orchestrator.process_utterance(event)
    assert duplicate.status == "DUPLICATE"


async def test_safety_floor_overrides_quiet_mode_and_model_wait(db):
    rt = build_runtime(db, response=_ExplodingResponse())
    person = await rt.registry.register(
        "Lin", "friend", allow_processing=True, allow_persistence=True
    )
    await rt.conversations.get_or_create("safe")
    await rt.conversations.set_quiet_mode("safe", True)

    event = make_utterance_event(
        conversation_id="safe",
        text="我想自杀",
        addressee_hint="unknown",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.RESPOND
    assert report.reason_code == ReasonCode.SAFETY_RESPONSE


async def test_safety_floor_respects_explicit_human_addressee(db):
    rt = build_runtime(db, response=_ExplodingResponse())
    person = await rt.registry.register(
        "Lin", "friend", allow_processing=True, allow_persistence=True
    )

    event = make_utterance_event(
        conversation_id="safe",
        text="我想自杀",
        addressee_hint="human",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.IGNORE
    assert report.response_text is None


def test_safety_detection_ignores_idioms_and_catches_explicit_language():
    assert not assess_safety("笑死我了，你也太逗了").triggered
    assert not assess_safety("我想死你了").triggered
    assert not assess_safety("今天忙死了").triggered

    self_harm = assess_safety("我不想活了")
    assert self_harm.triggered and self_harm.category == SELF_HARM

    english = assess_safety("I want to kill myself")
    assert english.triggered and english.category == SELF_HARM

    danger = assess_safety("我真的想杀了他")
    assert danger.triggered and danger.category == DANGER_TO_OTHERS


def test_safety_reply_language_matches_the_utterance():
    assert detect_language("我不想活了") == "zh"
    assert detect_language("I want to die") == "en"
    assert safety_response(SELF_HARM, "zh").startswith("我听到了")
    assert safety_response(SELF_HARM, "en").startswith("I hear you")


# -- memory vs temporary emotion -----------------------------------------


def test_temporary_emotion_filter_is_narrow():
    assert is_temporary_emotion_statement("我现在很焦虑")
    assert is_temporary_emotion_statement("我今天心情很差")
    assert is_temporary_emotion_statement("我最近很累")
    # Preferences and durative traits are durable.
    assert not is_temporary_emotion_statement("我喜欢安静")
    assert not is_temporary_emotion_statement("我一直很容易焦虑")
    assert not is_temporary_emotion_statement("我平时很累")


async def test_transient_emotion_candidate_is_not_written_as_a_fact(db):
    rt = build_runtime(db)
    writer = MemoryWriter(rt.repos, rt.memory_service)
    person = await rt.registry.register(
        "Lin",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    await rt.repos.conversation.get_or_create("c1")
    event = make_utterance_event(
        conversation_id="c1",
        text="我现在很焦虑",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    candidate = MemoryCandidate(
        candidate_id=new_id("memc"),
        source_event_id=event.event_id,
        subject_person_id=person.person_id,
        content="我现在很焦虑",
        operation=MemoryOperation.ADD,
        assertion_type=AssertionType.SELF_REPORT,
        certainty_category=CertaintyCategory.DIRECT,
        kind=MemoryKind.PERSONAL_FACT,
        topic="mood",
        source_evidence_span="我现在很焦虑",
    )
    await rt.repos.utterance.save(event)
    await rt.repos.analysis.save_candidate(candidate)

    outcome = await writer.consolidate_candidate(candidate, event)

    assert outcome["outcome"] == "NO_MEMORY"
    assert outcome["detail"] == "temporary_emotion_not_durable"
    assert outcome["memory_id"] is None
