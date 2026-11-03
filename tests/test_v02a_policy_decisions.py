"""V0.2A — structured cognitive decisions and the stronger Policy layer.

All providers are fakes. The tests pin the local rules that outrank the model:

- an explicit current-turn "don't reply" produces IGNORE with *no* response,
  while analysis and memory learning still happen;
- quoted or discussed silence is not a request and never suppresses a reply;
- an explicit human addressee is always IGNORE;
- quiet mode turns RESPOND into WAIT;
- an ordinary question is still answered;
- unaddressed suggestions clear the configured conservatism bar;
- the model's suggestion and the policy's final action are stored separately.
"""

from __future__ import annotations

import pytest

from companion.cognition.silence import (
    detect_explicit_remember,
    detect_explicit_silence,
)
from companion.core.policy import (
    SETTING_CURIOSITY_LEVEL,
    SETTING_INTERRUPTION_THRESHOLD,
    SETTING_RESPONSE_CONSERVATISM,
)
from companion.domain.enums import (
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    EventSource,
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
from companion.runtime import build_runtime


class ScriptedCognitive:
    def __init__(self, build):
        self.build = build
        self.calls = 0

    async def analyze(self, event, context, memories) -> CognitiveResult:
        self.calls += 1
        return self.build(event)


class RecordingResponse:
    def __init__(self, text="我在听。"):
        self.text = text
        self.calls: list[str] = []

    async def generate(self, event, analysis, context) -> str:
        self.calls.append(event.event_id)
        return self.text


def result(
    event,
    *,
    action: CognitiveAction = CognitiveAction.RESPOND,
    reason: ReasonCode = ReasonCode.DIRECT_AGENT_ADDRESS,
    necessity: ReplyNecessity = ReplyNecessity.REQUIRED,
    addressed_to: str | None = "agent",
    candidates=(),
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
        intent="question",
        addressed_to=addressed_to,
        silence_requested=False,
        reply_necessity=necessity,
        conversation_state="ongoing",
    )


async def _person(rt, name="Alice"):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
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


# -- detector units -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "这句话不要回复。",
        "不要回答我这条消息。",
        "接下来这句不用回复。",
        "Don't respond to this message.",
        "No need to reply.",
    ],
)
def test_direct_silence_requests_are_detected(text):
    assert detect_explicit_silence(text) is True


@pytest.mark.parametrize(
    "text",
    [
        '我的朋友刚才说："不要回答我"',
        "我的朋友刚才说：不要回答我",
        "为什么 AI 有时候不回答？",
        'She said "Don\'t respond to this message."',
        "我的名字是李约。",
    ],
)
def test_quoted_or_discussed_silence_is_not_a_request(text):
    assert detect_explicit_silence(text) is False


@pytest.mark.parametrize(
    "text", ["请记住：我叫李悦。", "Remember that I like tea.", "别忘了我不吃辣"]
)
def test_direct_remember_requests_are_detected(text):
    assert detect_explicit_remember(text) is True


@pytest.mark.parametrize(
    "text", ['他说："记住这个"', "我不记得了", "我的名字是李约。"]
)
def test_quoted_or_incidental_remember_is_not_a_request(text):
    assert detect_explicit_remember(text) is False


# -- J. explicit silence ------------------------------------------------------


async def test_j_explicit_silence_ignores_but_still_learns(db):
    response = RecordingResponse()
    cognitive = ScriptedCognitive(
        lambda event: result(
            event,
            candidates=[
                MemoryCandidate(
                    candidate_id=new_id("memc"),
                    source_event_id=event.event_id,
                    content="我叫李悦",
                    operation="ADD",
                    assertion_type=AssertionType.SELF_REPORT,
                    certainty_category=CertaintyCategory.DIRECT,
                    topic="name",
                    source_evidence_span="我叫李悦",
                )
            ],
        )
    )
    rt = build_runtime(db, cognitive=cognitive, response=response)
    person = await _person(rt)
    event = _event(person.person_id, "我叫李悦。这句话不要回复。")

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.IGNORE
    assert report.reason_code == ReasonCode.EXPLICIT_SILENCE_REQUEST
    assert report.response_text is None
    assert response.calls == []  # the response provider was never called
    assert cognitive.calls == 1  # analysis still happened

    # Analysis is durable, and the model's suggestion is preserved unedited.
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None
    assert analysis.action == CognitiveAction.RESPOND
    assert analysis.reply_necessity == ReplyNecessity.REQUIRED

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.action_type == CognitiveAction.IGNORE
    assert action.model_suggested_action == CognitiveAction.RESPOND
    assert action.policy_final_action == CognitiveAction.IGNORE
    assert action.override_reason == "explicit_silence_request"
    assert action.response_text is None

    # Silence suppresses the reply, not the memory write.
    await rt.worker.run_once()
    cur = await db.execute("SELECT status FROM memory_items")
    row = await cur.fetchone()
    assert row is not None and row["status"] == MemoryStatus.ACTIVE.value


# -- K. quoted / discussed silence -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '我的朋友刚才说："不要回答我"',
        "我的朋友刚才说：不要回答我",
        "为什么 AI 有时候不回答？",
        'She said "Don\'t respond to this message."',
    ],
)
async def test_k_quoted_silence_does_not_suppress_an_answer(db, text):
    response = RecordingResponse("这是我对你的回答。")
    rt = build_runtime(
        db,
        cognitive=ScriptedCognitive(lambda event: result(event)),
        response=response,
    )
    person = await _person(rt)
    event = _event(person.person_id, text)

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.RESPOND
    assert report.response_text == "这是我对你的回答。"
    assert response.calls == [event.event_id]


# -- L. human addressee -------------------------------------------------------


async def test_l_explicit_human_addressee_is_ignore(db):
    response = RecordingResponse()
    rt = build_runtime(
        db,
        cognitive=ScriptedCognitive(
            lambda event: result(
                event,
                reason=ReasonCode.DIRECT_HUMAN_ADDRESS,
            )
        ),
        response=response,
    )
    person = await _person(rt)
    event = _event(person.person_id, "你能帮我个忙吗？", to="human")

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.IGNORE
    assert report.reason_code == ReasonCode.DIRECT_HUMAN_ADDRESS
    assert response.calls == []

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.model_suggested_action == CognitiveAction.RESPOND
    assert action.policy_final_action == CognitiveAction.IGNORE
    assert action.override_reason == "explicit_human_addressee"


# -- M/N. quiet mode and an ordinary question --------------------------------


async def test_m_quiet_mode_downgrades_a_response(db):
    response = RecordingResponse()
    rt = build_runtime(
        db,
        cognitive=ScriptedCognitive(lambda event: result(event)),
        response=response,
    )
    person = await _person(rt)
    await rt.conversations.get_or_create("quiet")
    await rt.conversations.set_quiet_mode("quiet", True)
    event = _event(person.person_id, "你好，今天怎么样？", conversation="quiet")

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.WAIT
    assert report.reason_code == ReasonCode.QUIET_MODE
    assert response.calls == []

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.model_suggested_action == CognitiveAction.RESPOND
    assert action.policy_final_action == CognitiveAction.WAIT
    assert action.override_reason == "quiet_mode"


async def test_n_normal_question_is_answered(db):
    response = RecordingResponse("当然可以。")
    rt = build_runtime(
        db,
        cognitive=ScriptedCognitive(
            lambda event: result(event, necessity=ReplyNecessity.REQUIRED)
        ),
        response=response,
    )
    person = await _person(rt)
    event = _event(person.person_id, "你能帮我个忙吗？")

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.RESPOND
    assert report.response_text == "当然可以。"
    assert response.calls == [event.event_id]

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.model_suggested_action == CognitiveAction.RESPOND
    assert action.override_reason is None


# -- typed policy configuration ----------------------------------------------


async def test_unaddressed_responses_follow_the_configured_conservatism(db):
    response = RecordingResponse()
    rt = build_runtime(
        db,
        cognitive=ScriptedCognitive(
            lambda event: result(
                event,
                necessity=ReplyNecessity.USEFUL,
                addressed_to=None,
            )
        ),
        response=response,
    )
    person = await _person(rt)

    # Default is balanced: a USEFUL unaddressed suggestion may interject.
    useful = _event(person.person_id, "天气不错。", conversation="a", to="unknown")
    assert (
        await rt.orchestrator.process_utterance(useful)
    ).action == CognitiveAction.RESPOND

    # OPTIONAL does not clear the balanced bar.
    rt.cognitive.build = lambda event: result(
        event, necessity=ReplyNecessity.OPTIONAL, addressed_to=None
    )
    optional = _event(person.person_id, "嗯。", conversation="b", to="unknown")
    report = await rt.orchestrator.process_utterance(optional)
    assert report.action == CognitiveAction.WAIT
    assert report.reason_code == ReasonCode.CONSERVATIVE_AMBIGUITY

    # A stored "conservative" setting is picked up on the next utterance.
    await rt.repos.settings.set(SETTING_RESPONSE_CONSERVATISM, "conservative")
    rt.cognitive.build = lambda event: result(
        event, necessity=ReplyNecessity.USEFUL, addressed_to=None
    )
    strict = _event(person.person_id, "今天挺热的。", conversation="c", to="unknown")
    report = await rt.orchestrator.process_utterance(strict)
    assert report.action == CognitiveAction.WAIT
    assert report.reason_code == ReasonCode.CONSERVATIVE_AMBIGUITY
    assert response.calls == [useful.event_id]

    action = await rt.repos.analysis.get_action(strict.event_id)
    assert action.model_suggested_action == CognitiveAction.RESPOND
    assert action.policy_final_action == CognitiveAction.WAIT


async def test_policy_settings_are_typed_and_fall_back_on_garbage(db):
    from companion.core.policy import load_policy_config

    settings = make_settings(db)
    assert (await load_policy_config(settings)).response_conservatism == "balanced"

    await settings.set(SETTING_RESPONSE_CONSERVATISM, "nonsense")
    await settings.set(SETTING_INTERRUPTION_THRESHOLD, "medium")
    await settings.set(SETTING_CURIOSITY_LEVEL, "high")
    config = await load_policy_config(settings)
    assert config.response_conservatism == "balanced"  # default, never a crash
    assert config.curiosity_level == "high"


def make_settings(db):
    from companion.storage.repositories import SettingsRepository

    return SettingsRepository(db)
