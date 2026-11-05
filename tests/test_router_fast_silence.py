"""Fast silent path: a non-communicative turn never assembles a speaker.

The event is still committed atomically (analysis, action, trace, conversation
and event status). What is skipped is exactly the work that could only produce
a reply: speaker context, persona/emotion compilation, provider calls and
background learning. Normal communicative WAIT/IGNORE keeps the existing path.
"""

from __future__ import annotations

import pytest

from companion.core import orchestrator as orchestrator_module
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ProcessingStatus,
    ReportStatus,
)
from companion.domain.models import make_utterance_event
from companion.routing.models import RawRouteDecision
from companion.runtime import build_runtime


class FakeRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self, *, action="IGNORE", necessity="UNNECESSARY",
                 input_semantics="NON_COMMUNICATIVE"):
        self.calls = 0
        self._action = action
        self._necessity = necessity
        self._input_semantics = input_semantics

    async def route(self, context):
        self.calls += 1
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": self._action,
                "reply_necessity": self._necessity,
                "addressee_assessment": "UNKNOWN",
                "input_semantics": self._input_semantics,
                "turn_state": "CONTINUING",
                "tool_requests": [],
                "reason": "fast silence test",
            }
        )

    async def replan(self, context, previous, results):
        raise AssertionError("silent turns must not replan")


class ExplodingCognitive:
    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call the cognitive provider")


class RecordingResponse:
    def __init__(self):
        self.calls = 0

    async def generate(self, event, analysis, context):
        self.calls += 1
        return "should not happen"


@pytest.fixture
def semantic_runtime(db, monkeypatch):
    def _build(router):
        response = RecordingResponse()
        rt = build_runtime(
            db,
            provider="deepseek",
            router_mode="semantic",
            router_provider=router,
            deepseek_client=object(),
            cognitive=ExplodingCognitive(),
            response=response,
            allow_mixed_providers=True,
            fast_path_enabled=False,
        )
        return rt, response

    def _fail(*args, **kwargs):
        raise AssertionError("legacy text routing must not run in semantic mode")

    for name in (
        "assess_turn",
        "decide_route",
        "detect_explicit_silence",
        "detect_explicit_remember",
    ):
        monkeypatch.setattr(orchestrator_module, name, _fail)
    return _build


async def _person(rt, name="Lucas"):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _event(person_id, text="???", *, conversation="conv", event_id=None):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


def _forbid(monkeypatch, obj, name, message):
    if obj is None:
        return
    async def forbidden(*args, **kwargs):
        raise AssertionError(message)

    monkeypatch.setattr(obj, name, forbidden)


@pytest.mark.asyncio
async def test_silent_turn_skips_all_response_and_context_work(
    semantic_runtime, monkeypatch
):
    router = FakeRouter()
    rt, response = semantic_runtime(router)
    person = await _person(rt)
    monkeypatch.setattr(
        rt.orchestrator,
        "_prepare_dialogue",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("persona/scene compiled for silence")
        ),
    )
    _forbid(
        monkeypatch,
        rt.orchestrator,
        "_apply_emotion",
        "emotion appraised for silence",
    )
    _forbid(
        monkeypatch,
        rt.context_builder,
        "build_from_authorized_turns",
        "speaker context built for silence",
    )
    _forbid(
        monkeypatch,
        getattr(rt.orchestrator, "profile_learning", None),
        "observe",
        "profile learned from non-communicative input",
    )
    _forbid(
        monkeypatch,
        getattr(rt.scene_manager, "preview", None) and rt.scene_manager,
        "finalize",
        "scene advanced for silence",
    )

    event = _event(person.person_id)
    report = await rt.orchestrator.process_utterance(event)

    assert router.calls == 1
    assert response.calls == 0
    assert report.status == ReportStatus.COMPLETED.value
    assert report.action == CognitiveAction.IGNORE
    assert report.response_text is None

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.action_type == CognitiveAction.IGNORE
    assert action.response_text is None
    assert action.nonresponse_trace["cause_code"] == "NO_REPLY_NEEDED"
    assert action.nonresponse_trace["cause_source"] == "LOCAL_POLICY"
    assert action.nonresponse_trace["policy_override"] in {
        "non_communicative_input",
        "semantic_noncommunicative",
    }
    assert action.nonresponse_trace["spoken"] is False
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status == ProcessingStatus.COMPLETED
    assert await rt.repos.analysis.get_analysis(event.event_id) is not None
    assert (
        await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    ) is None


@pytest.mark.asyncio
async def test_communicative_wait_still_builds_context_but_never_speaks(
    semantic_runtime, monkeypatch
):
    router = FakeRouter(
        action="WAIT", necessity="OPTIONAL", input_semantics="COMMUNICATIVE"
    )
    rt, response = semantic_runtime(router)
    person = await _person(rt)
    built = []
    original = rt.context_builder.build_from_authorized_turns

    async def counted(*args, **kwargs):
        built.append(1)
        return await original(*args, **kwargs)

    monkeypatch.setattr(
        rt.context_builder, "build_from_authorized_turns", counted
    )

    event = _event(person.person_id, conversation="conv-wait")
    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.WAIT
    assert report.status == ReportStatus.COMPLETED.value
    assert built == [1]
    assert response.calls == 0
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.action_type == CognitiveAction.WAIT


@pytest.mark.asyncio
async def test_silent_turn_retry_is_idempotent(semantic_runtime):
    router = FakeRouter()
    rt, response = semantic_runtime(router)
    person = await _person(rt)
    event = _event(person.person_id, conversation="conv-retry")

    first = await rt.orchestrator.process_utterance(event)
    second = await rt.orchestrator.process_utterance(event)

    assert first.status == ReportStatus.COMPLETED.value
    assert second.status == ReportStatus.DUPLICATE.value
    assert second.action == CognitiveAction.IGNORE
    assert router.calls == 1
    assert response.calls == 0
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS n FROM agent_actions WHERE event_id = ?",
        (event.event_id,),
    )
    assert (await cur.fetchone())["n"] == 1


@pytest.mark.asyncio
async def test_uncertain_is_not_treated_as_silence(semantic_runtime):
    router = FakeRouter(
        action="WAIT", necessity="OPTIONAL", input_semantics="UNCERTAIN"
    )
    rt, response = semantic_runtime(router)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, conversation="conv-uncertain")
    )

    # An uncertain turn is not silently discarded; it still goes through the
    # normal decision path. The router suggested WAIT, so no reply is produced.
    assert report.status == ReportStatus.COMPLETED.value
    assert report.action == CognitiveAction.WAIT
    assert response.calls == 0
