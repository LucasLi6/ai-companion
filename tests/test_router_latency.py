"""Latency structure: one router call, one speaker call, observable stages.

These tests use fake providers with controlled waits. They verify the shape of
the pipeline and the presence of independent stage measurements — never a real
network latency claim.
"""

from __future__ import annotations

import asyncio

import pytest

from companion.core import orchestrator as orchestrator_module
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ReplyNecessity,
    ReportStatus,
)
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import StreamDelta
from companion.routing.models import (
    RawRouteDecision,
    RouteAction,
    RouteAddressee,
    RoutePhase,
    ValidatedRouteDecision,
)
from companion.runtime import build_runtime


class TimedRouter:
    provider_name = "timed_router"
    model = "timed-model"

    def __init__(self, *, action="RESPOND", necessity="REQUIRED",
                 input_semantics="COMMUNICATIVE", delay=0.01):
        self.calls = 0
        self.replans = 0
        self._action = action
        self._necessity = necessity
        self._input_semantics = input_semantics
        self._delay = delay

    def _raw(self):
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": self._action,
                "reply_necessity": self._necessity,
                "addressee_assessment": "AGENT",
                "input_semantics": self._input_semantics,
                "turn_state": "CONTINUING",
                "tool_requests": [],
                "reason": "timed semantic decision",
            }
        )

    async def route(self, context):
        self.calls += 1
        await asyncio.sleep(self._delay)
        return self._raw()

    async def replan(self, context, previous, results):
        self.replans += 1
        return self._raw()


class ExplodingCognitive:
    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call the cognitive provider")


class SlowResponse:
    def __init__(self, delay=0.01):
        self.calls = 0
        self._delay = delay

    async def generate(self, event, analysis, context):
        self.calls += 1
        await asyncio.sleep(self._delay)
        return "timed reply"


class StreamingResponse:
    def __init__(self):
        self.calls = 0

    async def generate_stream(self, event, analysis, context):
        self.calls += 1
        await asyncio.sleep(0.02)
        yield StreamDelta(content="你")
        await asyncio.sleep(0.01)
        yield StreamDelta(content="好", finish_reason="stop")


@pytest.fixture
def semantic_runtime(db, monkeypatch):
    def _build(router, *, response=None):
        response = response or SlowResponse()
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


def _event(person_id, text="hello", *, conversation="conv", event_id=None):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


@pytest.mark.asyncio
async def test_ordinary_reply_is_one_router_and_one_speaker_call(semantic_runtime):
    router = TimedRouter()
    rt, response = semantic_runtime(router)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(_event(person.person_id))

    assert report.status == ReportStatus.COMPLETED.value
    assert report.action == CognitiveAction.RESPOND
    assert router.calls == 1
    assert router.replans == 0
    assert response.calls == 1
    assert report.timings["router_llm_ms"] >= 5
    assert report.timings["total_ms"] >= report.timings["router_llm_ms"]


@pytest.mark.asyncio
async def test_no_tool_request_means_no_followup_and_no_tool_executor(
    semantic_runtime,
):
    router = TimedRouter()
    # Even if the model claims follow-up planning is useful, without an
    # executed tool there is nothing new to plan from.
    original = router._raw

    def raw_with_flag():
        model = original()
        return model.model_copy(update={"needs_followup_planning": True})

    router._raw = raw_with_flag
    rt, response = semantic_runtime(router)
    person = await _person(rt)
    executed = []

    async def record_execute(*args, **kwargs):
        executed.append(1)
        raise AssertionError("no tool should execute without a tool request")

    rt.routing_service.executor.execute_one = record_execute

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, conversation="conv-no-tools")
    )

    assert report.action == CognitiveAction.RESPOND
    assert router.calls == 1
    assert router.replans == 0
    assert executed == []
    assert response.calls == 1


@pytest.mark.asyncio
async def test_each_stage_is_recorded_independently(semantic_runtime):
    router = TimedRouter()
    rt, _ = semantic_runtime(router)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, conversation="conv-stages")
    )

    for stage in (
        "routing_context_ms",
        "router_llm_ms",
        "router_followup_ms",
        "tool_execution_ms",
        "speaker_context_ms",
        "policy_ms",
        "generation_ms",
        "commit_ms",
        "total_ms",
    ):
        assert stage in report.timings, stage
        assert report.timings[stage] >= 0
    assert report.timings["router_followup_ms"] == 0.0
    assert report.timings["tool_execution_ms"] == 0.0


@pytest.mark.asyncio
async def test_speaker_first_token_is_measured_on_the_stream(semantic_runtime):
    router = TimedRouter()
    response = StreamingResponse()
    rt, _ = semantic_runtime(router, response=response)
    person = await _person(rt)
    event = _event(person.person_id, conversation="conv-stream")

    events = [item async for item in rt.orchestrator.process_utterance_stream(event)]

    assert response.calls == 1
    assert [item.type for item in events] == [
        "response_started",
        "response_delta",
        "response_delta",
        "response_completed",
    ]
    snapshot = rt.metrics.snapshot()
    assert snapshot["speaker_first_token_ms"]["count"] == 1
    assert snapshot["speaker_first_token_ms"]["max_ms"] >= 15


@pytest.mark.asyncio
async def test_a_restored_route_costs_zero_router_model_time(semantic_runtime, db):
    router = TimedRouter(delay=0.05)
    rt, response = semantic_runtime(router)
    person = await _person(rt)
    event = _event(person.person_id, conversation="conv-restore")

    await rt.conversations.get_or_create(event.conversation_id)
    await rt.repos.utterance.save(event)
    route = ValidatedRouteDecision(
        action=RouteAction.RESPOND,
        reply_necessity=ReplyNecessity.USEFUL,
        addressee_assessment=RouteAddressee.AGENT,
        input_semantics="COMMUNICATIVE",
        reason="already committed by an earlier attempt",
    )
    await rt.routing_service.repository.save_decision(
        event_id=event.event_id,
        phase=RoutePhase.INITIAL,
        step_index=0,
        route=route,
    )

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == ReportStatus.COMPLETED.value
    assert router.calls == 0
    assert response.calls == 1
    assert report.timings["router_llm_ms"] == 0.0
    assert rt.metrics.snapshot()["router_llm_ms"]["max_ms"] == 0.0


@pytest.mark.asyncio
async def test_noncommunicative_silence_skips_speaker_context_work(
    semantic_runtime, monkeypatch
):
    router = TimedRouter(
        action="IGNORE",
        necessity="UNNECESSARY",
        input_semantics="NON_COMMUNICATIVE",
    )
    rt, response = semantic_runtime(router)
    person = await _person(rt)

    async def forbidden(*args, **kwargs):
        raise AssertionError("silent non-communicative turn built a speaker context")

    monkeypatch.setattr(
        rt.context_builder, "build_from_authorized_turns", forbidden
    )

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, conversation="conv-silent")
    )

    assert report.action == CognitiveAction.IGNORE
    assert response.calls == 0
    assert report.timings["router_llm_ms"] >= 5
    assert report.timings["speaker_context_ms"] == 0.0
    assert "generation_ms" not in report.timings
    assert report.timings["total_ms"] >= report.timings["router_llm_ms"]
