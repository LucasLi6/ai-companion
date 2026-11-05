"""End-to-end semantic pipeline: router first, no legacy text routing."""

from __future__ import annotations

import pytest

from companion.core import orchestrator as orchestrator_module
from companion.domain.enums import CognitiveAction, EventSource, ProcessingStatus, ReportStatus
from companion.domain.models import make_utterance_event
from companion.providers.mock import MockCognitiveProvider, MockResponseProvider
from companion.routing.errors import RouterTimeout
from companion.routing.models import RawRouteDecision
from companion.runtime import build_runtime


class FakeRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self, *, action="RESPOND", necessity="REQUIRED", addressee="AGENT",
                 tools=None, followup_tools=None, needs_followup=False,
                 raise_error=None):
        self.calls = 0
        self.replans = 0
        self._action = action
        self._necessity = necessity
        self._addressee = addressee
        self._tools = tools or []
        self._followup_tools = followup_tools or []
        self._needs_followup = needs_followup
        self._raise_error = raise_error

    def _raw(self, tools):
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": self._action,
                "reply_necessity": self._necessity,
                "addressee_assessment": self._addressee,
                "turn_state": "CONTINUING",
                "tool_requests": tools,
                "needs_followup_planning": self._needs_followup,
                "reason": "semantic test decision",
            }
        )

    async def route(self, context):
        self.calls += 1
        if self._raise_error is not None:
            raise self._raise_error
        return self._raw(self._tools)

    async def replan(self, context, previous, results):
        self.replans += 1
        return self._raw(self._followup_tools)


class ExplodingCognitive:
    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call the cognitive provider")


class RecordingResponse:
    def __init__(self):
        self.calls = []

    async def generate(self, event, analysis, context):
        self.calls.append((event, analysis, context))
        return "semantic reply"


@pytest.fixture
def semantic_runtime(db, monkeypatch):
    built = {}

    def _build(router, *, provider="mock", response=None):
        response = response or RecordingResponse()
        kwargs = dict(
            db=db,
            provider=provider,
            router_mode="semantic",
            router_provider=router,
            response=response,
        )
        if provider == "deepseek":
            kwargs.update(
                deepseek_client=object(),
                cognitive=ExplodingCognitive(),
                allow_mixed_providers=True,
            )
        rt = build_runtime(**kwargs)
        built["rt"] = rt
        built["response"] = response
        built["router"] = router
        return rt

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


async def _say(rt, person_id, text, *, to="agent", conversation="conv", event_id=None):
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )
    return event, await rt.orchestrator.process_utterance(event)


@pytest.mark.asyncio
async def test_semantic_respond_calls_router_then_speaker(semantic_runtime):
    router = FakeRouter()
    rt = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_persistence=True, allow_memory_learning=True, allow_external_ai=True
    )
    event, report = await _say(rt, person.person_id, "hello there")

    assert report.status == ReportStatus.COMPLETED.value
    assert report.action == CognitiveAction.RESPOND
    assert router.calls == 1
    assert len(rt.response.calls) == 1
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.action_type == CognitiveAction.RESPOND
    assert action.model_suggested_action == CognitiveAction.RESPOND
    decisions = await rt.repos.routing.list_decisions(event.event_id)
    assert decisions and decisions[0]["validation_status"] == "VALID"
    assert decisions[0]["provider"] == "fake_router"


@pytest.mark.asyncio
async def test_wait_and_ignore_never_call_the_speaker(semantic_runtime):
    for action in ("WAIT", "IGNORE"):
        router = FakeRouter(
            action=action, necessity="UNNECESSARY", addressee="UNKNOWN"
        )
        rt = semantic_runtime(router)
        person = await rt.registry.register(
            f"P-{action}", "friend", allow_processing=True, allow_persistence=True, allow_memory_learning=True, allow_external_ai=True
        )
        event, report = await _say(
            rt, person.person_id, "side remark", conversation=f"c-{action}"
        )
        assert report.status == ReportStatus.COMPLETED.value
        assert report.action == CognitiveAction(action)
        assert rt.response.calls == []
        stored = await rt.repos.analysis.get_action(event.event_id)
        assert stored.action_type == CognitiveAction(action)


@pytest.mark.asyncio
async def test_tool_results_and_route_reach_the_speaker_context(semantic_runtime):
    router = FakeRouter(
        tools=[
            {
                "tool_name": "runtime.current_time",
                "arguments": {},
                "purpose": "answer a time question",
                "required": True,
            }
        ]
    )
    rt = semantic_runtime(router, provider="deepseek")
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_persistence=True, allow_memory_learning=True, allow_external_ai=True
    )
    _, report = await _say(rt, person.person_id, "what time is it")

    assert report.action == CognitiveAction.RESPOND
    assert len(rt.response.calls) == 1
    context = rt.response.calls[0][2]
    assert context.route_context["action"] == "RESPOND"
    assert context.route_context["reason"] == "semantic test decision"
    assert context.tool_results[0]["tool_name"] == "runtime.current_time"
    assert context.tool_results[0]["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_router_failure_is_a_system_failure_not_silence(semantic_runtime):
    router = FakeRouter(raise_error=RouterTimeout("timed out"))
    rt = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_persistence=True, allow_memory_learning=True, allow_external_ai=True
    )
    event, report = await _say(rt, person.person_id, "hello")

    assert report.status == ReportStatus.FAILED.value
    assert rt.response.calls == []
    stored_event = await rt.repos.utterance.get(event.event_id)
    assert stored_event.processing_status == ProcessingStatus.FAILED
    assert await rt.repos.analysis.get_action(event.event_id) is None


@pytest.mark.asyncio
async def test_non_persistence_turn_writes_no_route_or_tool_rows(semantic_runtime):
    router = FakeRouter(
        tools=[{"tool_name": "runtime.current_time", "arguments": {}}]
    )
    rt = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas", "friend",
        allow_processing=True, allow_persistence=False, allow_memory_learning=False, allow_external_ai=True,
    )
    event, report = await _say(rt, person.person_id, "hello")

    assert report.status == ReportStatus.COMPLETED.value
    assert await rt.repos.routing.list_decisions(event.event_id) == []
    assert await rt.repos.routing.list_tool_calls(event.event_id) == []
    assert await rt.repos.analysis.get_action(event.event_id) is None
    assert await rt.repos.utterance.get(event.event_id) is None


@pytest.mark.asyncio
async def test_followup_planning_is_bounded(semantic_runtime):
    router = FakeRouter(
        tools=[{"tool_name": "runtime.current_time", "arguments": {}}],
        needs_followup=True,
        followup_tools=[{"tool_name": "history.recent", "arguments": {"limit": 2}}],
    )
    rt = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_persistence=True, allow_memory_learning=True, allow_external_ai=True
    )
    event, report = await _say(rt, person.person_id, "hello")

    assert report.status == ReportStatus.COMPLETED.value
    assert router.calls == 1
    assert router.replans == 1
    calls = await rt.repos.routing.list_tool_calls(event.event_id)
    assert len(calls) == 2
    assert {item["tool_name"] for item in calls} == {
        "runtime.current_time",
        "history.recent",
    }


@pytest.mark.asyncio
async def test_duplicate_event_does_not_reroute_or_reply_twice(semantic_runtime):
    router = FakeRouter()
    rt = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_persistence=True, allow_memory_learning=True, allow_external_ai=True
    )
    event, first = await _say(rt, person.person_id, "hello")
    _, second = await _say(rt, person.person_id, "hello", event_id=event.event_id)

    assert first.status == ReportStatus.COMPLETED.value
    assert second.status == "DUPLICATE"
    assert router.calls == 1
    assert len(rt.response.calls) == 1


@pytest.mark.asyncio
async def test_invalid_router_output_is_failed_and_not_persisted(semantic_runtime, db):
    from companion.routing.errors import RouterInvalidOutput

    router = FakeRouter(raise_error=RouterInvalidOutput("not json"))
    rt = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas", "friend", allow_processing=True, allow_persistence=True, allow_memory_learning=True, allow_external_ai=True
    )
    event, report = await _say(rt, person.person_id, "hello")

    assert report.status == ReportStatus.FAILED.value
    assert await rt.repos.routing.list_decisions(event.event_id) == []
    assert await rt.repos.analysis.get_action(event.event_id) is None


def test_policy_never_upgrades_semantic_wait_from_trusted_agent_hint():
    from companion.core.policy import Policy, TrustedTurnContext
    from companion.domain.enums import ReasonCode
    from companion.routing.models import (
        RouteAction,
        RouteAddressee,
        ValidatedRouteDecision,
    )
    from companion.domain.enums import ReplyNecessity

    route = ValidatedRouteDecision(
        action=RouteAction.WAIT,
        reply_necessity=ReplyNecessity.REQUIRED,
        addressee_assessment=RouteAddressee.AGENT,
        reason="speaker is still typing",
    )
    decision = Policy().decide_semantic(
        route,
        TrustedTurnContext(addressee_hint="agent", quiet_mode=False),
    )
    assert decision.action == CognitiveAction.WAIT
    assert decision.rule == "semantic_router"
    assert decision.model_action == CognitiveAction.WAIT
    assert decision.reason_code == ReasonCode.CONSERVATIVE_AMBIGUITY


def test_semantic_mode_is_not_the_mock_default_and_requires_a_router(db):
    from companion.runtime import build_runtime

    legacy = build_runtime(db, provider="mock")
    assert legacy.router_mode == "legacy"

    with pytest.raises(ValueError):
        build_runtime(db, provider="mock", router_mode="semantic")

    injected_client = build_runtime(
        db, provider="deepseek", deepseek_client=object()
    )
    assert injected_client.router_mode == "legacy"

