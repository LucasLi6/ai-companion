"""Input semantics: a validated NON_COMMUNICATIVE turn is never answered.

The router owns the semantic reading. These tests inject a structured status
through a fake router and assert that the execution chain honours it: the
same text can be answered or ignored depending only on the injected field, so
no local keyword, regex or character-shape rule exists in the path.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from companion.core import orchestrator as orchestrator_module
from companion.core.policy import Policy, TrustedTurnContext
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ProcessingStatus,
    ReasonCode,
    ReplyNecessity,
    ReportStatus,
)
from companion.domain.models import make_utterance_event
from companion.routing.errors import RouterValidationError
from companion.routing.models import (
    InputSemanticStatus,
    RawRouteDecision,
    RouteAction,
    RouteAddressee,
    RoutingContext,
    ValidatedRouteDecision,
)
from companion.routing.validator import RoutingValidator
from companion.runtime import build_runtime
from companion.tools.registry import build_default_tool_registry


# -- schema ----------------------------------------------------------------


def test_raw_decision_requires_input_semantics():
    with pytest.raises(ValidationError):
        RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": "RESPOND",
                "reply_necessity": "REQUIRED",
                "addressee_assessment": "AGENT",
            }
        )
    with pytest.raises(ValidationError):
        RawRouteDecision.model_validate(
            {
                "action": "RESPOND",
                "reply_necessity": "REQUIRED",
                "addressee_assessment": "AGENT",
                "input_semantics": "PROBABLY_TEXT",
            }
        )


def test_old_persisted_decision_reads_as_uncertain_not_noncommunicative():
    route = ValidatedRouteDecision.model_validate(
        {
            "action": "IGNORE",
            "reply_necessity": "UNNECESSARY",
            "addressee_assessment": "UNKNOWN",
        }
    )
    assert route.input_semantics == InputSemanticStatus.UNCERTAIN
    assert route.audit_dict()["input_semantics"] == "UNCERTAIN"


def test_core_fingerprint_includes_input_semantics():
    base = dict(
        action=RouteAction.IGNORE,
        reply_necessity=ReplyNecessity.UNNECESSARY,
        addressee_assessment=RouteAddressee.UNKNOWN,
    )
    communicative = ValidatedRouteDecision(
        **base, input_semantics=InputSemanticStatus.COMMUNICATIVE
    )
    uncertain = ValidatedRouteDecision(
        **base, input_semantics=InputSemanticStatus.UNCERTAIN
    )
    assert communicative.core_fingerprint() != uncertain.core_fingerprint()


# -- validator -------------------------------------------------------------


def _context() -> RoutingContext:
    return RoutingContext(
        conversation_id="conv",
        current_event_id="evt",
        current_speaker_id="p1",
        current_speaker_label="Lucas",
        current_text="irrelevant to the validator's semantic checks",
        trusted_addressee_hint="agent",
    )


def _raw(**overrides) -> RawRouteDecision:
    data = {
        "action": "IGNORE",
        "reply_necessity": "UNNECESSARY",
        "addressee_assessment": "UNKNOWN",
        "input_semantics": "NON_COMMUNICATIVE",
    }
    data.update(overrides)
    return RawRouteDecision.model_validate(data)


def test_validator_passes_the_status_through():
    registry = build_default_tool_registry(include_web=False)
    decision = RoutingValidator().validate(_raw(), _context(), registry)
    assert decision.input_semantics == InputSemanticStatus.NON_COMMUNICATIVE


@pytest.mark.parametrize(
    "overrides",
    [
        {"processing_intents": ["MEMORY_SAVE"]},
        {"processing_intents": ["MEMORY_CORRECTION"]},
        {"processing_intents": ["PRIVACY_REQUEST"]},
        {
            "tool_requests": [
                {"tool_name": "memory.search", "arguments": {"query": "x"}}
            ]
        },
        {"data_control_request": {"operation": "DELETE", "target_type": "MEMORY"}},
        {"reply_necessity": "REQUIRED"},
    ],
)
def test_structural_contradictions_are_rejected(overrides):
    registry = build_default_tool_registry(include_web=False)
    with pytest.raises(RouterValidationError):
        RoutingValidator().validate(_raw(**overrides), _context(), registry)


def test_uncertain_is_never_automatically_noncommunicative():
    registry = build_default_tool_registry(include_web=False)
    decision = RoutingValidator().validate(
        _raw(
            action="RESPOND",
            reply_necessity="USEFUL",
            addressee_assessment="AGENT",
            input_semantics="UNCERTAIN",
        ),
        _context(),
        registry,
    )
    assert decision.input_semantics == InputSemanticStatus.UNCERTAIN
    assert decision.action == RouteAction.RESPOND


# -- policy ----------------------------------------------------------------


def _validated(**overrides) -> ValidatedRouteDecision:
    data = {
        "action": RouteAction.RESPOND,
        "reply_necessity": ReplyNecessity.USEFUL,
        "addressee_assessment": RouteAddressee.AGENT,
        "input_semantics": InputSemanticStatus.NON_COMMUNICATIVE,
    }
    data.update(overrides)
    return ValidatedRouteDecision(**data)


def test_policy_ignores_noncommunicative_even_with_agent_hint():
    decision = Policy().decide_semantic(
        _validated(), TrustedTurnContext(addressee_hint="agent")
    )
    assert decision.action == CognitiveAction.IGNORE
    assert decision.model_action == CognitiveAction.RESPOND
    assert decision.rule == "semantic_noncommunicative"
    assert decision.reason_code == ReasonCode.RESPONSE_NOT_NECESSARY
    assert decision.override_reason == "non_communicative_input"


def test_policy_ignores_noncommunicative_even_in_quiet_mode():
    decision = Policy().decide_semantic(
        _validated(),
        TrustedTurnContext(addressee_hint="agent", quiet_mode=True),
    )
    assert decision.action == CognitiveAction.IGNORE
    assert decision.rule == "semantic_noncommunicative"


def test_policy_does_not_downgrade_uncertain_communication():
    decision = Policy().decide_semantic(
        _validated(input_semantics=InputSemanticStatus.UNCERTAIN),
        TrustedTurnContext(addressee_hint="agent"),
    )
    assert decision.action == CognitiveAction.RESPOND


def test_policy_still_respects_the_safety_floor():
    decision = Policy().decide_semantic(
        _validated(),
        TrustedTurnContext(addressee_hint="agent", safety_triggered=True),
    )
    assert decision.action == CognitiveAction.RESPOND
    assert decision.reason_code == ReasonCode.SAFETY_RESPONSE


# -- execution chain -------------------------------------------------------


class FakeRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self, *, action, necessity, addressee="UNKNOWN",
                 input_semantics="NON_COMMUNICATIVE", tools=None,
                 raise_error=None):
        self.calls = 0
        self.replans = 0
        self._action = action
        self._necessity = necessity
        self._addressee = addressee
        self._input_semantics = input_semantics
        self._tools = tools or []
        self._raise_error = raise_error

    def _raw(self):
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": self._action,
                "reply_necessity": self._necessity,
                "addressee_assessment": self._addressee,
                "input_semantics": self._input_semantics,
                "turn_state": "CONTINUING",
                "tool_requests": self._tools,
                "reason": "structured semantic test decision",
            }
        )

    async def route(self, context):
        self.calls += 1
        if self._raise_error is not None:
            raise self._raise_error
        return self._raw()

    async def replan(self, context, previous, results):
        self.replans += 1
        return self._raw()


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


async def _say(rt, person_id, text, *, event_id=None, conversation="conv"):
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )
    return event, await rt.orchestrator.process_utterance(event)


@pytest.mark.asyncio
async def test_noncommunicative_route_is_ignored_and_not_learned(semantic_runtime):
    router = FakeRouter(action="IGNORE", necessity="UNNECESSARY")
    rt, response = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    event, report = await _say(rt, person.person_id, "--- ??? ---")

    assert router.calls == 1
    assert response.calls == []
    assert report.status == ReportStatus.COMPLETED.value
    assert report.action == CognitiveAction.IGNORE
    assert report.response_text is None

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.action_type == CognitiveAction.IGNORE
    assert action.reason_code == ReasonCode.RESPONSE_NOT_NECESSARY
    assert action.model_suggested_action == CognitiveAction.IGNORE
    assert action.policy_final_action == CognitiveAction.IGNORE
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status == ProcessingStatus.COMPLETED
    # No background extraction, no memory candidate, no profile write.
    assert (
        await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    ) is None
    assert (
        await rt.repos.job.get_by_key(f"memory:{event.event_id}")
    ) is None
    assert await rt.repos.analysis.list_candidate_writes(event.event_id) == []


@pytest.mark.asyncio
async def test_agent_hint_cannot_turn_noncommunication_into_a_reply(semantic_runtime):
    # The router itself suggests RESPOND; the local policy must still ignore it.
    router = FakeRouter(action="RESPOND", necessity="USEFUL", addressee="AGENT")
    rt, response = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    event, report = await _say(rt, person.person_id, "???", conversation="conv-b")

    assert response.calls == []
    assert report.status == ReportStatus.COMPLETED.value
    assert report.action == CognitiveAction.IGNORE
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.action_type == CognitiveAction.IGNORE
    assert action.model_suggested_action == CognitiveAction.RESPOND
    assert action.override_reason == "non_communicative_input"


@pytest.mark.asyncio
async def test_uncertain_route_can_still_respond(semantic_runtime):
    router = FakeRouter(
        action="RESPOND",
        necessity="USEFUL",
        addressee="AGENT",
        input_semantics="UNCERTAIN",
    )
    rt, response = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    _, report = await _say(rt, person.person_id, "hmm", conversation="conv-c")

    assert report.action == CognitiveAction.RESPOND
    assert len(response.calls) == 1


@pytest.mark.asyncio
async def test_communicative_route_reaches_the_speaker(semantic_runtime):
    router = FakeRouter(
        action="RESPOND",
        necessity="REQUIRED",
        addressee="AGENT",
        input_semantics="COMMUNICATIVE",
    )
    rt, response = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    _, report = await _say(rt, person.person_id, "please tell me", conversation="conv-d")

    assert report.action == CognitiveAction.RESPOND
    assert len(response.calls) == 1


@pytest.mark.asyncio
async def test_contradictory_noncommunicative_payload_fails_without_writing(
    semantic_runtime,
):
    router = FakeRouter(
        action="IGNORE",
        necessity="UNNECESSARY",
        tools=[{"tool_name": "memory.search", "arguments": {"query": "x"}}],
    )
    rt, response = semantic_runtime(router)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    event, report = await _say(rt, person.person_id, "???", conversation="conv-e")

    assert response.calls == []
    assert report.status == ReportStatus.FAILED.value
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status == ProcessingStatus.FAILED
    assert await rt.repos.analysis.get_action(event.event_id) is None


# -- input-shape independence ---------------------------------------------

_INPUT_SHAPES = [
    "7 3 9 !!!",
    "أهلاً، كيف حالك؟",
    "def f(x):\n    return x * 2",
    "{\"machine\": true, \"value\": 42}",
    "嗯",
    "asdf qwer zxcv",
    "42",
    "……",
    "https://example.invalid/a?b=c",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", _INPUT_SHAPES)
async def test_outcome_follows_the_router_field_not_the_text_shape(
    semantic_runtime, text
):
    ignored = FakeRouter(action="IGNORE", necessity="UNNECESSARY")
    rt, response = semantic_runtime(ignored)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    _, report = await _say(rt, person.person_id, text, conversation="conv-shape")

    assert report.action == CognitiveAction.IGNORE
    assert response.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("text", _INPUT_SHAPES)
async def test_the_same_shape_is_answered_when_the_router_says_communicative(
    semantic_runtime, text
):
    answered = FakeRouter(
        action="RESPOND",
        necessity="USEFUL",
        addressee="AGENT",
        input_semantics="COMMUNICATIVE",
    )
    rt, response = semantic_runtime(answered)
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )

    _, report = await _say(rt, person.person_id, text, conversation="conv-shape-2")

    assert report.action == CognitiveAction.RESPOND
    assert len(response.calls) == 1
