"""V0.3.2 — validated silence motives and final-cause precedence.

All offline. These tests pin the boundary between a model *hint* and the local
Policy decision: the hint is validated, can never outrank local silence,
quiet mode, a human addressee or permissions, and is downgraded to UNKNOWN when
its evidence is not real.
"""

from __future__ import annotations

import json
import math

import pytest

from companion.cognition.models import CognitiveContext
from companion.cognition.nonresponse import derive_nonresponse_trace
from companion.core.policy import Policy
from companion.domain.enums import (
    AddresseeHint,
    CognitiveAction,
    EventSource,
    ReasonCode,
    ReplyNecessity,
)
from companion.domain.models import (
    CognitiveResult,
    ConversationContext,
    NonresponseMotive,
    make_utterance_event,
    validate_nonresponse_motive,
)
from companion.providers.deepseek_client import ChatResponse
from companion.providers.deepseek_cognitive import DeepSeekCognitiveProvider


def _event(text="你觉得呢", *, to="unknown"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint=to,
        speaker_person_id="person_1",
        source=EventSource.TEST,
    )


def _analysis(
    event,
    *,
    action=CognitiveAction.WAIT,
    reason=ReasonCode.AMBIGUOUS_ADDRESSEE,
    necessity=ReplyNecessity.OPTIONAL,
    motive=None,
    addressed_to=None,
):
    return CognitiveResult(
        event_id=event.event_id,
        meaning="fake analysis",
        action=action,
        reason_code=reason,
        response_style="plain",
        provider="fake",
        reply_necessity=necessity,
        addressed_to=addressed_to,
        nonresponse_motive=motive,
    )


class _Decision:
    def __init__(
        self,
        *,
        action,
        reason,
        model_action=None,
        rule="model_suggestion",
        override=None,
    ):
        self.action = action
        self.reason_code = reason
        self.model_action = model_action or action
        self.rule = rule
        self.override_reason = override

    @property
    def overrode_model(self):
        return self.action != self.model_action


def _decision(action, reason, *, model_action=None, rule="model_suggestion", override=None):
    return _Decision(
        action=action,
        reason=reason,
        model_action=model_action,
        rule=rule,
        override=override,
    )


# -- local validation --------------------------------------------------------


def test_valid_motive_is_accepted():
    motive = validate_nonresponse_motive(
        {
            "motive": "UNCERTAIN_CONTEXT",
            "confidence": 0.86,
            "evidence_span": "你觉得呢",
            "reentry_hint": "WHEN_CONTEXT_CLEAR",
            "person_id": "person_injected",
            "event_id": "evt_injected",
            "memory_id": "mem_injected",
        },
        "你觉得呢",
    )
    assert motive is not None
    assert motive.motive == "UNCERTAIN_CONTEXT"
    assert motive.confidence == pytest.approx(0.86)
    assert motive.evidence_span == "你觉得呢"
    assert motive.reentry_hint == "WHEN_CONTEXT_CLEAR"
    dumped = motive.model_dump()
    assert "person_id" not in dumped
    assert "event_id" not in dumped
    assert "memory_id" not in dumped


def test_missing_motive_is_none_for_backward_compatibility():
    assert validate_nonresponse_motive(None, "随便") is None


@pytest.mark.parametrize(
    "raw",
    [
        "not-an-object",
        {"motive": "MADE_UP"},
        {"motive": "UNCERTAIN_CONTEXT", "confidence": float("nan"), "evidence_span": "随便"},
        {"motive": "UNCERTAIN_CONTEXT", "confidence": float("inf"), "evidence_span": "随便"},
        {"motive": "UNCERTAIN_CONTEXT", "confidence": -0.1, "evidence_span": "随便"},
        {"motive": "UNCERTAIN_CONTEXT", "confidence": 1.1, "evidence_span": "随便"},
        {"motive": "UNCERTAIN_CONTEXT", "confidence": 0.9, "evidence_span": "不在原文"},
        {"motive": "UNCERTAIN_CONTEXT", "confidence": 0.9, "evidence_span": ""},
        {"motive": "UNCERTAIN_CONTEXT", "confidence": 0.9, "evidence_span": "x" * 513},
    ],
)
def test_invalid_motive_degrades_to_unknown(raw):
    motive = validate_nonresponse_motive(raw, "随便")
    assert motive is not None
    assert motive.motive == "UNKNOWN"
    assert motive.confidence == 0.0
    assert motive.evidence_span is None


def test_unknown_needs_no_evidence_and_bad_reentry_is_ignored():
    motive = validate_nonresponse_motive(
        {"motive": "UNKNOWN", "confidence": 0.4, "reentry_hint": "NONSENSE"},
        "普通输入",
    )
    assert motive is not None
    assert motive.motive == "UNKNOWN"
    assert motive.reentry_hint == "NO_FOLLOWUP"


# -- provider trust boundary -------------------------------------------------


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        return ChatResponse(
            content=json.dumps(self.payload, ensure_ascii=False),
            model="fake",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


def _context(event):
    return CognitiveContext(
        conversation_id=event.conversation_id,
        current_event_id=event.event_id,
        current_speaker_id=event.speaker_person_id,
        current_speaker_label="Alice",
        current_text=event.text,
        current_addressee_hint=event.addressee_hint.value,
    )


def _payload(**overrides):
    payload = {
        "meaning": "用户在问",
        "action": "WAIT",
        "reason_code": "ambiguous_addressee",
        "reply_necessity": "OPTIONAL",
        "nonresponse_motive": {
            "motive": "UNCERTAIN_CONTEXT",
            "confidence": 0.9,
            "evidence_span": "你觉得呢",
            "reentry_hint": "WHEN_CONTEXT_CLEAR",
        },
    }
    payload.update(overrides)
    return payload


async def test_provider_keeps_a_valid_motive():
    event = _event()
    provider = DeepSeekCognitiveProvider(_Client(_payload()))
    result = await provider.analyze(event, _context(event), [])
    assert result.action == CognitiveAction.WAIT
    assert result.nonresponse_motive is not None
    assert result.nonresponse_motive.motive == "UNCERTAIN_CONTEXT"


async def test_provider_downgrades_an_invalid_motive():
    payload = _payload(
        nonresponse_motive={
            "motive": "UNCERTAIN_CONTEXT",
            "confidence": 0.9,
            "evidence_span": "不在输入里",
        }
    )
    event = _event()
    provider = DeepSeekCognitiveProvider(_Client(payload))
    result = await provider.analyze(event, _context(event), [])
    assert result.nonresponse_motive is not None
    assert result.nonresponse_motive.motive == "UNKNOWN"


async def test_provider_ignores_motive_when_action_is_respond():
    payload = _payload(
        action="RESPOND",
        reason_code="direct_agent_address",
        reply_necessity="REQUIRED",
    )
    event = _event("我升职了", to="agent")
    provider = DeepSeekCognitiveProvider(_Client(payload))
    result = await provider.analyze(event, _context(event), [])
    assert result.action == CognitiveAction.RESPOND
    assert result.nonresponse_motive is None


async def test_provider_without_motive_still_works():
    payload = _payload()
    payload.pop("nonresponse_motive")
    event = _event()
    provider = DeepSeekCognitiveProvider(_Client(payload))
    result = await provider.analyze(event, _context(event), [])
    assert result.action == CognitiveAction.WAIT
    assert result.nonresponse_motive is None


# -- Policy: direct question correction --------------------------------------


def _context_obj(quiet=False):
    return ConversationContext(conversation_id="c1", quiet_mode=quiet)


def test_direct_required_question_corrects_ambiguous_model_wait():
    event = _event("你在听吗？", to="agent")
    analysis = _analysis(
        event,
        action=CognitiveAction.WAIT,
        reason=ReasonCode.AMBIGUOUS_ADDRESSEE,
        necessity=ReplyNecessity.REQUIRED,
    )
    decision = Policy().decide(
        analysis,
        _context_obj(),
        addressee_hint=event.addressee_hint.value,
        explicit_silence=False,
    )
    assert decision.action == CognitiveAction.RESPOND
    assert decision.rule == "direct_question_correction"
    assert decision.override_reason == "model_wait_on_direct_question"


def test_correction_requires_trusted_agent_hint_not_model_claim():
    event = _event("你在听吗？", to="unknown")
    analysis = _analysis(
        event,
        action=CognitiveAction.WAIT,
        reason=ReasonCode.AMBIGUOUS_ADDRESSEE,
        necessity=ReplyNecessity.REQUIRED,
        addressed_to="agent",  # model claim alone is not enough
    )
    decision = Policy().decide(
        analysis,
        _context_obj(),
        addressee_hint=event.addressee_hint.value,
        explicit_silence=False,
    )
    assert decision.action == CognitiveAction.WAIT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"necessity": ReplyNecessity.USEFUL},
        {"necessity": ReplyNecessity.OPTIONAL},
        {"reason": ReasonCode.RESPONSE_NOT_NECESSARY},
        {"quiet": True},
        {"explicit_silence": True},
    ],
)
def test_correction_respects_local_limits(kwargs):
    event = _event("你在听吗？", to="agent")
    analysis = _analysis(
        event,
        action=CognitiveAction.WAIT,
        reason=kwargs.get("reason", ReasonCode.AMBIGUOUS_ADDRESSEE),
        necessity=kwargs.get("necessity", ReplyNecessity.REQUIRED),
    )
    decision = Policy().decide(
        analysis,
        _context_obj(quiet=kwargs.get("quiet", False)),
        addressee_hint=event.addressee_hint.value,
        explicit_silence=kwargs.get("explicit_silence", False),
    )
    assert decision.action != CognitiveAction.RESPOND


def test_explicit_human_and_permission_are_never_overridden():
    event = _event("你好", to="human")
    human = _analysis(
        event,
        action=CognitiveAction.RESPOND,
        reason=ReasonCode.DIRECT_AGENT_ADDRESS,
        necessity=ReplyNecessity.REQUIRED,
    )
    decision = Policy().decide(
        human, _context_obj(), addressee_hint="human", explicit_silence=False
    )
    assert decision.action == CognitiveAction.IGNORE
    assert decision.reason_code == ReasonCode.DIRECT_HUMAN_ADDRESS

    denied = _analysis(
        event,
        action=CognitiveAction.RESPOND,
        reason=ReasonCode.PERMISSION_DENIED,
        necessity=ReplyNecessity.REQUIRED,
    )
    decision = Policy().decide(
        denied, _context_obj(), addressee_hint="agent", explicit_silence=False
    )
    assert decision.action == CognitiveAction.WAIT
    assert decision.reason_code == ReasonCode.PERMISSION_DENIED


# -- derive_nonresponse_trace precedence -------------------------------------


def _trace_for(event, analysis, decision, context=None):
    return derive_nonresponse_trace(
        event=event,
        analysis=analysis,
        policy_decision=decision,
        conversation_context=context or _context_obj(),
    )


def test_respond_has_no_trace():
    event = _event("你好", to="agent")
    analysis = _analysis(
        event,
        action=CognitiveAction.RESPOND,
        reason=ReasonCode.DIRECT_AGENT_ADDRESS,
        necessity=ReplyNecessity.REQUIRED,
    )
    decision = _decision(
        CognitiveAction.RESPOND, ReasonCode.DIRECT_AGENT_ADDRESS
    )
    assert _trace_for(event, analysis, decision) is None


def test_quiet_mode_outranks_model_motive():
    event = _event()
    motive = NonresponseMotive(
        motive="AWAITING_CONTINUATION",
        confidence=0.9,
        evidence_span="你觉得呢",
        reentry_hint="NEXT_TOPIC_CONTINUATION",
    )
    analysis = _analysis(event, motive=motive)
    decision = _decision(CognitiveAction.WAIT, ReasonCode.AMBIGUOUS_ADDRESSEE)
    trace = _trace_for(event, analysis, decision, _context_obj(quiet=True))
    assert trace is not None
    assert trace.cause_code == "QUIET_MODE"
    assert trace.cause_source == "LOCAL_POLICY"
    assert trace.model_motive is None
    assert trace.policy_override == "quiet_mode"
    assert trace.pending_followup is False


def test_explicit_human_addressee_outranks_model():
    event = _event(to="human")
    analysis = _analysis(
        event,
        action=CognitiveAction.IGNORE,
        reason=ReasonCode.DIRECT_HUMAN_ADDRESS,
        motive=NonresponseMotive(
            motive="AVOID_INTERRUPTION",
            confidence=0.8,
            evidence_span="你好",
        ),
    )
    decision = _decision(
        CognitiveAction.IGNORE,
        ReasonCode.DIRECT_HUMAN_ADDRESS,
        model_action=CognitiveAction.RESPOND,
        rule="explicit_human",
        override="explicit_human_addressee",
    )
    trace = _trace_for(event, analysis, decision)
    assert trace is not None
    assert trace.cause_code == "HUMAN_ADDRESSEE"
    assert trace.cause_source == "LOCAL_POLICY"
    assert trace.model_motive is None
    assert trace.policy_override == "explicit_human_addressee"


def test_explicit_silence_and_permission_have_local_causes():
    event = _event()
    analysis = _analysis(event)
    silent = _trace_for(
        event,
        analysis,
        _decision(
            CognitiveAction.IGNORE,
            ReasonCode.EXPLICIT_SILENCE_REQUEST,
            model_action=CognitiveAction.RESPOND,
            rule="explicit_silence",
            override="explicit_silence_request",
        ),
    )
    assert silent.cause_code == "EXPLICIT_SILENCE"
    assert silent.cause_source == "LOCAL_POLICY"

    denied = _trace_for(
        event,
        _analysis(event, reason=ReasonCode.PERMISSION_DENIED),
        _decision(CognitiveAction.WAIT, ReasonCode.PERMISSION_DENIED),
    )
    assert denied.cause_code == "PERMISSION_DENIED"
    assert denied.cause_source == "LOCAL_POLICY"


@pytest.mark.parametrize(
    "reason,cause",
    [
        (ReasonCode.EXPLICIT_SILENCE_REQUEST, "EXPLICIT_SILENCE"),
        (ReasonCode.QUIET_MODE, "QUIET_MODE"),
        (ReasonCode.DIRECT_HUMAN_ADDRESS, "HUMAN_ADDRESSEE"),
        (ReasonCode.CONVERSATION_BETWEEN_HUMANS, "HUMAN_ADDRESSEE"),
    ],
)
def test_model_only_reason_codes_stay_model_assessment(reason, cause):
    """A model's reason alone must not be relabelled as a local decision."""
    event = _event(to="unknown")
    analysis = _analysis(
        event, action=CognitiveAction.IGNORE, reason=reason
    )
    trace = _trace_for(
        event, analysis, _decision(CognitiveAction.IGNORE, reason)
    )
    assert trace is not None
    assert trace.cause_code == cause
    assert trace.cause_source == "MODEL_ASSESSMENT"
    assert trace.policy_override is None


def test_local_policy_downgrade_does_not_borrow_model_motive():
    event = _event(to="agent", text="我成功了")
    analysis = _analysis(
        event,
        action=CognitiveAction.RESPOND,
        reason=ReasonCode.DIRECT_AGENT_ADDRESS,
        necessity=ReplyNecessity.REQUIRED,
    )
    decision = _decision(
        CognitiveAction.WAIT,
        ReasonCode.RESPONSE_NOT_NECESSARY,
        model_action=CognitiveAction.RESPOND,
        rule="response_not_necessary",
        override="reply_necessity_unnecessary",
    )
    trace = _trace_for(event, analysis, decision)
    assert trace is not None
    assert trace.cause_code == "NO_REPLY_NEEDED"
    assert trace.cause_source == "LOCAL_POLICY"
    assert trace.model_motive is None
    assert trace.policy_override == "reply_necessity_unnecessary"


@pytest.mark.parametrize(
    "motive,cause,pending,reentry",
    [
        ("NOT_ADDRESSED", "NO_REPLY_NEEDED", False, "NO_FOLLOWUP"),
        ("AWAITING_CONTINUATION", "AWAITING_CONTINUATION", True, "NEXT_TOPIC_CONTINUATION"),
        ("AVOID_INTERRUPTION", "AVOID_INTERRUPTION", False, "NEXT_DIRECT_ADDRESS"),
        ("LISTEN_WITHOUT_REPLY", "LISTEN_WITHOUT_REPLY", False, "NO_FOLLOWUP"),
        ("NO_USEFUL_CONTRIBUTION", "NO_REPLY_NEEDED", False, "NO_FOLLOWUP"),
        ("UNCERTAIN_CONTEXT", "AMBIGUOUS_ADDRESSEE", False, "WHEN_CONTEXT_CLEAR"),
        ("UNKNOWN", "UNKNOWN", False, "NO_FOLLOWUP"),
    ],
)
def test_motive_mapping_and_pending(motive, cause, pending, reentry):
    event = _event()
    analysis = _analysis(
        event,
        motive=NonresponseMotive(
            motive=motive,
            confidence=0.9,
            evidence_span="你觉得呢",
            reentry_hint=reentry,
        ),
    )
    trace = _trace_for(
        event,
        analysis,
        _decision(CognitiveAction.WAIT, ReasonCode.AMBIGUOUS_ADDRESSEE),
    )
    assert trace is not None
    assert trace.cause_code == cause
    assert trace.cause_source == "MODEL_ASSESSMENT"
    assert trace.model_motive == motive
    assert trace.pending_followup is pending
    assert trace.reentry_condition == (
        "NONE" if reentry == "NO_FOLLOWUP" else reentry
    )


def test_uncertain_context_can_be_pending_only_with_a_continuation_hint():
    event = _event()
    analysis = _analysis(
        event,
        motive=NonresponseMotive(
            motive="UNCERTAIN_CONTEXT",
            confidence=0.8,
            evidence_span="你觉得呢",
            reentry_hint="NEXT_TOPIC_CONTINUATION",
        ),
    )
    trace = _trace_for(
        event,
        analysis,
        _decision(CognitiveAction.WAIT, ReasonCode.AMBIGUOUS_ADDRESSEE),
    )
    assert trace.pending_followup is True
    assert trace.reentry_condition == "NEXT_TOPIC_CONTINUATION"


def test_no_motive_uses_reason_code_but_never_invents_evidence():
    event = _event()
    analysis = _analysis(
        event, action=CognitiveAction.IGNORE, reason=ReasonCode.SOCIAL_RESPONSE_NOT_NEEDED
    )
    trace = _trace_for(
        event, analysis, _decision(CognitiveAction.IGNORE, analysis.reason_code)
    )
    assert trace is not None
    assert trace.cause_code == "NO_REPLY_NEEDED"
    assert trace.cause_source == "MODEL_ASSESSMENT"
    assert trace.model_motive is None
    assert trace.evidence_span is None

    unknown = _analysis(event, reason=ReasonCode.UNKNOWN)
    trace = _trace_for(
        event, unknown, _decision(CognitiveAction.WAIT, ReasonCode.UNKNOWN)
    )
    assert trace.cause_code == "UNKNOWN"
    assert trace.cause_source == "MODEL_ASSESSMENT"
