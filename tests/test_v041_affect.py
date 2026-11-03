"""V0.4.1 event semantics, target validation and appraisal diagnostics.

These tests are offline and deterministic. They verify the local trust boundary:
a model hint is only used when it names a supported event, an explicit target,
an explicit social intent, and verbatim evidence. Profanity alone never creates
hostility, and a single joke never raises relationship tension.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from companion.domain.models import AffectiveSignal, validate_affective_signal
from companion.emotion.appraisal import (
    APPLIED,
    INVALID_EVIDENCE,
    LOW_CONFIDENCE,
    NO_EVENT,
    NO_RULE_MATCHED,
    RULE_DISABLED,
    UNSUPPORTED_EVENT,
    AppraisalEngine,
)
from companion.emotion.models import EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


async def _rules(rt):
    return await rt.emotion_service.list_rules(await _agent_id(rt))


def _signal(
    event_type,
    evidence,
    *,
    target="AGENT",
    intent="HOSTILE",
    factuality="asserted",
    confidence=0.9,
    subject="speaker",
):
    return AffectiveSignal(
        event_type=event_type,
        target=target,
        social_intent=intent,
        subject=subject,
        factuality=factuality,
        confidence=confidence,
        evidence_span=evidence,
    )


def _appraise(rules, signal, text, *, now=None, addressee="agent", state=None):
    return AppraisalEngine().appraise(
        text,
        rules=rules,
        signal=signal,
        current_state=state
        or EmotionState(updated_at=now or datetime.now(timezone.utc)),
        now=now or datetime.now(timezone.utc),
        addressee_hint=addressee,
        config=DEFAULT_CONFIG,
    )


async def test_direct_insult_at_agent_applies_frustration_and_moderate_tension(rt):
    now = datetime.now(timezone.utc)
    text = "你这个没用的东西"
    result = _appraise(
        await _rules(rt),
        _signal("direct_insult", text),
        text,
        now=now,
    )
    assert result.diagnostic == APPLIED
    assert result.reason == "direct_insult"
    assert result.emotion == "FRUSTRATION"
    assert result.target == "AGENT"
    assert result.social_intent == "HOSTILE"
    # One hostile event raises tension moderately; it is not instantly "high".
    assert 0.0 < result.state.relationship_tension < 0.50


async def test_playful_teasing_does_not_raise_tension(rt):
    now = datetime.now(timezone.utc)
    text = "哈哈你今天有点笨啊"
    result = _appraise(
        await _rules(rt),
        _signal("playful_teasing", text, intent="PLAYFUL"),
        text,
        now=now,
    )
    assert result.diagnostic == APPLIED
    assert result.reason == "playful_teasing"
    assert result.emotion == "JOY"
    assert result.state.relationship_tension == 0.0


async def test_criticism_raises_much_less_tension_than_insult(rt):
    rules = await _rules(rt)
    now = datetime.now(timezone.utc)
    text = "你刚才那个回答根本没回答我的问题"
    criticism = _appraise(
        rules,
        _signal("criticism", text, intent="SINCERE"),
        text,
        now=now,
    )
    insult_text = "你真蠢"
    insult = _appraise(
        rules,
        _signal("direct_insult", insult_text),
        insult_text,
        now=now,
    )
    assert criticism.reason == "agent_criticism"
    assert 0.0 < criticism.state.relationship_tension
    assert criticism.state.relationship_tension < insult.state.relationship_tension


async def test_agent_apology_and_reconciliation_lower_existing_tension(rt):
    rules = await _rules(rt)
    now = datetime.now(timezone.utc)
    charged = EmotionState(
        emotion="FRUSTRATION",
        intensity=0.5,
        activations={"FRUSTRATION": 0.5},
        relationship_tension=0.6,
        tension_updated_at=now,
        updated_at=now,
    )
    text = "刚才是我说得太重了，对不起"
    apology = _appraise(
        rules,
        _signal("apology", text, intent="SINCERE"),
        text,
        now=now,
        state=charged,
    )
    assert apology.reason == "agent_apology"
    assert apology.emotion == "RELIEF"
    assert apology.state.relationship_tension < 0.6

    text2 = "我们别为这个吵了，我还是想好好聊"
    repair = _appraise(
        rules,
        _signal("reconciliation", text2, intent="SINCERE"),
        text2,
        now=now,
        state=charged,
    )
    assert repair.reason == "relationship_repair"
    assert repair.state.relationship_tension < 0.6


async def test_user_self_distress_requires_speaker_target(rt):
    rules = await _rules(rt)
    now = datetime.now(timezone.utc)
    text = "我最近真的撑不住了"
    result = _appraise(
        rules,
        _signal(
            "self_distress",
            text,
            target="SPEAKER",
            intent="SINCERE",
        ),
        text,
        now=now,
    )
    assert result.reason == "user_self_distress"
    assert result.state.relationship_tension == 0.0

    wrong = _appraise(
        rules,
        _signal("self_distress", text, target="AGENT", intent="SINCERE"),
        text,
        now=now,
    )
    assert wrong.diagnostic == UNSUPPORTED_EVENT
    assert wrong.validation == "unsupported_target"


async def test_direct_praise_and_support_request_are_distinct(rt):
    rules = await _rules(rt)
    now = datetime.now(timezone.utc)
    praise_text = "你这次讲得真清楚，谢谢你"
    praise = _appraise(
        rules,
        _signal("direct_praise", praise_text, intent="SINCERE"),
        praise_text,
        now=now,
    )
    assert praise.reason == "direct_praise_joy"
    assert praise.emotion == "JOY"

    support_text = "你能帮我看看这段吗"
    support = _appraise(
        rules,
        _signal("support_request", support_text, intent="SINCERE"),
        support_text,
        now=now,
    )
    assert support.reason == "support_request_concern"


@pytest.mark.parametrize(
    "target,intent,factuality,confidence,expected,validation",
    [
        ("OTHER", "HOSTILE", "asserted", 0.9, UNSUPPORTED_EVENT, "unsupported_target"),
        ("UNKNOWN", "HOSTILE", "asserted", 0.9, UNSUPPORTED_EVENT, "unsupported_target"),
        ("AGENT", "PLAYFUL", "asserted", 0.9, UNSUPPORTED_EVENT, "unsupported_social_intent"),
        ("AGENT", "UNCERTAIN", "asserted", 0.9, UNSUPPORTED_EVENT, "unsupported_social_intent"),
        ("AGENT", "HOSTILE", "quoted", 0.9, UNSUPPORTED_EVENT, "not_asserted"),
        ("AGENT", "HOSTILE", "negated", 0.9, UNSUPPORTED_EVENT, "not_asserted"),
        ("AGENT", "HOSTILE", "asserted", 0.4, LOW_CONFIDENCE, "low_confidence"),
    ],
)
async def test_insult_validation_rejects_untrusted_claims(
    rt, target, intent, factuality, confidence, expected, validation
):
    text = "你这个没用的东西"
    signal = _signal(
        "direct_insult",
        text,
        target=target,
        intent=intent,
        factuality=factuality,
        confidence=confidence,
    )
    result = _appraise(await _rules(rt), signal, text)
    assert result.diagnostic == expected
    assert result.validation == validation
    assert result.trigger_rule_id is None
    assert result.state.relationship_tension == 0.0


async def test_evidence_must_be_verbatim(rt):
    text = "你今天真差劲"
    signal = _signal("direct_insult", "你真差劲")
    result = _appraise(await _rules(rt), signal, text)
    assert result.diagnostic == INVALID_EVIDENCE
    assert result.validation == "evidence_not_in_text"


async def test_missing_evidence_is_invalid_not_guessed(rt):
    signal = _signal("direct_insult", "")
    result = _appraise(await _rules(rt), signal, "你这个坏蛋")
    assert result.diagnostic == INVALID_EVIDENCE


async def test_none_event_is_no_event(rt):
    result = _appraise(
        await _rules(rt),
        AffectiveSignal(event_type="none", confidence=0.99),
        "今天天气不错",
    )
    assert result.diagnostic == NO_EVENT
    assert result.trigger_rule_id is None


async def test_profanity_without_signal_never_creates_hostility(rt):
    text = "靠，这破电脑又卡了"
    result = _appraise(await _rules(rt), None, text)
    assert result.diagnostic == NO_RULE_MATCHED
    assert result.state.relationship_tension == 0.0


async def test_model_cannot_target_agent_when_trusted_addressee_is_human(rt):
    text = "你真蠢"
    result = _appraise(
        await _rules(rt),
        _signal("direct_insult", text),
        text,
        addressee="human",
    )
    assert result.diagnostic == UNSUPPORTED_EVENT
    assert result.validation == "not_addressed_to_agent"
    assert result.state.relationship_tension == 0.0


async def test_disabled_rule_outcome_is_explicit(rt):
    agent_id = await _agent_id(rt)
    rules = await _rules(rt)
    rule = next(r for r in rules if r.name == "direct_insult")
    await rt.emotion_service.set_rule_status(agent_id, rule.rule_id, "DISABLED")
    text = "你真蠢"
    result = _appraise(
        await _rules(rt), _signal("direct_insult", text), text
    )
    assert result.diagnostic == RULE_DISABLED
    assert result.trigger_rule_id is None


async def test_repeated_hostility_requires_local_repeat_evidence(rt):
    rules = await _rules(rt)
    now = datetime.now(timezone.utc)
    text = "你真蠢"
    first = _appraise(rules, _signal("direct_insult", text), text, now=now)
    assert first.reason == "direct_insult"

    second = _appraise(
        rules,
        _signal("direct_insult", text),
        text,
        now=now + timedelta(seconds=30),
        state=first.state,
    )
    assert second.reason == "repeated_hostility"
    assert second.state.relationship_tension > first.state.relationship_tension

    # ``repeated_hostility`` is a locally derived rule, not an event a model may
    # claim, so it cannot enter through the signal channel at all.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _signal("repeated_hostility", text)


def test_model_signal_named_repeated_hostility_is_not_an_event_type():
    """The only route to repeated hostility is local repeat evidence."""
    from companion.domain.models import _AFT_EVENT_TYPES

    assert "repeated_hostility" not in _AFT_EVENT_TYPES


async def test_old_hostility_does_not_count_as_repeated(rt):
    rules = await _rules(rt)
    now = datetime.now(timezone.utc)
    text = "你真蠢"
    first = _appraise(rules, _signal("direct_insult", text), text, now=now)
    later = _appraise(
        rules,
        _signal("direct_insult", text),
        text,
        now=now + timedelta(seconds=DEFAULT_CONFIG.repeated_event_window_seconds + 1),
        state=first.state,
    )
    assert later.reason == "direct_insult"


def test_validate_signal_normalises_lowercase_enums_and_derives_target():
    text = "谢谢你"
    signal = validate_affective_signal(
        {
            "event_type": "gratitude",
            "subject": "speaker",
            "target": "agent",
            "social_intent": "sincere",
            "factuality": "asserted",
            "confidence": 0.9,
            "evidence_span": "谢谢",
        },
        text,
    )
    assert signal is not None
    assert signal.target == "AGENT"
    assert signal.social_intent == "SINCERE"

    legacy = validate_affective_signal(
        {
            "event_type": "achievement",
            "subject": "speaker",
            "factuality": "asserted",
            "confidence": 0.9,
            "evidence_span": "升职",
        },
        "我升职了",
    )
    assert legacy is not None
    assert legacy.target == "SPEAKER"
    assert legacy.social_intent == "UNCERTAIN"
