"""Explainable nonresponse: why the companion chose not to speak.

The cognitive model may suggest ``WAIT``/``IGNORE`` and may offer a
:class:`NonresponseMotive`, but it never owns the final reason. This module
derives one persisted :class:`NonresponseTrace` from trusted local policy first
and the validated model hint second. It also turns stored actions from earlier
turns into :class:`ActionHistoryRecord` entries, so the next model call can
know what the companion *did* without pretending the silence was spoken text.

No function here calls a model, writes to the database, or changes an action.
"""

from __future__ import annotations

from typing import Literal, Optional

from companion.cognition.models import ActionHistoryRecord
from companion.domain.enums import AddresseeHint, CognitiveAction, ReasonCode
from companion.domain.models import DomainModel, NonresponseMotive


class NonresponseTrace(DomainModel):
    """The final, auditable explanation for one WAIT/IGNORE action."""

    final_action: Literal["WAIT", "IGNORE"]
    cause_code: str
    cause_source: Literal[
        "LOCAL_POLICY",
        "MODEL_ASSESSMENT",
        "LEGACY_UNKNOWN",
    ]
    model_motive: Optional[str] = None
    policy_override: Optional[str] = None
    evidence_span: Optional[str] = None
    pending_followup: bool = False
    reentry_condition: str = "NONE"
    spoken: bool = False


# Local policy reason -> final cause. A local reason always outranks whatever
# motive the model offered.
_REASON_TO_CAUSE = {
    ReasonCode.PERMISSION_DENIED: "PERMISSION_DENIED",
    ReasonCode.EXPLICIT_SILENCE_REQUEST: "EXPLICIT_SILENCE",
    ReasonCode.QUIET_MODE: "QUIET_MODE",
    ReasonCode.DIRECT_HUMAN_ADDRESS: "HUMAN_ADDRESSEE",
    ReasonCode.CONVERSATION_BETWEEN_HUMANS: "HUMAN_ADDRESSEE",
    ReasonCode.AMBIGUOUS_ADDRESSEE: "AMBIGUOUS_ADDRESSEE",
    ReasonCode.UNKNOWN_ADDRESSEE: "AMBIGUOUS_ADDRESSEE",
    ReasonCode.RESPONSE_NOT_NECESSARY: "NO_REPLY_NEEDED",
    ReasonCode.SOCIAL_RESPONSE_NOT_NEEDED: "NO_REPLY_NEEDED",
    ReasonCode.CONSERVATIVE_AMBIGUITY: "AMBIGUOUS_ADDRESSEE",
}

_MOTIVE_TO_CAUSE = {
    "NOT_ADDRESSED": "NO_REPLY_NEEDED",
    "AWAITING_CONTINUATION": "AWAITING_CONTINUATION",
    "AVOID_INTERRUPTION": "AVOID_INTERRUPTION",
    "LISTEN_WITHOUT_REPLY": "LISTEN_WITHOUT_REPLY",
    "NO_USEFUL_CONTRIBUTION": "NO_REPLY_NEEDED",
    "UNCERTAIN_CONTEXT": "AMBIGUOUS_ADDRESSEE",
    "UNKNOWN": "UNKNOWN",
}

_REENTRY_TO_CONDITION = {
    "NEXT_DIRECT_ADDRESS": "NEXT_DIRECT_ADDRESS",
    "NEXT_TOPIC_CONTINUATION": "NEXT_TOPIC_CONTINUATION",
    "WHEN_CONTEXT_CLEAR": "WHEN_CONTEXT_CLEAR",
    "NO_FOLLOWUP": "NONE",
}


def derive_nonresponse_trace(
    *,
    event,
    analysis,
    policy_decision,
    conversation_context,
) -> Optional[NonresponseTrace]:
    """Derive the final explanation for a WAIT/IGNORE decision.

    Priority is deliberately fixed and local-first:

    1. permission / processing gate
    2. explicit silence request
    3. quiet mode
    4. trusted human addressee
    5. other local policy restrictions
    6. validated model motive
    7. unknown

    Returns ``None`` for RESPOND. A model motive is never copied into the trace
    when a local rule decided the outcome; ``policy_override`` records the
    actual local cause instead. The model's motive is also never presented as a
    fact about the physical environment.
    """
    action = policy_decision.action
    if action == CognitiveAction.RESPOND:
        return None
    if action not in (CognitiveAction.WAIT, CognitiveAction.IGNORE):
        return None

    reason = policy_decision.reason_code
    model_action = policy_decision.model_action
    override = policy_decision.override_reason or policy_decision.rule

    # 1. Permission / processing gate.
    if reason == ReasonCode.PERMISSION_DENIED:
        return _trace(action, "PERMISSION_DENIED", "LOCAL_POLICY", override=override)

    # 2. Explicit, locally detected silence request. A model-supplied
    # ``explicit_silence_request`` reason without the local detector is only a
    # suggestion and is handled at step 6/7 instead.
    if policy_decision.rule == "explicit_silence":
        return _trace(action, "EXPLICIT_SILENCE", "LOCAL_POLICY", override=override)

    # 3. Quiet mode. It is a local state even when the model also suggested
    # WAIT; a quiet room is the reason no speech happened.
    if conversation_context is not None and getattr(
        conversation_context, "quiet_mode", False
    ):
        return _trace(action, "QUIET_MODE", "LOCAL_POLICY", override="quiet_mode")
    if policy_decision.rule == "quiet_mode":
        return _trace(action, "QUIET_MODE", "LOCAL_POLICY", override="quiet_mode")

    # 4. Trusted explicit human addressee. Again, the trusted caller hint or
    # the local policy rule, never the model's own claim, makes this local.
    if (
        event.addressee_hint == AddresseeHint.HUMAN
        or policy_decision.rule == "explicit_human"
    ):
        return _trace(action, "HUMAN_ADDRESSEE", "LOCAL_POLICY", override=override)

    # 5. Other local policy restrictions (including a RESPOND suggestion
    # downgraded to WAIT, and any local assessment). The model's motive is not
    # reused here: it did not decide this outcome.
    local_provider = str(getattr(analysis, "provider", "")) == "local_fast_path"
    if (
        local_provider
        and getattr(analysis, "conversation_state", "") == "local_wait_request"
    ):
        return _trace(
            action,
            "AWAITING_CONTINUATION",
            "LOCAL_POLICY",
            override=override,
            pending=True,
            reentry="NEXT_TOPIC_CONTINUATION",
        )
    if (
        policy_decision.overrode_model
        or policy_decision.rule != "model_suggestion"
        or local_provider
    ):
        cause = _REASON_TO_CAUSE.get(reason, "UNKNOWN")
        return _trace(action, cause, "LOCAL_POLICY", override=override)

    # 6. A validated model motive, used only when local policy did not override.
    motive = getattr(analysis, "nonresponse_motive", None)
    if isinstance(motive, NonresponseMotive):
        cause = _MOTIVE_TO_CAUSE.get(motive.motive, "UNKNOWN")
        reentry = _REENTRY_TO_CONDITION.get(motive.reentry_hint, "NONE")
        pending = cause == "AWAITING_CONTINUATION"
        if motive.motive == "UNCERTAIN_CONTEXT" and reentry == "NEXT_TOPIC_CONTINUATION":
            pending = True
        return NonresponseTrace(
            final_action=action.value,
            cause_code=cause,
            cause_source="MODEL_ASSESSMENT",
            model_motive=motive.motive,
            policy_override=None,
            evidence_span=motive.evidence_span,
            pending_followup=pending,
            reentry_condition=reentry,
            spoken=False,
        )

    # 7. No model motive. A known model reason code is still a truthful,
    # conservative explanation; otherwise record UNKNOWN rather than inventing
    # a psychological cause.
    cause = _REASON_TO_CAUSE.get(reason, "UNKNOWN")
    return _trace(action, cause, "MODEL_ASSESSMENT", override=None)


def _trace(
    action: CognitiveAction,
    cause: str,
    source: str,
    *,
    override: Optional[str],
    pending: bool = False,
    reentry: str = "NONE",
) -> NonresponseTrace:
    return NonresponseTrace(
        final_action=action.value,
        cause_code=cause,
        cause_source=source,
        model_motive=None,
        policy_override=override,
        evidence_span=None,
        pending_followup=pending,
        reentry_condition=reentry,
        spoken=False,
    )


def build_action_history_record(action) -> Optional[ActionHistoryRecord]:
    """Convert a stored WAIT/IGNORE action into a model-facing record.

    New rows use their persisted trace verbatim. Legacy rows (schema 12) get a
    conservative reason-code mapping with ``LEGACY_UNKNOWN`` provenance; no
    model motive or evidence is invented for them.
    """
    action_type = getattr(action, "action_type", None)
    if action_type not in (CognitiveAction.WAIT, CognitiveAction.IGNORE):
        return None

    trace = getattr(action, "nonresponse_trace", None)
    if isinstance(trace, dict):
        try:
            parsed = NonresponseTrace.model_validate(trace)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and parsed.final_action == action_type.value:
            return ActionHistoryRecord(
                event_id=action.event_id,
                final_action=parsed.final_action,
                cause_code=parsed.cause_code,
                cause_source=parsed.cause_source,
                pending_followup=parsed.pending_followup,
                reentry_condition=parsed.reentry_condition,
            )

    cause = _legacy_cause(action)
    return ActionHistoryRecord(
        event_id=action.event_id,
        final_action=action_type.value,
        cause_code=cause,
        cause_source="LEGACY_UNKNOWN",
        pending_followup=False,
        reentry_condition="NONE",
    )


def _legacy_cause(action) -> str:
    """Map old agent_actions columns to a cause without inventing evidence."""
    override = getattr(action, "override_reason", None) or ""
    reason = getattr(action, "reason_code", None)
    if override == "explicit_silence_request":
        return "EXPLICIT_SILENCE"
    if override == "quiet_mode":
        return "QUIET_MODE"
    if override == "explicit_human_addressee":
        return "HUMAN_ADDRESSEE"
    if reason is not None:
        return _REASON_TO_CAUSE.get(reason, "UNKNOWN")
    return "UNKNOWN"


def action_history_lines(context) -> list[str]:
    """Render real stored action history for a prompt, if any.

    Only records whose source utterance is still present in the filtered,
    truncated conversation are rendered, so a privacy-dropped or truncated turn
    cannot leave an orphaned action behind. The record describes an action, not
    a spoken utterance; callers add that caveat to the prompt.
    """
    records = getattr(context, "action_history", None) or []
    if not records:
        return []
    texts = {
        turn.event_id: turn.text
        for turn in getattr(context, "turns", [])
        if turn.role == "human"
    }
    lines: list[str] = []
    for record in records[:8]:
        text = texts.get(record.event_id)
        if text is None:
            continue
        lines.append(f'- 用户曾说：“{text}”')
        lines.append("- 你没有回复。")
        lines.append(f"- 当时最终决定：{record.final_action}")
        lines.append(f"- 原因代码：{record.cause_code}")
        lines.append(f"- 原因来源：{record.cause_source}")
        lines.append(
            "- 是否有未完成交流："
            + ("是" if record.pending_followup else "否")
        )
        if record.reentry_condition and record.reentry_condition != "NONE":
            lines.append(f"- 重新进入条件：{record.reentry_condition}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines
