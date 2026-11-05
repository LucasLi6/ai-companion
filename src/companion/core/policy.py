"""Behaviour policy: the final arbiter of whether a suggestion may execute.

The cognitive provider *suggests* an action; the Policy layer decides whether
that suggestion is allowed to run. The model can never grant itself
permission, silence or an explicit human addressee.

Rules are applied in this fixed priority order (first match wins):

1. **Permission** — a denied provider result is never turned into a response.
2. **Explicit current-turn silence** — detected locally from the utterance
   (and only locally: a quoted or discussed "don't reply" does not count).
   The result is ``IGNORE`` with no output at all.
3. **Quiet mode** — a ``RESPOND`` suggestion becomes ``WAIT``.
4. **Explicit human addressee** — always ``IGNORE``.
5. **Trusted control state** — currently quiet mode; reserved for future
   conversation controls.
6. **Model suggestion** — used only if none of the above applies.
7. **Conservative ambiguity** — an unaddressed ``RESPOND`` must clear the
   configured necessity/conservatism bar, otherwise it becomes ``WAIT``.

The model's original suggestion is never overwritten; the final action and any
override are recorded next to it for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from companion.domain.enums import CognitiveAction, ReasonCode, ReplyNecessity
from companion.domain.models import CognitiveResult, ConversationContext

RESPONSE_CONSERVATISM_VALUES = ("conservative", "balanced", "responsive")
INTERRUPTION_THRESHOLD_VALUES = ("low", "medium", "high")
CURIOSITY_LEVEL_VALUES = ("off", "low", "medium", "high")

# Documented defaults. All are explicit settings, never random probabilities.
DEFAULT_RESPONSE_CONSERVATISM = "balanced"
DEFAULT_INTERRUPTION_THRESHOLD = "medium"
DEFAULT_CURIOSITY_LEVEL = "low"

SETTING_RESPONSE_CONSERVATISM = "policy.response_conservatism"
SETTING_INTERRUPTION_THRESHOLD = "policy.interruption_threshold"
SETTING_CURIOSITY_LEVEL = "policy.curiosity_level"


@dataclass(frozen=True)
class PolicyConfig:
    response_conservatism: str = DEFAULT_RESPONSE_CONSERVATISM
    interruption_threshold: str = DEFAULT_INTERRUPTION_THRESHOLD
    curiosity_level: str = DEFAULT_CURIOSITY_LEVEL

    def validated(self) -> "PolicyConfig":
        return PolicyConfig(
            response_conservatism=_one_of(
                self.response_conservatism,
                RESPONSE_CONSERVATISM_VALUES,
                DEFAULT_RESPONSE_CONSERVATISM,
            ),
            interruption_threshold=_one_of(
                self.interruption_threshold,
                INTERRUPTION_THRESHOLD_VALUES,
                DEFAULT_INTERRUPTION_THRESHOLD,
            ),
            curiosity_level=_one_of(
                self.curiosity_level,
                CURIOSITY_LEVEL_VALUES,
                DEFAULT_CURIOSITY_LEVEL,
            ),
        )


@dataclass(frozen=True)
class PolicyDecision:
    action: CognitiveAction
    reason_code: ReasonCode
    model_action: CognitiveAction
    override_reason: Optional[str] = None
    rule: str = "model_suggestion"
    # ``"model"`` when the suggestion came from a cognitive provider, or
    # ``"local_fast_path"`` when the turn was routed locally. The field keeps a
    # local decision auditable without pretending a model analysed it.
    source: str = "model"

    @property
    def overrode_model(self) -> bool:
        return self.action != self.model_action


@dataclass(frozen=True)
class TrustedTurnContext:
    """Local facts the semantic policy is allowed to enforce.

    Every field is supplied by trusted program code. Model output can never
    fill these values; in particular the addressee hint comes from the event
    metadata, not from the router's ``addressee_assessment``.
    """

    addressee_hint: Optional[str] = None
    quiet_mode: bool = False
    permission_denied: bool = False
    safety_triggered: bool = False
    local_silence: bool = False


def _one_of(value: str | None, allowed: tuple[str, ...], default: str) -> str:
    if value is None:
        return default
    lowered = value.strip().lower()
    return lowered if lowered in allowed else default


async def load_policy_config(settings) -> PolicyConfig:
    """Read the typed policy configuration from ``app_settings``.

    Unknown or invalid stored values fall back to the documented default
    rather than raising, so a hand-edited database never breaks the pipeline.
    """
    return PolicyConfig(
        response_conservatism=await settings.get(SETTING_RESPONSE_CONSERVATISM)
        or DEFAULT_RESPONSE_CONSERVATISM,
        interruption_threshold=await settings.get(SETTING_INTERRUPTION_THRESHOLD)
        or DEFAULT_INTERRUPTION_THRESHOLD,
        curiosity_level=await settings.get(SETTING_CURIOSITY_LEVEL)
        or DEFAULT_CURIOSITY_LEVEL,
    ).validated()


class Policy:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = (config or PolicyConfig()).validated()

    async def refresh(self, settings) -> PolicyConfig:
        """Reload typed settings; called once per utterance by the orchestrator."""
        self.config = await load_policy_config(settings)
        return self.config

    def decide(
        self,
        analysis: CognitiveResult,
        context: ConversationContext,
        *,
        addressee_hint: str | None = None,
        explicit_silence: bool = False,
    ) -> PolicyDecision:
        model_action = analysis.action
        hint = addressee_hint or "unknown"
        provider_name = str(getattr(analysis, "provider", ""))
        local_source = (
            provider_name
            if provider_name in ("local_fast_path", "local_policy")
            else "model"
        )

        def decision(
            action: CognitiveAction,
            reason: ReasonCode,
            rule: str,
            override: str | None = None,
        ) -> PolicyDecision:
            return PolicyDecision(
                action=action,
                reason_code=reason,
                model_action=model_action,
                override_reason=override if action != model_action else None,
                rule=rule,
                source=local_source,
            )

        # 1. Permission is a hard gate.
        if analysis.reason_code == ReasonCode.PERMISSION_DENIED:
            return decision(
                CognitiveAction.WAIT, ReasonCode.PERMISSION_DENIED, "permission"
            )

        # 2. Explicit, current-turn silence: no output at all.
        if explicit_silence:
            return decision(
                CognitiveAction.IGNORE,
                ReasonCode.EXPLICIT_SILENCE_REQUEST,
                "explicit_silence",
                "explicit_silence_request",
            )

        # 3. Quiet mode downgrades a response, but does not erase understanding.
        if context.quiet_mode and model_action == CognitiveAction.RESPOND:
            return decision(
                CognitiveAction.WAIT, ReasonCode.QUIET_MODE, "quiet_mode", "quiet_mode"
            )

        # 4. Explicit human addressee is a hard constraint.
        if hint == "human":
            return decision(
                CognitiveAction.IGNORE,
                ReasonCode.DIRECT_HUMAN_ADDRESS,
                "explicit_human",
                "explicit_human_addressee",
            )

        # 5. Trusted control state: quiet mode handled above.

        # 5b. Narrow correction for a direct, trusted, required question that
        # the model wrongly deferred for an ambiguous reason. Every hard local
        # gate above has already passed (permission, explicit silence, quiet
        # mode, human addressee), and only the model's *ambiguous* reasons are
        # eligible. This can never let the model's own ``addressed_to`` field
        # or a human-to-human utterance bypass local policy.
        if (
            model_action != CognitiveAction.RESPOND
            and hint == "agent"
            and analysis.reply_necessity == ReplyNecessity.REQUIRED
            and not explicit_silence
            and not context.quiet_mode
            and analysis.reason_code
            in (
                ReasonCode.AMBIGUOUS_ADDRESSEE,
                ReasonCode.UNKNOWN_ADDRESSEE,
                ReasonCode.UNKNOWN,
            )
        ):
            return decision(
                CognitiveAction.RESPOND,
                ReasonCode.DIRECT_AGENT_ADDRESS,
                "direct_question_correction",
                "model_wait_on_direct_question",
            )

        # 6. Model suggestion, then 7. conservative checks on it. A local
        # assessment is labelled as local even when it chose WAIT/IGNORE, so
        # the non-response trace never attributes a local decision to a model.
        if model_action != CognitiveAction.RESPOND:
            return decision(
                model_action,
                analysis.reason_code,
                "local_assessment"
                if local_source != "model"
                else "model_suggestion",
            )

        if analysis.reply_necessity == ReplyNecessity.UNNECESSARY:
            return decision(
                CognitiveAction.WAIT,
                ReasonCode.RESPONSE_NOT_NECESSARY,
                "response_not_necessary",
                "reply_necessity_unnecessary",
            )

        if hint != "agent":
            if not self._may_interject(analysis.reply_necessity):
                return decision(
                    CognitiveAction.WAIT,
                    ReasonCode.CONSERVATIVE_AMBIGUITY,
                    "conservative_ambiguity",
                    "unaddressed_response_below_threshold",
                )

        return decision(
            CognitiveAction.RESPOND,
            analysis.reason_code,
            "fast_path_local_policy"
            if local_source == "local_fast_path"
            else ("local_policy" if local_source != "model" else "model_suggestion"),
        )

    def decide_semantic(
        self,
        route,
        trusted: TrustedTurnContext,
    ) -> PolicyDecision:
        """Apply local authority on top of a validated semantic route.

        The router owns the semantic reading (reply necessity, addressee
        assessment, internal intents). This method cannot re-run text
        classification and does not contain a "direct question" correction:
        a semantic WAIT/IGNORE is never upgraded from a trusted agent hint.

        Local gates, in order: permission, independent safety, quiet mode,
        trusted human addressee, then the semantic action itself.
        """
        model_action = CognitiveAction(route.action.value)
        hint = trusted.addressee_hint or "unknown"
        source = "semantic_router"
        from companion.routing.adapter import route_reason_code

        reason = route_reason_code(route)

        def decision(
            action: CognitiveAction,
            reason: ReasonCode,
            rule: str,
            override: str | None = None,
        ) -> PolicyDecision:
            return PolicyDecision(
                action=action,
                reason_code=reason,
                model_action=model_action,
                override_reason=override if action != model_action else None,
                rule=rule,
                source=source,
            )

        # 1. Permission is a hard gate.
        if trusted.permission_denied:
            return decision(
                CognitiveAction.WAIT, ReasonCode.PERMISSION_DENIED, "permission"
            )

        # 2. Independent safety floor. The local safety reply is generated
        # outside the model; the semantic router does not get to suppress it.
        if trusted.safety_triggered:
            return decision(
                CognitiveAction.RESPOND,
                ReasonCode.SAFETY_RESPONSE,
                "safety_response",
                "local_safety_floor",
            )

        # 3. Explicit, current-turn silence supplied by trusted program code.
        if trusted.local_silence:
            return decision(
                CognitiveAction.IGNORE,
                ReasonCode.EXPLICIT_SILENCE_REQUEST,
                "explicit_silence",
                "explicit_silence_request",
            )

        # 3b. A validated NON_COMMUNICATIVE turn with no internal operation is
        # not a turn to answer. This reads only the router's validated enum and
        # the (already validated) intent/tool lists; it never inspects the raw
        # text. The model's own RESPOND suggestion cannot override this, and a
        # trusted "agent" hint is not evidence of communication either.
        from companion.routing.models import InputSemanticStatus

        if (
            route.input_semantics == InputSemanticStatus.NON_COMMUNICATIVE
            and not route.processing_intents
            and not route.tool_requests
        ):
            return decision(
                CognitiveAction.IGNORE,
                ReasonCode.RESPONSE_NOT_NECESSARY,
                "semantic_noncommunicative",
                "non_communicative_input",
            )

        # 4. Quiet mode downgrades a visible response, but the turn is still
        # understood and internal operations may proceed.
        if trusted.quiet_mode and model_action == CognitiveAction.RESPOND:
            return decision(
                CognitiveAction.WAIT, ReasonCode.QUIET_MODE, "quiet_mode", "quiet_mode"
            )

        # 5. Trusted human addressee is a hard constraint on output.
        if hint == "human" and model_action != CognitiveAction.IGNORE:
            return decision(
                CognitiveAction.IGNORE,
                ReasonCode.CONVERSATION_BETWEEN_HUMANS,
                "trusted_human_addressee",
                "trusted_human_addressee",
            )

        # 6. The semantic decision itself.
        if model_action != CognitiveAction.RESPOND:
            return decision(model_action, reason, "semantic_router")

        if route.reply_necessity == ReplyNecessity.UNNECESSARY:
            return decision(
                CognitiveAction.WAIT,
                ReasonCode.RESPONSE_NOT_NECESSARY,
                "response_not_necessary",
                "reply_necessity_unnecessary",
            )

        if hint != "agent":
            if not self._may_interject(route.reply_necessity):
                return decision(
                    CognitiveAction.WAIT,
                    ReasonCode.CONSERVATIVE_AMBIGUITY,
                    "conservative_ambiguity",
                    "unaddressed_response_below_threshold",
                )

        return decision(CognitiveAction.RESPOND, reason, "semantic_router")

    def _may_interject(self, necessity: ReplyNecessity) -> bool:
        """Whether an unaddressed RESPOND suggestion clears the local bar."""
        if necessity == ReplyNecessity.REQUIRED:
            return True
        if self.config.response_conservatism == "conservative":
            return False
        if self.config.interruption_threshold == "high":
            return False
        if self.config.curiosity_level == "off":
            return False
        if self.config.response_conservatism == "responsive":
            return necessity in (ReplyNecessity.USEFUL, ReplyNecessity.OPTIONAL)
        # balanced
        return necessity == ReplyNecessity.USEFUL


def normalize_addressee(value: str | None) -> str:
    """Only ``agent``/``human`` are meaningful; anything else is unknown."""
    if value and value.strip().lower() in ("agent", "human"):
        return value.strip().lower()
    return "unknown"
