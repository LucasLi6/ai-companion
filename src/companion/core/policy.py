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
        local_source = (
            "local_fast_path"
            if str(getattr(analysis, "provider", "")) == "local_fast_path"
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
                if local_source == "local_fast_path"
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
            "fast_path_local_policy" if local_source == "local_fast_path" else "model_suggestion",
        )

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
