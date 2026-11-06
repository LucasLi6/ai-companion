"""DeepSeek-backed Proactive Judge (V0.9.8).

This is one small semantic judgment, not a second agent. It reuses the
existing :class:`DeepSeekClient` and the ``analysis`` request type. It never
generates Ava's words, never writes memory/profile, and never calls a tool.

Invalid or missing output means "no decision": the manager stays silent and
logs a metadata-only diagnostic. There is deliberately no repair loop and no
retry here — the check is best-effort and silent by design.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from companion.prompts.proactive import (
    PROACTIVE_JUDGE_PROMPT_VERSION,
    build_proactive_judge_messages,
)
from companion.proactive.judge import (
    ProactiveJudgeContext,
    ProactiveJudgeDecision,
    ProactiveJudgeError,
)

_LOGGER = logging.getLogger("companion.proactive.judge.provider")

_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})


class DeepSeekProactiveJudgeProvider:
    """One-shot JSON judgment over the recent real conversation."""

    name = "deepseek-proactive-judge"
    prompt_version = PROACTIVE_JUDGE_PROMPT_VERSION

    def __init__(
        self,
        client,
        *,
        max_tokens: int = 192,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.logger = logger or _LOGGER
        self.last_model = ""

    async def decide(self, payload: ProactiveJudgeContext) -> ProactiveJudgeDecision:
        messages = build_proactive_judge_messages(payload)
        try:
            response = await self.client.complete(
                messages,
                request_type="analysis",
                json_mode=True,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            self.logger.warning(
                "proactive_judge_failed conversation_id=%s error=%s",
                payload.conversation_id,
                type(exc).__name__,
            )
            raise ProactiveJudgeError("judge call failed") from exc

        self.last_model = getattr(response, "model", "") or self.last_model
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason in _TRUNCATED_FINISH_REASONS:
            self._diagnose(payload, finish_reason, "truncated")
            raise ProactiveJudgeError("truncated")

        try:
            decision = self._decode(response.content)
        except (ValueError, TypeError) as exc:
            self._diagnose(payload, finish_reason, type(exc).__name__)
            raise ProactiveJudgeError("invalid output") from exc

        self.logger.info(
            "proactive_judge_decision conversation_id=%s model=%s"
            " prompt_version=%s finish_reason=%s should_initiate=%s",
            payload.conversation_id,
            getattr(response, "model", "") or self.last_model or "unknown",
            self.prompt_version,
            finish_reason or "unknown",
            decision.should_initiate,
        )
        return decision

    @staticmethod
    def _decode(content: str) -> ProactiveJudgeDecision:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("non_object")
        value = data.get("should_initiate")
        if not isinstance(value, bool):
            raise ValueError("should_initiate_not_bool")
        reason = data.get("reason")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            raise ValueError("reason_not_string")
        return ProactiveJudgeDecision(
            should_initiate=value, reason=reason.strip()[:300]
        )

    def _diagnose(
        self, payload: ProactiveJudgeContext, finish_reason, error: str
    ) -> None:
        """Metadata only: no turn text, no scene, no profile."""
        self.logger.warning(
            "proactive_judge_output conversation_id=%s model=%s"
            " prompt_version=%s finish_reason=%s error=%s",
            payload.conversation_id,
            getattr(self, "last_model", "") or "unknown",
            self.prompt_version,
            finish_reason or "unknown",
            error,
        )


class ScriptedProactiveJudgeProvider:
    """Test helper: replay decisions; ``None`` simulates silence on failure."""

    name = "scripted-proactive-judge"
    prompt_version = PROACTIVE_JUDGE_PROMPT_VERSION

    def __init__(self, decisions=None, *, error: Optional[Exception] = None):
        self.decisions = list(decisions or [])
        self.error = error
        self.payloads: list[ProactiveJudgeContext] = []

    async def decide(self, payload: ProactiveJudgeContext) -> ProactiveJudgeDecision:
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        item = self.decisions.pop(0) if self.decisions else False
        if isinstance(item, ProactiveJudgeDecision):
            return item
        return ProactiveJudgeDecision(should_initiate=bool(item), reason="scripted")


__all__ = [
    "DeepSeekProactiveJudgeProvider",
    "ScriptedProactiveJudgeProvider",
]
