"""The one-shot Proactive Judge decision (V0.9.8).

The judge answers exactly one question: should Ava speak again right now?
It never produces the words Ava would say. A failed or invalid judgment is
treated as "no decision" and therefore as silence; there is no retry loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol


class ProactiveJudgeError(Exception):
    """The judge could not produce a usable decision."""


@dataclass(frozen=True)
class ProactiveJudgeDecision:
    should_initiate: bool
    reason: str = ""


@dataclass(frozen=True)
class ProactiveJudgeContext:
    """Everything the judge may see: recent real turns + committed scene.

    There is deliberately no full memory retrieval and no full profile here.
    The task is to read the current exchange, not to re-learn the person.
    """

    conversation_id: str
    person_id: str
    recent_turns: list[dict] = field(default_factory=list)
    last_ava_response: str = ""
    scene_context: Optional[dict] = None
    seconds_since_last_response: float = 0.0

    def as_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "person_id": self.person_id,
            "recent_turns": self.recent_turns,
            "last_ava_response": self.last_ava_response,
            "scene_context": self.scene_context,
            "seconds_since_last_response": round(
                float(self.seconds_since_last_response), 3
            ),
        }


class ProactiveJudgeProvider(Protocol):
    async def decide(
        self, context: ProactiveJudgeContext
    ) -> ProactiveJudgeDecision: ...


class ProactiveJudge:
    """Validate-and-contain wrapper around one judge provider call."""

    def __init__(
        self,
        provider: ProactiveJudgeProvider,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.provider = provider
        self.logger = logger or logging.getLogger("companion.proactive.judge")

    async def decide(
        self, context: ProactiveJudgeContext
    ) -> Optional[ProactiveJudgeDecision]:
        """Return a decision, or ``None`` when the check must stay silent."""
        try:
            decision = await self.provider.decide(context)
        except ProactiveJudgeError as exc:
            self.logger.info(
                "proactive_judge_no_decision conversation_id=%s reason=%s",
                context.conversation_id,
                exc,
            )
            return None
        except Exception as exc:  # never let a judge failure affect chat
            self.logger.warning(
                "proactive_judge_error conversation_id=%s error=%s",
                context.conversation_id,
                type(exc).__name__,
            )
            return None
        if not isinstance(decision, ProactiveJudgeDecision):
            self.logger.warning(
                "proactive_judge_invalid_type conversation_id=%s",
                context.conversation_id,
            )
            return None
        return decision


__all__ = [
    "ProactiveJudge",
    "ProactiveJudgeContext",
    "ProactiveJudgeDecision",
    "ProactiveJudgeError",
    "ProactiveJudgeProvider",
]
