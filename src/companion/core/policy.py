"""Behaviour policy: the final arbiter of whether a suggestion may execute.

The cognitive provider *suggests* an action; the Policy layer decides whether
that suggestion is allowed to run. This is the boundary that keeps mock (or,
later, DeepSeek) output from being executed unchecked. V0.0 enforces one rule:
a ``RESPOND`` suggestion is suppressed while the conversation is in quiet mode.
"""

from __future__ import annotations

from companion.domain.enums import CognitiveAction, ReasonCode
from companion.domain.models import CognitiveResult, ConversationContext


class Policy:
    def decide(
        self, analysis: CognitiveResult, context: ConversationContext
    ) -> tuple[CognitiveAction, ReasonCode]:
        if analysis.action == CognitiveAction.RESPOND and context.quiet_mode:
            return CognitiveAction.WAIT, ReasonCode.QUIET_MODE
        return analysis.action, analysis.reason_code
