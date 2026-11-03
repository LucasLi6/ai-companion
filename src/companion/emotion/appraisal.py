"""Deterministic appraisal: map a turn to an emotion using persistent rules.

Appraisal is *local and deterministic*. It never calls an external model and
never asserts anything about a human's real feelings — it decides the
companion's own simulated affect from:

- the current utterance text;
- the retrieved memories (which may carry a matched ``memory_id``);
- the agent persona/identity;
- the relationship tier to the current speaker;
- the persisted appraisal rules (most specific, highest priority first);
- the current decayed emotion state.

A rule's ``pattern`` is a small JSON schema validated against those inputs. The
default rules handle the common situations directly (milestone, setback,
explicit request, clarification); a model may only *suggest* a new rule through
the reflection engine, and it is never applied to the active profile without
human approval. The ``reason`` string on the result records exactly which rule
fired so every emotional shift is auditable.
"""

from __future__ import annotations

import logging
import re
from typing import Mapping, Optional

from companion.emotion.dynamics import clamp, label_for
from companion.emotion.models import (
    AppraisalResult,
    AppraisalRule,
    EmotionState,
)

# Default, evidence-free rules seeded for every agent. They encode only the
# most ordinary, reversible situations and are the baseline every profile
# starts from. ``source = "system"`` marks them as not learned from any event.
DEFAULT_RULES: list[dict] = [
    {
        "name": "milestone_joy",
        "trigger_type": "utterance",
        "pattern": {
            "keywords": [
                "成功", "完成", "通过了", "搞定", "赢了", "恭喜", "顺利", "好了",
                "第一名", "太棒了", "开心", "达成",
            ]
        },
        "emotion": "JOY",
        "valence_delta": 0.4,
        "arousal_delta": 0.2,
        "intensity": 0.5,
        "priority": 20,
    },
    {
        "name": "setback_concern",
        "trigger_type": "utterance",
        "pattern": {"keywords": ["失败", "坏了", "出问题", "不行", "麻烦", "崩溃", "搞砸", "错误"]},
        "emotion": "CONCERN",
        "valence_delta": -0.3,
        "arousal_delta": 0.1,
        "intensity": 0.4,
        "priority": 20,
    },
    {
        "name": "explicit_gratitude",
        "trigger_type": "utterance",
        "pattern": {"keywords": ["谢谢", "感谢", "太好了", "真棒"]},
        "emotion": "JOY",
        "valence_delta": 0.25,
        "arousal_delta": 0.1,
        "intensity": 0.35,
        "priority": 15,
    },
    {
        "name": "urgent_concern",
        "trigger_type": "utterance",
        "pattern": {"keywords": ["紧急", "救命", "快点", "立刻", "马上", "危险"]},
        "emotion": "CONCERN",
        "valence_delta": -0.25,
        "arousal_delta": 0.5,
        "intensity": 0.7,
        "priority": 30,
    },
    {
        "name": "curiosity_question",
        "trigger_type": "utterance",
        "pattern": {"question": True},
        "emotion": "CURIOSITY",
        "valence_delta": 0.1,
        "arousal_delta": 0.2,
        "intensity": 0.3,
        "priority": 5,
    },
]


class AppraisalEngine:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("companion.emotion.appraisal")

    def appraise(
        self,
        text: str,
        *,
        rules: list[AppraisalRule],
        memories: list[dict] = (),
        agent_context: Optional[Mapping] = None,
        current_state: Optional[EmotionState] = None,
    ) -> AppraisalResult:
        """Choose the best-matching ACTIVE rule and compute the resulting state.

        ``memories`` are the already-authorised, retrieved memories for this
        turn (list of dicts with at least ``memory_id`` and ``content``).
        ``agent_context`` supplies the persona/relationship for expression
        tuning. ``current_state`` (decayed) seeds the intensity/valence, so the
        new state composes with what the companion is already feeling.
        """
        active = sorted(
            (r for r in rules if r.status == "ACTIVE"),
            key=lambda r: (r.priority, r.updated_at),
            reverse=True,
        )
        matched = next((r for r in active if self._matches(r, text, memories)), None)

        if matched is None:
            # No rule fired: hold the decayed state (drift toward baseline).
            state = current_state or EmotionState()
            return AppraisalResult(
                emotion=state.emotion,
                valence=state.valence,
                arousal=state.arousal,
                intensity=state.intensity,
                reason="no_rule_matched",
            )

        base = current_state or EmotionState(
            valence=0.0, arousal=0.0, intensity=0.0, emotion="NEUTRAL"
        )
        valence = clamp(base.valence + matched.valence_delta, -1.0, 1.0)
        arousal = clamp(base.arousal + matched.arousal_delta, 0.0, 1.0)
        intensity = clamp(matched.intensity, 0.0, 1.0)

        matched_memory_id = self._first_matched_memory(matched, memories)

        return AppraisalResult(
            emotion=matched.emotion,
            valence=valence,
            arousal=arousal,
            intensity=intensity,
            trigger_rule_id=matched.rule_id,
            reason=matched.name,
            matched_memory_id=matched_memory_id,
        )

    def _matches(
        self, rule: AppraisalRule, text: str, memories: list[dict]
    ) -> bool:
        pattern = rule.pattern or {}
        if not pattern:
            return False
        if rule.trigger_type == "utterance":
            return self._matches_utterance(pattern, text)
        if rule.trigger_type == "memory":
            return self._matches_memory(pattern, memories)
        return False

    @staticmethod
    def _matches_utterance(pattern: dict, text: str) -> bool:
        keywords = pattern.get("keywords") or []
        if keywords and not any(k.lower() in text.lower() for k in keywords):
            return False
        if pattern.get("question") and not re.search(r"[?？]|吗$|呢$|什么|怎么|为什么", text):
            return False
        if pattern.get("exclamation") and not re.search(r"[!！]", text):
            return False
        return bool(keywords or pattern.get("question") or pattern.get("exclamation"))

    @staticmethod
    def _matches_memory(pattern: dict, memories: list[dict]) -> bool:
        kinds = pattern.get("kinds") or []
        if not kinds:
            return False
        return any(m.get("kind") in kinds for m in memories)

    @staticmethod
    def _first_matched_memory(
        rule: AppraisalRule, memories: list[dict]
    ) -> Optional[str]:
        if rule.trigger_type != "memory":
            return None
        kinds = (rule.pattern or {}).get("kinds") or []
        for memory in memories:
            if memory.get("kind") in kinds:
                return memory.get("memory_id")
        return None


def label_state(valence: float, arousal: float, intensity: float) -> str:
    """Public helper: derive a categorical label from continuous affect."""
    return label_for(valence, arousal, intensity)
