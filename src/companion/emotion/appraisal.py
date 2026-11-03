"""Deterministic appraisal: map a turn to an emotion using persistent rules.

Appraisal is *local and deterministic*. It never calls an external model and
never asserts anything about a human's real feelings — it decides the
companion's own simulated affect from:

- a locally validated :class:`AffectiveSignal` (the model's semantic event hint);
- the current utterance text (checked for negation/hypothesis/quotation when no
  signal is available);
- the retrieved, already-authorised memories;
- the agent persona/relationship;
- the persisted ACTIVE appraisal rules;
- the current decayed emotion state and profile.

The model's signal is a *suggestion*, never authority: this engine re-checks
its whitelist values, confidence, subject, factuality and evidence span before
using it. A rule's ``pattern`` is a small JSON schema validated against those
inputs. A model may only suggest a new rule through reflection; it is stored
DISABLED and never applied without human approval.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Mapping, Optional

from companion.emotion.dynamics import (
    apply_emotional_stimulus,
    clamp,
    label_for,
    reaction_gain,
)
from companion.emotion.models import (
    AppraisalResult,
    AppraisalRule,
    EmotionProfile,
    EmotionState,
)

# A signal below this confidence is ignored (conservative initial threshold,
# deliberately a named constant so it can be tuned from data later).
SIGNAL_MIN_CONFIDENCE = 0.75

# Semantic event -> the persisted default rule that expresses it. A custom rule
# may instead declare ``pattern.event_type`` (or trigger_type="semantic_event").
EVENT_TO_RULE = {
    "achievement": "milestone_joy",
    "setback": "setback_concern",
    "gratitude": "explicit_gratitude",
    "urgency": "urgent_concern",
    "question": "curiosity_question",
}

# Words whose "asserted" status is particularly easy to get wrong: seeing them
# inside a question or a negation must not create a positive/negative event.
_AFFECT_WORDS = {
    "开心", "高兴", "快乐", "幸福", "难过", "伤心", "成功", "失败",
    "完成", "搞定", "赢了", "输", "恭喜",
}
_NEGATION_MARKERS = (
    "没有", "没", "未", "不", "别", "尚未", "还没", "并不", "从没",
    "不想", "不愿", "不要", "避免", "害怕", "担心", "差点", "差一点", "几乎",
)
_HYPOTHETICAL_MARKERS = ("如果", "假如", "要是", "若是", "万一", "希望", "梦想", "想要")
_REPORT_MARKERS = ("他说", "她说", "他说过", "她说过", "据说", "听说", "别人说", "转述", "引用")
_UNCERTAIN_MARKERS = ("好像", "似乎", "可能", "也许", "大概", "不确定")
_QUESTION_MARKERS = re.compile(r"[?？]|吗$|呢$|为什么|怎么|是不是|觉得|认为")

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
        signal=None,
        memories: list[dict] = (),
        agent_context: Optional[Mapping] = None,
        current_state: Optional[EmotionState] = None,
        profile: Optional[EmotionProfile] = None,
        traits: Optional[Mapping[str, int]] = None,
        now: Optional[datetime] = None,
    ) -> AppraisalResult:
        """Choose the eligible rule and compute the resulting emotional state.

        A *valid* signal is authoritative: none/negated/hypothetical/quoted/
        third-party/low-confidence signals hold the current state. Only when no
        signal exists at all does the legacy keyword path run, and that path
        applies local negation/hypothesis/quotation guards.
        """
        active = sorted(
            (r for r in rules if r.status == "ACTIVE"),
            key=lambda r: (r.priority, r.updated_at),
            reverse=True,
        )
        base = current_state or EmotionState(
            valence=0.0, arousal=0.0, intensity=0.0, emotion="NEUTRAL"
        )

        if signal is not None:
            if not self._signal_actionable(signal, text):
                return self._unchanged(base, "signal_not_actionable")
            matched = self._resolve_semantic_rule(signal.event_type, active)
            if matched is None:
                return self._unchanged(base, "no_semantic_rule")
            return self._stimulate(
                base, matched, profile, traits, now, memories=memories
            )

        matched = next(
            (r for r in active if self._matches(r, text, memories)), None
        )
        if matched is None:
            return self._unchanged(base, "no_rule_matched")
        return self._stimulate(
            base, matched, profile, traits, now, memories=memories
        )

    # -- signal handling ----------------------------------------------------

    @staticmethod
    def _signal_actionable(signal, text: str) -> bool:
        if getattr(signal, "event_type", "none") == "none":
            return False
        if getattr(signal, "factuality", "uncertain") != "asserted":
            return False
        if getattr(signal, "subject", "unknown") != "speaker":
            return False
        try:
            confidence = float(getattr(signal, "confidence", 0.0))
        except (TypeError, ValueError):
            return False
        if not (confidence >= SIGNAL_MIN_CONFIDENCE):
            return False
        evidence = getattr(signal, "evidence_span", None)
        return bool(evidence) and evidence in text

    @staticmethod
    def _resolve_semantic_rule(
        event_type: str, active: list[AppraisalRule]
    ) -> Optional[AppraisalRule]:
        expected_name = EVENT_TO_RULE.get(event_type)
        candidates = [
            rule
            for rule in active
            if rule.name == expected_name
            or (rule.pattern or {}).get("event_type") == event_type
            or (
                rule.trigger_type == "semantic_event"
                and (rule.pattern or {}).get("event") == event_type
            )
        ]
        # ``active`` is already priority-ordered; a DISABLED rule is absent.
        return candidates[0] if candidates else None

    # -- state calculation --------------------------------------------------

    def _stimulate(
        self,
        base: EmotionState,
        rule: AppraisalRule,
        profile: Optional[EmotionProfile],
        traits: Optional[Mapping[str, int]],
        now: Optional[datetime],
        *,
        memories: list[dict],
    ) -> AppraisalResult:
        gain, empathy = reaction_gain(profile, traits)
        empathy_factor = (
            0.8 + 0.4 * empathy if rule.emotion in ("CONCERN", "SADNESS") else 1.0
        )
        effective_gain = clamp(gain * empathy_factor, 0.0, 2.0)
        if effective_gain <= 0.0:
            # A completely unresponsive profile produces no emotional stimulus.
            return self._unchanged(base, "no_reaction")

        moment = now or base.updated_at or datetime.now(timezone.utc)
        state = apply_emotional_stimulus(base, rule, profile, traits, moment)
        return AppraisalResult(
            emotion=state.emotion,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
            trigger_rule_id=rule.rule_id,
            reason=rule.name,
            matched_memory_id=self._first_matched_memory(rule, memories),
        )

    @staticmethod
    def _unchanged(state: EmotionState, reason: str) -> AppraisalResult:
        return AppraisalResult(
            emotion=state.emotion,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
            reason=reason,
        )

    # -- legacy keyword matching -------------------------------------------

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

    @classmethod
    def _matches_utterance(cls, pattern: dict, text: str) -> bool:
        keywords = pattern.get("keywords") or []
        if keywords and not any(
            cls._keyword_is_asserted(text, keyword) for keyword in keywords
        ):
            return False
        if pattern.get("question") and not _QUESTION_MARKERS.search(text):
            return False
        if pattern.get("exclamation") and not re.search(r"[!！]", text):
            return False
        return bool(keywords or pattern.get("question") or pattern.get("exclamation"))

    @classmethod
    def _keyword_is_asserted(cls, text: str, keyword: str) -> bool:
        """Whether ``keyword`` occurs as a genuine first-person assertion.

        Small and deliberately conservative: it rejects occurrences inside
        quotes, after negation/hypothetical/report/uncertainty markers, and
        affect words inside interrogative framing. When in doubt it does *not*
        claim an event happened.
        """
        if not keyword:
            return False
        lowered = text.lower()
        needle = keyword.lower()
        start = lowered.find(needle)
        while start != -1:
            end = start + len(needle)
            prefix = text[max(0, start - 8) : start]
            suffix = text[end : end + 5]

            if cls._inside_quotes(text, start):
                start = lowered.find(needle, start + 1)
                continue
            if any(marker in prefix for marker in _NEGATION_MARKERS):
                start = lowered.find(needle, start + 1)
                continue
            if any(marker in prefix for marker in _HYPOTHETICAL_MARKERS):
                start = lowered.find(needle, start + 1)
                continue
            if any(marker in prefix for marker in _REPORT_MARKERS):
                start = lowered.find(needle, start + 1)
                continue
            if cls._has_other_subject(text, start):
                start = lowered.find(needle, start + 1)
                continue
            if any(marker in prefix for marker in _UNCERTAIN_MARKERS):
                start = lowered.find(needle, start + 1)
                continue
            if any(marker in suffix for marker in ("就好了", "的话", "就好了")):
                start = lowered.find(needle, start + 1)
                continue
            if (
                keyword in _AFFECT_WORDS
                and _QUESTION_MARKERS.search(text)
                and not re.search(r"[!！]", text)
            ):
                start = lowered.find(needle, start + 1)
                continue
            return True
        return False

    @staticmethod
    def _has_other_subject(text: str, start: int) -> bool:
        """Whether the clause's last explicit subject is a third party.

        "他成功了" must not become the companion's JOY, while
        "我告诉他我成功了" still refers to the speaker. A clause boundary resets
        the check, and a later first-person pronoun wins.
        """
        boundary = max(
            (text.rfind(mark, 0, start) for mark in "，。！？；,.!?;"), default=-1
        )
        clause = text[boundary + 1 : start]
        last_first = max(clause.rfind(m) for m in ("我", "我们"))
        last_third = max(
            clause.rfind(m) for m in ("他", "她", "它", "别人", "对方", "那个人")
        )
        return last_third != -1 and last_third > last_first

    @staticmethod
    def _inside_quotes(text: str, start: int) -> bool:
        for opening, closing in (("“", "”"), ("「", "」"), ("『", "』")):
            if text.rfind(opening, 0, start) > text.rfind(closing, 0, start):
                return True
        # ASCII double quotes use the same character for both ends.
        return text.count('"', 0, start) % 2 == 1

    @staticmethod
    def _matches_memory(pattern: dict, memories: list[dict]) -> bool:
        """A memory rule needs an explicit, verified memory association.

        Matching by ``kind`` alone cannot prove the memory caused this turn, so
        it is no longer accepted. A rule must name the memory ids it applies to.
        """
        memory_ids = set(pattern.get("memory_ids") or [])
        if not memory_ids:
            return False
        return any(
            memory.get("memory_id") in memory_ids and memory.get("status") == "ACTIVE"
            for memory in memories
        )

    @staticmethod
    def _first_matched_memory(
        rule: AppraisalRule, memories: list[dict]
    ) -> Optional[str]:
        if rule.trigger_type != "memory":
            return None
        memory_ids = set((rule.pattern or {}).get("memory_ids") or [])
        for memory in memories:
            if (
                memory.get("memory_id") in memory_ids
                and memory.get("status") == "ACTIVE"
            ):
                return memory.get("memory_id")
        return None


def label_state(valence: float, arousal: float, intensity: float) -> str:
    """Public helper: derive a categorical label from continuous affect."""
    return label_for(valence, arousal, intensity)
