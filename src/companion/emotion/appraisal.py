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

The model's signal is a *suggestion*, never authority. V0.4.1 re-checks the
event's **target** and **social intent** per event type: an insult aimed at a
third party, a quoted insult, playful teasing, or an "insult" with uncertain
intent cannot raise tension toward the agent. Profanity alone is never treated
as hostility; there is no profanity keyword table.

A rule's ``pattern`` is a small JSON schema validated against those inputs. A
model may only suggest a new rule through reflection; it is stored DISABLED and
never applied without human approval. Every invocation reports one of the
outcomes in :data:`OUTCOMES`, so a miss is distinguishable from a change.
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
from companion.emotion.parameters import (
    DEFAULT_CONFIG,
    EmotionDynamicsConfig,
)

# A signal below this confidence is ignored (conservative initial threshold,
# deliberately a named constant so it can be tuned from data later).
SIGNAL_MIN_CONFIDENCE = 0.75

# Auditable outcome taxonomy. The first group means "no change" and names the
# exact reason instead of collapsing every miss into NO_CHANGE.
NO_EVENT = "NO_EVENT"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
INVALID_EVIDENCE = "INVALID_EVIDENCE"
UNSUPPORTED_EVENT = "UNSUPPORTED_EVENT"
RULE_DISABLED = "RULE_DISABLED"
NO_RULE_MATCHED = "NO_RULE_MATCHED"
APPLIED = "APPLIED"

OUTCOMES = (
    NO_EVENT,
    LOW_CONFIDENCE,
    INVALID_EVIDENCE,
    UNSUPPORTED_EVENT,
    RULE_DISABLED,
    NO_RULE_MATCHED,
    APPLIED,
)

# Semantic event -> the persisted default rule that expresses it. A custom rule
# may instead declare ``pattern.event_type`` (or trigger_type="semantic_event").
EVENT_TO_RULE = {
    "achievement": "milestone_joy",
    "setback": "setback_concern",
    "gratitude": "explicit_gratitude",
    "urgency": "urgent_concern",
    "question": "curiosity_question",
    "direct_insult": "direct_insult",
    "repeated_hostility": "repeated_hostility",
    "criticism": "agent_criticism",
    "playful_teasing": "playful_teasing",
    "apology": "agent_apology",
    "reconciliation": "relationship_repair",
    "direct_praise": "direct_praise_joy",
    "self_distress": "user_self_distress",
    "support_request": "support_request_concern",
}

# Which target each event must have to act on *this* conversation. ``UNKNOWN``
# is accepted only for the V0.3 events, where older signals had no target
# field; it is never enough for an interpersonal event.
_TARGET_RULES: dict[str, set[str]] = {
    "achievement": {"SPEAKER", "UNKNOWN"},
    "setback": {"SPEAKER", "UNKNOWN"},
    "gratitude": {"AGENT", "UNKNOWN"},
    "urgency": {"SPEAKER", "UNKNOWN"},
    "question": {"AGENT", "SPEAKER", "OTHER", "UNKNOWN"},
    "direct_insult": {"AGENT"},
    "repeated_hostility": {"AGENT"},
    "criticism": {"AGENT"},
    "playful_teasing": {"AGENT"},
    "apology": {"AGENT"},
    "reconciliation": {"AGENT"},
    "direct_praise": {"AGENT"},
    "self_distress": {"SPEAKER"},
    "support_request": {"AGENT", "UNKNOWN"},
}

# Social intent required per event. In particular, ``direct_insult`` requires
# explicit HOSTILE intent; PLAYFUL/UNCERTAIN stays playful or unsupported and
# never becomes hostility by keyword inference.
_INTENT_RULES: dict[str, set[str]] = {
    "direct_insult": {"HOSTILE"},
    "repeated_hostility": {"HOSTILE"},
    "criticism": {"HOSTILE", "SINCERE", "NEUTRAL", "UNCERTAIN"},
    "playful_teasing": {"PLAYFUL"},
    "apology": {"SINCERE"},
    "reconciliation": {"SINCERE"},
    "direct_praise": {"SINCERE", "PLAYFUL", "NEUTRAL"},
    "self_distress": {"SINCERE", "NEUTRAL", "UNCERTAIN"},
    "support_request": {"SINCERE", "NEUTRAL", "UNCERTAIN"},
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
# None contains profanity or an insult word list: interpersonal events require a
# validated semantic signal with a target and intent.
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
    # -- V0.4.1 interpersonal rules (semantic-event only) -------------------
    {
        "name": "direct_insult",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "direct_insult"},
        "emotion": "FRUSTRATION",
        "valence_delta": -0.3,
        "arousal_delta": 0.35,
        "intensity": 0.55,
        "priority": 60,
    },
    {
        "name": "repeated_hostility",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "repeated_hostility"},
        "emotion": "FRUSTRATION",
        "valence_delta": -0.4,
        "arousal_delta": 0.45,
        "intensity": 0.75,
        "priority": 80,
    },
    {
        "name": "agent_criticism",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "criticism"},
        "emotion": "CONCERN",
        "valence_delta": -0.15,
        "arousal_delta": 0.15,
        "intensity": 0.3,
        "priority": 40,
    },
    {
        "name": "playful_teasing",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "playful_teasing"},
        "emotion": "JOY",
        "valence_delta": 0.2,
        "arousal_delta": 0.2,
        "intensity": 0.25,
        "priority": 35,
    },
    {
        "name": "agent_apology",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "apology"},
        "emotion": "RELIEF",
        "valence_delta": 0.2,
        "arousal_delta": 0.05,
        "intensity": 0.45,
        "priority": 50,
    },
    {
        "name": "relationship_repair",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "reconciliation"},
        "emotion": "RELIEF",
        "valence_delta": 0.25,
        "arousal_delta": 0.05,
        "intensity": 0.4,
        "priority": 50,
    },
    {
        "name": "user_self_distress",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "self_distress"},
        "emotion": "CONCERN",
        "valence_delta": -0.2,
        "arousal_delta": 0.2,
        "intensity": 0.45,
        "priority": 45,
    },
    {
        "name": "direct_praise_joy",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "direct_praise"},
        "emotion": "JOY",
        "valence_delta": 0.3,
        "arousal_delta": 0.15,
        "intensity": 0.4,
        "priority": 45,
    },
    {
        "name": "support_request_concern",
        "trigger_type": "semantic_event",
        "pattern": {"event_type": "support_request"},
        "emotion": "CONCERN",
        "valence_delta": -0.1,
        "arousal_delta": 0.15,
        "intensity": 0.3,
        "priority": 40,
    },
]


class AppraisalEngine:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        *,
        config: EmotionDynamicsConfig | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("companion.emotion.appraisal")
        self.config = config or DEFAULT_CONFIG

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
        config: EmotionDynamicsConfig | None = None,
        addressee_hint: Optional[str] = None,
    ) -> AppraisalResult:
        """Choose the eligible rule and compute the resulting emotional state.

        A *valid* signal is authoritative: none/negated/hypothetical/quoted/
        third-party/low-confidence signals hold the current state. Only when no
        signal exists at all does the legacy keyword path run, and that path
        applies local negation/hypothesis/quotation guards.
        """
        cfg = config or self.config
        active = sorted(
            (r for r in rules if r.status == "ACTIVE"),
            key=lambda r: (r.priority, r.updated_at),
            reverse=True,
        )
        base = current_state or EmotionState(
            valence=0.0, arousal=0.0, intensity=0.0, emotion="NEUTRAL"
        )
        moment = now or datetime.now(timezone.utc)

        if signal is not None:
            diagnostic, validation = self._signal_diagnostic(
                signal, text, base, moment, cfg, addressee_hint=addressee_hint
            )
            if diagnostic is not None:
                return self._unchanged(
                    base,
                    self._legacy_reason(diagnostic),
                    diagnostic=diagnostic,
                    validation=validation,
                    signal=signal,
                )
            event_type = signal.event_type
            if event_type == "direct_insult" and self._hostility_is_repeated(
                base, moment, cfg
            ):
                event_type = "repeated_hostility"
                signal = signal.model_copy(
                    update={"event_type": "repeated_hostility"}
                )
            matched = self._resolve_semantic_rule(event_type, active)
            if matched is None:
                disabled = self._resolve_semantic_rule(event_type, rules)
                if disabled is not None:
                    return self._unchanged(
                        base,
                        self._legacy_reason(RULE_DISABLED),
                        diagnostic=RULE_DISABLED,
                        validation="valid",
                        signal=signal,
                    )
                return self._unchanged(
                    base,
                    self._legacy_reason(NO_RULE_MATCHED),
                    diagnostic=NO_RULE_MATCHED,
                    validation="valid",
                    signal=signal,
                )
            return self._stimulate(
                base,
                matched,
                profile,
                traits,
                moment,
                memories=memories,
                signal=signal,
                cfg=cfg,
            )

        matched = next(
            (r for r in active if self._matches(r, text, memories)), None
        )
        if matched is None:
            return self._unchanged(
                base,
                "no_rule_matched",
                diagnostic=NO_RULE_MATCHED,
            )
        return self._stimulate(
            base,
            matched,
            profile,
            traits,
            moment,
            memories=memories,
            cfg=cfg,
        )

    # -- signal handling ----------------------------------------------------

    def _signal_diagnostic(
        self,
        signal,
        text: str,
        state: EmotionState,
        now: datetime,
        config: EmotionDynamicsConfig,
        *,
        addressee_hint: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        """Return ``(outcome, validation)``; ``outcome=None`` means actionable."""
        event_type = getattr(signal, "event_type", "none")
        if event_type == "none":
            return NO_EVENT, "none"
        if event_type not in EVENT_TO_RULE:
            return UNSUPPORTED_EVENT, "unsupported_event"
        try:
            confidence = float(getattr(signal, "confidence", 0.0))
        except (TypeError, ValueError):
            return LOW_CONFIDENCE, "invalid_confidence"
        if not (confidence >= SIGNAL_MIN_CONFIDENCE):
            return LOW_CONFIDENCE, "low_confidence"
        if getattr(signal, "factuality", "uncertain") != "asserted":
            return UNSUPPORTED_EVENT, "not_asserted"
        if getattr(signal, "subject", "unknown") != "speaker":
            return UNSUPPORTED_EVENT, "unsupported_subject"

        target = str(getattr(signal, "target", "UNKNOWN") or "UNKNOWN").upper()
        intent = str(
            getattr(signal, "social_intent", "UNCERTAIN") or "UNCERTAIN"
        ).upper()
        if target not in _TARGET_RULES.get(event_type, set()):
            return UNSUPPORTED_EVENT, "unsupported_target"
        if intent not in _INTENT_RULES.get(event_type, {intent}):
            return UNSUPPORTED_EVENT, "unsupported_social_intent"
        # Trusted addressee routing: when the session says this turn is meant
        # for a human/unknown listener, a model claim that it targets the agent
        # is not enough. ``None`` keeps old callers backward compatible; the
        # orchestrator always supplies the trusted value.
        if (
            target == "AGENT"
            and addressee_hint is not None
            and str(addressee_hint).strip().lower() != "agent"
        ):
            return UNSUPPORTED_EVENT, "not_addressed_to_agent"
        if (
            event_type == "repeated_hostility"
            and not self._hostility_is_repeated(state, now, config)
        ):
            return UNSUPPORTED_EVENT, "no_local_repeat_evidence"

        evidence = getattr(signal, "evidence_span", None)
        if not evidence or not isinstance(evidence, str) or evidence not in text:
            return INVALID_EVIDENCE, "evidence_not_in_text"
        return None, "valid"

    @staticmethod
    def _legacy_reason(diagnostic: str) -> str:
        return {
            NO_EVENT: "signal_not_actionable",
            LOW_CONFIDENCE: "signal_not_actionable",
            INVALID_EVIDENCE: "signal_not_actionable",
            UNSUPPORTED_EVENT: "signal_not_actionable",
            RULE_DISABLED: "no_semantic_rule",
            NO_RULE_MATCHED: "no_semantic_rule",
        }.get(diagnostic, diagnostic.lower())

    @staticmethod
    def _hostility_is_repeated(
        state: EmotionState,
        now: datetime,
        config: EmotionDynamicsConfig,
    ) -> bool:
        """Local repeat evidence, independent of the model's claim.

        A single hostile turn is not "repeated hostility". The previous event
        must be a hostile event within the configured window; a stale event or
        a missing timestamp never counts.
        """
        if state.last_event_type not in ("direct_insult", "repeated_hostility"):
            return False
        if state.last_event_at is None:
            return False
        left = state.last_event_at
        if left.tzinfo is None:
            left = left.replace(tzinfo=timezone.utc)
        right = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        elapsed = (right.astimezone(timezone.utc) - left.astimezone(timezone.utc)).total_seconds()
        return 0.0 <= elapsed <= config.repeated_event_window_seconds

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
        signal=None,
        cfg: EmotionDynamicsConfig | None = None,
    ) -> AppraisalResult:
        cfg = cfg or self.config
        gain, empathy = reaction_gain(profile, traits, cfg)
        empathy_factor = (
            0.8 + 0.4 * empathy if rule.emotion in ("CONCERN", "SADNESS") else 1.0
        )
        effective_gain = clamp(gain * empathy_factor, 0.0, 2.0)
        if effective_gain <= 0.0:
            # A completely unresponsive profile produces no emotional stimulus.
            return self._unchanged(
                base,
                "no_reaction",
                diagnostic=NO_RULE_MATCHED,
                signal=signal,
            )

        moment = now or base.updated_at or datetime.now(timezone.utc)
        state = apply_emotional_stimulus(
            base, rule, profile, traits, moment, config=cfg
        )
        return AppraisalResult(
            emotion=state.emotion,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
            trigger_rule_id=rule.rule_id,
            reason=rule.name,
            matched_memory_id=self._first_matched_memory(rule, memories),
            diagnostic=APPLIED,
            event_type=getattr(signal, "event_type", None),
            target=getattr(signal, "target", None),
            social_intent=getattr(signal, "social_intent", None),
            validation="valid" if signal is not None else "keyword",
            acute_before=base.intensity,
            acute_after=state.intensity,
            mood_after=state.mood_intensity,
            tension_after=state.relationship_tension,
            state=state,
        )

    @staticmethod
    def _unchanged(
        state: EmotionState,
        reason: str,
        *,
        diagnostic: str = "NO_CHANGE",
        validation: str = "none",
        signal=None,
    ) -> AppraisalResult:
        return AppraisalResult(
            emotion=state.emotion,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
            reason=reason,
            diagnostic=diagnostic,
            event_type=getattr(signal, "event_type", None),
            target=getattr(signal, "target", None),
            social_intent=getattr(signal, "social_intent", None),
            validation=validation,
            acute_before=state.intensity,
            acute_after=state.intensity,
            mood_after=state.mood_intensity,
            tension_after=state.relationship_tension,
            state=state,
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
