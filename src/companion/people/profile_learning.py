"""Deterministic profile learning from one utterance.

Every turn feeds this module *locally*; it never calls a model, so learning
adds no latency and no extra API cost. The existing background extraction job
remains responsible for long-term memory facts; this layer only interprets
explicit self-report and records unconfirmed inferences.

Decision rules (V0.7):

- A stated communication preference (``不用这么正式``) updates the profile
  immediately and is attributed to a direct self report.
- A stated fact (``我现在在学习嵌入式开发``) is *not* promoted to an effective
  attribute. It becomes a candidate, anchored to the event, and is linked to
  the confirmed memory later. This is what keeps "learning about embedded
  systems" from silently becoming ``occupation = embedded engineer``.
- Ava's own inference (tone, brevity) also becomes a candidate. A single turn
  or a model-generated confidence value can never confirm it.
- Repeated independent evidence may confirm a candidate. A user correction
  outranks that inference and rejects its pending candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from companion.people.profile_models import (
    LIST_FIELDS,
    ProfileFieldKey,
    ProfileSource,
)
from companion.people.profile_service import (
    PersonProfileService,
    ProfileUnauthorized,
)

# Three independent turns is the minimum for automatic confirmation. Model
# confidence is deliberately not part of this decision.
CONFIRMATION_EVIDENCE_EVENTS = 3

_MISSING = object()

_CORRECTION_MARKERS = (
    "不是",
    "不对",
    "说错",
    "更正",
    "纠正",
    "改一下",
    "其实我",
    "不用这么",
    "别再",
    "以后不要",
    "记错了",
)

_PREFERENCE_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            r"(?:不用|无需|别|不要)(?:这么|那么)?正式"
            r"|(?:随意|随便|放松|自然)(?:一点|点|些)?"
        ),
        ProfileFieldKey.FORMALITY.value,
        "casual",
    ),
    (
        re.compile(r"(?:正式|严肃)(?:一点|点|些)?"),
        ProfileFieldKey.FORMALITY.value,
        "formal",
    ),
    (
        re.compile(r"(?:友好|亲切|温柔)(?:一点|点|些)?"),
        ProfileFieldKey.TONE.value,
        "friendly",
    ),
    (
        re.compile(r"(?:专业)(?:一点|点|些)?"),
        ProfileFieldKey.TONE.value,
        "professional",
    ),
    (
        re.compile(r"(?:俏皮|活泼|轻松)(?:一点|点|些)?"),
        ProfileFieldKey.TONE.value,
        "playful",
    ),
    (
        re.compile(
            r"(?:不用|无需|别|不要)(?:再)?开玩笑"
            r"|(?:少|不要)(?:一点|点)?幽默|不要幽默"
        ),
        ProfileFieldKey.HUMOR.value,
        "avoid",
    ),
    (
        re.compile(r"(?:可以|多)(?:一点|点)?开玩笑|多点幽默|幽默(?:一点|点)?"),
        ProfileFieldKey.HUMOR.value,
        "welcome",
    ),
    (
        re.compile(
            r"(?:简洁|简短|简单|精简)(?:一点|点|些)?"
            r"|别(?:太|那么)?长|说重点|短一点"
        ),
        ProfileFieldKey.DEPTH.value,
        "concise",
    ),
    (
        re.compile(r"(?:详细|具体|展开)(?:一点|点|些)?|多说一点|讲详细"),
        ProfileFieldKey.DEPTH.value,
        "detailed",
    ),
    (
        re.compile(r"(?:用|说|讲)英文|英文(?:回答|交流|回复)"),
        ProfileFieldKey.LANGUAGE.value,
        "en",
    ),
    (
        re.compile(r"(?:用|说|讲)中文|中文(?:回答|交流|回复)"),
        ProfileFieldKey.LANGUAGE.value,
        "zh",
    ),
)

_NEGATION_BEFORE = ("不要", "别", "不用", "无需", "不想")

_ADDRESS_RE = re.compile(
    r"(?:以后|请)?(?:叫我|称呼我|喊我)\s*([^\s，。！？,.!?]{1,20})"
)

_FACT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"我的(?:职业|工作)是\s*([^，。！？,.!?]{1,20})"),
        ProfileFieldKey.OCCUPATION.value,
    ),
    (
        re.compile(
            r"我是(?:一个|一名)?\s*([^，。！？,.!?]{1,20}?"
            r"(?:工程师|老师|医生|护士|设计师|程序员|学生|律师|会计))"
        ),
        ProfileFieldKey.OCCUPATION.value,
    ),
    (
        re.compile(r"我(?:想|打算|计划|希望|准备)([^，。！？,.!?]{2,40})"),
        ProfileFieldKey.PERSONAL_GOALS.value,
    ),
    (
        re.compile(r"我(?:经常|平时|每天|通常|习惯)([^，。！？,.!?]{2,40})"),
        ProfileFieldKey.USUAL_ACTIVITIES.value,
    ),
    (
        re.compile(
            r"(?:我)?(?:现在|目前|最近|这阵子)"
            r"(?:正在|在|一直)?(?:做|开发|写|搞|研究|学习|准备)"
            r"([^，。！？,.!?]{2,40})"
        ),
        ProfileFieldKey.ONGOING_PROJECTS.value,
    ),
    (
        re.compile(r"我(?:喜欢|爱好是|感兴趣的是|对)([^，。！？,.!?]{2,40})"),
        ProfileFieldKey.INTERESTS.value,
    ),
)


@dataclass(frozen=True)
class ProfileSignal:
    field_key: str
    value: Any
    kind: str  # "preference" or "fact"
    explicit: bool = True


@dataclass
class LearningResult:
    authorized: bool = False
    preferences: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def is_correction(text: str) -> bool:
    value = str(text or "")
    return any(marker in value for marker in _CORRECTION_MARKERS)


def _negated(text: str, start: int) -> bool:
    window = text[max(0, start - 5) : start]
    return any(marker in window for marker in _NEGATION_BEFORE)


def extract_signals(text: str) -> list[ProfileSignal]:
    """Extract explicit preference and fact signals, deterministically.

    Only one value per field is kept (the first explicit statement), so a
    sentence cannot fight with itself. Nothing here infers attributes from
    demographics or from the model's interpretation.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return []
    signals: list[ProfileSignal] = []
    seen: set[str] = set()

    address = _ADDRESS_RE.search(value)
    if address is not None:
        label = address.group(1).strip(" ，。！？,.!?")
        if label:
            signals.append(
                ProfileSignal(ProfileFieldKey.ADDRESS.value, label, "preference")
            )
            seen.add(ProfileFieldKey.ADDRESS.value)

    for pattern, field_key, field_value in _PREFERENCE_RULES:
        if field_key in seen:
            continue
        match = pattern.search(value)
        if match is None:
            continue
        if _negated(value, match.start()):
            continue
        signals.append(ProfileSignal(field_key, field_value, "preference"))
        seen.add(field_key)

    for pattern, field_key in _FACT_RULES:
        if field_key in seen:
            continue
        match = pattern.search(value)
        if match is None:
            continue
        label = match.group(1).strip(" ，。！？,.!?")
        if not label:
            continue
        signals.append(ProfileSignal(field_key, label, "fact"))
        seen.add(field_key)

    return signals


class PersonProfileLearningService:
    def __init__(
        self,
        profiles: PersonProfileService,
        *,
        confirmation_threshold: int = CONFIRMATION_EVIDENCE_EVENTS,
    ) -> None:
        self.profiles = profiles
        self.confirmation_threshold = confirmation_threshold

    def observe_signals(self, text: str) -> list[ProfileSignal]:
        """Extract optional language signals *without* writing anything.

        The normal conversation path uses this only for diagnostics. A regex
        match is never permission to change a permanent profile: the single
        authoritative path is the background Profile Review, which reads the
        complete profile first and lets local rules validate every change.
        """
        return extract_signals(text)

    async def observe(
        self,
        person_id: str,
        text: str,
        *,
        event_id: Optional[str] = None,
        actor: str = "learning",
    ) -> LearningResult:
        """Learn from one speaker turn, within their explicit consent."""
        result = LearningResult(authorized=await self.profiles.can_learn(person_id))
        if not result.authorized:
            return result

        correction = is_correction(text)
        for signal in extract_signals(text):
            try:
                if signal.kind == "preference":
                    changed = await self._apply_preference(
                        person_id,
                        signal,
                        event_id=event_id,
                        correction=correction,
                        actor=actor,
                    )
                    if changed:
                        if correction:
                            result.corrections.append(signal.field_key)
                        else:
                            result.preferences.append(signal.field_key)
                    else:
                        result.skipped.append(signal.field_key)
                else:
                    candidate = await self.profiles.propose_candidate(
                        person_id,
                        signal.field_key,
                        signal.value,
                        source=(
                            ProfileSource.USER_CORRECTION
                            if correction
                            else ProfileSource.DIRECT_SELF_REPORT
                        ),
                        confidence=0.9 if correction else 0.7,
                        event_id=event_id,
                        reason="explicit_self_report",
                        actor=actor,
                    )
                    if candidate is None:
                        result.skipped.append(signal.field_key)
                    else:
                        result.candidates.append(candidate)
                        await self._maybe_confirm(candidate, actor=actor)
            except (ProfileUnauthorized, ValueError):
                result.skipped.append(signal.field_key)
        return result

    async def _apply_preference(
        self,
        person_id: str,
        signal: ProfileSignal,
        *,
        event_id: Optional[str],
        correction: bool,
        actor: str,
    ) -> bool:
        profile = await self.profiles.get(person_id)
        active = profile.fields.get(signal.field_key) if profile is not None else None
        if correction and active is not None and active.value != signal.value:
            await self.profiles.correct_field(
                person_id,
                signal.field_key,
                value=signal.value,
                event_id=event_id,
                reason="explicit_correction",
                actor=actor,
            )
            return True
        before = getattr(profile, "metadata", None)
        await self.profiles.set_field(
            person_id,
            signal.field_key,
            signal.value,
            source=ProfileSource.DIRECT_SELF_REPORT,
            confidence=1.0,
            event_id=event_id,
            reason="explicit_self_report",
            actor=actor,
        )
        after = await self.profiles.get(person_id)
        return (
            before is None
            or after is None
            or after.metadata.revision != profile.metadata.revision
        )

    async def record_inference(
        self,
        person_id: str,
        field_key: str,
        value: Any,
        *,
        event_id: Optional[str] = None,
        confidence: float = 0.5,
        memory_ids: Iterable[str] = (),
        reason: Optional[str] = None,
    ) -> Optional[str]:
        """Record Ava's own inference as a candidate (never effective)."""
        candidate = await self.profiles.propose_candidate(
            person_id,
            field_key,
            value,
            source=ProfileSource.INTERACTION_INFERENCE,
            confidence=confidence,
            event_id=event_id,
            memory_ids=list(memory_ids),
            reason=reason or "interaction_inference",
        )
        if candidate is not None:
            await self._maybe_confirm(candidate)
        return candidate

    async def apply_correction(
        self,
        person_id: str,
        field_key: str,
        value: Any = _MISSING,
        *,
        revoke: bool = False,
        event_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        """Correction path used when the correction target is known.

        An effective value is corrected immediately; a still-pending fact is
        rejected and replaced with a correction-sourced candidate instead of
        being promoted without confirmation.
        """
        profile = await self.profiles.get(person_id)
        active = profile.fields.get(field_key) if profile is not None else None
        if revoke or active is not None:
            kwargs: dict[str, Any] = {
                "revoke": revoke,
                "event_id": event_id,
                "reason": reason or "user_correction",
            }
            if value is not _MISSING:
                kwargs["value"] = value
            await self.profiles.correct_field(person_id, field_key, **kwargs)
            return "corrected"
        if value is _MISSING:
            return "ignored"
        for candidate in await self.profiles.repo.find_pending(person_id, field_key):
            await self.profiles.reject_candidate(
                candidate["candidate_id"], reason="user_correction"
            )
        await self.profiles.propose_candidate(
            person_id,
            field_key,
            value,
            source=ProfileSource.USER_CORRECTION,
            confidence=1.0,
            event_id=event_id,
            reason=reason or "user_correction",
        )
        return "candidate"

    async def link_output_memory(
        self, person_id: str, event_id: str, memory_id: str
    ) -> None:
        """Attach a confirmed memory id to profile evidence from this event."""
        if not await self.profiles.can_learn(person_id):
            return
        affected = await self.profiles.repo.link_event_evidence(
            person_id, event_id, memory_id
        )
        if not affected:
            return
        fields = {item["field_key"] for item in affected}
        candidates = {
            item["candidate_id"]
            for item in affected
            if item["candidate_id"] is not None
        }
        async with self.profiles.repo.db.transaction():
            for field_key in fields:
                field = await self.profiles.repo.get_field(person_id, field_key)
                if field is None or field["status"] != "ACTIVE":
                    continue
                if memory_id in field["evidence_memory_ids"]:
                    continue
                value = field["value"]
                if field_key in LIST_FIELDS and isinstance(value, list):
                    for entry in value:
                        if entry.get("event_id") == event_id and not entry.get(
                            "memory_id"
                        ):
                            entry["memory_id"] = memory_id
                await self.profiles.repo.upsert_field(
                    person_id,
                    field_key,
                    value=value,
                    source=field["source"],
                    confidence=field["confidence"],
                    evidence_memory_ids=field["evidence_memory_ids"]
                    + [memory_id],
                    evidence_event_ids=field["evidence_event_ids"],
                    evidence_candidate_ids=field["evidence_candidate_ids"],
                    revision=field["revision"],
                )
            for candidate_id in candidates:
                await self.profiles.repo.add_candidate_evidence(
                    candidate_id,
                    memory_ids=[memory_id],
                    event_ids=[],
                    candidate_ids=[],
                )

    async def _maybe_confirm(
        self, candidate_id: str, *, actor: str = "learning"
    ) -> bool:
        """Confirm only after several independent evidence events."""
        candidate = await self.profiles.repo.get_candidate(candidate_id)
        if candidate is None:
            return False
        events = set(candidate["evidence_event_ids"])
        if len(events) < self.confirmation_threshold:
            return False
        try:
            await self.profiles.confirm_candidate(
                candidate_id,
                reason=f"repeated_evidence:{len(events)}",
                actor=actor,
            )
            return True
        except (ProfileUnauthorized, ValueError):
            return False


__all__ = [
    "CONFIRMATION_EVIDENCE_EVENTS",
    "LearningResult",
    "PersonProfileLearningService",
    "ProfileSignal",
    "extract_signals",
    "is_correction",
]
