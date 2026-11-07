"""V0.9.9 single-turn confirmation of critical profile information.

A person saying "我叫李约" is an explicit self-report, but the profile must not
be written silently and the companion should not interrogate them later. This
module keeps one small, local state machine:

    explicit self-report detected before glyph generation
        -> pending Candidate (DIRECT_SELF_REPORT) + one confirmation question
        -> question is appended to the next *deliverable* normal reply
        -> the next real user message is read as confirm / deny / correct /
           ignore / ambiguous
        -> trusted ProfileService write or reject

Silence is never treated as an explicit confirmation. The only silence rule is
the narrow "clear first-person self-report" fallback, which writes with the
real ``DIRECT_SELF_REPORT`` source and never with ``TRUSTED_CONFIRMATION``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from companion.domain.models import new_id, utcnow
from companion.people.profile_service import (
    PersonProfileService,
    ProfileUnauthorized,
)
from companion.storage.repositories import ProfileConfirmationRecord

STATE_PENDING_QUESTION = "PENDING_QUESTION"
STATE_AWAITING_REPLY = "AWAITING_REPLY"
STATE_CONFIRMED = "CONFIRMED"
STATE_REJECTED = "REJECTED"
STATE_ACCEPTED_SELF_REPORT = "ACCEPTED_SELF_REPORT"
STATE_NEEDS_REVIEW = "NEEDS_REVIEW"

_ACTIVE_STATES = (STATE_PENDING_QUESTION, STATE_AWAITING_REPLY)

KIND_CONFIRM = "CONFIRM"
KIND_DENY = "DENY"
KIND_CORRECT = "CORRECT"
KIND_IGNORED = "IGNORED"
KIND_AMBIGUOUS = "AMBIGUOUS"

_CJK = r"一-鿿"
_NAME_TOKEN = (
    rf"(?:[A-Za-z][A-Za-z0-9.'’\-]*(?:\s+[A-Za-z][A-Za-z0-9.'’\-]*){{0,2}}"
    rf"|[{_CJK}·]{{1,12}})"
)

_QUESTION_END = re.compile(r"[?？]\s*$")
_QUESTION_PARTICLE = re.compile(r"(吗|呢|么|嘛)\s*[。.!！~～]*\s*$")
_HYPOTHETICAL = re.compile(r"(假如|如果|假设|万一|要是|比如说|比如|若是|倘若)")
_QUOTED = re.compile(
    r"(他说|她说|他们说|她们说|别人说|有人说|据说|听说|转述说|"
    r"朋友说|同事说|同学说|老师说)|说[：:]|说[“\"']"
)
_THIRD_PERSON = re.compile(
    r"(他|她|他们|她们|朋友|同事|同学|老师|别人|有人)\s*(?:说|叫|让|希望|想)"
)
_THIRD_PERSON_SUBJECT = re.compile(
    r"(?:^|[^你您])(?:他|她|他们|她们|朋友|同事|同学|老师|别人|有人)\s*$"
)

_CONFIRM_ONLY = re.compile(
    r"^(?:(?:嗯+|哦+|噢+|啊+)[，,、]?\s*)?"
    r"(?:嗯+|对(?:的|呀|啊)?|是(?:的|啊|呀)?|没错|确认|确定|"
    r"好(?:的|呀)?|可以|没问题|正确|ok|okay|yes|yeah|yep|sure|当然)"
    r"[。.!！~～\s]*$",
    re.IGNORECASE,
)
# A reply that pushes the question aside is not a yes and is not unrelated
# small talk: the confirmation stays pending instead of being silently
# accepted. "嗯，先说别的" and "对了，我们聊其他话题" land here.
_DEFERRAL = re.compile(
    r"(先说别的|先聊别的|先不说这个|先不说|等会(?:儿)?|待会(?:儿)?|"
    r"一会儿再说|回头再说|以后再说|换个话题|换话题|聊别的|聊点别的|"
    r"说点别的|说别的|聊其他|说其他|先这样|现在不方便|不想说)"
)
_DENY = re.compile(
    r"(不是|不对|错了|记错|搞错|弄错|不叫|别叫|没有|并没有|否|"
    r"^不$|^no\b|^nope\b)",
    re.IGNORECASE,
)
_MENTIONS_IDENTITY = re.compile(
    r"(名字|中文名|英文名|昵称|外号|小名|称呼|叫你|喊你|叫我|确认)"
)

_TRAILING_CUT = re.compile(
    r"(?:以后|之后|今后|平时|现在|就好|就行|就可以|然后|吧|呀|啊|哦|嘛|呢|你|您)"
)


@dataclass(frozen=True)
class DetectedSelfReport:
    field_key: str
    value: str
    quote: str


@dataclass(frozen=True)
class _Interpretation:
    kind: str
    detected: tuple[DetectedSelfReport, ...] = ()


_SELF_REPORT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # A direct introduction using "我是李约" was not previously recognised,
    # even though the memory extractor recorded it as a name. Keep this
    # narrower than an unrestricted "我是..." so occupations and roles such as
    # "我是学生" do not become names.
    (
        "identity.name",
        re.compile(
            r"(?<![\\w])我是\\s*(?P<value>"
            r"[李王张刘陈杨黄赵周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文]"
            r"[一-鿿]{1,2})(?=$|[，。.!！?？\\s])"
        ),
    ),
    # identity.name: "我叫李约" / "我的名字是李约"
    (
        "identity.name",
        re.compile(
            rf"我(?:的)?(?:中文)?(?:名字|名)(?:是|叫|为)\s*(?P<value>{_NAME_TOKEN})"
        ),
    ),
    ("identity.name", re.compile(rf"我(?:叫|名叫)\s*(?P<value>{_NAME_TOKEN})")),
    # preferred name: English name, or an explicit future form of address
    (
        "identity.preferred_name",
        re.compile(rf"(?:我的)?英文名(?:是|叫|为)?\s*(?P<value>{_NAME_TOKEN})"),
    ),
    (
        "identity.preferred_name",
        re.compile(
            rf"(?:以后|之后|今后|平时|你可以|你可以直接)(?:叫我|喊我)\s*"
            rf"(?P<value>{_NAME_TOKEN})"
        ),
    ),
    # nicknames
    (
        "identity.nicknames",
        re.compile(
            rf"我的(?:昵称|外号|小名)(?:是|叫|为)\s*(?P<value>{_NAME_TOKEN})"
        ),
    ),
    # explicit address preference
    (
        "communication.address_preference",
        re.compile(rf"(?:请)?(?:称呼我|称我)(?:为|作)?\s*(?P<value>{_NAME_TOKEN})"),
    ),
    (
        "communication.address_preference",
        re.compile(
            rf"(?:以后)?(?:叫我|喊我)\s*(?P<value>{_NAME_TOKEN})\s*(?:就行|就好|吧)"
        ),
    ),
    # a bare "叫我李约" is a preferred name, not an address syntax
    (
        "identity.preferred_name",
        re.compile(rf"(?:叫我|喊我)\s*(?P<value>{_NAME_TOKEN})"),
    ),
)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split())


def _is_question(text: str) -> bool:
    return bool(_QUESTION_END.search(text) or _QUESTION_PARTICLE.search(text))


def _clean_name(raw: str) -> Optional[str]:
    text = str(raw or "").strip()
    text = text.strip("，,。.!！?？:：;；\"'“”‘’（）()【】[]、 ")
    # "李约以后你就叫我约约" must not become the name "李约以后".
    text = _TRAILING_CUT.split(text, maxsplit=1)[0].strip()
    text = text.strip("，,。.!！?？:：;；\"'“”‘’（）()【】[]、 ")
    if not text:
        return None
    if any(ch.isdigit() for ch in text):
        return None
    if len(text) > 40:
        return None
    return text


def detect_self_reports(text: str) -> tuple[DetectedSelfReport, ...]:
    """Return explicit first-person identity statements in one message.

    The function is deliberately narrow: questions, hypotheticals, quotations
    and third-person reports are refused, and nothing is inferred from context.
    """
    value = _normalise(text)
    if not value:
        return ()
    if _is_question(value) or _HYPOTHETICAL.search(value) or _QUOTED.search(value):
        return ()

    found: dict[str, DetectedSelfReport] = {}
    for field_key, pattern in _SELF_REPORT_PATTERNS:
        if field_key in found:
            continue
        for match in pattern.finditer(value):
            prefix = value[max(0, match.start() - 8) : match.start()]
            if _THIRD_PERSON.search(prefix) or _THIRD_PERSON_SUBJECT.search(prefix):
                continue
            cleaned = _clean_name(match.group("value"))
            if not cleaned:
                continue
            # Guard role descriptions resembling a Chinese name in a bare
            # "我是..." introduction (e.g. "我是高三生").
            if match.group(0).startswith("我是") and cleaned in {
                "高三生", "高中生", "大学生", "研究生", "初中生", "小学生",
                "王老师", "李老师", "程序员", "中国人", "巴西人",
            }:
                continue
            found[field_key] = DetectedSelfReport(
                field_key=field_key, value=cleaned, quote=value[:200]
            )
            break
    return tuple(found.values())


def _is_latin(value: str) -> bool:
    return bool(value) and all(ord(ch) < 128 for ch in value)


def build_confirmation_question(
    records: list[ProfileConfirmationRecord],
) -> Optional[str]:
    """One short, natural question for all fields detected in one turn."""
    parts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        value = str(record.value)
        marker = (record.field_key, value)
        if marker in seen:
            continue
        seen.add(marker)
        if record.field_key == "identity.name":
            if _is_latin(value):
                parts.append(f"你的名字是 {value}")
            else:
                parts.append(f"你的中文名是{value}")
        elif record.field_key == "identity.preferred_name":
            if _is_latin(value):
                parts.append(f"你的英文名是 {value}")
            else:
                parts.append(f"你希望我叫你{value}")
        elif record.field_key == "identity.nicknames":
            parts.append(f"可以叫你{value}")
        elif record.field_key == "communication.address_preference":
            parts.append(f"你希望我称呼你为{value}")
    if not parts:
        return None
    return "确认一下，" + "，".join(parts) + "，对吗？"


def interpret_reply(
    text: str, records: list[ProfileConfirmationRecord]
) -> _Interpretation:
    """Read the next real reply as confirm / deny / correct / ignore / ambiguous."""
    value = _normalise(text)
    if not value:
        return _Interpretation(KIND_AMBIGUOUS)
    if (
        _is_question(value)
        or _HYPOTHETICAL.search(value)
        or _QUOTED.search(value)
    ):
        # "李约是谁？" / "假如我是李约" is not an answer, and must never count
        # as a yes.
        return _Interpretation(KIND_AMBIGUOUS)

    detected = detect_self_reports(value)
    if detected:
        pending = {(record.field_key, str(record.value)) for record in records}
        if any((item.field_key, item.value) not in pending for item in detected):
            return _Interpretation(KIND_CORRECT, detected)
        return _Interpretation(KIND_CONFIRM)
    if _DENY.search(value):
        return _Interpretation(KIND_DENY)
    # Only a short, whole-message affirmation counts as a yes. A longer reply
    # that merely starts with "对" / "嗯" ("嗯，先说别的") is a different
    # message, not agreement.
    if _CONFIRM_ONLY.fullmatch(value):
        return _Interpretation(KIND_CONFIRM)
    if _DEFERRAL.search(value):
        return _Interpretation(KIND_AMBIGUOUS)
    if _MENTIONS_IDENTITY.search(value) or any(
        str(record.value) in value for record in records
    ):
        return _Interpretation(KIND_AMBIGUOUS)
    return _Interpretation(KIND_IGNORED)


class ProfileConfirmationService:
    """Detect, ask, interpret and resolve one identity confirmation at a time."""

    def __init__(
        self,
        *,
        profile_service: PersonProfileService,
        repos,
        no_reply_timeout_seconds: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.profile_service = profile_service
        self.repos = repos
        self.confirmations = repos.profile_confirmation
        self.no_reply_timeout_seconds = max(0.0, float(no_reply_timeout_seconds))
        self.logger = logger or logging.getLogger(
            "companion.people.profile_confirmation"
        )
        self._timers: dict[str, asyncio.Task] = {}
        self._closed = False

    # -- turn hook (before Speaker generation) ----------------------------

    async def begin_turn(
        self, *, event, person_id: str
    ) -> Optional[str]:
        """Resolve a pending answer and detect a new self-report.

        Returns the question that may be appended to this turn's normal reply,
        or ``None``. The caller decides whether the reply is actually
        deliverable; until then the state stays ``PENDING_QUESTION``.
        """
        if self._closed or not getattr(event, "text", ""):
            return None
        try:
            if not await self.profile_service.can_learn(person_id):
                return None
        except Exception as exc:
            self.logger.warning(
                "profile_confirmation_consent_failed person=%s error=%s",
                person_id,
                type(exc).__name__,
            )
            return None

        records = await self._records(person_id, event.conversation_id)
        awaiting = [
            record
            for record in records
            if record.state == STATE_AWAITING_REPLY
            and record.source_event_id != event.event_id
        ]
        allow_detection = True
        if awaiting:
            interpretation = interpret_reply(event.text, awaiting)
            if interpretation.kind == KIND_CONFIRM:
                for record in awaiting:
                    await self._apply_record(
                        record,
                        confirmed=True,
                        event_id=event.event_id,
                        delivered_state=STATE_CONFIRMED,
                        reason="confirmed_by_person",
                    )
            elif interpretation.kind == KIND_DENY:
                for record in awaiting:
                    await self._reject_record(record)
            elif interpretation.kind == KIND_CORRECT:
                for record in awaiting:
                    await self._reject_record(record)
            elif interpretation.kind == KIND_IGNORED:
                for record in awaiting:
                    await self._apply_record(
                        record,
                        confirmed=False,
                        event_id=record.source_event_id,
                        delivered_state=STATE_ACCEPTED_SELF_REPORT,
                        reason="accepted_direct_self_report",
                    )
            else:  # AMBIGUOUS: stay pending, never count as a yes
                allow_detection = False

        if allow_detection:
            for detected in detect_self_reports(event.text):
                await self._record_detection(event, person_id, detected)

        pending = [
            record
            for record in await self._records(person_id, event.conversation_id)
            if record.state == STATE_PENDING_QUESTION
        ]
        return build_confirmation_question(pending)

    async def mark_question_delivered(
        self, *, person_id: str, conversation_id: str, question_message_id: str
    ) -> None:
        """The reply carrying the question was actually delivered."""
        records = [
            record
            for record in await self._records(person_id, conversation_id)
            if record.state == STATE_PENDING_QUESTION
        ]
        if not records:
            return
        for record in records:
            await self.confirmations.update_state(
                record.confirmation_id,
                STATE_AWAITING_REPLY,
                question_message_id=question_message_id,
            )
        self._schedule_no_reply(person_id, conversation_id)

    async def shutdown(self) -> None:
        self._closed = True
        tasks = [task for task in self._timers.values() if not task.done()]
        for task in tasks:
            task.cancel()
        self._timers.clear()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def resume_pending(self) -> int:
        """Restart no-reply timers for questions asked before a restart.

        The timer lives only in memory, so a process restart would otherwise
        leave a delivered confirmation question in ``AWAITING_REPLY`` forever.
        Each person/conversation keeps at most one timer; it is based on the
        most recent pending record so every pending question gets its full
        window and a long downtime fires promptly.
        """
        if self._closed:
            return 0
        records = await self.confirmations.list_by_state(STATE_AWAITING_REPLY)
        latest: dict[tuple[str, str], ProfileConfirmationRecord] = {}
        for record in records:
            key = (record.person_id, record.conversation_id)
            current = latest.get(key)
            if current is None or _after(record.updated_at, current.updated_at):
                latest[key] = record
        for (person_id, conversation_id), record in latest.items():
            remaining = self.no_reply_timeout_seconds - _elapsed(
                record.updated_at
            )
            self._schedule_no_reply(
                person_id, conversation_id, delay=max(0.0, remaining)
            )
        return len(latest)

    # -- internals ---------------------------------------------------------

    async def _records(
        self, person_id: str, conversation_id: str
    ) -> list[ProfileConfirmationRecord]:
        records = await self.confirmations.list_for_person(
            person_id, states=_ACTIVE_STATES
        )
        return [
            record
            for record in records
            if record.conversation_id == conversation_id
        ]

    async def _record_detection(
        self, event, person_id: str, detected: DetectedSelfReport
    ) -> Optional[str]:
        try:
            candidate_id = await self.profile_service.propose_self_report(
                person_id,
                detected.field_key,
                detected.value,
                event_id=event.event_id,
                quote=detected.quote,
            )
        except (ValueError, ProfileUnauthorized) as exc:
            self.logger.info(
                "profile_confirmation_propose_skipped person=%s field=%s"
                " error=%s",
                person_id,
                detected.field_key,
                type(exc).__name__,
            )
            return None
        if not candidate_id:
            return None

        existing = await self.confirmations.get_by_candidate(candidate_id)
        if existing is not None and existing.state in _ACTIVE_STATES:
            return existing.confirmation_id

        # A new explicit value for the same field replaces the old question.
        for record in await self._records(person_id, event.conversation_id):
            if record.field_key != detected.field_key:
                continue
            if record.candidate_id == candidate_id:
                return record.confirmation_id
            await self.confirmations.update_state(
                record.confirmation_id, STATE_REJECTED
            )

        now = utcnow()
        record = ProfileConfirmationRecord(
            confirmation_id=new_id("pcfm"),
            person_id=person_id,
            conversation_id=event.conversation_id,
            source_event_id=event.event_id,
            question_message_id=None,
            candidate_id=candidate_id,
            field_key=detected.field_key,
            value=detected.value,
            quote=detected.quote,
            state=STATE_PENDING_QUESTION,
            created_at=now,
            updated_at=now,
        )
        await self.confirmations.save(record)
        self.logger.info(
            "profile_confirmation_pending person=%s conversation=%s"
            " field=%s candidate=%s",
            person_id,
            event.conversation_id,
            detected.field_key,
            candidate_id,
        )
        return record.confirmation_id

    async def _apply_record(
        self,
        record: ProfileConfirmationRecord,
        *,
        confirmed: bool,
        event_id: Optional[str],
        delivered_state: str,
        reason: str,
    ) -> None:
        if not record.candidate_id:
            await self.confirmations.update_state(
                record.confirmation_id, STATE_NEEDS_REVIEW
            )
            return
        try:
            revision = await self.profile_service.accept_self_report(
                record.candidate_id,
                confirmed=confirmed,
                event_id=event_id,
                reason=reason,
            )
        except ProfileUnauthorized as exc:
            self.logger.info(
                "profile_confirmation_unauthorized person=%s field=%s error=%s",
                record.person_id,
                record.field_key,
                type(exc).__name__,
            )
            return
        except Exception as exc:  # local validation only
            self.logger.warning(
                "profile_confirmation_apply_failed person=%s field=%s error=%s",
                record.person_id,
                record.field_key,
                type(exc).__name__,
            )
            await self.confirmations.update_state(
                record.confirmation_id, STATE_NEEDS_REVIEW
            )
            return

        if revision is None:
            stored = await self.repos.profile.get_candidate(record.candidate_id)
            state = (
                STATE_NEEDS_REVIEW
                if stored is not None and stored["status"] == "NEEDS_REVIEW"
                else delivered_state
            )
        else:
            state = delivered_state
        await self.confirmations.update_state(record.confirmation_id, state)
        self.logger.info(
            "profile_confirmation_%s person=%s field=%s revision=%s",
            state.lower(),
            record.person_id,
            record.field_key,
            revision,
        )

    async def _reject_record(self, record: ProfileConfirmationRecord) -> None:
        if record.candidate_id:
            await self.profile_service.reject_candidate(
                record.candidate_id, reason="rejected_by_person"
            )
        await self.confirmations.update_state(record.confirmation_id, STATE_REJECTED)

    def _schedule_no_reply(
        self,
        person_id: str,
        conversation_id: str,
        *,
        delay: Optional[float] = None,
    ) -> None:
        key = f"{person_id}:{conversation_id}"
        old = self._timers.get(key)
        if old is not None and not old.done():
            old.cancel()
        task = asyncio.create_task(
            self._no_reply_cycle(person_id, conversation_id, delay)
        )
        self._timers[key] = task
        task.add_done_callback(
            lambda finished: self._timers.pop(key, None)
            if self._timers.get(key) is finished
            else None
        )

    async def _no_reply_cycle(
        self,
        person_id: str,
        conversation_id: str,
        delay: Optional[float] = None,
    ) -> None:
        """Silence fallback: only a clear direct self-report may be applied.

        This is not an explicit confirmation. The write keeps the real
        ``DIRECT_SELF_REPORT`` source, and a conflicting confirmed identity is
        still refused.
        """
        try:
            wait = (
                self.no_reply_timeout_seconds if delay is None else delay
            )
            await asyncio.sleep(max(0.0, wait))
            if self._closed:
                return
            records = [
                record
                for record in await self._records(person_id, conversation_id)
                if record.state == STATE_AWAITING_REPLY
            ]
            if not records:
                return
            latest = await self.repos.utterance.recent_context(
                conversation_id, exclude_event_id=None, limit=1
            )
            latest_at = latest[-1].occurred_at if latest else None
            for record in records:
                if latest_at is not None and _after(latest_at, record.updated_at):
                    # A real user message arrived after the question; begin_turn
                    # owns the answer, or will on its next run.
                    continue
                await self._apply_record(
                    record,
                    confirmed=False,
                    event_id=record.source_event_id,
                    delivered_state=STATE_ACCEPTED_SELF_REPORT,
                    reason="accepted_direct_self_report_after_silence",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                "profile_confirmation_timeout_failed person=%s conversation=%s"
                " error=%s",
                person_id,
                conversation_id,
                type(exc).__name__,
            )


def _after(left: datetime, right: datetime) -> bool:
    """Compare two stored timestamps, tolerating naive/aware mixing."""
    if left.tzinfo is None:
        left = left.replace(tzinfo=timezone.utc)
    if right.tzinfo is None:
        right = right.replace(tzinfo=timezone.utc)
    return left > right


def _elapsed(timestamp: datetime) -> float:
    """Seconds since a stored timestamp, tolerating naive/aware mixing."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return max(0.0, (utcnow() - timestamp).total_seconds())


__all__ = [
    "DetectedSelfReport",
    "ProfileConfirmationService",
    "STATE_ACCEPTED_SELF_REPORT",
    "STATE_AWAITING_REPLY",
    "STATE_CONFIRMED",
    "STATE_NEEDS_REVIEW",
    "STATE_PENDING_QUESTION",
    "STATE_REJECTED",
    "build_confirmation_question",
    "detect_self_reports",
    "interpret_reply",
]
