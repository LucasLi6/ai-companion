"""Temporal continuity: Last Heard vs Last Interaction vs Meaningful Context.

These are three different questions and must not be collapsed into one:

- **Last Heard** — when this speaker last said anything to the companion.
- **Last Interaction** — when the companion last actually *replied* to this
  speaker (a visible RESPOND). WAIT/IGNORE turns are not interactions: a run of
  silence says nothing about when the two last talked.
- **Last Meaningful Context** — currently the same RESPOND timestamp, kept as a
  separate field so a future notion of "meaningful" can narrow it without
  changing the answer to "how long since we talked?".

Gaps are data. They are never converted into a fixed reply rule such as
``if gap > 86400: reply = "好久不见"``. The generation model receives the facts
and decides whether mentioning them is natural in this conversation. The
re-entry state is also just a label (``NEW`` / ``ACTIVE`` / ``CONTINUING`` /
``RETURNING``); it does not authorise a greeting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

DEFAULT_ACTIVE_GAP_SECONDS = 10 * 60
DEFAULT_REENTRY_GAP_SECONDS = 6 * 60 * 60

REENTRY_NEW = "NEW"
REENTRY_ACTIVE = "ACTIVE"
REENTRY_CONTINUING = "CONTINUING"
REENTRY_RETURNING = "RETURNING"

WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


class ConversationContinuity(str, Enum):
    """How the current turn relates to the previous user message in time.

    This is a pure classification of ``last_heard_gap_seconds`` — the gap
    since the *person's own* last message, never Ava's proactive speech. It
    decides only the default context continuity (whether old topics still
    count as an unfinished discussion); it never decides what Ava must say,
    and it never authorises a greeting.

    NEW             no earlier user message at all
    ACTIVE          within 10 minutes: ordinary turn-by-turn exchange
    SHORT_PAUSE     10 minutes – 2 hours
    EXTENDED_PAUSE  2 – 6 hours
    RETURNING       6 – 24 hours
    NEW_DAY         24 hours or more: a new exchange stage
    """

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    SHORT_PAUSE = "SHORT_PAUSE"
    EXTENDED_PAUSE = "EXTENDED_PAUSE"
    RETURNING = "RETURNING"
    NEW_DAY = "NEW_DAY"


CONTINUITY_SHORT_PAUSE_SECONDS = 10 * 60
CONTINUITY_EXTENDED_PAUSE_SECONDS = 2 * 60 * 60
CONTINUITY_RETURNING_SECONDS = 6 * 60 * 60
CONTINUITY_NEW_DAY_SECONDS = 24 * 60 * 60

CONTINUITY_LABELS = {
    ConversationContinuity.NEW: "尚无历史",
    ConversationContinuity.ACTIVE: "连续交流",
    ConversationContinuity.SHORT_PAUSE: "短暂停顿",
    ConversationContinuity.EXTENDED_PAUSE: "几小时间隔",
    ConversationContinuity.RETURNING: "间隔后重新交流",
    ConversationContinuity.NEW_DAY: "新的交流阶段",
}


def classify_continuity(
    last_heard_gap_seconds: Optional[int],
) -> ConversationContinuity:
    """Pure classifier over the *user* gap. No LLM, no clock, no I/O."""
    if last_heard_gap_seconds is None:
        return ConversationContinuity.NEW
    try:
        gap = max(0, int(last_heard_gap_seconds))
    except (TypeError, ValueError):
        return ConversationContinuity.NEW
    if gap < CONTINUITY_SHORT_PAUSE_SECONDS:
        return ConversationContinuity.ACTIVE
    if gap < CONTINUITY_EXTENDED_PAUSE_SECONDS:
        return ConversationContinuity.SHORT_PAUSE
    if gap < CONTINUITY_RETURNING_SECONDS:
        return ConversationContinuity.EXTENDED_PAUSE
    if gap < CONTINUITY_NEW_DAY_SECONDS:
        return ConversationContinuity.RETURNING
    return ConversationContinuity.NEW_DAY


def weekday_label(moment: datetime) -> str:
    """Chinese weekday for a local datetime; purely presentational."""
    return WEEKDAY_LABELS[moment.weekday()]


def day_phase(moment: datetime) -> str:
    """Local daytime phase, computed from the trusted local hour.

    The six labels are deliberately coarse: the model receives a fact about
    the hour, never an instruction about how to greet someone.
    """
    hour = moment.hour
    if hour < 5:
        return "凌晨"
    if hour < 11:
        return "早晨"
    if hour < 13:
        return "中午"
    if hour < 18:
        return "下午"
    if hour < 23:
        return "晚上"
    return "深夜"


@dataclass(frozen=True)
class TemporalSignals:
    last_heard_at: Optional[datetime] = None
    last_interaction_at: Optional[datetime] = None
    last_meaningful_context_at: Optional[datetime] = None
    last_heard_gap_seconds: Optional[int] = None
    last_interaction_gap_seconds: Optional[int] = None
    last_meaningful_context_gap_seconds: Optional[int] = None
    reentry_state: str = REENTRY_NEW
    # V0.9.9.27: six-level continuity derived from the user gap only. It is
    # the fact the assembler and the Speaker use to decide whether an old
    # topic still counts as an unfinished discussion.
    continuity: ConversationContinuity = ConversationContinuity.NEW
    # True only when the person's previous turn was a structurally explicit
    # departure. A gap alone never means the person left.
    previous_turn_departure: bool = False

    @property
    def reentry(self) -> bool:
        return self.reentry_state == REENTRY_RETURNING


def gap_seconds(now: datetime, then: Optional[datetime]) -> Optional[int]:
    if then is None:
        return None
    return max(0, int((now - then).total_seconds()))


def reentry_state(
    last_heard_gap: Optional[int],
    *,
    active_gap_seconds: int = DEFAULT_ACTIVE_GAP_SECONDS,
    reentry_gap_seconds: int = DEFAULT_REENTRY_GAP_SECONDS,
) -> str:
    if last_heard_gap is None:
        return REENTRY_NEW
    if last_heard_gap <= active_gap_seconds:
        return REENTRY_ACTIVE
    if last_heard_gap < reentry_gap_seconds:
        return REENTRY_CONTINUING
    return REENTRY_RETURNING


def build_temporal_signals(
    now: datetime,
    *,
    last_heard_at: Optional[datetime] = None,
    last_interaction_at: Optional[datetime] = None,
    last_meaningful_context_at: Optional[datetime] = None,
    active_gap_seconds: int = DEFAULT_ACTIVE_GAP_SECONDS,
    reentry_gap_seconds: int = DEFAULT_REENTRY_GAP_SECONDS,
    previous_turn_departure: bool = False,
) -> TemporalSignals:
    heard_gap = gap_seconds(now, last_heard_at)
    interaction_gap = gap_seconds(now, last_interaction_at)
    meaningful_gap = gap_seconds(now, last_meaningful_context_at)
    return TemporalSignals(
        last_heard_at=last_heard_at,
        last_interaction_at=last_interaction_at,
        last_meaningful_context_at=last_meaningful_context_at,
        last_heard_gap_seconds=heard_gap,
        last_interaction_gap_seconds=interaction_gap,
        last_meaningful_context_gap_seconds=meaningful_gap,
        reentry_state=reentry_state(
            heard_gap,
            active_gap_seconds=active_gap_seconds,
            reentry_gap_seconds=reentry_gap_seconds,
        ),
        continuity=classify_continuity(heard_gap),
        previous_turn_departure=bool(previous_turn_departure),
    )


def humanize_gap(seconds: Optional[int]) -> Optional[str]:
    """Render a gap compactly, without inventing precision."""
    if seconds is None:
        return None
    if seconds < 60:
        return "刚刚"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} 分钟"
    hours = minutes // 60
    if hours < 24:
        rest = minutes % 60
        return f"{hours} 小时" + (f" {rest} 分钟" if rest else "")
    days = hours // 24
    rest = hours % 24
    return f"{days} 天" + (f" {rest} 小时" if rest else "")
