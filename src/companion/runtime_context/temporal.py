"""Temporal continuity: Last Heard vs Last Interaction vs Meaningful Context.

These are three different questions and must not be collapsed into one:

- **Last Heard** — when this speaker last said anything to the companion.
- **Last Interaction** — when the companion last *did* something for this
  speaker, including choosing silence (WAIT/IGNORE). It answers "did we have a
  turn recently?", not "did we talk?".
- **Last Meaningful Context** — when the companion last actually *said*
  something (a visible RESPOND). A run of WAITs is not a conversation.

Gaps are data. They are never converted into a fixed reply rule such as
``if gap > 86400: reply = "好久不见"``. The generation model receives the facts
and decides whether mentioning them is natural in this conversation. The
re-entry state is also just a label (``NEW`` / ``ACTIVE`` / ``CONTINUING`` /
``RETURNING``); it does not authorise a greeting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

DEFAULT_ACTIVE_GAP_SECONDS = 10 * 60
DEFAULT_REENTRY_GAP_SECONDS = 6 * 60 * 60

REENTRY_NEW = "NEW"
REENTRY_ACTIVE = "ACTIVE"
REENTRY_CONTINUING = "CONTINUING"
REENTRY_RETURNING = "RETURNING"


@dataclass(frozen=True)
class TemporalSignals:
    last_heard_at: Optional[datetime] = None
    last_interaction_at: Optional[datetime] = None
    last_meaningful_context_at: Optional[datetime] = None
    last_heard_gap_seconds: Optional[int] = None
    last_interaction_gap_seconds: Optional[int] = None
    last_meaningful_context_gap_seconds: Optional[int] = None
    reentry_state: str = REENTRY_NEW

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
