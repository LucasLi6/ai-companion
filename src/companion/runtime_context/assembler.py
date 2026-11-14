"""Compress a runtime snapshot into a low-token prompt block.

The block is trusted application data. It is appended *after* the fixed
persona prefix, so the stable identity text stays a cache-friendly prefix.
Only fields that are actually known are rendered; an unknown location or gap
is omitted rather than filled with a placeholder.

V0.6.1 separates two instants that used to be conflated:

- ``event_time`` is fixed when the input event is accepted and never changes on
  a retry;
- ``time`` is read when the response is generated, so a reply always knows the
  real current time even if it is produced later.

The wording reports state, not a script. It never says "greet the user"; a long
gap is a fact the model may use if the moment calls for it, and no fixed
opening sentence is prescribed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from companion.domain.models import utcnow

from .snapshot import RuntimeSnapshot
from .temporal import day_phase, humanize_gap, weekday_label

MAX_RUNTIME_BLOCK_CHARS = 1200

_SOURCE_LABEL = {
    "user_config": "用户设置",
    "trusted_device": "设备时区",
    "ip_geo": "网络出口推断",
    "system_default": "系统时区",
    "utc": "UTC",
}

# Continuity states that describe a real interruption, with the one factual
# line that explains what the older topic means now. These are context facts,
# not instructions about what to say (and never a fixed greeting).
_LONG_CONTINUITY = {
    "SHORT_PAUSE": "stage=短暂停顿后继续：刚才的话题仍可能未结束",
    "EXTENDED_PAUSE": "stage=几小时间隔后继续：旧话题仍是记忆背景，不默认双方停在同一个问题上",
    "RETURNING": "stage=间隔后重新交流：旧话题仍是记忆背景，不默认仍是当前未结束的讨论",
    "NEW_DAY": "stage=新的交流阶段：旧话题只是记忆背景，除非本次输入提到它，否则不是当前未结束的讨论",
}


def _value(snapshot, name: str, default=None):
    if isinstance(snapshot, dict):
        return snapshot.get(name, default)
    return getattr(snapshot, name, default)


def _as_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _zone_of(snapshot):
    name = str(_value(snapshot, "timezone") or "").strip()
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _local(moment: Optional[datetime], zone, snapshot) -> Optional[datetime]:
    if moment is None:
        return None
    if zone is not None:
        return moment.astimezone(zone)
    try:
        offset = int(_value(snapshot, "utc_offset_seconds", 0) or 0)
    except (TypeError, ValueError):
        offset = 0
    return moment.astimezone(timezone(timedelta(seconds=offset)))


def _stamp(moment: Optional[datetime]) -> str:
    return f"{moment:%Y-%m-%d %H:%M}" if moment is not None else ""


def _single_line(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def compile_runtime_context(
    snapshot: Optional[RuntimeSnapshot | dict],
    *,
    now: Optional[datetime] = None,
    max_chars: int = MAX_RUNTIME_BLOCK_CHARS,
) -> str:
    if snapshot is None:
        return ""

    zone = _zone_of(snapshot)
    zone_name = _single_line(_value(snapshot, "timezone"), 64)
    source = str(_value(snapshot, "timezone_source") or "")
    current = _local(now or utcnow(), zone, snapshot)
    event_time = _local(
        _as_datetime(_value(snapshot, "occurred_at_utc")), zone, snapshot
    )
    speaker = _single_line(_value(snapshot, "speaker_label"), 60)
    last_user = _local(
        _as_datetime(_value(snapshot, "last_heard_at_utc")), zone, snapshot
    )
    last_interaction = _local(
        _as_datetime(_value(snapshot, "last_interaction_at_utc")), zone, snapshot
    )
    mood = _single_line(_value(snapshot, "ava_emotion"), 40)

    lines: list[str] = ["【运行状态】（本地可信数据）"]
    if current is not None:
        source_label = _SOURCE_LABEL.get(source)
        detail = "，".join(
            part for part in (zone_name, source_label) if part
        )
        stamp = (
            f"{current:%Y-%m-%d %H:%M} {weekday_label(current)}"
            f"（{day_phase(current)}"
        )
        if source == "utc" or zone is None:
            # UTC is a storage fallback, not the person's local time. Say so
            # instead of presenting server time as their daytime.
            stamp += "；时区未确定，非用户当地时间"
        elif source == "system_default":
            # The host zone is a fallback; on a server it may not be the
            # person's zone, so it is labelled rather than asserted as local.
            host = f"{zone_name}，" if zone_name else ""
            stamp += f"；{host}系统时区，可能非用户当地时间"
        elif detail:
            stamp += f"；{detail}"
        lines.append(f"time={stamp}）")
    elif zone_name or source:
        source_label = _SOURCE_LABEL.get(source)
        detail = "，".join(part for part in (zone_name, source_label) if part)
        lines.append(f"timezone={detail}")
    if event_time is not None:
        lines.append(f"event_time={_stamp(event_time)}")
    # What the turn actually arrived as. A text turn carries no acoustic
    # information, and the model must not invent having heard a tone.
    modality = _single_line(_value(snapshot, "source"), 16)
    voice_features = _single_line(_value(snapshot, "voice_features"), 40)
    if modality:
        lines.append(f"source={modality}")
    if voice_features:
        lines.append(f"voice_features={voice_features}")
    if speaker:
        lines.append(f"speaker={speaker}")
    if last_user is not None:
        # The person's own last message, which is not the same fact as when
        # Ava last replied — and never updated by Ava's proactive speech.
        gap = humanize_gap(_value(snapshot, "last_heard_gap_seconds"))
        suffix = f"（{gap}前）" if gap else ""
        lines.append(f"last_user_message={_stamp(last_user)}{suffix}")
    if last_interaction is not None:
        gap = humanize_gap(
            _value(snapshot, "last_interaction_gap_seconds")
        )
        suffix = f"（{gap}前）" if gap else ""
        lines.append(f"last_interaction={_stamp(last_interaction)}{suffix}")
    continuity = str(_value(snapshot, "continuity") or "NEW")
    if last_user is not None and continuity in _LONG_CONTINUITY:
        # Only a real interruption adds these lines; an ordinary exchange
        # keeps the same compact block.
        lines.append(f"continuity={continuity}")
        lines.append(_LONG_CONTINUITY[continuity])
    if _value(snapshot, "previous_turn_departure", False):
        # Only an explicit previous departure is reported; a long silence is
        # not evidence that the person left.
        lines.append("return=上次发言为明确离开，现在是再次出现")
    if mood:
        lines.append(f"mood={mood}")

    location = _value(snapshot, "location")
    if location:
        parts = [
            str(_value(location, name))
            for name in ("city", "subdivision", "country")
            if _value(location, name)
        ]
        if parts:
            # IP-derived, coarse, and explicitly not the person's location.
            lines.append(
                "ip_location=" + "，".join(parts) + "（网络出口推断，不代表本人位置）"
            )

    if len(lines) == 1:
        return ""
    return "\n".join(lines)[:max_chars]


def runtime_snapshot_with_state(snapshot, emotion_state) -> Optional[dict]:
    """Return an in-memory snapshot dict carrying the turn's emotion state."""
    if snapshot is None:
        return None
    data = (
        snapshot.as_dict()
        if isinstance(snapshot, RuntimeSnapshot)
        else dict(snapshot)
    )
    if emotion_state:
        data["ava_emotion"] = emotion_state
    return data


__all__ = [
    "compile_runtime_context",
    "runtime_snapshot_with_state",
]
