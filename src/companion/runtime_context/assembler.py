"""Compress a runtime snapshot into a low-token prompt block.

The block is trusted application data. It is appended *after* the fixed
persona prefix, so the stable identity text stays a cache-friendly prefix.
Only fields that are actually known are rendered; an unknown location or gap
is omitted rather than filled with a placeholder.

The wording deliberately reports state, not a script. It never says "greet the
user"; a long gap is a fact the model may use if the moment calls for it.
"""

from __future__ import annotations

from typing import Optional

from .snapshot import RuntimeSnapshot
from .temporal import humanize_gap, reentry_state as classify_gap

MAX_RUNTIME_BLOCK_CHARS = 1200

_REENTRY_LABEL = {
    "NEW": "首次交流",
    "ACTIVE": "正在对话",
    "CONTINUING": "间歇交流",
    "RETURNING": "间隔较久后回来",
}


def _value(snapshot, name: str, default=None):
    if isinstance(snapshot, dict):
        return snapshot.get(name, default)
    return getattr(snapshot, name, default)


def compile_runtime_context(
    snapshot: Optional[RuntimeSnapshot | dict],
    *,
    max_chars: int = MAX_RUNTIME_BLOCK_CHARS,
) -> str:
    if snapshot is None:
        return ""

    local_time = " ".join(str(_value(snapshot, "local_time") or "").split())
    zone = " ".join(str(_value(snapshot, "timezone") or "").split())
    zone_source = str(_value(snapshot, "timezone_source") or "")
    lines: list[str] = []

    clock: list[str] = []
    if local_time:
        clock.append(f"现在 {local_time}")
    if zone:
        suffix = ""
        if zone_source == "ip_geo":
            suffix = "（按网络出口推断）"
        elif zone_source == "trusted_device":
            suffix = "（设备时区）"
        elif zone_source == "user_config":
            suffix = "（用户设置）"
        clock.append(f"时区 {zone}{suffix}")
    if clock:
        lines.append("；".join(clock))

    heard = humanize_gap(_value(snapshot, "last_heard_gap_seconds"))
    interaction = humanize_gap(_value(snapshot, "last_interaction_gap_seconds"))
    meaningful = humanize_gap(
        _value(snapshot, "last_meaningful_context_gap_seconds")
    )
    gaps: list[str] = []
    if heard:
        gaps.append(f"上次听对方说话: {heard}前")
    if interaction:
        gaps.append(f"上次轮到回应: {interaction}前")
    if meaningful and meaningful != heard:
        gaps.append(f"上次有实质交流: {meaningful}前")
    if gaps:
        lines.append("；".join(gaps))

    state = str(_value(snapshot, "reentry_state") or "NEW")
    label = _REENTRY_LABEL.get(state)
    if label:
        lines.append(f"连续性: {label}")

    location = _value(snapshot, "location")
    if location:
        def field(name):
            return _value(location, name)

        parts = [
            str(part)
            for part in (
                field("city"),
                field("subdivision"),
                field("country"),
            )
            if part
        ]
        if parts:
            lines.append(
                "网络出口位置（近似，不代表本人当前所在地）: " + "，".join(parts)
            )

    if not lines:
        return ""
    text = "【当前时间与状态】（本地可信数据）\n" + "\n".join(lines)
    text += (
        "\n时间间隔只是事实状态，不是固定话术；是否需要提到由当前对话决定，"
        "不要每轮重复“好久不见”之类的开场。"
    )
    return text[:max_chars]


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
    "classify_gap",
]
