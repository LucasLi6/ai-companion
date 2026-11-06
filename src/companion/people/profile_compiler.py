"""Compile one person's understanding profile into short prompt blocks.

Two blocks are produced:

``PROFILE_CORE``
    Who this person is to Ava: confirmed identity, relationship tier, stable
    communication preferences and a brief understanding. Identity is the first
    priority and is never taken from the registration label.

``PROFILE_RELEVANT``
    Only the understanding entries plausibly relevant to the current question:
    interests, projects, skills, experience and goals.

The blocks are background data, never instructions and never a second
persona. Each item is budgeted individually; the assembled result is never
sliced mid-sentence. Pending candidates and other people's profiles never
appear.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from companion.people.profile_models import (
    PersonInteractionProfile,
    ProfileEntry,
    ProfileFieldKey,
)

MAX_BLOCK_CHARS = 600
MAX_CORE_CHARS = 420
MAX_RELEVANT_CHARS = 300
MAX_LABEL_CHARS = 40
MAX_LIST_ITEMS_IN_PROMPT = 3

_LEGACY_RELEVANT = (
    (ProfileFieldKey.INTERESTS, "interests"),
    (ProfileFieldKey.ONGOING_PROJECTS, "ongoing"),
    (ProfileFieldKey.USUAL_ACTIVITIES, "activities"),
    (ProfileFieldKey.PERSONAL_GOALS, "goals"),
    (ProfileFieldKey.RECURRING_TOPICS, "topics"),
)

_ENTRY_RELEVANT = (
    ("interests.topics", "interests"),
    ("interests.likes", "likes"),
    ("projects.active", "ongoing"),
    ("projects.paused", "paused"),
    ("projects.completed", "completed"),
    ("goals.short_term", "goals"),
    ("goals.long_term", "goals"),
    ("goals.general", "goals"),
    ("habits.stable_habits", "activities"),
    ("knowledge.skills", "skills"),
    ("knowledge.domains", "domains"),
    ("knowledge.experience", "experience"),
    ("interaction.recurring_patterns", "topics"),
)

_WORD = re.compile(r"[A-Za-z0-9_+\-]{2,}|[一-鿿]{2,}")


def _short(value: object, limit: int = MAX_LABEL_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _tokens(text: str) -> set[str]:
    value = str(text or "").casefold()
    tokens: set[str] = set()
    for match in _WORD.findall(value):
        if re.fullmatch(r"[一-鿿]+", match):
            tokens.update(match[index : index + 2] for index in range(len(match) - 1))
        else:
            tokens.add(match)
    return tokens


def _entry_texts(profile: PersonInteractionProfile, field_key: str) -> list[str]:
    return [
        entry.content
        for entry in profile.entries.get(field_key, [])
        if entry.is_effective() and entry.content.strip()
    ]


def _labels(profile: PersonInteractionProfile, field: ProfileFieldKey) -> list[str]:
    labels: list[str] = []
    for entry in profile.list_entries(field):
        label = _short(entry.get("label") if isinstance(entry, dict) else entry)
        if label and label not in labels:
            labels.append(label)
    return labels


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _short(item)
        marker = text.casefold()
        if text and marker not in seen:
            seen.add(marker)
            out.append(text)
    return out


def _list_segment(label: str, items: list[str], budget: int) -> Optional[str]:
    """Add list items one at a time until this segment's budget is used."""
    if not items or budget <= len(label) + 3:
        return None
    chosen: list[str] = []
    used = len(label) + 1
    for item in items[:MAX_LIST_ITEMS_IN_PROMPT]:
        piece = item if not chosen else f", {item}"
        if used + len(piece) > budget:
            if not chosen:
                room = budget - used - 1
                if room >= 4:
                    chosen.append(_short(item, room))
            break
        chosen.append(item)
        used += len(piece)
    if not chosen:
        return None
    return f"{label}=" + ", ".join(chosen)


def _fit(segments: list[Optional[str]], budget: int) -> str:
    """Join complete segments under a budget, shortening at most one item."""
    out: list[str] = []
    used = 0
    for segment in segments:
        if not segment:
            continue
        extra = len(segment) + (1 if out else 0)
        if used + extra <= budget:
            out.append(segment)
            used += extra
            continue
        room = budget - used - (1 if out else 0)
        if room >= 12:
            out.append(segment[: room - 1].rstrip() + "…")
        break
    return " ".join(out)


def compile_profile_core(
    profile: Optional[PersonInteractionProfile],
    *,
    max_chars: int = MAX_CORE_CHARS,
) -> Optional[str]:
    if profile is None:
        return None
    identity = profile.identity or {}
    confirmed_name = str(identity.get("name") or "UNKNOWN")
    segments: list[Optional[str]] = [
        "【当前交流对象】",
        f"person_id={_short(profile.person_id, 64)}",
        f"relationship={_short(profile.relationship_tier, 32).lower()}",
    ]
    if confirmed_name and confirmed_name != "UNKNOWN":
        segments.append(f"name={_short(confirmed_name, 40)}")
    preferred = str(identity.get("preferred_name") or "UNKNOWN")
    if preferred and preferred != "UNKNOWN" and preferred != confirmed_name:
        segments.append(f"preferred_name={_short(preferred, 40)}")
    pronouns = str(identity.get("pronouns") or "UNKNOWN")
    if pronouns and pronouns != "UNKNOWN":
        segments.append(f"pronouns={_short(pronouns, 24)}")

    for field, label in (
        (ProfileFieldKey.FORMALITY, "formality"),
        (ProfileFieldKey.TONE, "tone"),
        (ProfileFieldKey.HUMOR, "humor"),
        (ProfileFieldKey.DEPTH, "depth"),
        (ProfileFieldKey.LANGUAGE, "language"),
        (ProfileFieldKey.ADDRESS, "address"),
    ):
        value = profile.communication_value(field)
        if value and str(value).lower() != "unknown":
            segments.append(f"{label}={_short(value)}")

    occupation = profile.communication_value(
        ProfileFieldKey.OCCUPATION, default=""
    )
    if not occupation:
        occupation = " ".join(_entry_texts(profile, "background.occupation")[:1])
    segments.append(
        f"occupation={_short(occupation) if occupation else 'unknown'}"
    )

    brief: list[str] = []
    for field_key in ("personality.traits", "interests.topics", "projects.active"):
        brief.extend(_entry_texts(profile, field_key)[:1])
        if len(brief) >= 2:
            break
    if brief:
        segments.append("about=" + "; ".join(_short(item, 60) for item in brief[:2]))

    block = _fit(segments, max_chars)
    return block or None


def compile_profile_relevant(
    profile: Optional[PersonInteractionProfile],
    *,
    query: str = "",
    max_chars: int = MAX_RELEVANT_CHARS,
) -> Optional[str]:
    if profile is None:
        return None
    query_tokens = _tokens(query)
    grouped: dict[str, list[str]] = {}

    for field_key, label in _ENTRY_RELEVANT:
        grouped.setdefault(label, []).extend(_entry_texts(profile, field_key))
    for field, label in _LEGACY_RELEVANT:
        grouped.setdefault(label, []).extend(_labels(profile, field))

    if query_tokens:
        def rank(item: str) -> tuple[int, int]:
            overlap = len(query_tokens & _tokens(item))
            return (-overlap, 0)

        for label, items in grouped.items():
            ordered = sorted(_dedupe(items), key=rank)
            # Keep relevant items first, but drop wholly unrelated ones when a
            # question clearly points at a subset.
            matched = [item for item in ordered if query_tokens & _tokens(item)]
            grouped[label] = matched or ordered[:1]
    else:
        grouped = {label: _dedupe(items) for label, items in grouped.items()}

    segments: list[Optional[str]] = []
    remaining = max_chars
    for label, items in grouped.items():
        if remaining <= 0:
            break
        segment = _list_segment(label, items, remaining)
        if segment is None:
            continue
        segments.append(segment)
        remaining -= len(segment) + 1
    block = _fit(segments, max_chars)
    return block or None


def compile_profile_context(
    profile: Optional[PersonInteractionProfile],
    *,
    max_chars: int = MAX_BLOCK_CHARS,
    query: str = "",
) -> Optional[str]:
    """Backwards-compatible combined block (core first, relevant second)."""
    if profile is None:
        return None
    core_budget = min(MAX_CORE_CHARS, max_chars)
    core = compile_profile_core(profile, max_chars=core_budget)
    if core is None:
        return None
    remaining = max_chars - len(core) - 1
    relevant = (
        compile_profile_relevant(profile, query=query, max_chars=remaining)
        if remaining > 20
        else None
    )
    if relevant:
        return f"{core}\n{relevant}"
    return core


__all__ = [
    "MAX_BLOCK_CHARS",
    "MAX_CORE_CHARS",
    "MAX_RELEVANT_CHARS",
    "compile_profile_context",
    "compile_profile_core",
    "compile_profile_relevant",
]
