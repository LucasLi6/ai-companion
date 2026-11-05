"""Compile one person's confirmed understanding profile into a short block.

The block is *background data*, never an instruction and never a second
persona. It is bounded (a few hundred characters), includes only ACTIVE,
confirmed values, and never serialises the profile JSON or pending candidates.

Example::

    【当前交流对象】 person_id=P01 relationship=familiar formality=casual
    tone=friendly interests=robotics, embedded systems occupation=unknown
"""

from __future__ import annotations

from typing import Optional

from companion.people.profile_models import (
    PersonInteractionProfile,
    ProfileFieldKey,
)

MAX_BLOCK_CHARS = 600
MAX_LABEL_CHARS = 40
MAX_LIST_ITEMS_IN_PROMPT = 3


def _short(value: object, limit: int = MAX_LABEL_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _labels(profile: PersonInteractionProfile, field: ProfileFieldKey) -> list[str]:
    labels: list[str] = []
    for entry in profile.list_entries(field):
        label = _short(entry.get("label") if isinstance(entry, dict) else entry)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= MAX_LIST_ITEMS_IN_PROMPT:
            break
    return labels


def compile_profile_context(
    profile: Optional[PersonInteractionProfile],
    *,
    max_chars: int = MAX_BLOCK_CHARS,
) -> Optional[str]:
    """Return the compact prompt block, or ``None`` when there is nothing to say.

    Only effective (ACTIVE) values are read. A pending candidate fact — for
    example an occupation inferred from one conversation — never appears.
    """
    if profile is None:
        return None

    parts = [
        "【当前交流对象】",
        f"person_id={_short(profile.person_id, 64)}",
        f"relationship={_short(profile.relationship_tier, 32).lower()}",
    ]
    if profile.display_name and profile.display_name != "unknown":
        parts.append(f"name={_short(profile.display_name, 40)}")

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
            parts.append(f"{label}={_short(value)}")

    occupation = profile.communication_value(ProfileFieldKey.OCCUPATION, default="")
    parts.append(f"occupation={_short(occupation) if occupation else 'unknown'}")

    for field, label in (
        (ProfileFieldKey.INTERESTS, "interests"),
        (ProfileFieldKey.ONGOING_PROJECTS, "ongoing"),
        (ProfileFieldKey.USUAL_ACTIVITIES, "activities"),
        (ProfileFieldKey.PERSONAL_GOALS, "goals"),
        (ProfileFieldKey.RECURRING_TOPICS, "topics"),
    ):
        labels = _labels(profile, field)
        if labels:
            parts.append(f"{label}={', '.join(labels)}")

    block = " ".join(parts)
    if len(block) > max_chars:
        block = block[: max_chars - 1].rstrip() + "…"
    return block


__all__ = ["MAX_BLOCK_CHARS", "compile_profile_context"]
