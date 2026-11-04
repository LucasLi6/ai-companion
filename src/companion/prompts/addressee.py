"""Trusted speaker/addressee rendering shared by both prompts.

``addressee_hint`` is recorded by the local event pipeline (an explicit human
addressee, the agent, or unknown). It is trusted metadata: neither prompt may
guess a different target from the message text, and a model inference may not
override it.

The response prompt uses the human-readable target (the agent's trusted name,
"other human", or "unknown"). The cognitive prompt keeps the canonical
``agent`` / ``human`` / ``unknown`` tokens. Both share the same normalisation,
so the two prompts cannot drift apart.
"""

from __future__ import annotations

from typing import Any

CANONICAL_ADDRESSEES = frozenset({"agent", "human", "unknown"})


def normalize_addressee_hint(value: Any) -> str:
    """Return one of ``agent`` / ``human`` / ``unknown``.

    A missing or unrecognised value is ``unknown``. It is never inferred from
    the text.
    """
    text = str(value or "").strip().lower()
    return text if text in CANONICAL_ADDRESSEES else "unknown"


def safe_speaker_label(value: Any, fallback: str = "unknown") -> str:
    """A speaker/agent name is metadata, never prompt markup or a new line."""
    text = " ".join(str(value or "").split())
    text = text.replace("【", "[").replace("】", "]")
    text = text[:40]
    return text or fallback


def addressee_target(hint: Any, *, agent_name: str = "Companion") -> str:
    """Human-readable target for a response-prompt label."""
    normalized = normalize_addressee_hint(hint)
    if normalized == "agent":
        return safe_speaker_label(agent_name, "Companion")
    if normalized == "human":
        return "other human"
    return "unknown"


def needs_speaker_label(*, multi_person: bool, addressee_hint: Any) -> bool:
    """Whether a human turn must carry an explicit speaker/target label.

    A one-to-one conversation stays plain text. A multi-speaker view always
    labels turns. A single-speaker history still keeps an explicitly
    human-directed utterance labelled, because its target is meaningful even
    when only one person is currently visible.
    """
    return bool(
        multi_person or normalize_addressee_hint(addressee_hint) == "human"
    )


def render_human_turn(
    turn, *, multi_person: bool, agent_name: str = "Companion"
) -> str:
    """Render one human history turn without merging or inventing a target."""
    text = str(getattr(turn, "text", "") or "")
    hint = getattr(turn, "addressee_hint", None)
    if not needs_speaker_label(multi_person=multi_person, addressee_hint=hint):
        return text
    speaker = safe_speaker_label(getattr(turn, "speaker_label", None))
    target = addressee_target(hint, agent_name=agent_name)
    return f"[{speaker} → {target}] {text}"


def render_current_turn(
    context, *, multi_person: bool, agent_name: str = "Companion"
) -> str:
    """Render the current human utterance with the same rules as history."""
    text = str(getattr(context, "current_text", "") or "")
    hint = getattr(context, "current_addressee_hint", None)
    if not needs_speaker_label(multi_person=multi_person, addressee_hint=hint):
        return text
    speaker = safe_speaker_label(
        getattr(context, "current_speaker_label", None), "user"
    )
    target = addressee_target(hint, agent_name=agent_name)
    return f"[{speaker} → {target}] {text}"
