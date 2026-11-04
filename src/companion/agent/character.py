"""Trusted per-turn character snapshot (V0.4.3).

The cognitive and response providers must describe the *same* person. Before
V0.4.3 each layer re-derived identity separately (the cognitive prompt only saw
coarse traits, the response prompt re-read the persona core), which allowed a
silent revision skew if an admin edited the character between the two calls.

This module builds one immutable :class:`CharacterSnapshot` per turn from the
trusted ``AgentService`` effective context. It adds no storage and accepts no
model input. Relevance filtering for stances is deliberately local: an opinion
about music is not injected into a conversation about a debugger.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional

from companion.agent.models import (
    DEFAULT_EMOTIONAL_DISPOSITIONS,
    CharacterSnapshot,
    EffectiveAgentContext,
)
from companion.agent.stance import filter_relevant_stances

MAX_DISPOSITION_KEYS = 8


def parse_emotional_dispositions(raw: Any) -> dict[str, float]:
    """Validate an admin-configured disposition map; clamp to [0, 1].

    Accepts a mapping or a JSON object string. Unknown keys are ignored and
    every value falls back to the conservative midpoint, so a malformed
    configuration can never crash a turn or amplify affect.
    """
    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, TypeError):
            data = {}
    if not isinstance(data, Mapping):
        data = {}

    result = dict(DEFAULT_EMOTIONAL_DISPOSITIONS)
    for key in list(DEFAULT_EMOTIONAL_DISPOSITIONS)[:MAX_DISPOSITION_KEYS]:
        if key not in data:
            continue
        value = data.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        result[key] = max(0.0, min(1.0, number))
    return result


async def load_emotional_dispositions(settings_repo) -> dict[str, float]:
    """Read the optional trusted disposition map from application settings."""
    if settings_repo is None:
        return dict(DEFAULT_EMOTIONAL_DISPOSITIONS)
    try:
        raw = await settings_repo.get("persona.emotional_dispositions")
    except Exception:  # pragma: no cover - settings must never fail a turn
        return dict(DEFAULT_EMOTIONAL_DISPOSITIONS)
    return parse_emotional_dispositions(raw)


def build_character_snapshot(
    effective: EffectiveAgentContext,
    *,
    stances: Optional[list] = None,
    query_text: str = "",
    topic: Optional[str] = None,
    dispositions: Optional[Mapping[str, float]] = None,
) -> CharacterSnapshot:
    """Compile the one character view both provider calls will share."""
    core = dict(effective.persona_core or {})
    try:
        revision = int(core.get("revision") or effective.revision or 1)
    except (TypeError, ValueError):
        revision = int(effective.revision or 1)

    relevant = filter_relevant_stances(
        stances or [], topic=topic, query_text=query_text
    )
    stance_dicts: list[dict] = []
    for record in relevant:
        if isinstance(record, Mapping):
            stance_dicts.append(
                {
                    "topic": str(record.get("topic") or "UNKNOWN")[:120],
                    "stance": str(record.get("stance") or "")[:240],
                    "source": str(record.get("source") or "RESPONSE")[:16],
                }
            )
        else:
            stance_dicts.append(
                {
                    "topic": str(getattr(record, "topic", "UNKNOWN") or "UNKNOWN")[
                        :120
                    ],
                    "stance": str(getattr(record, "stance", "") or "")[:240],
                    "source": str(getattr(record, "source", "RESPONSE") or "RESPONSE")[
                        :16
                    ],
                }
            )

    return CharacterSnapshot(
        agent_id=effective.agent_id,
        revision=revision,
        display_name=effective.display_name or "Companion",
        description=effective.description or "",
        traits=dict(effective.traits or {}),
        persona_core=core,
        relationship_tier=effective.familiarity or "STRANGER",
        relevant_stances=tuple(stance_dicts),
        emotional_dispositions=parse_emotional_dispositions(
            dispositions or DEFAULT_EMOTIONAL_DISPOSITIONS
        ),
    )
