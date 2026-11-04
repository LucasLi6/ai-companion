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

import hashlib
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

_SNAPSHOT_STRING_FIELDS = (
    "agent_id",
    "display_name",
    "description",
    "character_description",
    "relationship_tier",
)
_SNAPSHOT_MAPPING_FIELDS = ("traits", "persona_core", "emotional_dispositions")


class CharacterSnapshotValidationError(ValueError):
    """A persisted character snapshot is missing, corrupt or inconsistent."""


def character_snapshot_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over the complete character snapshot.

    The fingerprint covers every field in ``CharacterSnapshot.as_dict()``, not
    just the Persona Core revision: traits, description, relationship tier,
    stances and emotional dispositions can all change while the revision stays
    the same. Key order is irrelevant, no transient field is included, and
    values that cannot be serialized deterministically (NaN, custom objects)
    are a hard error rather than a silently reduced snapshot.
    """
    if not isinstance(snapshot, Mapping):
        raise CharacterSnapshotValidationError(
            "character snapshot must be a mapping"
        )
    try:
        serialized = json.dumps(
            dict(snapshot),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CharacterSnapshotValidationError(
            f"character snapshot is not serializable: {exc}"
        ) from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_character_snapshot(snapshot: Any) -> dict[str, Any]:
    """Validate a persisted snapshot's structure and return it as a dict.

    Only the fields written by :meth:`CharacterSnapshot.as_dict` are accepted.
    Missing or wrongly typed fields are an explicit failure: a partial snapshot
    must never be used as if it were the character that produced an analysis.
    """
    if not isinstance(snapshot, Mapping):
        raise CharacterSnapshotValidationError(
            "stored character snapshot is not a mapping"
        )

    validated = dict(snapshot)
    for name in _SNAPSHOT_STRING_FIELDS:
        value = validated.get(name)
        if not isinstance(value, str):
            raise CharacterSnapshotValidationError(
                f"stored character snapshot field {name!r} is not text"
            )
    if not validated["agent_id"].strip():
        raise CharacterSnapshotValidationError(
            "stored character snapshot has an empty agent_id"
        )

    revision = validated.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise CharacterSnapshotValidationError(
            "stored character snapshot revision is not an integer"
        )
    if revision < 0:
        raise CharacterSnapshotValidationError(
            "stored character snapshot revision is negative"
        )

    for name in _SNAPSHOT_MAPPING_FIELDS:
        if not isinstance(validated.get(name), Mapping):
            raise CharacterSnapshotValidationError(
                f"stored character snapshot field {name!r} is not a mapping"
            )

    traits = validated["traits"]
    if any(not isinstance(key, str) for key in traits):
        raise CharacterSnapshotValidationError(
            "stored character snapshot traits have non-text keys"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in traits.values()
    ):
        raise CharacterSnapshotValidationError(
            "stored character snapshot traits have non-integer values"
        )

    dispositions = validated["emotional_dispositions"]
    for key, value in dispositions.items():
        if not isinstance(key, str):
            raise CharacterSnapshotValidationError(
                "stored character snapshot dispositions have non-text keys"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CharacterSnapshotValidationError(
                "stored character snapshot dispositions are not numeric"
            )
        if not math.isfinite(float(value)):
            raise CharacterSnapshotValidationError(
                "stored character snapshot dispositions are not finite"
            )

    stances = validated.get("relevant_stances")
    if not isinstance(stances, (list, tuple)):
        raise CharacterSnapshotValidationError(
            "stored character snapshot relevant_stances is not a list"
        )
    for stance in stances:
        if not isinstance(stance, Mapping):
            raise CharacterSnapshotValidationError(
                "stored character snapshot stance is not a mapping"
            )
        for field_name in ("topic", "stance", "source"):
            if not isinstance(stance.get(field_name), str):
                raise CharacterSnapshotValidationError(
                    "stored character snapshot stance field"
                    f" {field_name!r} is not text"
                )

    validated["relevant_stances"] = [dict(item) for item in stances]
    return validated


def validate_stored_snapshot(stored: Any) -> dict[str, Any]:
    """Validate the analysis's stored snapshot against its persisted metadata.

    ``stored`` is the repository result:
    ``{"agent_id": ..., "revision": ..., "snapshot": ...}``. A legacy row has
    no metadata; a corrupt JSON column has no snapshot. Neither is ever
    assumed to match the current character.
    """
    if not isinstance(stored, Mapping):
        raise CharacterSnapshotValidationError(
            "the persisted analysis has no persona revision (legacy row)"
        )
    metadata_agent = stored.get("agent_id")
    metadata_revision = stored.get("revision")
    if not metadata_agent or metadata_revision is None:
        raise CharacterSnapshotValidationError(
            "the persisted analysis has no persona revision (legacy row)"
        )
    if isinstance(metadata_revision, bool) or not isinstance(
        metadata_revision, int
    ):
        raise CharacterSnapshotValidationError(
            "the persisted analysis has an invalid persona revision"
        )

    snapshot = stored.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise CharacterSnapshotValidationError(
            "the original snapshot is unavailable"
        )
    validated = validate_character_snapshot(snapshot)
    if validated["agent_id"] != str(metadata_agent):
        raise CharacterSnapshotValidationError(
            "the stored character snapshot does not match the persisted"
            " agent_id"
        )
    if validated["revision"] != metadata_revision:
        raise CharacterSnapshotValidationError(
            "the stored character snapshot does not match the persisted"
            " revision"
        )
    return validated


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
