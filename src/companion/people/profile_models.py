"""Per-person understanding profile (V0.7).

This is an extension of the existing CMF person layer, not a second database.
It answers one question the stable persona does not: *how does Ava understand
this particular person?* The stable persona never changes with the speaker;
runtime state changes every turn; this profile changes only when there is
evidence.

Design rules:

- **Bounded, validated values.** Communication preferences are enums, not
  free-form model prose. Unknown stays ``UNKNOWN``.
- **Facts reference memory.** The understanding layer keeps short labels plus
  the ``memory_id``/event that supports them. ``MemoryService`` remains the
  authority for the fact itself; a profile must not become a second, divergent
  copy of someone's biography.
- **No invented attributes.** Occupation, age, health, schedule and similar
  personal attributes are only stored when explicitly reported and confirmed.
- **Provenance and revision.** Every effective value records its source,
  confidence category, evidence and the revision it first appeared in.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_SHORT_TEXT = 160
MAX_LIST_ITEMS = 8
MAX_ADDRESS_PREFERENCE = 40

# -- field registry ---------------------------------------------------------


class ProfileFieldKey(str, Enum):
    FORMALITY = "communication.formality"
    TONE = "communication.tone"
    HUMOR = "communication.humor_preference"
    DEPTH = "communication.explanation_depth"
    LANGUAGE = "communication.preferred_language"
    ADDRESS = "communication.address_preference"
    OCCUPATION = "understanding.occupation"
    INTERESTS = "understanding.interests"
    ONGOING_PROJECTS = "understanding.ongoing_projects"
    USUAL_ACTIVITIES = "understanding.usual_activities"
    PERSONAL_GOALS = "understanding.personal_goals"
    RECURRING_TOPICS = "interaction.recurring_topics"
    COMMUNICATION_FEEDBACK = "interaction.communication_feedback"
    FAMILIARITY_NOTES = "interaction.familiarity_notes"


COMMUNICATION_FIELDS: dict[str, tuple[str, ...]] = {
    ProfileFieldKey.FORMALITY.value: ("casual", "neutral", "formal", "UNKNOWN"),
    ProfileFieldKey.TONE.value: (
        "friendly",
        "neutral",
        "professional",
        "playful",
        "UNKNOWN",
    ),
    ProfileFieldKey.HUMOR.value: ("welcome", "neutral", "avoid", "UNKNOWN"),
    ProfileFieldKey.DEPTH.value: ("concise", "adaptive", "detailed", "UNKNOWN"),
    ProfileFieldKey.LANGUAGE.value: ("zh", "en", "other", "UNKNOWN"),
}

LIST_FIELDS: frozenset[str] = frozenset(
    {
        ProfileFieldKey.INTERESTS.value,
        ProfileFieldKey.ONGOING_PROJECTS.value,
        ProfileFieldKey.USUAL_ACTIVITIES.value,
        ProfileFieldKey.PERSONAL_GOALS.value,
        ProfileFieldKey.RECURRING_TOPICS.value,
        ProfileFieldKey.COMMUNICATION_FEEDBACK.value,
    }
)

ALL_FIELD_KEYS: frozenset[str] = frozenset(key.value for key in ProfileFieldKey)


class ProfileSource(str, Enum):
    DEFAULT = "DEFAULT"
    DIRECT_SELF_REPORT = "DIRECT_SELF_REPORT"
    USER_CORRECTION = "USER_CORRECTION"
    INTERACTION_INFERENCE = "INTERACTION_INFERENCE"
    ADMIN = "ADMIN"


class ProfileFieldStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    RETIRED = "RETIRED"


class CandidateStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


def validate_field_value(field_key: str, value: Any) -> Any:
    """Normalise one field value or raise ``ValueError``.

    Enum-like fields only accept their documented values. List fields accept a
    bounded list of short labels (or evidence-backed entries); anything else is
    a programming error, not something to silently store.
    """
    key = str(field_key)
    if key in COMMUNICATION_FIELDS:
        text = str(value).strip().lower()
        if text not in COMMUNICATION_FIELDS[key]:
            raise ValueError(
                f"invalid {key} value {value!r}; expected"
                f" {COMMUNICATION_FIELDS[key]}"
            )
        return text
    if key == ProfileFieldKey.ADDRESS.value:
        text = " ".join(str(value or "").split())[:MAX_ADDRESS_PREFERENCE]
        return text or "UNKNOWN"
    if key == ProfileFieldKey.OCCUPATION.value:
        text = " ".join(str(value or "").split())[:MAX_SHORT_TEXT]
        return text or "unknown"
    if key == ProfileFieldKey.FAMILIARITY_NOTES.value:
        text = " ".join(str(value or "").split())[:400]
        return text or "unknown"
    if key in LIST_FIELDS:
        return _normalise_list(value)
    raise ValueError(f"unknown profile field {key!r}")


def _normalise_list(value: Any) -> list[dict]:
    items = value if isinstance(value, list) else [value]
    normalised: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            label = " ".join(str(item.get("label") or "").split())
            entry = {
                "label": label[:MAX_SHORT_TEXT],
                "memory_id": _optional_id(item.get("memory_id")),
                "event_id": _optional_id(item.get("event_id")),
                "candidate_id": _optional_id(item.get("candidate_id")),
            }
        else:
            label = " ".join(str(item or "").split())
            entry = {
                "label": label[:MAX_SHORT_TEXT],
                "memory_id": None,
                "event_id": None,
                "candidate_id": None,
            }
        if not entry["label"]:
            continue
        marker = entry["label"].casefold()
        if marker in seen:
            continue
        seen.add(marker)
        normalised.append(entry)
        if len(normalised) >= MAX_LIST_ITEMS:
            break
    return normalised


def _optional_id(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text[:128] or None


# -- aggregate models -------------------------------------------------------


class ProfileMetadata(BaseModel):
    revision: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_reviewed_at: Optional[datetime] = None
    updated_by: str = "system"


class ProfileField(BaseModel):
    field_key: str
    value: Any
    source: ProfileSource = ProfileSource.DEFAULT
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: ProfileFieldStatus = ProfileFieldStatus.ACTIVE
    evidence_memory_ids: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_candidate_ids: list[str] = Field(default_factory=list)
    revision: int = 1
    updated_at: Optional[datetime] = None


class ProfileCandidate(BaseModel):
    candidate_id: str
    person_id: str
    field_key: str
    candidate_value: Any
    source: ProfileSource = ProfileSource.INTERACTION_INFERENCE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: CandidateStatus = CandidateStatus.PENDING
    evidence_memory_ids: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_candidate_ids: list[str] = Field(default_factory=list)
    reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProfileRevision(BaseModel):
    revision_id: str
    person_id: str
    revision: int
    changes: dict[str, Any] = Field(default_factory=dict)
    changed_by: str
    source_event_id: Optional[str] = None
    reason: Optional[str] = None
    created_at: Optional[datetime] = None


class ProfileEvidence(BaseModel):
    evidence_id: str
    person_id: str
    field_key: str
    memory_id: Optional[str] = None
    source_event_id: Optional[str] = None
    candidate_id: Optional[str] = None
    evidence_type: str = "EVENT"
    created_at: Optional[datetime] = None


class PersonInteractionProfile(BaseModel):
    """The effective, compiled view used by prompts and the admin UI."""

    person_id: str
    display_name: str = "unknown"
    relationship_tier: str = "STRANGER"
    fields: dict[str, ProfileField] = Field(default_factory=dict)
    candidates: list[ProfileCandidate] = Field(default_factory=list)
    compact_summary: str = ""
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)

    # -- typed accessors ---------------------------------------------------

    def communication_value(self, field_key: ProfileFieldKey, default="UNKNOWN"):
        field = self.fields.get(field_key.value)
        if field is None or field.status != ProfileFieldStatus.ACTIVE:
            return default
        return field.value

    def list_entries(self, field_key: ProfileFieldKey) -> list[dict]:
        field = self.fields.get(field_key.value)
        if field is None or field.status != ProfileFieldStatus.ACTIVE:
            return []
        value = field.value
        return value if isinstance(value, list) else []


__all__ = [
    "ALL_FIELD_KEYS",
    "COMMUNICATION_FIELDS",
    "CandidateStatus",
    "LIST_FIELDS",
    "MAX_ADDRESS_PREFERENCE",
    "MAX_LIST_ITEMS",
    "MAX_SHORT_TEXT",
    "PersonInteractionProfile",
    "ProfileCandidate",
    "ProfileEvidence",
    "ProfileField",
    "ProfileFieldKey",
    "ProfileFieldStatus",
    "ProfileMetadata",
    "ProfileRevision",
    "ProfileSource",
    "validate_field_value",
]
