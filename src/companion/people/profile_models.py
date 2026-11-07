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

from dataclasses import dataclass
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


class ProfileEntryStatus(str, Enum):
    """Lifecycle of one independent understanding entry.

    ``ACTIVE`` is the only status the Speaker context uses. ``TENTATIVE`` is
    still visible to a review but not presented as settled understanding.
    ``MERGED``/``ARCHIVED``/``RETIRED`` keep the row for traceability; entry
    content is never destroyed by a refinement.
    """

    ACTIVE = "ACTIVE"
    TENTATIVE = "TENTATIVE"
    PENDING_REVIEW = "PENDING_REVIEW"
    MERGED = "MERGED"
    ARCHIVED = "ARCHIVED"
    RETIRED = "RETIRED"


class ProfileFieldStorage(str, Enum):
    """Where a registry field physically lives.

    ``ENTRY`` fields use independent ``person_profile_entries`` rows, so one
    item can be added, corrected, merged or archived without rewriting the
    others. ``SINGLE`` is the pre-V0.9.7 enum-validated field table.
    ``DERIVED`` is read from the authoritative source (relationship tier) and
    can never be written by the refinement model.
    """

    ENTRY = "ENTRY"
    SINGLE = "SINGLE"
    DERIVED = "DERIVED"


@dataclass(frozen=True)
class ProfileFieldSpec:
    key: str
    category: str
    storage: ProfileFieldStorage
    label: str
    # Confirmed identity fields: a model may only ever propose a pending
    # confirmation. Local code writes them after explicit confirmation.
    strict_identity: bool = False
    # False for derived/authoritative values. A proposal is rejected, not
    # silently ignored.
    model_writable: bool = True
    single_value: bool = False
    max_entries: int = 40
    max_chars: int = 240


# V0.9.7 evolving-understanding registry. Keys are stable storage identifiers;
# a category groups related entries for the prompt but never merges them.
_PROFILE_FIELD_SPECS: tuple[ProfileFieldSpec, ...] = (
    # -- confirmed identity (never auto-written) -------------------------
    ProfileFieldSpec("identity.name", "identity", ProfileFieldStorage.ENTRY,
                     "name", strict_identity=True, model_writable=False,
                     single_value=True, max_entries=1, max_chars=80),
    ProfileFieldSpec("identity.preferred_name", "identity", ProfileFieldStorage.ENTRY,
                     "preferred_name", strict_identity=True, model_writable=False,
                     single_value=True, max_entries=1, max_chars=80),
    ProfileFieldSpec("identity.nicknames", "identity", ProfileFieldStorage.ENTRY,
                     "nicknames", strict_identity=True, model_writable=False,
                     max_entries=8, max_chars=80),
    ProfileFieldSpec("identity.gender_identity", "identity", ProfileFieldStorage.ENTRY,
                     "gender_identity", strict_identity=True, model_writable=False,
                     single_value=True, max_entries=1, max_chars=80),
    ProfileFieldSpec("identity.pronouns", "identity", ProfileFieldStorage.ENTRY,
                     "pronouns", strict_identity=True, model_writable=False,
                     single_value=True, max_entries=1, max_chars=80),
    # The relationship tier is the single authoritative source. The derived
    # value exists for snapshot readability only.
    ProfileFieldSpec("relationship.familiarity", "relationship",
                     ProfileFieldStorage.DERIVED, "familiarity",
                     strict_identity=True, model_writable=False),
    ProfileFieldSpec("relationship.label", "relationship", ProfileFieldStorage.ENTRY,
                     "label", strict_identity=True, model_writable=False,
                     max_entries=8, max_chars=120),
    ProfileFieldSpec("relationship.boundaries", "relationship",
                     ProfileFieldStorage.ENTRY, "boundaries",
                     strict_identity=True, model_writable=False,
                     max_entries=20, max_chars=240),
    # -- communication ---------------------------------------------------
    ProfileFieldSpec("communication.language", "communication",
                     ProfileFieldStorage.ENTRY, "language", max_entries=4),
    ProfileFieldSpec("communication.preferred_tone", "communication",
                     ProfileFieldStorage.ENTRY, "preferred_tone", max_entries=8),
    ProfileFieldSpec("communication.response_length", "communication",
                     ProfileFieldStorage.ENTRY, "response_length", max_entries=8),
    ProfileFieldSpec("communication.explanation_style", "communication",
                     ProfileFieldStorage.ENTRY, "explanation_style", max_entries=8),
    ProfileFieldSpec("communication.humor_preference", "communication",
                     ProfileFieldStorage.ENTRY, "humor_preference", max_entries=8),
    ProfileFieldSpec("communication.boundaries", "communication",
                     ProfileFieldStorage.ENTRY, "communication_boundaries",
                     max_entries=20),
    # -- personality -----------------------------------------------------
    ProfileFieldSpec("personality.traits", "personality",
                     ProfileFieldStorage.ENTRY, "traits", max_entries=20),
    ProfileFieldSpec("personality.behavior_patterns", "personality",
                     ProfileFieldStorage.ENTRY, "behavior_patterns",
                     max_entries=20),
    ProfileFieldSpec("personality.thinking_style", "personality",
                     ProfileFieldStorage.ENTRY, "thinking_style", max_entries=8),
    ProfileFieldSpec("personality.decision_style", "personality",
                     ProfileFieldStorage.ENTRY, "decision_style", max_entries=8),
    # -- interests -------------------------------------------------------
    ProfileFieldSpec("interests.topics", "interests",
                     ProfileFieldStorage.ENTRY, "topics", max_entries=40),
    ProfileFieldSpec("interests.likes", "interests",
                     ProfileFieldStorage.ENTRY, "likes", max_entries=40),
    ProfileFieldSpec("interests.dislikes", "interests",
                     ProfileFieldStorage.ENTRY, "dislikes", max_entries=40),
    # -- knowledge -------------------------------------------------------
    ProfileFieldSpec("knowledge.domains", "knowledge",
                     ProfileFieldStorage.ENTRY, "domains", max_entries=30),
    ProfileFieldSpec("knowledge.skills", "knowledge",
                     ProfileFieldStorage.ENTRY, "skills", max_entries=40),
    ProfileFieldSpec("knowledge.experience", "knowledge",
                     ProfileFieldStorage.ENTRY, "experience", max_entries=40),
    # -- background ------------------------------------------------------
    ProfileFieldSpec("background.education", "background",
                     ProfileFieldStorage.ENTRY, "education", max_entries=12),
    ProfileFieldSpec("background.occupation", "background",
                     ProfileFieldStorage.ENTRY, "occupation", max_entries=8),
    ProfileFieldSpec("background.history", "background",
                     ProfileFieldStorage.ENTRY, "history", max_entries=40),
    ProfileFieldSpec("background.stable_facts", "background",
                     ProfileFieldStorage.ENTRY, "stable_facts", max_entries=40),
    # -- projects --------------------------------------------------------
    ProfileFieldSpec("projects.active", "projects",
                     ProfileFieldStorage.ENTRY, "active_projects", max_entries=20),
    ProfileFieldSpec("projects.paused", "projects",
                     ProfileFieldStorage.ENTRY, "paused_projects", max_entries=20),
    ProfileFieldSpec("projects.completed", "projects",
                     ProfileFieldStorage.ENTRY, "completed_projects",
                     max_entries=40),
    # -- goals -----------------------------------------------------------
    ProfileFieldSpec("goals.short_term", "goals",
                     ProfileFieldStorage.ENTRY, "short_term_goals", max_entries=20),
    ProfileFieldSpec("goals.long_term", "goals",
                     ProfileFieldStorage.ENTRY, "long_term_goals", max_entries=20),
    # Migration target for the pre-V0.9.7 generic "personal goals" list.
    ProfileFieldSpec("goals.general", "goals",
                     ProfileFieldStorage.ENTRY, "goals", max_entries=20),
    # -- habits ----------------------------------------------------------
    ProfileFieldSpec("habits.stable_habits", "habits",
                     ProfileFieldStorage.ENTRY, "stable_habits", max_entries=30),
    ProfileFieldSpec("habits.working_preferences", "habits",
                     ProfileFieldStorage.ENTRY, "working_preferences",
                     max_entries=20),
    # -- values ----------------------------------------------------------
    ProfileFieldSpec("values.priorities", "values",
                     ProfileFieldStorage.ENTRY, "priorities", max_entries=20),
    ProfileFieldSpec("values.motivations", "values",
                     ProfileFieldStorage.ENTRY, "motivations", max_entries=20),
    # -- interaction -----------------------------------------------------
    ProfileFieldSpec("interaction.communication_feedback", "interaction",
                     ProfileFieldStorage.ENTRY, "communication_feedback",
                     max_entries=30),
    ProfileFieldSpec("interaction.recurring_patterns", "interaction",
                     ProfileFieldStorage.ENTRY, "recurring_patterns",
                     max_entries=30),
    ProfileFieldSpec("interaction.shared_context", "interaction",
                     ProfileFieldStorage.ENTRY, "shared_context",
                     max_entries=30),
    ProfileFieldSpec("interaction.familiarity_notes", "interaction",
                     ProfileFieldStorage.ENTRY, "familiarity_notes",
                     max_entries=10),
)

_PROFILE_FIELD_INDEX: dict[str, ProfileFieldSpec] = {
    spec.key: spec for spec in _PROFILE_FIELD_SPECS
}

# Legacy pre-V0.9.7 fields stay readable. A refinement may not target them
# with free-form prose because they carry enum-validated interaction settings.
LEGACY_ENTRY_MIGRATION: dict[str, str] = {
    ProfileFieldKey.INTERESTS.value: "interests.topics",
    ProfileFieldKey.ONGOING_PROJECTS.value: "projects.active",
    ProfileFieldKey.USUAL_ACTIVITIES.value: "habits.stable_habits",
    ProfileFieldKey.PERSONAL_GOALS.value: "goals.general",
    ProfileFieldKey.RECURRING_TOPICS.value: "interaction.recurring_patterns",
    ProfileFieldKey.COMMUNICATION_FEEDBACK.value: "interaction.communication_feedback",
    ProfileFieldKey.OCCUPATION.value: "background.occupation",
    ProfileFieldKey.FAMILIARITY_NOTES.value: "interaction.familiarity_notes",
}

STRICT_IDENTITY_FIELDS: frozenset[str] = frozenset(
    spec.key for spec in _PROFILE_FIELD_SPECS if spec.strict_identity
)

ENTRY_FIELD_KEYS: frozenset[str] = frozenset(
    spec.key
    for spec in _PROFILE_FIELD_SPECS
    if spec.storage is ProfileFieldStorage.ENTRY
)

PROFILE_FIELD_SPECS: tuple[ProfileFieldSpec, ...] = _PROFILE_FIELD_SPECS


def profile_field_spec(field_key: str) -> Optional[ProfileFieldSpec]:
    return _PROFILE_FIELD_INDEX.get(str(field_key or ""))


def is_strict_identity_field(field_key: str) -> bool:
    return str(field_key or "") in STRICT_IDENTITY_FIELDS


def is_entry_field(field_key: str) -> bool:
    return str(field_key or "") in ENTRY_FIELD_KEYS


class ProfileEntry(BaseModel):
    """One independent, traceable understanding item.

    ``content`` is a short human-readable statement (not a JSON blob). The
    original row stays in place when it is superseded: ``status`` changes and
    ``superseded_by`` names the newer entry.
    """

    entry_id: str
    person_id: str
    field_key: str
    content: str
    status: str = ProfileEntryStatus.ACTIVE.value
    source: str = ProfileSource.INTERACTION_INFERENCE.value
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    merged_from: list[str] = Field(default_factory=list)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    superseded_by: Optional[str] = None

    def is_effective(self) -> bool:
        return self.status in {
            ProfileEntryStatus.ACTIVE.value,
            ProfileEntryStatus.TENTATIVE.value,
        }


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
    # V0.9.7 independent understanding entries, grouped by field key. The
    # pre-V0.9.7 ``fields`` map stays intact so existing readers keep working.
    entries: dict[str, list[ProfileEntry]] = Field(default_factory=dict)
    # Confirmed identity only. The registration label is ``display_name`` and
    # is deliberately never used as the person's confirmed name.
    identity: dict[str, Any] = Field(default_factory=dict)
    relationship: dict[str, Any] = Field(default_factory=dict)
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

    def entry_texts(self, field_key: str) -> list[str]:
        """Effective entry contents for one registry key, oldest first."""
        return [
            entry.content
            for entry in self.entries.get(str(field_key), [])
            if entry.is_effective()
        ]


__all__ = [
    "ALL_FIELD_KEYS",
    "COMMUNICATION_FIELDS",
    "ENTRY_FIELD_KEYS",
    "LEGACY_ENTRY_MIGRATION",
    "LIST_FIELDS",
    "MAX_ADDRESS_PREFERENCE",
    "MAX_LIST_ITEMS",
    "MAX_SHORT_TEXT",
    "PROFILE_FIELD_SPECS",
    "PersonInteractionProfile",
    "ProfileEntry",
    "ProfileEntryStatus",
    "ProfileEvidence",
    "ProfileField",
    "ProfileFieldKey",
    "ProfileFieldSpec",
    "ProfileFieldStatus",
    "ProfileFieldStorage",
    "ProfileMetadata",
    "ProfileRevision",
    "ProfileSource",
    "STRICT_IDENTITY_FIELDS",
    "is_entry_field",
    "is_strict_identity_field",
    "profile_field_spec",
    "validate_field_value",
]
