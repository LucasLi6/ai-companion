"""Stable constants for the Companion Memory Format (CMF) 1.0.

The CMF *format version* is independent of the SQLite *schema version*:

- ``CMF_FORMAT_VERSION`` describes the portable file contract. A reader that
  does not understand a newer major format must refuse to open the file.
- ``companion.storage.schema.SCHEMA_VERSION`` describes the append-only
  migration level inside that format. Older schema versions are upgraded on
  open; newer ones are refused.

``APPLICATION_ID`` is written to SQLite's ``application_id`` header field so an
arbitrary SQLite database cannot be mistaken for a Companion memory file.
"""

from __future__ import annotations

CMF_FORMAT_VERSION = "1.0"
CMF_FORMAT_MAJOR = 1
CMF_EXTENSION = ".memory"
LEGACY_EXTENSIONS = (".db", ".sqlite", ".sqlite3")

# 0x434D4601 = "CMF" + format major 1.
APPLICATION_ID = 0x434D4601
APPLICATION_ID_NAME = "CMF1"

# Tables that every valid CMF 1.0 file must contain.
REQUIRED_TABLES = frozenset(
    {
        "schema_version",
        "cmf_manifest",
        "agent_identity",
        "agent_persona",
        "persons",
        "conversations",
        "utterances",
        "memory_items",
        "memory_subjects",
        "memory_sources",
        "memory_evidence",
        "memory_revisions",
        "memory_provenance",
        "memory_provenance_events",
        "memory_assignments",
        "memory_candidate_writes",
        "person_profiles",
        "relationship_presets",
        "person_interaction_profiles",
        "relationship_history",
        "research_tasks",
        "research_projects",
        "research_runs",
        "web_sources",
        "knowledge_claims",
        "curiosity_queue",
        "reflection_records",
        "learning_runs",
        "goals",
        "media_assets",
        "embedding_models",
        "memory_embeddings",
        "embedding_index_metadata",
        "retrieval_feedback",
        "emotion_profiles",
        "emotion_appraisal_rules",
        "emotion_events",
        "emotion_state_snapshots",
        "emotion_feedback",
        "voice_profiles",
    }
)


def format_major(version: str | None) -> int | None:
    """Parse the major component of a CMF format version."""
    if not version:
        return None
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError):
        return None
