"""Stable character identity (Persona Core) for V0.4.1.

The existing ``agent_persona`` row stores nine numeric expression traits. That
is enough to tune *how much* warmth or humour is expressed, but not *who the
companion is*: what it values, what it prefers, the contradictions that make it
feel like one character rather than a mood generator.

``PersonaCore`` is deliberately a separate, admin-maintained layer:

- it is written only by a trusted CLI/admin path, never by a model;
- every update is revisioned and audited;
- it contains character commitments, not biography: the fields are general
  dispositions and boundaries, and nothing is presented as a verified fact
  about a real person or a made-up life story;
- the response prompt renders it as *the character speaking*, not as a list of
  rules to mention. A conflicting user request still wins on style, while
  values/boundaries remain stable.

Being direct, humorous, stubborn, choosy or contradictory is supported by
``temperament``, ``preferences``, ``quirks`` and ``contradictions``. The
anti-sycophancy stance is a disposition ("can disagree without performing
disagreement"), not forced contrarianism.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from companion.storage.database import Database

MAX_LIST_ITEMS = 10
MAX_ITEM_LENGTH = 240

DEFAULT_TEMPERAMENT: dict[str, str] = {
    "directness": "moderate",
    "humor": "dry",
    "stubbornness": "moderate",
    "energy": "calm",
    "curiosity": "genuine",
}

DEFAULT_BOUNDARIES: list[str] = [
    "不编造个人经历、身份或现实见闻。",
    "不把一次情绪或玩笑写成永久人格或亲密关系。",
    "不声称拥有人类身体、感官或现实中的行动能力。",
]


def _clean_items(values: Iterable[str], field: str) -> list[str]:
    clean: list[str] = []
    for value in values:
        text = " ".join(str(value).split()).strip()
        if not text:
            continue
        if len(text) > MAX_ITEM_LENGTH:
            raise ValueError(f"{field} items must be at most {MAX_ITEM_LENGTH} chars")
        if text not in clean:
            clean.append(text)
    if len(clean) > MAX_LIST_ITEMS:
        raise ValueError(f"{field} supports at most {MAX_LIST_ITEMS} items")
    return clean


class PersonaCore(BaseModel):
    """Admin-maintained character commitments. Never model-authored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1, max_length=128)
    core_values: list[str] = Field(default_factory=list)
    temperament: dict[str, str] = Field(default_factory=dict)
    preferences: dict[str, str] = Field(default_factory=dict)
    quirks: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    conversational_boundaries: list[str] = Field(
        default_factory=lambda: list(DEFAULT_BOUNDARIES)
    )
    voice_identity: dict[str, str] = Field(default_factory=dict)
    revision: int = 1
    updated_at: datetime | None = None
    updated_by: str = "admin"

    @field_validator("core_values", "quirks", "contradictions")
    @classmethod
    def _validate_lists(cls, value: list[str], info) -> list[str]:
        return _clean_items(value, info.field_name)

    @field_validator("conversational_boundaries")
    @classmethod
    def _validate_boundaries(cls, value: list[str]) -> list[str]:
        return _clean_items(value, "conversational_boundaries")

    @field_validator("temperament", "preferences", "voice_identity")
    @classmethod
    def _validate_dicts(cls, value: dict[str, str], info) -> dict[str, str]:
        clean: dict[str, str] = {}
        for key, item in value.items():
            name = " ".join(str(key).split()).strip()
            text = " ".join(str(item).split()).strip()
            if not name or not text:
                continue
            if len(name) > 64 or len(text) > MAX_ITEM_LENGTH:
                raise ValueError(f"{info.field_name} entries are too long")
            clean[name] = text
        if len(clean) > MAX_LIST_ITEMS:
            raise ValueError(f"{info.field_name} supports at most {MAX_LIST_ITEMS} entries")
        return clean

    def as_prompt_dict(self) -> dict:
        """Compact trusted rendering for the response prompt."""
        return {
            "core_values": list(self.core_values),
            "temperament": dict(self.temperament),
            "preferences": dict(self.preferences),
            "quirks": list(self.quirks),
            "contradictions": list(self.contradictions),
            "conversational_boundaries": list(self.conversational_boundaries),
            "voice_identity": dict(self.voice_identity),
            "revision": self.revision,
        }


def default_persona_core(agent_id: str, now: datetime | None = None) -> PersonaCore:
    """Neutral defaults: no invented biography, stable boundaries."""
    return PersonaCore(
        agent_id=agent_id,
        core_values=[],
        temperament=dict(DEFAULT_TEMPERAMENT),
        preferences={},
        quirks=[],
        contradictions=[],
        conversational_boundaries=list(DEFAULT_BOUNDARIES),
        voice_identity={},
        revision=1,
        updated_at=now or datetime.now(timezone.utc),
        updated_by="system",
    )


def _json_list(raw: str | None) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _json_dict(raw: str | None) -> dict[str, str]:
    try:
        value = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


class PersonaCoreRepository:
    """Persistence for the single character identity row per agent."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get(self, agent_id: str) -> PersonaCore | None:
        cur = await self.db.execute(
            "SELECT agent_id, core_values_json, temperament_json,"
            " preferences_json, quirks_json, contradictions_json,"
            " conversational_boundaries_json, voice_identity_json, revision,"
            " updated_at, updated_by FROM agent_persona_core WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return PersonaCore(
            agent_id=row["agent_id"],
            core_values=_json_list(row["core_values_json"]),
            temperament=_json_dict(row["temperament_json"]),
            preferences=_json_dict(row["preferences_json"]),
            quirks=_json_list(row["quirks_json"]),
            contradictions=_json_list(row["contradictions_json"]),
            conversational_boundaries=_json_list(
                row["conversational_boundaries_json"]
            ),
            voice_identity=_json_dict(row["voice_identity_json"]),
            revision=int(row["revision"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            updated_by=row["updated_by"],
        )

    async def get_or_default(self, agent_id: str) -> PersonaCore:
        existing = await self.get(agent_id)
        if existing is not None:
            return existing
        now = datetime.now(timezone.utc)
        core = default_persona_core(agent_id, now)
        await self._insert_if_absent(core)
        return await self.get(agent_id) or core

    async def _insert_if_absent(self, core: PersonaCore) -> None:
        now = core.updated_at or datetime.now(timezone.utc)
        await self.db.execute(
            "INSERT OR IGNORE INTO agent_persona_core(agent_id, core_values_json,"
            " temperament_json, preferences_json, quirks_json,"
            " contradictions_json, conversational_boundaries_json,"
            " voice_identity_json, revision, updated_at, updated_by)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                core.agent_id,
                json.dumps(core.core_values, ensure_ascii=False),
                json.dumps(core.temperament, ensure_ascii=False, sort_keys=True),
                json.dumps(core.preferences, ensure_ascii=False, sort_keys=True),
                json.dumps(core.quirks, ensure_ascii=False),
                json.dumps(core.contradictions, ensure_ascii=False),
                json.dumps(core.conversational_boundaries, ensure_ascii=False),
                json.dumps(core.voice_identity, ensure_ascii=False, sort_keys=True),
                core.revision,
                now.isoformat(),
                core.updated_by,
            ),
        )

    async def update(
        self,
        core: PersonaCore,
        *,
        expected_revision: int,
        changed_by: str = "admin",
        now: datetime | None = None,
    ) -> bool:
        """Optimistic update; returns False when another admin won the race."""
        moment = now or datetime.now(timezone.utc)
        cur = await self.db.execute(
            "UPDATE agent_persona_core SET core_values_json = ?,"
            " temperament_json = ?, preferences_json = ?, quirks_json = ?,"
            " contradictions_json = ?, conversational_boundaries_json = ?,"
            " voice_identity_json = ?, revision = revision + 1, updated_at = ?,"
            " updated_by = ? WHERE agent_id = ? AND revision = ?",
            (
                json.dumps(core.core_values, ensure_ascii=False),
                json.dumps(core.temperament, ensure_ascii=False, sort_keys=True),
                json.dumps(core.preferences, ensure_ascii=False, sort_keys=True),
                json.dumps(core.quirks, ensure_ascii=False),
                json.dumps(core.contradictions, ensure_ascii=False),
                json.dumps(core.conversational_boundaries, ensure_ascii=False),
                json.dumps(core.voice_identity, ensure_ascii=False, sort_keys=True),
                moment.isoformat(),
                changed_by,
                core.agent_id,
                expected_revision,
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1


class PersonaCoreService:
    """Admin-facing read/update path. No model can call this class."""

    def __init__(
        self,
        repo: PersonaCoreRepository,
        *,
        audit=None,
        logger=None,
    ) -> None:
        self.repo = repo
        self.audit = audit
        self.logger = logger

    async def get(self, agent_id: str) -> PersonaCore:
        return await self.repo.get_or_default(agent_id)

    async def update(
        self,
        agent_id: str,
        *,
        core_values: Optional[list[str]] = None,
        temperament: Optional[dict[str, str]] = None,
        preferences: Optional[dict[str, str]] = None,
        quirks: Optional[list[str]] = None,
        contradictions: Optional[list[str]] = None,
        conversational_boundaries: Optional[list[str]] = None,
        voice_identity: Optional[dict[str, str]] = None,
        changed_by: str = "admin",
    ) -> PersonaCore:
        current = await self.repo.get_or_default(agent_id)
        candidate = PersonaCore(
            agent_id=agent_id,
            core_values=current.core_values
            if core_values is None
            else core_values,
            temperament=current.temperament
            if temperament is None
            else temperament,
            preferences=current.preferences
            if preferences is None
            else preferences,
            quirks=current.quirks if quirks is None else quirks,
            contradictions=current.contradictions
            if contradictions is None
            else contradictions,
            conversational_boundaries=current.conversational_boundaries
            if conversational_boundaries is None
            else conversational_boundaries,
            voice_identity=current.voice_identity
            if voice_identity is None
            else voice_identity,
            revision=current.revision,
            updated_at=current.updated_at,
            updated_by=changed_by,
        )
        written = await self.repo.update(
            candidate,
            expected_revision=current.revision,
            changed_by=changed_by,
        )
        if not written:
            raise RuntimeError(
                f"persona core for {agent_id!r} changed concurrently"
            )
        updated = await self.repo.get(agent_id) or candidate
        if self.audit is not None:
            await self.audit.record(
                actor=changed_by,
                action="persona_core_update",
                entity_type="agent",
                entity_id=agent_id,
                detail=f"revision={updated.revision}",
            )
        if self.logger is not None:
            self.logger.info(
                "persona_core_updated agent_id=%s revision=%d by=%s",
                agent_id,
                updated.revision,
                changed_by,
            )
        return updated
