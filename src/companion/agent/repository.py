"""SQLite persistence for the Agent Core and person relationships."""

from __future__ import annotations

import json
import uuid
from typing import Any

from companion.agent.constants import (
    DEFAULT_FAMILIARITY,
    DEFAULT_PERSONA,
    DEFAULT_PRESET_DELTAS,
    FAMILIARITY_TIERS,
    PERSONA_TRAITS,
)
from companion.domain.models import utcnow
from companion.storage.database import Database


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _json(value: str | None) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class AgentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def ensure_defaults(self) -> None:
        now = utcnow().isoformat()
        async with self.db.transaction():
            cur = await self.db.conn.execute(
                "SELECT agent_id FROM cmf_manifest WHERE singleton = 1"
            )
            row = await cur.fetchone()
            if row is None:
                agent_id = _new("agent")
                await self.db.conn.execute(
                    "INSERT INTO cmf_manifest(singleton, format_version, agent_id,"
                    " display_name, created_at, updated_at, lineage_id)"
                    " VALUES (1, '1.0', ?, 'Companion', ?, ?, ?)",
                    (agent_id, now, now, _new("lineage")),
                )
            else:
                agent_id = row["agent_id"]

            cur = await self.db.conn.execute(
                "SELECT 1 FROM agent_identity WHERE agent_id = ?", (agent_id,)
            )
            if await cur.fetchone() is None:
                await self.db.conn.execute(
                    "INSERT INTO agent_identity(agent_id, display_name, pronouns,"
                    " description, created_at, updated_at, revision)"
                    " VALUES (?, 'Companion', NULL,"
                    " 'A persistent AI companion identity stored in this memory file.',"
                    " ?, ?, 1)",
                    (agent_id, now, now),
                )

            cur = await self.db.conn.execute(
                "SELECT 1 FROM agent_persona WHERE agent_id = ?", (agent_id,)
            )
            if await cur.fetchone() is None:
                columns = ", ".join(PERSONA_TRAITS)
                placeholders = ", ".join("?" for _ in PERSONA_TRAITS)
                await self.db.conn.execute(
                    f"INSERT INTO agent_persona(agent_id, {columns}, updated_at)"
                    f" VALUES (?, {placeholders}, ?)",
                    (agent_id, *[DEFAULT_PERSONA[t] for t in PERSONA_TRAITS], now),
                )
                await self.db.conn.execute(
                    "INSERT OR IGNORE INTO persona_revisions(revision_id, agent_id,"
                    " revision, changes, changed_by, changed_at, reason)"
                    " VALUES (?, ?, 1, '{}', 'system', ?, 'initial_defaults')",
                    (_new("rev"), agent_id, now),
                )

            await self.db.conn.execute(
                "INSERT OR IGNORE INTO embedding_index_metadata(singleton, status, extra)"
                " VALUES (1, 'EMPTY', '{}')"
            )

    # -- identity / persona ------------------------------------------------

    async def get_manifest(self) -> dict[str, Any] | None:
        cur = await self.db.execute(
            "SELECT * FROM cmf_manifest WHERE singleton = 1"
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def get_identity(self) -> dict[str, Any] | None:
        cur = await self.db.execute("SELECT * FROM agent_identity LIMIT 1")
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def get_persona(self) -> dict[str, Any] | None:
        cur = await self.db.execute("SELECT * FROM agent_persona LIMIT 1")
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def set_identity(
        self,
        *,
        display_name: str | None = None,
        pronouns: str | None = None,
        description: str | None = None,
        changed_by: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        now = utcnow().isoformat()
        async with self.db.transaction():
            cur = await self.db.conn.execute("SELECT * FROM agent_identity LIMIT 1")
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("agent identity is missing")
            new_name = display_name if display_name is not None else row["display_name"]
            new_pronouns = pronouns if pronouns is not None else row["pronouns"]
            new_description = (
                description if description is not None else row["description"]
            )
            await self.db.conn.execute(
                "UPDATE agent_identity SET display_name = ?, pronouns = ?,"
                " description = ?, updated_at = ?, revision = revision + 1"
                " WHERE agent_id = ?",
                (
                    new_name,
                    new_pronouns,
                    new_description,
                    now,
                    row["agent_id"],
                ),
            )
            await self.db.conn.execute(
                "UPDATE cmf_manifest SET display_name = ?, updated_at = ?"
                " WHERE singleton = 1",
                (new_name, now),
            )
        return {
            "display_name": new_name,
            "pronouns": new_pronouns,
            "description": new_description,
            "revision": int(row["revision"]) + 1,
            "changed_by": changed_by,
            "reason": reason,
        }

    async def update_persona(
        self,
        updates: dict[str, int],
        *,
        changed_by: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        now = utcnow().isoformat()
        async with self.db.transaction():
            cur = await self.db.conn.execute("SELECT * FROM agent_persona LIMIT 1")
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("agent persona is missing")
            columns = ", ".join(f"{name} = ?" for name in updates)
            await self.db.conn.execute(
                f"UPDATE agent_persona SET {columns}, updated_at = ?,"
                f" revision = revision + 1 WHERE agent_id = ?",
                (*updates.values(), now, row["agent_id"]),
            )
            revision = int(row["revision"]) + 1
            await self.db.conn.execute(
                "INSERT INTO persona_revisions(revision_id, agent_id, revision,"
                " changes, changed_by, changed_at, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _new("rev"),
                    row["agent_id"],
                    revision,
                    json.dumps(updates, ensure_ascii=False, sort_keys=True),
                    changed_by,
                    now,
                    reason,
                ),
            )
        result = {name: row[name] for name in PERSONA_TRAITS}
        result.update(updates)
        result["revision"] = revision
        return result

    async def list_persona_revisions(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = await self.db.execute(
            "SELECT * FROM persona_revisions ORDER BY revision DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in await cur.fetchall()]
        for row in rows:
            row["changes"] = _json(row.get("changes"))
        return rows

    # -- relationships -----------------------------------------------------

    async def get_presets(self) -> dict[str, dict[str, int]]:
        cur = await self.db.execute("SELECT * FROM relationship_presets")
        presets: dict[str, dict[str, int]] = {}
        for row in await cur.fetchall():
            deltas = _json(row["style_deltas"])
            presets[row["tier"]] = {
                key: int(value)
                for key, value in deltas.items()
                if key in PERSONA_TRAITS and isinstance(value, (int, float))
            }
        for tier in FAMILIARITY_TIERS:
            presets.setdefault(tier, dict(DEFAULT_PRESET_DELTAS[tier]))
        return presets

    async def get_profile(self, person_id: str) -> dict[str, Any]:
        cur = await self.db.execute(
            "SELECT p.person_id, p.familiarity, p.notes,"
            " i.relationship_tier, i.style_overrides"
            " FROM person_profiles p"
            " LEFT JOIN person_interaction_profiles i ON i.person_id = p.person_id"
            " WHERE p.person_id = ?",
            (person_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return {
                "person_id": person_id,
                "familiarity": DEFAULT_FAMILIARITY,
                "notes": None,
                "relationship_tier": DEFAULT_FAMILIARITY,
                "style_overrides": {},
            }
        result = dict(row)
        result["style_overrides"] = {
            key: int(value)
            for key, value in _json(result.get("style_overrides")).items()
            if key in PERSONA_TRAITS and isinstance(value, (int, float))
        }
        return result

    async def set_familiarity(
        self,
        person_id: str,
        tier: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> tuple[str, str]:
        now = utcnow().isoformat()
        async with self.db.transaction():
            cur = await self.db.conn.execute(
                "SELECT p.familiarity FROM person_profiles p WHERE p.person_id = ?",
                (person_id,),
            )
            row = await cur.fetchone()
            old_tier = row["familiarity"] if row is not None else None
            await self.db.conn.execute(
                "INSERT INTO person_profiles(person_id, familiarity, created_at,"
                " updated_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(person_id) DO UPDATE SET familiarity = excluded.familiarity,"
                " updated_at = excluded.updated_at",
                (person_id, tier, now, now),
            )
            await self.db.conn.execute(
                "INSERT INTO person_interaction_profiles(person_id,"
                " relationship_tier, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(person_id) DO UPDATE SET"
                " relationship_tier = excluded.relationship_tier,"
                " updated_at = excluded.updated_at",
                (person_id, tier, now),
            )
            await self.db.conn.execute(
                "INSERT INTO relationship_history(history_id, person_id, old_tier,"
                " new_tier, actor, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_new("rel"), person_id, old_tier, tier, actor, reason, now),
            )
        return old_tier or DEFAULT_FAMILIARITY, tier

    async def set_overrides(
        self,
        person_id: str,
        overrides: dict[str, int],
        *,
        actor: str,
        reason: str | None = None,
    ) -> dict[str, int]:
        now = utcnow().isoformat()
        async with self.db.transaction():
            cur = await self.db.conn.execute(
                "SELECT style_overrides FROM person_interaction_profiles"
                " WHERE person_id = ?",
                (person_id,),
            )
            row = await cur.fetchone()
            current = _json(row["style_overrides"]) if row is not None else {}
            cur = await self.db.conn.execute(
                "SELECT familiarity FROM person_profiles WHERE person_id = ?",
                (person_id,),
            )
            profile = await cur.fetchone()
            tier = (
                profile["familiarity"] if profile is not None else DEFAULT_FAMILIARITY
            )
            for name, value in overrides.items():
                if value is None:
                    current.pop(name, None)
                else:
                    current[name] = value
            await self.db.conn.execute(
                "INSERT INTO person_profiles(person_id, familiarity, created_at,"
                " updated_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(person_id) DO NOTHING",
                (person_id, DEFAULT_FAMILIARITY, now, now),
            )
            await self.db.conn.execute(
                "INSERT INTO person_interaction_profiles(person_id,"
                " relationship_tier, style_overrides, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(person_id) DO UPDATE SET"
                " style_overrides = excluded.style_overrides,"
                " updated_at = excluded.updated_at",
                (
                    person_id,
                    tier,
                    json.dumps(current, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
        return current

    async def list_relationship_history(self, person_id: str) -> list[dict[str, Any]]:
        cur = await self.db.execute(
            "SELECT * FROM relationship_history WHERE person_id = ?"
            " ORDER BY created_at DESC",
            (person_id,),
        )
        return [dict(r) for r in await cur.fetchall()]
