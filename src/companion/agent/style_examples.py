"""Trusted, admin-authored dialogue style examples (V0.4).

A style example says "a response in this situation should meet this standard".
It is never a real conversation turn and never a fixed reply. Only the trusted
CLI/admin writes here: the cognitive/response providers have no write path to
this table.

Scoping rules:

- ``person_id IS NULL`` is a global example for the agent.
- a person-scoped example is returned only for that person;
- at most three are ever handed to the prompt, and disabled rows are skipped.
"""

from __future__ import annotations

from typing import Optional

from companion.domain.models import new_id, utcnow
from companion.storage.database import Database
from companion.storage.repositories import _iso


class StyleExampleError(ValueError):
    pass


class StyleExampleRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def add(
        self,
        *,
        agent_id: str,
        scenario: str,
        user_text: str,
        ideal_response: str,
        person_id: Optional[str] = None,
        created_by: str = "admin",
    ) -> dict:
        scenario = _clean(scenario, 80, "scenario")
        user_text = _clean(user_text, 4000, "user_text")
        ideal_response = _clean(ideal_response, 8000, "ideal_response")
        if not agent_id:
            raise StyleExampleError("agent_id is required")
        if person_id is not None:
            cur = await self.db.execute(
                "SELECT 1 FROM persons WHERE person_id = ?", (person_id,)
            )
            if await cur.fetchone() is None:
                raise StyleExampleError(f"unknown person: {person_id}")
        now = _iso(utcnow())
        example_id = new_id("style")
        await self.db.execute(
            "INSERT INTO dialogue_style_examples(example_id, agent_id, person_id,"
            " scenario, user_text, ideal_response, status, created_by, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)",
            (
                example_id,
                agent_id,
                person_id,
                scenario,
                user_text,
                ideal_response,
                created_by,
                now,
                now,
            ),
        )
        return await self.get(example_id)

    async def get(self, example_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM dialogue_style_examples WHERE example_id = ?",
            (example_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list(
        self,
        *,
        agent_id: Optional[str] = None,
        person_id: Optional[str] = None,
        include_disabled: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        where = []
        params: list = []
        if agent_id is not None:
            where.append("agent_id = ?")
            params.append(agent_id)
        if person_id is not None:
            where.append("(person_id IS NULL OR person_id = ?)")
            params.append(person_id)
        if not include_disabled:
            where.append("status = 'ACTIVE'")
        sql = "SELECT * FROM dialogue_style_examples"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        cur = await self.db.execute(sql, tuple(params))
        return [dict(r) for r in await cur.fetchall()]

    async def list_for_prompt(
        self,
        *,
        agent_id: str,
        person_id: Optional[str],
        scenario: Optional[str] = None,
        limit: int = 3,
    ) -> list[dict]:
        """Return at most ``limit`` active examples visible to this person.

        A person-scoped row is preferred over a global row when both exist for
        the same scenario, but a global row is never person-specific evidence.
        """
        if not agent_id:
            return []
        limit = max(0, min(int(limit), 3))
        if limit == 0:
            return []
        params: list = [agent_id]
        scope = "(person_id IS NULL"
        if person_id is not None:
            scope += " OR person_id = ?"
            params.append(person_id)
        scope += ")"
        sql = (
            "SELECT example_id, agent_id, person_id, scenario, user_text,"
            " ideal_response, status FROM dialogue_style_examples"
            f" WHERE agent_id = ? AND status = 'ACTIVE' AND {scope}"
        )
        if scenario:
            sql += " ORDER BY CASE WHEN scenario = ? THEN 0 ELSE 1 END,"
            sql += " CASE WHEN person_id IS NULL THEN 1 ELSE 0 END, created_at DESC"
            params.append(scenario)
        else:
            sql += " ORDER BY CASE WHEN person_id IS NULL THEN 1 ELSE 0 END, created_at DESC"
        sql += " LIMIT ?"
        params.append(limit)
        cur = await self.db.execute(sql, tuple(params))
        return [
            {
                "example_id": row["example_id"],
                "scenario": row["scenario"],
                "user_text": row["user_text"],
                "ideal_response": row["ideal_response"],
            }
            for row in await cur.fetchall()
        ]

    async def disable(self, example_id: str, *, actor: str = "admin") -> dict:
        now = _iso(utcnow())
        await self.db.execute(
            "UPDATE dialogue_style_examples SET status = 'DISABLED', updated_at = ?"
            " WHERE example_id = ?",
            (now, example_id),
        )
        row = await self.get(example_id)
        if row is None:
            raise StyleExampleError(f"style example not found: {example_id}")
        return row


def _clean(value: object, limit: int, name: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise StyleExampleError(f"{name} must not be empty")
    if len(text) > limit:
        raise StyleExampleError(f"{name} must be at most {limit} characters")
    return text
