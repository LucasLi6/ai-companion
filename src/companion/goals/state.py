"""Durable goals and checkpoints.

The memory file stores *intent*; it never executes anything. A goal's
``next_action`` is a plan, and a goal restored from a file must still have any
tool action re-authorised by the runtime before it runs. Dependencies are only
planning data: :meth:`GoalRepository.ready` reports goals whose dependencies
are done, it does not start them.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from companion.domain.models import utcnow
from companion.storage.database import Database

GOAL_STATUSES = ("OPEN", "IN_PROGRESS", "BLOCKED", "DONE", "CANCELLED")


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _iso(value: datetime) -> str:
    return value.isoformat()


class GoalRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def create(
        self,
        title: str,
        *,
        description: str | None = None,
        priority: int = 0,
        next_action: str | None = None,
        due_at: datetime | None = None,
    ) -> str:
        goal_id = _new("goal")
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO goals(goal_id, title, description, status, priority,"
            " next_action, due_at, created_at, updated_at)"
            " VALUES (?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)",
            (
                goal_id,
                title,
                description,
                priority,
                next_action,
                _iso(due_at) if due_at is not None else None,
                now,
                now,
            ),
        )
        return goal_id

    async def get(self, goal_id: str) -> dict | None:
        cur = await self.db.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,))
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list(self, status: str | None = None) -> list[dict]:
        if status is None:
            cur = await self.db.execute(
                "SELECT * FROM goals ORDER BY priority DESC, created_at"
            )
        else:
            cur = await self.db.execute(
                "SELECT * FROM goals WHERE status = ?"
                " ORDER BY priority DESC, created_at",
                (status,),
            )
        return [dict(r) for r in await cur.fetchall()]

    async def set_status(
        self, goal_id: str, status: str, *, next_action: str | None = None
    ) -> None:
        if status not in GOAL_STATUSES:
            raise ValueError(f"unknown goal status: {status}")
        await self.db.execute(
            "UPDATE goals SET status = ?,"
            " next_action = COALESCE(?, next_action), updated_at = ?"
            " WHERE goal_id = ?",
            (status, next_action, _iso(utcnow()), goal_id),
        )

    async def set_next_action(self, goal_id: str, next_action: str | None) -> None:
        await self.db.execute(
            "UPDATE goals SET next_action = ?, updated_at = ? WHERE goal_id = ?",
            (next_action, _iso(utcnow()), goal_id),
        )

    async def add_checkpoint(
        self, goal_id: str, status: str, *, summary: str | None = None
    ) -> str:
        if status not in GOAL_STATUSES:
            raise ValueError(f"unknown goal status: {status}")
        checkpoint_id = _new("gckpt")
        await self.db.execute(
            "INSERT INTO goal_checkpoints(checkpoint_id, goal_id, status, summary,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (checkpoint_id, goal_id, status, summary, _iso(utcnow())),
        )
        return checkpoint_id

    async def list_checkpoints(self, goal_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM goal_checkpoints WHERE goal_id = ? ORDER BY created_at",
            (goal_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def add_dependency(self, goal_id: str, depends_on_goal_id: str) -> None:
        if goal_id == depends_on_goal_id:
            raise ValueError("a goal cannot depend on itself")
        for gid in (goal_id, depends_on_goal_id):
            if await self.get(gid) is None:
                raise ValueError(f"goal not found: {gid}")
        if await self._depends_on(depends_on_goal_id, goal_id):
            raise ValueError("dependency would create a cycle")
        await self.db.execute(
            "INSERT OR IGNORE INTO goal_dependencies(goal_id, depends_on_goal_id,"
            " created_at) VALUES (?, ?, ?)",
            (goal_id, depends_on_goal_id, _iso(utcnow())),
        )

    async def _depends_on(self, goal_id: str, target: str) -> bool:
        """True when ``goal_id`` transitively depends on ``target``."""
        seen: set[str] = set()
        stack = [goal_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            cur = await self.db.execute(
                "SELECT depends_on_goal_id FROM goal_dependencies WHERE goal_id = ?",
                (current,),
            )
            for row in await cur.fetchall():
                dep = row["depends_on_goal_id"]
                if dep == target:
                    return True
                stack.append(dep)
        return False

    async def list_dependencies(self, goal_id: str) -> list[str]:
        cur = await self.db.execute(
            "SELECT depends_on_goal_id FROM goal_dependencies WHERE goal_id = ?",
            (goal_id,),
        )
        return [r["depends_on_goal_id"] for r in await cur.fetchall()]

    async def ready(self) -> list[dict]:
        """Open goals whose dependencies are all DONE (planning only)."""
        goals = await self.list()
        ready = []
        for goal in goals:
            if goal["status"] not in ("OPEN", "IN_PROGRESS"):
                continue
            deps = await self.list_dependencies(goal["goal_id"])
            if not deps:
                ready.append(goal)
                continue
            done = True
            for dep in deps:
                row = await self.get(dep)
                if row is None or row["status"] != "DONE":
                    done = False
                    break
            if done:
                ready.append(goal)
        return ready
