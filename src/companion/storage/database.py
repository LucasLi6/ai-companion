"""Asynchronous SQLite connection wrapper.

The connection is opened with foreign keys enforced and WAL journaling. The
connection runs in autocommit mode (``isolation_level=None``); multi-statement
atomicity is opt-in via :meth:`Database.transaction`, which is used only for
short, local writes. Transactions are never held across an external provider
call.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from .schema import SCHEMA_SQL, SCHEMA_VERSION


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> "Database":
        self._conn = await aiosqlite.connect(self.path, isolation_level=None)
        # aiosqlite runs the real sqlite3 connection on a worker thread, so
        # attribute-style setters on it would hit SQLite's same-thread check.
        # Set row_factory through the worker thread instead; isolation_level
        # is supplied at connect time (autocommit) above.
        await self._conn._execute(setattr, self._conn._conn, "row_factory", aiosqlite.Row)
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.execute("PRAGMA journal_mode = WAL")
        return self

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected; call connect() first")
        return self._conn

    async def initialize(self) -> None:
        """Create the schema idempotently and record the schema version."""
        await self.conn.executescript(SCHEMA_SQL)
        cur = await self.conn.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        )
        row = await cur.fetchone()
        if row["v"] is None:
            from companion.domain.models import utcnow

            await self.conn.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, utcnow().isoformat()),
            )

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        return await self.conn.execute(sql, params)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = self.conn
        await conn.execute("BEGIN")
        try:
            yield conn
            await conn.execute("COMMIT")
        except BaseException:
            await conn.execute("ROLLBACK")
            raise
