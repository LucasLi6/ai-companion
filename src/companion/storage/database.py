"""Asynchronous SQLite connection wrapper.

The connection is opened with foreign keys enforced and WAL journaling. The
connection runs in autocommit mode (``isolation_level=None``); multi-statement
atomicity is opt-in via :meth:`Database.transaction`.

Concurrency model
-----------------

All access to the shared connection is serialised through a re-entrant async
lock (:class:`_ReentrantAsyncLock`):

- :meth:`execute` acquires the lock for a single statement, so one statement
  can never be injected into another task's open transaction.
- :meth:`transaction` holds the lock for the whole ``BEGIN … COMMIT/ROLLBACK``
  span, so no other task can run a statement on the connection until the
  transaction finishes.

The lock is re-entrant because a repository method invoked *inside* a
transaction may itself call :meth:`execute` on the same task without
deadlocking. Cross-*connection* correctness is *not* provided by this lock —
that relies on database-level ``UNIQUE`` constraints and atomic ``UPDATE …
WHERE`` statements (see ``JobRepository.claim``).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from companion.cmf.constants import (
    APPLICATION_ID,
    CMF_FORMAT_MAJOR,
    format_major,
)

from .schema import MIGRATIONS, SCHEMA_VERSION


class _ReentrantAsyncLock:
    """A re-entrant ``asyncio.Lock`` for a single task.

    The same task may acquire it repeatedly; it is only released once the
    acquire count returns to zero.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._count = 0

    async def __aenter__(self) -> "_ReentrantAsyncLock":
        task = asyncio.current_task()
        if self._task is task:
            self._count += 1
            return self
        await self._lock.acquire()
        self._task = task
        self._count = 1
        return self

    async def __aexit__(self, *exc) -> bool:
        self._count -= 1
        if self._count == 0:
            self._task = None
            self._lock.release()
        return False


def _statements(sql: str) -> list[str]:
    """Split a migration script into individual SQL statements.

    Full-line ``--`` comments and blank lines are dropped, then the remainder
    is split on ``;``. This is deliberately simple: migrations must not contain
    ``PRAGMA`` statements, string literals with embedded semicolons, or
    ``BEGIN … END`` trigger bodies (none of the shipped migrations do).
    """
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    body = "\n".join(lines)
    return [part.strip() for part in body.split(";") if part.strip()]


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = _ReentrantAsyncLock()
        self._tx_depth = 0

    async def connect(self) -> "Database":
        self._conn = await aiosqlite.connect(self.path, isolation_level=None)
        # aiosqlite runs the real sqlite3 connection on a worker thread, so
        # attribute-style setters on it would hit SQLite's same-thread check.
        # Set row_factory through the worker thread instead; isolation_level
        # is supplied at connect time (autocommit) above.
        try:
            await self._conn._execute(setattr, self._conn._conn, "row_factory", aiosqlite.Row)
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.execute("PRAGMA journal_mode = WAL")
            await self._conn.execute("PRAGMA busy_timeout = 5000")
        except BaseException:
            # A corrupt/unreadable file can fail partway through the PRAGMAs.
            # Close the connection so its worker thread is never leaked; the
            # caller's ``open()`` has no Database to close yet.
            await self._conn.close()
            self._conn = None
            raise
        return self

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        """The raw connection.

        Only used inside :meth:`transaction` (where the lock is already held)
        or during single-threaded startup (:meth:`initialize`). Regular
        repository access must go through :meth:`execute`.
        """
        if self._conn is None:
            raise RuntimeError("Database is not connected; call connect() first")
        return self._conn

    async def initialize(self) -> None:
        """Bring the schema up to date, in version order.

        - A fresh database gets every migration applied in order.
        - A database already at ``SCHEMA_VERSION`` is left untouched.
        - A database at a *higher* version is refused without modification.
        - An arbitrary non-empty SQLite database is refused: a corrupt or
          unrelated file must never silently become a new, empty companion.
        - A CMF file declaring a newer *format* major version is refused.
        """
        await self._refuse_foreign_database()

        # The version table must exist before we can read the current version.
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        cur = await self.conn.execute("SELECT MAX(version) AS v FROM schema_version")
        row = await cur.fetchone()
        current = row["v"]

        if current is None:
            applied = 0
        elif current > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema version {current} is newer than the supported"
                f" version {SCHEMA_VERSION}; refusing to open it"
            )
        else:
            applied = current

        from companion.domain.models import utcnow

        for version, sql in MIGRATIONS:
            if version > applied:
                # Each migration is atomic: every one of its statements and its
                # schema-version stamp commit together or not at all. SQLite
                # DDL is transactional, so a mid-migration error rolls back the
                # partial DDL *and* leaves the version unchanged. This replaces
                # the old ``executescript`` approach, which committed outside a
                # transaction and could leave half-applied DDL without a
                # matching version record.
                async with self.transaction():
                    for statement in _statements(sql):
                        await self.conn.execute(statement)
                    await self.conn.execute(
                        "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                        (version, utcnow().isoformat()),
                    )

        await self._ensure_application_id()
        await self._refuse_newer_format()

    async def _refuse_foreign_database(self) -> None:
        """Refuse a non-empty SQLite file that is not a Companion database.

        A legacy Companion database has a ``schema_version`` table even before
        it gains the CMF manifest. An unrelated SQLite file has tables but no
        such table, and is left untouched.
        """
        cur = await self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name NOT LIKE 'sqlite_%'"
        )
        tables = {row["name"] for row in await cur.fetchall()}
        if tables and "schema_version" not in tables:
            raise RuntimeError(
                "refusing to open: this SQLite file is not a Companion memory"
                " database (no schema_version table); it has not been modified"
            )

        cur = await self.conn.execute("PRAGMA application_id")
        row = await cur.fetchone()
        app_id = int(row[0]) if row is not None else 0
        if app_id not in (0, APPLICATION_ID):
            raise RuntimeError(
                f"refusing to open: SQLite application_id {app_id} is not a"
                f" Companion/CMF file (expected {APPLICATION_ID})"
            )

    async def _ensure_application_id(self) -> None:
        """Stamp the CMF application id on a fresh or legacy database.

        SQLite's ``application_id`` is a header field, not a table row, so it
        is written here rather than in a migration. A pre-CMF Companion file
        (application_id 0) is stamped in place after its migrations commit.
        """
        cur = await self.conn.execute("PRAGMA application_id")
        row = await cur.fetchone()
        if row is not None and int(row[0]) == APPLICATION_ID:
            return
        await self.conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")

    async def _refuse_newer_format(self) -> None:
        """Refuse a file written by a newer major CMF format."""
        cur = await self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cmf_manifest'"
        )
        if await cur.fetchone() is None:
            return
        cur = await self.conn.execute(
            "SELECT format_version FROM cmf_manifest WHERE singleton = 1"
        )
        row = await cur.fetchone()
        if row is None:
            return
        major = format_major(row["format_version"])
        if major is not None and major > CMF_FORMAT_MAJOR:
            raise RuntimeError(
                f"memory file format {row['format_version']} is newer than the"
                f" supported CMF {CMF_FORMAT_MAJOR}.x; refusing to open it"
            )

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        async with self._lock:
            return await self.conn.execute(sql, params)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._lock:
            if self._tx_depth > 0:
                # Already inside a transaction on this connection (the lock is
                # re-entrant per task). A nested transaction is a no-op: the
                # *outermost* block owns BEGIN/COMMIT/ROLLBACK, so any inner
                # failure rolls the whole thing back atomically.
                self._tx_depth += 1
                try:
                    yield self.conn
                finally:
                    self._tx_depth -= 1
                return
            self._tx_depth = 1
            conn = self.conn
            await conn.execute("BEGIN")
            try:
                yield conn
                await conn.execute("COMMIT")
            except BaseException:
                await conn.execute("ROLLBACK")
                raise
            finally:
                self._tx_depth = 0
