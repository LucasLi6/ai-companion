"""CMF memory-file operations: create, open, inspect, validate, export, clone.

All file-level work goes through this service instead of ad-hoc SQLite file
copies. The rules it enforces:

- **Atomic replacement.** New/exported/cloned files are written to a temporary
  file in the same directory, validated, then atomically replaced. A failure
  never leaves a half-written memory file at the destination.
- **No accidental overwrite.** ``overwrite=True`` is required explicitly; the
  source of an export can never be its own destination.
- **WAL-safe copies.** Exports and clones use SQLite's backup API, which
  produces a consistent snapshot even while the source is in WAL mode. Copying
  the ``.db`` file alone is not sufficient.
- **Read-only inspection.** ``inspect`` and ``validate`` never migrate or write
  to the file they are checking.
- **Clone identity.** A clone receives a new ``agent_id`` and records its
  lineage (``lineage_id`` shared, ``lineage_parent_agent_id``/``clone_of`` set
  to the source id). Memories and people are preserved.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from companion.cmf.constants import (
    APPLICATION_ID,
    CMF_FORMAT_MAJOR,
    CMF_FORMAT_VERSION,
    REQUIRED_TABLES,
    format_major,
)
from companion.domain.models import utcnow
from companion.storage.database import Database
from companion.storage.schema import SCHEMA_VERSION


class CmfError(Exception):
    """Base class for memory-file errors."""


class CmfFileExistsError(CmfError):
    pass


class CmfValidationError(CmfError):
    def __init__(self, message: str, result: "CmfValidation | None" = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class CmfValidation:
    path: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class CmfInfo:
    path: str
    ok: bool
    agent_id: str | None
    display_name: str | None
    format_version: str | None
    schema_version: int | None
    lineage_id: str | None
    lineage_parent_agent_id: str | None
    counts: dict[str, int]
    errors: list[str] = field(default_factory=list)


def _ro_conn(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    wanted = (
        "persons",
        "utterances",
        "memory_items",
        "memory_assignments",
        "memory_candidate_writes",
        "agent_identity",
        "agent_persona",
        "person_profiles",
        "relationship_presets",
        "research_tasks",
        "research_projects",
        "web_sources",
        "knowledge_claims",
        "curiosity_queue",
        "reflection_records",
        "learning_runs",
        "goals",
        "media_assets",
        "memory_embeddings",
    )
    existing = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    out: dict[str, int] = {}
    for table in wanted:
        if table in existing:
            out[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return out


def _validate_sync(path: Path, *, deep: bool = True) -> CmfValidation:
    result = CmfValidation(path=str(path), ok=False)
    if not path.exists():
        result.errors.append("file does not exist")
        return result
    if not path.is_file():
        result.errors.append("path is not a regular file")
        return result

    try:
        conn = _ro_conn(path)
    except sqlite3.Error as exc:
        result.errors.append(f"not a readable SQLite database ({exc})")
        return result

    try:
        app_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
        result.info["application_id"] = app_id
        if app_id != APPLICATION_ID:
            result.errors.append(
                f"application_id is {app_id}, expected {APPLICATION_ID}"
            )

        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            result.errors.append("missing tables: " + ", ".join(missing))

        if deep and "schema_version" in tables:
            check = conn.execute("PRAGMA integrity_check").fetchall()
            integrity = "; ".join(str(r[0]) for r in check)
            result.info["integrity_check"] = integrity
            if integrity.lower() != "ok":
                result.errors.append(f"integrity_check failed: {integrity}")
            fk = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk:
                result.errors.append(f"foreign_key_check reported {len(fk)} violation(s)")
        elif deep:
            result.errors.append("no schema_version table")

        if "schema_version" in tables:
            row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            schema_version = row["v"] if row is not None else None
            result.info["schema_version"] = schema_version
            if schema_version is None:
                result.errors.append("schema_version is empty")
            elif int(schema_version) > SCHEMA_VERSION:
                result.errors.append(
                    f"schema version {schema_version} is newer than supported"
                    f" {SCHEMA_VERSION}"
                )

        if "cmf_manifest" in tables:
            rows = conn.execute("SELECT * FROM cmf_manifest").fetchall()
            if len(rows) != 1:
                result.errors.append(
                    f"manifest must have exactly one row, found {len(rows)}"
                )
            else:
                manifest = rows[0]
                result.info["agent_id"] = manifest["agent_id"]
                result.info["display_name"] = manifest["display_name"]
                result.info["format_version"] = manifest["format_version"]
                result.info["lineage_id"] = manifest["lineage_id"]
                result.info["lineage_parent_agent_id"] = manifest[
                    "lineage_parent_agent_id"
                ]
                if not manifest["agent_id"]:
                    result.errors.append("manifest agent_id is empty")
                major = format_major(manifest["format_version"])
                if major is None:
                    result.errors.append("manifest format_version is invalid")
                elif major > CMF_FORMAT_MAJOR:
                    result.errors.append(
                        f"format {manifest['format_version']} is newer than"
                        f" supported {CMF_FORMAT_VERSION}"
                    )
                elif major < CMF_FORMAT_MAJOR:
                    result.warnings.append(
                        f"older format {manifest['format_version']} will be"
                        " migrated on open"
                    )

        if "agent_identity" in tables and "cmf_manifest" in tables:
            row = conn.execute(
                "SELECT i.agent_id AS identity_id, m.agent_id AS manifest_id"
                " FROM agent_identity i CROSS JOIN cmf_manifest m"
            ).fetchone()
            if row is None:
                result.errors.append("agent_identity row is missing")
            elif row["identity_id"] != row["manifest_id"]:
                result.errors.append("agent_identity does not match manifest agent_id")

        if "agent_persona" in tables:
            for row in conn.execute("SELECT * FROM agent_persona").fetchall():
                for key in (
                    "humor",
                    "warmth",
                    "curiosity",
                    "formality",
                    "verbosity",
                    "assertiveness",
                    "empathy",
                    "playfulness",
                    "patience",
                ):
                    value = row[key]
                    if not isinstance(value, int) or not 0 <= value <= 100:
                        result.errors.append(
                            f"persona trait {key}={value!r} is outside 0..100"
                        )

        result.info["counts"] = _counts(conn)
    except sqlite3.Error as exc:
        result.errors.append(f"SQLite error: {exc}")
    finally:
        conn.close()

    result.ok = not result.errors
    return result


def _backup_sync(source: Path, destination: Path) -> None:
    """Consistent snapshot via the SQLite backup API (WAL-safe)."""
    src = _ro_conn(source)
    try:
        dst = sqlite3.connect(str(destination))
        try:
            src.backup(dst)
            dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()


def _replace_verified(temp: Path, destination: Path, *, overwrite: bool) -> None:
    validation = _validate_sync(temp)
    if not validation.ok:
        _cleanup(temp)
        raise CmfValidationError(
            "generated file failed validation: " + "; ".join(validation.errors),
            validation,
        )
    if destination.exists():
        if not overwrite:
            _cleanup(temp)
            raise CmfFileExistsError(f"destination already exists: {destination}")
        destination.unlink()
    os.replace(temp, destination)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(temp) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def _cleanup(path: Path) -> None:
    for candidate in (
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
    ):
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            pass


def _temp_path(destination: Path) -> Path:
    token = uuid.uuid4().hex[:10]
    return destination.with_name(f".{destination.name}.{token}.tmp")


class CmfFileService:
    """High-level memory-file service. All methods are safe to call async."""

    @staticmethod
    async def create(
        path: str | Path,
        *,
        display_name: str = "Companion",
        overwrite: bool = False,
    ) -> CmfInfo:
        import asyncio

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise CmfFileExistsError(f"destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = _temp_path(destination)
        try:
            db = await Database(temp).connect()
            try:
                await db.initialize()
                now = utcnow().isoformat()
                await db.execute(
                    "UPDATE cmf_manifest SET display_name = ?, updated_at = ?"
                    " WHERE singleton = 1",
                    (display_name.strip() or "Companion", now),
                )
                await db.execute(
                    "UPDATE agent_identity SET display_name = ?, updated_at = ?",
                    (display_name.strip() or "Companion", now),
                )
                await db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                await db.close()
            await asyncio.to_thread(_replace_verified, temp, destination, overwrite=overwrite)
        except BaseException:
            _cleanup(temp)
            raise
        return await CmfFileService.inspect(destination)

    @staticmethod
    async def open(path: str | Path) -> Database:
        """Open and migrate a memory file. Validation happens on initialize."""
        db = await Database(path).connect()
        try:
            await db.initialize()
        except BaseException:
            await db.close()
            raise
        return db

    @staticmethod
    async def inspect(path: str | Path) -> CmfInfo:
        import asyncio

        destination = Path(path)
        validation = await asyncio.to_thread(_validate_sync, destination, deep=False)
        info = validation.info
        return CmfInfo(
            path=str(destination),
            ok=validation.ok,
            agent_id=info.get("agent_id"),
            display_name=info.get("display_name"),
            format_version=info.get("format_version"),
            schema_version=info.get("schema_version"),
            lineage_id=info.get("lineage_id"),
            lineage_parent_agent_id=info.get("lineage_parent_agent_id"),
            counts=info.get("counts", {}),
            errors=list(validation.errors),
        )

    @staticmethod
    async def validate(path: str | Path) -> CmfValidation:
        import asyncio

        return await asyncio.to_thread(_validate_sync, Path(path), deep=True)

    @staticmethod
    async def export(
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> CmfValidation:
        import asyncio

        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            raise CmfError(f"source does not exist: {src}")
        if src.resolve() == dst.resolve():
            raise CmfError("source and destination are the same file")
        if dst.exists() and not overwrite:
            raise CmfFileExistsError(f"destination already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = _temp_path(dst)
        try:
            await asyncio.to_thread(_backup_sync, src, temp)
            await asyncio.to_thread(_replace_verified, temp, dst, overwrite=overwrite)
        except BaseException:
            _cleanup(temp)
            raise
        return await CmfFileService.validate(dst)

    @staticmethod
    async def clone(
        source: str | Path,
        destination: str | Path,
        *,
        display_name: str | None = None,
        overwrite: bool = False,
    ) -> CmfInfo:
        import asyncio

        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            raise CmfError(f"source does not exist: {src}")
        if src.resolve() == dst.resolve():
            raise CmfError("source and destination are the same file")
        if dst.exists() and not overwrite:
            raise CmfFileExistsError(f"destination already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = _temp_path(dst)
        try:
            await asyncio.to_thread(_backup_sync, src, temp)
            await asyncio.to_thread(
                _reidentify_clone, temp, display_name=display_name
            )
            await asyncio.to_thread(_replace_verified, temp, dst, overwrite=overwrite)
        except BaseException:
            _cleanup(temp)
            raise
        return await CmfFileService.inspect(dst)

    @staticmethod
    async def migrate(path: str | Path) -> dict[str, Any]:
        """Open a legacy ``.db`` in place and upgrade it to CMF 1.0."""
        destination = Path(path)
        if not destination.exists():
            raise CmfError(f"file does not exist: {destination}")

        def _before() -> int | None:
            try:
                conn = _ro_conn(destination)
            except sqlite3.Error:
                return None
            try:
                row = conn.execute(
                    "SELECT MAX(version) AS v FROM schema_version"
                ).fetchone()
                return row["v"] if row is not None else None
            except sqlite3.Error:
                return None
            finally:
                conn.close()

        from_version = await asyncio.to_thread(_before)
        db = await Database(destination).connect()
        try:
            await db.initialize()
            cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
            to_version = (await cur.fetchone())["v"]
            cur = await db.execute(
                "SELECT agent_id, format_version FROM cmf_manifest WHERE singleton = 1"
            )
            manifest = await cur.fetchone()
        finally:
            await db.close()
        return {
            "path": str(destination),
            "from_schema_version": from_version,
            "to_schema_version": to_version,
            "agent_id": manifest["agent_id"] if manifest else None,
            "format_version": manifest["format_version"] if manifest else None,
            "migrated": from_version != to_version,
        }


def _reidentify_clone(path: Path, *, display_name: str | None) -> None:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        manifest = conn.execute(
            "SELECT * FROM cmf_manifest WHERE singleton = 1"
        ).fetchone()
        if manifest is None:
            raise CmfError("source file has no manifest")
        old_agent = manifest["agent_id"]
        new_agent = "agent_" + uuid.uuid4().hex[:16]
        new_name = (display_name or f"{manifest['display_name'] or 'Companion'} (clone)").strip()
        now = utcnow().isoformat()
        conn.execute("BEGIN")
        conn.execute("PRAGMA defer_foreign_keys = ON")
        conn.execute(
            "UPDATE cmf_manifest SET agent_id = ?, display_name = ?,"
            " lineage_parent_agent_id = ?, clone_of = ?, updated_at = ?"
            " WHERE singleton = 1",
            (new_agent, new_name, old_agent, old_agent, now),
        )
        conn.execute(
            "UPDATE agent_identity SET agent_id = ?, display_name = ?,"
            " updated_at = ?, revision = revision + 1 WHERE agent_id = ?",
            (new_agent, new_name, now, old_agent),
        )
        conn.execute(
            "UPDATE agent_persona SET agent_id = ?, updated_at = ? WHERE agent_id = ?",
            (new_agent, now, old_agent),
        )
        conn.execute(
            "INSERT INTO persona_revisions(revision_id, agent_id, revision,"
            " changes, changed_by, changed_at, reason)"
            " SELECT ?, ?, COALESCE(MAX(revision), 0) + 1, '{}', 'system', ?,"
            " 'cloned_from_parent' FROM persona_revisions WHERE agent_id = ?",
            ("rev_clone_" + uuid.uuid4().hex[:16], new_agent, now, old_agent),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
