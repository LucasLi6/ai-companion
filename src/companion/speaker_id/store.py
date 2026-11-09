"""Local SQLite store for anonymous speaker profiles and voice prints.

This is a separate database from Ava's: it never reads or writes the
application's ``persons`` or ``memories`` tables. The only writer is the
engine's recognition worker (the demo also opens it for explicit list/delete
commands); one lock and one connection serialise every operation so a
multi-threaded write race cannot corrupt the numbering.

Privacy rules enforced here:

* raw audio is never stored — only normalized embeddings and diagnostics;
* a temporary ID is allocated by the database, atomically, never by a Python
  counter;
* one utterance can produce at most one observation row;
* embeddings from an incompatible model version are never compared;
* every identity can be deleted, and the whole store can be cleared.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from companion.speaker_id.matcher import SpeakerCandidate


class SpeakerStoreError(Exception):
    """Storage failure (message is display-safe)."""


class IncompatibleModelError(SpeakerStoreError):
    """An embedding cannot be mixed with this profile's model version."""


QUALITY_RANK_SQL = (
    "CASE quality WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 2 ELSE 1 END"
)


class SpeakerStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_embeddings_per_speaker: int = 8,
        clock=time.time,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_embeddings_per_speaker = max(1, int(max_embeddings_per_speaker))
        self._clock = clock
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------- lifecycle
    def initialize(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(
                    str(self.path),
                    check_same_thread=False,
                    isolation_level=None,
                )
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.executescript(_SCHEMA)
            except (OSError, sqlite3.Error) as exc:
                raise SpeakerStoreError(
                    "无法打开声纹数据库（" + type(exc).__name__ + "）。"
                ) from exc
            self._connection = connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except sqlite3.Error:
                    pass
                self._connection = None

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise SpeakerStoreError("声纹数据库尚未初始化。")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._conn
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    # ---------------------------------------------------------------- writes
    def create_candidate(self) -> str:
        """Atomically allocate the next ``TEMP-0001`` style identity."""
        with self._lock, self._transaction() as connection:
            rows = connection.execute(
                "SELECT speaker_id FROM speaker_profiles"
                " WHERE speaker_id LIKE 'TEMP-%'"
            ).fetchall()
            highest = 0
            for row in rows:
                suffix = str(row["speaker_id"])[len("TEMP-"):]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
            speaker_id = f"TEMP-{highest + 1:04d}"
            now = self._clock()
            connection.execute(
                "INSERT INTO speaker_profiles"
                " (speaker_id, status, created_at, updated_at,"
                "  observation_count, model_id, embedding_version)"
                " VALUES (?, 'CANDIDATE', ?, ?, 0, ?, ?)",
                (speaker_id, now, now, "", ""),
            )
            return speaker_id

    def add_embedding(
        self,
        speaker_id: str,
        vector,
        *,
        quality: str,
        voiced_seconds: float,
        model_id: str,
        embedding_version: str,
    ) -> Optional[int]:
        """Store one normalized voice print, keeping the sample count bounded."""
        from companion.speaker_id.embedding import validate_embedding

        array = validate_embedding(vector)
        with self._lock, self._transaction() as connection:
            profile = connection.execute(
                "SELECT model_id, embedding_version FROM speaker_profiles"
                " WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
            if profile is None:
                raise SpeakerStoreError("声纹身份不存在。")
            stored_model = str(profile["model_id"] or "")
            if stored_model and (
                stored_model != model_id
                or str(profile["embedding_version"] or "") != embedding_version
            ):
                raise IncompatibleModelError(
                    "声纹模型版本不一致，拒绝混合比较。"
                )
            now = self._clock()
            if not stored_model:
                connection.execute(
                    "UPDATE speaker_profiles SET model_id = ?,"
                    " embedding_version = ?, updated_at = ?"
                    " WHERE speaker_id = ?",
                    (model_id, embedding_version, now, speaker_id),
                )
            cursor = connection.execute(
                "INSERT INTO speaker_embeddings"
                " (speaker_id, vector, quality, voiced_seconds, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    speaker_id,
                    array.astype("<f4").tobytes(),
                    str(quality),
                    float(voiced_seconds),
                    now,
                ),
            )
            connection.execute(
                "UPDATE speaker_profiles SET updated_at = ? WHERE speaker_id = ?",
                (now, speaker_id),
            )
            self._trim_embeddings(connection, speaker_id)
            return int(cursor.lastrowid)

    def _trim_embeddings(
        self, connection: sqlite3.Connection, speaker_id: str
    ) -> None:
        rows = connection.execute(
            "SELECT embedding_id FROM speaker_embeddings"
            f" WHERE speaker_id = ? ORDER BY {QUALITY_RANK_SQL} ASC,"
            " created_at ASC",
            (speaker_id,),
        ).fetchall()
        overflow = len(rows) - self.max_embeddings_per_speaker
        if overflow <= 0:
            return
        for row in rows[:overflow]:
            connection.execute(
                "DELETE FROM speaker_embeddings WHERE embedding_id = ?",
                (int(row["embedding_id"]),),
            )

    def record_observation(
        self,
        utterance_id: int,
        *,
        speaker_id: Optional[str],
        match_status: str,
        similarity: Optional[float],
        margin: Optional[float],
    ) -> None:
        """One row per utterance; a later good verdict may refine it.

        A later low-information verdict (``INSUFFICIENT_AUDIO``/``PENDING``/
        ``ERROR``) never overwrites an already recorded match, and re-processing
        the same result is idempotent.
        """
        with self._lock, self._transaction() as connection:
            previous = self._observation_speaker(
                connection, int(utterance_id)
            )
            connection.execute(
                "INSERT INTO speaker_observations"
                " (utterance_id, speaker_id, match_status, similarity,"
                "  margin, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(utterance_id) DO UPDATE SET"
                "  speaker_id = excluded.speaker_id,"
                "  match_status = excluded.match_status,"
                "  similarity = excluded.similarity,"
                "  margin = excluded.margin"
                " WHERE excluded.match_status NOT IN"
                "       ('INSUFFICIENT_AUDIO', 'PENDING', 'ERROR')",
                (
                    int(utterance_id),
                    speaker_id,
                    str(match_status),
                    None if similarity is None else float(similarity),
                    None if margin is None else float(margin),
                    self._clock(),
                ),
            )
            for profile_id in (speaker_id, previous):
                if profile_id:
                    connection.execute(
                        "UPDATE speaker_profiles SET observation_count ="
                        " (SELECT COUNT(*) FROM speaker_observations"
                        "  WHERE speaker_id = ?) WHERE speaker_id = ?",
                        (profile_id, profile_id),
                    )

    @staticmethod
    def _observation_speaker(
        connection: sqlite3.Connection, utterance_id: int
    ) -> Optional[str]:
        row = connection.execute(
            "SELECT speaker_id FROM speaker_observations WHERE utterance_id = ?",
            (utterance_id,),
        ).fetchone()
        return None if row is None else row["speaker_id"]

    # ----------------------------------------------------------------- reads
    def load_candidates(
        self, *, model_id: str, embedding_version: str
    ) -> tuple[list[SpeakerCandidate], int]:
        """Profiles compatible with the current model, plus an incompatible count."""
        with self._lock:
            profiles = self._conn.execute(
                "SELECT speaker_id, status, model_id, embedding_version"
                " FROM speaker_profiles ORDER BY created_at ASC, speaker_id ASC"
            ).fetchall()
            candidates: list[SpeakerCandidate] = []
            incompatible = 0
            for profile in profiles:
                stored_model = str(profile["model_id"] or "")
                stored_version = str(profile["embedding_version"] or "")
                if not stored_model:
                    # No sample was ever enrolled; nothing to compare.
                    continue
                if (
                    stored_model != model_id
                    or stored_version != embedding_version
                ):
                    incompatible += 1
                    continue
                vectors = self._load_vectors(str(profile["speaker_id"]))
                if not vectors:
                    continue
                candidates.append(
                    SpeakerCandidate(
                        speaker_id=str(profile["speaker_id"]),
                        status=str(profile["status"]),
                        vectors=tuple(vectors),
                    )
                )
            return candidates, incompatible

    def _load_vectors(self, speaker_id: str) -> list:
        import numpy as np

        rows = self._conn.execute(
            "SELECT vector FROM speaker_embeddings WHERE speaker_id = ?"
            " ORDER BY embedding_id ASC",
            (speaker_id,),
        ).fetchall()
        vectors = []
        for row in rows:
            try:
                vector = np.frombuffer(row["vector"], dtype="<f4").astype(
                    np.float32
                )
            except (TypeError, ValueError):
                continue
            if vector.ndim != 1 or vector.size == 0:
                continue
            if not np.all(np.isfinite(vector)):
                continue
            norm = float(np.linalg.norm(vector))
            if norm < 1e-6:
                continue
            vectors.append((vector / norm).astype(np.float32))
        return vectors

    def list_speakers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.speaker_id, p.status, p.created_at, p.updated_at,"
                " p.observation_count, p.model_id, p.embedding_version,"
                " (SELECT COUNT(*) FROM speaker_embeddings e"
                "  WHERE e.speaker_id = p.speaker_id) AS samples"
                " FROM speaker_profiles p ORDER BY p.created_at ASC,"
                " p.speaker_id ASC"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_profile(self, speaker_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM speaker_profiles WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
            return None if row is None else dict(row)

    def aliases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT alias_id, canonical_speaker_id, alias_speaker_id,"
                " created_at FROM speaker_aliases ORDER BY alias_id ASC"
            ).fetchall()
            return [dict(row) for row in rows]

    # --------------------------------------------------------------- deletion
    def delete_speaker(self, speaker_id: str) -> bool:
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                "SELECT 1 FROM speaker_profiles WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "DELETE FROM speaker_embeddings WHERE speaker_id = ?",
                (speaker_id,),
            )
            connection.execute(
                "DELETE FROM speaker_observations WHERE speaker_id = ?",
                (speaker_id,),
            )
            connection.execute(
                "DELETE FROM speaker_aliases WHERE canonical_speaker_id = ?"
                " OR alias_speaker_id = ?",
                (speaker_id, speaker_id),
            )
            connection.execute(
                "DELETE FROM speaker_profiles WHERE speaker_id = ?",
                (speaker_id,),
            )
            return True

    def clear(self) -> int:
        """Delete every profile, print, observation and alias."""
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM speaker_profiles"
            ).fetchone()
            connection.execute("DELETE FROM speaker_embeddings")
            connection.execute("DELETE FROM speaker_observations")
            connection.execute("DELETE FROM speaker_aliases")
            connection.execute("DELETE FROM speaker_profiles")
            return int(row["n"]) if row is not None else 0

    # ------------------------------------------------------------------ merge
    def merge_speakers(
        self, canonical_speaker_id: str, alias_speaker_id: str
    ) -> bool:
        """Merge a duplicate identity into the stable one, keeping a mapping."""
        if canonical_speaker_id == alias_speaker_id:
            return False
        with self._lock, self._transaction() as connection:
            canonical = connection.execute(
                "SELECT 1 FROM speaker_profiles WHERE speaker_id = ?",
                (canonical_speaker_id,),
            ).fetchone()
            alias = connection.execute(
                "SELECT 1 FROM speaker_profiles WHERE speaker_id = ?",
                (alias_speaker_id,),
            ).fetchone()
            if canonical is None or alias is None:
                return False
            connection.execute(
                "UPDATE speaker_embeddings SET speaker_id = ?"
                " WHERE speaker_id = ?",
                (canonical_speaker_id, alias_speaker_id),
            )
            connection.execute(
                "UPDATE speaker_observations SET speaker_id = ?"
                " WHERE speaker_id = ?",
                (canonical_speaker_id, alias_speaker_id),
            )
            now = self._clock()
            connection.execute(
                "INSERT OR IGNORE INTO speaker_aliases"
                " (canonical_speaker_id, alias_speaker_id, created_at)"
                " VALUES (?, ?, ?)",
                (canonical_speaker_id, alias_speaker_id, now),
            )
            connection.execute(
                "DELETE FROM speaker_profiles WHERE speaker_id = ?",
                (alias_speaker_id,),
            )
            connection.execute(
                "UPDATE speaker_profiles SET updated_at = ?,"
                " observation_count = (SELECT COUNT(*) FROM speaker_observations"
                "  WHERE speaker_id = ?) WHERE speaker_id = ?",
                (now, canonical_speaker_id, canonical_speaker_id),
            )
            self._trim_embeddings(connection, canonical_speaker_id)
            return True

    def promote_speaker(self, speaker_id: str) -> Optional[str]:
        """Upgrade a temporary identity to a stable ``S-0001`` id."""
        with self._lock, self._transaction() as connection:
            profile = connection.execute(
                "SELECT * FROM speaker_profiles WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
            if profile is None or str(profile["status"]) == "KNOWN":
                return None
            rows = connection.execute(
                "SELECT speaker_id FROM speaker_profiles"
                " WHERE speaker_id LIKE 'S-%'"
            ).fetchall()
            highest = 0
            for row in rows:
                suffix = str(row["speaker_id"])[len("S-"):]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
            new_id = f"S-{highest + 1:04d}"
            now = self._clock()
            connection.execute(
                "INSERT INTO speaker_profiles"
                " (speaker_id, status, created_at, updated_at,"
                "  observation_count, model_id, embedding_version)"
                " VALUES (?, 'KNOWN', ?, ?, ?, ?, ?)",
                (
                    new_id,
                    float(profile["created_at"]),
                    now,
                    int(profile["observation_count"]),
                    str(profile["model_id"] or ""),
                    str(profile["embedding_version"] or ""),
                ),
            )
            connection.execute(
                "UPDATE speaker_embeddings SET speaker_id = ?"
                " WHERE speaker_id = ?",
                (new_id, speaker_id),
            )
            connection.execute(
                "UPDATE speaker_observations SET speaker_id = ?"
                " WHERE speaker_id = ?",
                (new_id, speaker_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO speaker_aliases"
                " (canonical_speaker_id, alias_speaker_id, created_at)"
                " VALUES (?, ?, ?)",
                (new_id, speaker_id, now),
            )
            connection.execute(
                "DELETE FROM speaker_profiles WHERE speaker_id = ?",
                (speaker_id,),
            )
            return new_id


_SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_profiles (
    speaker_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 0,
    model_id TEXT NOT NULL,
    embedding_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speaker_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id TEXT NOT NULL,
    vector BLOB NOT NULL,
    quality TEXT NOT NULL,
    voiced_seconds REAL NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_speaker
    ON speaker_embeddings(speaker_id);

CREATE TABLE IF NOT EXISTS speaker_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    utterance_id INTEGER NOT NULL UNIQUE,
    speaker_id TEXT,
    match_status TEXT NOT NULL,
    similarity REAL,
    margin REAL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS speaker_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_speaker_id TEXT NOT NULL,
    alias_speaker_id TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);
"""


__all__ = [
    "IncompatibleModelError",
    "SpeakerStore",
    "SpeakerStoreError",
]
