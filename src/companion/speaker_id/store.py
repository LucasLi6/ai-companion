"""Local SQLite store for anonymous speaker profiles and voice prints.

This is a separate database from Ava's: it never reads or writes the
application's ``persons`` or ``memories`` tables. The only writer is the
engine's recognition worker (the demo also opens it for explicit list/delete
commands); one lock and one connection serialise every operation so a
multi-threaded write race cannot corrupt the numbering.

Privacy rules enforced here:

* raw audio is never stored — only normalized embeddings and diagnostics;
* a temporary ID is allocated by a persistent monotonic counter, never by a
  Python counter and never reused after deletion;
* one utterance can produce at most one observation row *per session*; the
  ASR manager restarts utterance numbering from 0, so the identity of an
  observation is ``(session_id, utterance_id)``;
* embeddings from an incompatible model version are never compared;
* a sample records the session/utterance it came from, so a post-commit
  cancellation can revoke it;
* every identity can be deleted, and the whole store can be cleared.

V0.9.9.24 migrates old stores in place: legacy observations become
``session_id='legacy'``, new embedding metadata columns are nullable, and the
migration runs in one transaction with a row-count check. No table is dropped
until the copy has been verified.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from companion.speaker_id.matcher import SpeakerCandidate

SCHEMA_VERSION = 2
LEGACY_SESSION_ID = "legacy"


class SpeakerStoreError(Exception):
    """Storage failure (message is display-safe)."""


class IncompatibleModelError(SpeakerStoreError):
    """An embedding cannot be mixed with this profile's model version."""


_QUALITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# What ``add_embedding`` did with the offered sample. Only ``SAMPLE_ADDED``
# means the identity gained a reference vector.
SAMPLE_ADDED = "ADDED"
SAMPLE_DUPLICATE = "DUPLICATE"


class SpeakerStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_embeddings_per_speaker: int = 8,
        duplicate_sample_similarity: float = 0.995,
        minimum_enrollment_quality: str = "MEDIUM",
        clock=time.time,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_embeddings_per_speaker = max(1, int(max_embeddings_per_speaker))
        self.duplicate_sample_similarity = float(duplicate_sample_similarity)
        self.minimum_enrollment_quality = str(
            minimum_enrollment_quality or "MEDIUM"
        ).upper()
        self._clock = clock
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------- lifecycle
    def initialize(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            connection: Optional[sqlite3.Connection] = None
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
                self._migrate(connection)
            except SpeakerStoreError:
                if connection is not None:
                    try:
                        connection.close()
                    except sqlite3.Error:
                        pass
                raise
            except (OSError, sqlite3.Error) as exc:
                if connection is not None:
                    try:
                        connection.close()
                    except sqlite3.Error:
                        pass
                raise SpeakerStoreError(
                    "无法打开声纹数据库（" + type(exc).__name__ + "）。"
                ) from exc
            self._connection = connection


    # -------------------------------------------------------------- migration
    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Bring an old store to ``SCHEMA_VERSION`` in one verified transaction.

        The version is recorded in ``speaker_meta``; a store created before
        V0.9.9.24 has no such row, so the shape itself is inspected as well.
        Columns are only ever added (never dropped), and the observations table
        is rebuilt only when it lacks ``session_id``. The copy is counted before
        the old table is dropped; any mismatch rolls the whole transaction back
        and leaves the original data untouched.
        """
        has_meta = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table'"
            " AND name='speaker_meta'"
        ).fetchone()
        if has_meta is None:
            connection.executescript(_META_SCHEMA)

        embedding_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(speaker_embeddings)"
            ).fetchall()
        }
        observation_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(speaker_observations)"
            ).fetchall()
        }

        try:
            connection.execute("BEGIN IMMEDIATE")
            if "is_verified_reference" not in embedding_columns:
                connection.execute(
                    "ALTER TABLE speaker_embeddings ADD COLUMN"
                    " is_verified_reference INTEGER NOT NULL DEFAULT 0"
                )
            if "source_session_id" not in embedding_columns:
                connection.execute(
                    "ALTER TABLE speaker_embeddings ADD COLUMN"
                    " source_session_id TEXT"
                )
            if "source_utterance_id" not in embedding_columns:
                connection.execute(
                    "ALTER TABLE speaker_embeddings ADD COLUMN"
                    " source_utterance_id INTEGER"
                )

            if "session_id" not in observation_columns:
                self._rebuild_observations(connection)

            # Sentinel samples are not automatically trusted after a version
            # upgrade: old rows keep ``is_verified_reference = 0``.

            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS"
                " idx_speaker_embeddings_source"
                " ON speaker_embeddings"
                " (speaker_id, source_session_id, source_utterance_id)"
                " WHERE source_session_id IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_session"
                " ON speaker_observations(session_id, utterance_id)"
            )
            self._seed_sequences(connection)
            connection.execute(
                "INSERT INTO speaker_meta(key, value)"
                " VALUES('schema_version', ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

        # Never advertise a migrated store that fails SQLite's own checks.
        check = connection.execute("PRAGMA integrity_check").fetchone()
        if check is not None and str(check[0]).lower() != "ok":
            raise SpeakerStoreError(
                "声纹数据库迁移后完整性检查失败。"
            )

    @staticmethod
    def _rebuild_observations(connection: sqlite3.Connection) -> None:
        before = connection.execute(
            "SELECT COUNT(*) AS n FROM speaker_observations"
        ).fetchone()
        expected = int(before["n"]) if before is not None else 0
        connection.execute(
            "CREATE TABLE speaker_observations_v2 ("
            " observation_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session_id TEXT NOT NULL DEFAULT 'legacy',"
            " utterance_id INTEGER NOT NULL,"
            " speaker_id TEXT,"
            " match_status TEXT NOT NULL,"
            " similarity REAL,"
            " margin REAL,"
            " created_at REAL NOT NULL,"
            " UNIQUE(session_id, utterance_id)"
            ")"
        )
        connection.execute(
            "INSERT INTO speaker_observations_v2"
            " (session_id, utterance_id, speaker_id, match_status,"
            "  similarity, margin, created_at)"
            " SELECT ?, utterance_id, speaker_id, match_status,"
            "  similarity, margin, created_at"
            " FROM speaker_observations",
            (LEGACY_SESSION_ID,),
        )
        copied = connection.execute(
            "SELECT COUNT(*) AS n FROM speaker_observations_v2"
        ).fetchone()
        copied_count = int(copied["n"]) if copied is not None else -1
        if copied_count != expected:
            raise SpeakerStoreError(
                "声纹观察记录迁移数量不一致，已回滚。"
            )
        connection.execute("DROP TABLE speaker_observations")
        connection.execute(
            "ALTER TABLE speaker_observations_v2"
            " RENAME TO speaker_observations"
        )

    @staticmethod
    def _seed_sequences(connection: sqlite3.Connection) -> None:
        """Persist monotonic counters so a deleted last ID is never reused."""
        for key, prefix, column in (
            ("candidate_seq", "TEMP-", "speaker_id"),
            ("promoted_seq", "S-", "speaker_id"),
        ):
            row = connection.execute(
                "SELECT value FROM speaker_meta WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                continue
            highest = 0
            for profile in connection.execute(
                f"SELECT {column} AS value FROM speaker_profiles"
                f" WHERE {column} LIKE ?",
                (prefix + "%",),
            ).fetchall():
                suffix = str(profile["value"])[len(prefix):]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
            connection.execute(
                "INSERT INTO speaker_meta(key, value) VALUES(?, ?)",
                (key, str(highest)),
            )

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection, key: str) -> int:
        row = connection.execute(
            "SELECT value FROM speaker_meta WHERE key = ?", (key,)
        ).fetchone()
        value = int(row["value"]) if row is not None else 0
        value += 1
        connection.execute(
            "INSERT INTO speaker_meta(key, value) VALUES(?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        return value


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
        """Atomically allocate the next ``TEMP-0001`` style identity.

        The counter lives in ``speaker_meta`` and never moves backwards, so
        deleting ``TEMP-0007`` does not let a later candidate reuse that id.
        """
        with self._lock, self._transaction() as connection:
            number = self._next_sequence(connection, "candidate_seq")
            speaker_id = f"TEMP-{number:04d}"
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
        source_session_id: Optional[str] = None,
        source_utterance_id: Optional[int] = None,
        is_verified_reference: bool = False,
    ) -> str:
        """Store one normalized voice print, keeping the sample set useful.

        The write is idempotent per ``(speaker, source session, source
        utterance)``. A near-duplicate (cosine >= ``duplicate_sample_similarity``)
        is normally discarded; it may replace the matching sample only when it
        has strictly better quality, and a verified reference is never
        replaced. After insertion the set is trimmed to
        ``max_embeddings_per_speaker`` with verified references protected and
        the rest chosen by deterministic farthest-first diversity.

        Returns ``SAMPLE_ADDED`` only when a reference vector was actually
        kept, otherwise ``SAMPLE_DUPLICATE`` (including a same-source retry
        and a sample evicted by the trim in the same transaction).
        """
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

            if source_session_id is not None and source_utterance_id is not None:
                existing = connection.execute(
                    "SELECT embedding_id FROM speaker_embeddings"
                    " WHERE speaker_id = ? AND source_session_id = ?"
                    "   AND source_utterance_id = ?",
                    (speaker_id, str(source_session_id),
                     int(source_utterance_id)),
                ).fetchone()
                if existing is not None:
                    # A retry after a commit (or an ASR replay) must not add a
                    # second sample for the same spoken utterance.
                    return SAMPLE_DUPLICATE

            duplicate = self._find_duplicate(
                connection, speaker_id, array
            )
            if duplicate is not None:
                existing_id, existing_vector, existing_rank, protected = duplicate
                new_rank = _QUALITY_RANK.get(str(quality).upper(), 0)
                if protected or new_rank <= existing_rank:
                    return SAMPLE_DUPLICATE
                connection.execute(
                    "DELETE FROM speaker_embeddings WHERE embedding_id = ?",
                    (existing_id,),
                )

            cursor = connection.execute(
                "INSERT INTO speaker_embeddings"
                " (speaker_id, vector, quality, voiced_seconds, created_at,"
                "  is_verified_reference, source_session_id,"
                "  source_utterance_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    speaker_id,
                    array.astype("<f4").tobytes(),
                    str(quality),
                    float(voiced_seconds),
                    now,
                    1 if is_verified_reference else 0,
                    None if source_session_id is None
                    else str(source_session_id),
                    None if source_utterance_id is None
                    else int(source_utterance_id),
                ),
            )
            connection.execute(
                "UPDATE speaker_profiles SET updated_at = ? WHERE speaker_id = ?",
                (now, speaker_id),
            )
            new_id = int(cursor.lastrowid)
            self._trim_embeddings(connection, speaker_id)
            # A sample inserted and evicted by the same trim did not add value.
            kept = connection.execute(
                "SELECT 1 FROM speaker_embeddings WHERE embedding_id = ?",
                (new_id,),
            ).fetchone()
            return SAMPLE_ADDED if kept is not None else SAMPLE_DUPLICATE

    def _find_duplicate(
        self,
        connection: sqlite3.Connection,
        speaker_id: str,
        array,
    ):
        """Return ``(id, vector, rank, protected)`` for a near-identical sample."""
        import numpy as np

        rows = connection.execute(
            "SELECT embedding_id, vector, quality, is_verified_reference"
            " FROM speaker_embeddings WHERE speaker_id = ?"
            " ORDER BY embedding_id ASC",
            (speaker_id,),
        ).fetchall()
        best = None
        best_score = -2.0
        for row in rows:
            vector = self._decode_vector(row["vector"])
            if vector is None:
                continue
            score = float(array @ vector)
            if score > best_score:
                best_score = score
                best = (row, vector)
        if best is None:
            return None
        if best_score < self.duplicate_sample_similarity:
            return None
        row, vector = best
        rank = _QUALITY_RANK.get(str(row["quality"]).upper(), 0)
        return (
            int(row["embedding_id"]),
            vector,
            rank,
            bool(int(row["is_verified_reference"] or 0)),
        )

    def _trim_embeddings(
        self, connection: sqlite3.Connection, speaker_id: str
    ) -> None:
        """Keep the best, most diverse bounded sample set for one identity.

        Verified references are protected. If more than the limit exist, the
        best-quality (then oldest) are kept. Remaining slots are filled by
        farthest-first diversity over samples that meet the enrollment quality
        floor, so a low-quality outlier is never admitted merely for being
        different. Selection is deterministic for the same rows.
        """
        limit = self.max_embeddings_per_speaker
        rows = connection.execute(
            "SELECT embedding_id, vector, quality, is_verified_reference,"
            " created_at FROM speaker_embeddings WHERE speaker_id = ?"
            " ORDER BY embedding_id ASC",
            (speaker_id,),
        ).fetchall()
        parsed = []
        for row in rows:
            vector = self._decode_vector(row["vector"])
            if vector is None:
                # Corrupt or legacy-invalid rows are removed, never compared.
                connection.execute(
                    "DELETE FROM speaker_embeddings WHERE embedding_id = ?",
                    (int(row["embedding_id"]),),
                )
                continue
            parsed.append(
                {
                    "id": int(row["embedding_id"]),
                    "vector": vector,
                    "rank": _QUALITY_RANK.get(
                        str(row["quality"]).upper(), 0
                    ),
                    "verified": bool(int(row["is_verified_reference"] or 0)),
                    "created_at": float(row["created_at"]),
                }
            )
        if len(parsed) <= limit:
            return

        verified = sorted(
            (item for item in parsed if item["verified"]),
            key=lambda item: (
                -item["rank"], item["created_at"], item["id"]
            ),
        )
        keep = verified[:limit]
        keep_ids = {item["id"] for item in keep}

        if len(keep) < limit:
            min_rank = _QUALITY_RANK.get(
                self.minimum_enrollment_quality, 2
            )
            pool = [item for item in parsed if item["id"] not in keep_ids]
            eligible = sorted(
                (item for item in pool if item["rank"] >= min_rank),
                key=lambda item: (
                    -item["rank"], item["created_at"], item["id"]
                ),
            )
            weak = sorted(
                (item for item in pool if item["rank"] < min_rank),
                key=lambda item: (
                    -item["rank"], item["created_at"], item["id"]
                ),
            )
            selected: list[dict[str, Any]] = []
            reference = [item["vector"] for item in keep]
            while len(keep) + len(selected) < limit and eligible:
                if not reference:
                    chosen = eligible.pop(0)
                else:
                    import numpy as np

                    best_index = 0
                    best_key = None
                    for index, item in enumerate(eligible):
                        distance = min(
                            1.0 - float(item["vector"] @ anchor)
                            for anchor in reference
                        )
                        key = (
                            distance,
                            item["rank"],
                            -item["created_at"],
                            -item["id"],
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_index = index
                    chosen = eligible.pop(best_index)
                selected.append(chosen)
                reference.append(chosen["vector"])
            # Only after every eligible sample is kept may weaker samples fill
            # the remaining slots, in quality order (never for being diverse).
            for item in weak:
                if len(keep) + len(selected) >= limit:
                    break
                selected.append(item)
            keep.extend(selected)
            keep_ids.update(item["id"] for item in selected)

        delete_ids = [
            item["id"] for item in parsed if item["id"] not in keep_ids
        ]
        for embedding_id in delete_ids:
            connection.execute(
                "DELETE FROM speaker_embeddings WHERE embedding_id = ?",
                (embedding_id,),
            )

    @staticmethod
    def _decode_vector(blob):
        import numpy as np

        try:
            vector = np.frombuffer(blob, dtype="<f4").astype(np.float32)
        except (TypeError, ValueError):
            return None
        if vector.ndim != 1 or vector.size == 0:
            return None
        if not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return (vector / norm).astype(np.float32)

    def revoke_embedding(
        self,
        speaker_id: str,
        *,
        source_session_id: str,
        source_utterance_id: int,
    ) -> bool:
        """Remove a sample that was committed before a late cancellation."""
        with self._lock, self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM speaker_embeddings"
                " WHERE speaker_id = ? AND source_session_id = ?"
                "   AND source_utterance_id = ?"
                "   AND is_verified_reference = 0",
                (speaker_id, str(source_session_id), int(source_utterance_id)),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE speaker_profiles SET updated_at = ?"
                    " WHERE speaker_id = ?",
                    (self._clock(), speaker_id),
                )
                return True
            return False


    def record_observation(
        self,
        utterance_id: int,
        *,
        speaker_id: Optional[str],
        match_status: str,
        similarity: Optional[float],
        margin: Optional[float],
        session_id: str = LEGACY_SESSION_ID,
    ) -> None:
        """One row per ``(session, utterance)``; a later verdict may refine it.

        ASR utterance numbering restarts at 0 in every process, so the session
        id is part of the row's identity. A later low-information verdict
        (``INSUFFICIENT_AUDIO``/``PENDING``/``ERROR``) never overwrites an
        already recorded match, and re-processing the same result is
        idempotent.
        """
        with self._lock, self._transaction() as connection:
            previous = self._observation_speaker(
                connection, str(session_id), int(utterance_id)
            )
            connection.execute(
                "INSERT INTO speaker_observations"
                " (session_id, utterance_id, speaker_id, match_status,"
                "  similarity, margin, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(session_id, utterance_id) DO UPDATE SET"
                "  speaker_id = excluded.speaker_id,"
                "  match_status = excluded.match_status,"
                "  similarity = excluded.similarity,"
                "  margin = excluded.margin"
                " WHERE excluded.match_status NOT IN"
                "       ('INSUFFICIENT_AUDIO', 'PENDING', 'ERROR')",
                (
                    str(session_id),
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
        connection: sqlite3.Connection,
        session_id: str,
        utterance_id: int,
    ) -> Optional[str]:
        row = connection.execute(
            "SELECT speaker_id FROM speaker_observations"
            " WHERE session_id = ? AND utterance_id = ?",
            (session_id, utterance_id),
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
                "UPDATE OR IGNORE speaker_embeddings SET speaker_id = ?"
                " WHERE speaker_id = ?",
                (canonical_speaker_id, alias_speaker_id),
            )
            # Samples that could not move because the canonical identity
            # already has a sample from the same source utterance are exact
            # duplicates and are removed with the alias.
            connection.execute(
                "DELETE FROM speaker_embeddings WHERE speaker_id = ?",
                (alias_speaker_id,),
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
            number = self._next_sequence(connection, "promoted_seq")
            new_id = f"S-{number:04d}"
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


_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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
    created_at REAL NOT NULL,
    is_verified_reference INTEGER NOT NULL DEFAULT 0,
    source_session_id TEXT,
    source_utterance_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_speaker
    ON speaker_embeddings(speaker_id);

CREATE TABLE IF NOT EXISTS speaker_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'legacy',
    utterance_id INTEGER NOT NULL,
    speaker_id TEXT,
    match_status TEXT NOT NULL,
    similarity REAL,
    margin REAL,
    created_at REAL NOT NULL,
    UNIQUE(session_id, utterance_id)
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
    "LEGACY_SESSION_ID",
    "SCHEMA_VERSION",
    "SpeakerStore",
    "SpeakerStoreError",
]
