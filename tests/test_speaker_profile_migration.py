"""Schema v2 -> v3 migration: additive, transactional, lossless.

Every fixture here is a hand-built v2 database with real 192-dim vectors.
No model is loaded, so these tests cannot accidentally claim recognition
accuracy; they only assert that stored identity data survives the upgrade.
"""

from __future__ import annotations

import shutil
import sqlite3

import numpy as np
import pytest

from companion.speaker_id.store import (
    SCHEMA_VERSION,
    SAMPLE_ADDED,
    SpeakerStore,
    SpeakerStoreError,
)

MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common@v1.0.1"
VERSION = "eres2netv2-192-v1"
DIM = 192


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[: len(values)] = values
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def blob(*values: float) -> bytes:
    return unit(*values).astype("<f4").tobytes()


def build_v2(path) -> None:
    """A v2 store: session-aware observations, no origin/quarantined columns."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE speaker_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE speaker_profiles (
            speaker_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 0,
            model_id TEXT NOT NULL,
            embedding_version TEXT NOT NULL
        );
        CREATE TABLE speaker_embeddings (
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
        CREATE TABLE speaker_observations (
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
        CREATE TABLE speaker_aliases (
            alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_speaker_id TEXT NOT NULL,
            alias_speaker_id TEXT NOT NULL UNIQUE,
            created_at REAL NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO speaker_meta VALUES ('schema_version', '2')"
    )
    connection.execute("INSERT INTO speaker_meta VALUES ('candidate_seq', '1')")
    connection.execute("INSERT INTO speaker_meta VALUES ('promoted_seq', '0')")
    connection.execute(
        "INSERT INTO speaker_profiles VALUES"
        " ('TEMP-0001', 'CANDIDATE', 1.0, 1.0, 1, ?, ?)",
        (MODEL, VERSION),
    )
    connection.execute(
        "INSERT INTO speaker_embeddings"
        " (speaker_id, vector, quality, voiced_seconds, created_at,"
        "  is_verified_reference, source_session_id, source_utterance_id)"
        " VALUES ('TEMP-0001', ?, 'HIGH', 2.0, 1.0, 1, 'session-a', 0)",
        (blob(1, 0),),
    )
    connection.execute(
        "INSERT INTO speaker_embeddings"
        " (speaker_id, vector, quality, voiced_seconds, created_at,"
        "  is_verified_reference, source_session_id, source_utterance_id)"
        " VALUES ('TEMP-0001', ?, 'MEDIUM', 1.5, 2.0, 0, 'session-a', 1)",
        (blob(0, 1),),
    )
    connection.execute(
        "INSERT INTO speaker_observations"
        " (session_id, utterance_id, speaker_id, match_status, similarity,"
        "  margin, created_at)"
        " VALUES ('session-a', 0, 'TEMP-0001', 'CANDIDATE_MATCH', 0.9, 0.2, 1.0)"
    )
    connection.execute(
        "INSERT INTO speaker_aliases"
        " (canonical_speaker_id, alias_speaker_id, created_at)"
        " VALUES ('TEMP-0001', 'OLD-0009', 1.0)"
    )
    connection.commit()
    connection.close()


def columns(path, table: str) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {
            str(row[1])
            for row in connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }


def test_v2_store_migrates_to_current_schema_preserving_every_row(
    tmp_path,
) -> None:
    path = tmp_path / "v2.sqlite3"
    build_v2(path)

    store = SpeakerStore(path)
    store.initialize()
    try:
        stage = "schema setup"
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM speaker_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        samples = connection.execute(
            "SELECT embedding_id, speaker_id, quality, is_verified_reference,"
            " source_session_id, source_utterance_id, origin, quarantined"
            " FROM speaker_embeddings ORDER BY embedding_id"
        ).fetchall()
        observation = connection.execute(
            "SELECT session_id, utterance_id, speaker_id, match_status"
            " FROM speaker_observations"
        ).fetchone()
        alias = connection.execute(
            "SELECT canonical_speaker_id, alias_speaker_id"
            " FROM speaker_aliases"
        ).fetchone()

    assert stage == "schema setup"
    assert version == str(SCHEMA_VERSION) == "5"
    assert samples == [
        (1, "TEMP-0001", "HIGH", 1, "session-a", 0, "legacy", 0),
        (2, "TEMP-0001", "MEDIUM", 0, "session-a", 1, "legacy", 0),
    ]
    assert observation == ("session-a", 0, "TEMP-0001", "CANDIDATE_MATCH")
    assert alias == ("TEMP-0001", "OLD-0009")


def test_v4_consent_rows_migrate_as_grants_and_revocation_is_durable(
    tmp_path,
) -> None:
    path = tmp_path / "v4.sqlite3"
    build_v2(path)
    connection = sqlite3.connect(path)
    # The v4 consent table had no status column: every row was a grant.
    connection.execute(
        "CREATE TABLE speaker_enrollment_consents ("
        " speaker_id TEXT PRIMARY KEY,"
        " granted_by TEXT NOT NULL,"
        " created_at REAL NOT NULL,"
        " updated_at REAL NOT NULL)"
    )
    connection.execute(
        "INSERT INTO speaker_enrollment_consents VALUES"
        " ('TEMP-0001', 'local_cli', 1.0, 1.0)"
    )
    connection.execute(
        "UPDATE speaker_meta SET value = '4' WHERE key = 'schema_version'"
    )
    connection.commit()
    connection.close()

    store = SpeakerStore(path)
    store.initialize()
    try:
        rows = store.enrollment_consents()
        assert [(row["speaker_id"], row["status"]) for row in rows] == [
            ("TEMP-0001", "GRANTED")
        ]
        # A refusal is written as REVOKED, never by deleting the row, so a
        # later restart cannot mistake silence for permission.
        assert store.revoke_enrollment_consent("TEMP-0001") is True
        assert store.enrollment_consent("TEMP-0001")["status"] == "REVOKED"
    finally:
        store.close()

    reopened = SpeakerStore(path)
    reopened.initialize()
    try:
        assert reopened.enrollment_consent("TEMP-0001")["status"] == "REVOKED"
    finally:
        reopened.close()


def test_v3_columns_are_added_only_once_and_migration_is_idempotent(tmp_path) -> None:
    path = tmp_path / "v2.sqlite3"
    build_v2(path)

    first = SpeakerStore(path)
    first.initialize()
    first.close()
    before = {
        name: columns(path, name)
        for name in ("speaker_embeddings", "speaker_observations")
    }

    second = SpeakerStore(path)
    second.initialize()
    try:
        assert second.create_candidate() == "TEMP-0002"
    finally:
        second.close()

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM speaker_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        count = connection.execute(
            "SELECT COUNT(*) FROM speaker_embeddings"
        ).fetchone()[0]
    assert version == str(SCHEMA_VERSION)
    assert count == 2
    assert columns(path, "speaker_embeddings") == before["speaker_embeddings"]


def test_a_failed_migration_rolls_back_and_the_retry_succeeds(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "v2.sqlite3"
    build_v2(path)

    def explode(connection):
        raise SpeakerStoreError("simulated migration failure")

    monkeypatch.setattr(SpeakerStore, "_seed_sequences", staticmethod(explode))
    failing = SpeakerStore(path)
    with pytest.raises(SpeakerStoreError):
        failing.initialize()
    failing.close()

    # The ALTER TABLE statements ran inside the rolled-back transaction, so the
    # original v2 shape and version must still be intact.
    assert "origin" not in columns(path, "speaker_embeddings")
    assert "quarantined" not in columns(path, "speaker_embeddings")
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM speaker_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == "2"
        assert connection.execute(
            "SELECT COUNT(*) FROM speaker_embeddings"
        ).fetchone()[0] == 2

    monkeypatch.undo()
    retry = SpeakerStore(path)
    retry.initialize()
    try:
        assert "origin" in columns(path, "speaker_embeddings")
    finally:
        retry.close()


def test_new_samples_record_their_origin_and_quarantine_defaults_to_zero(
    tmp_path,
) -> None:
    store = SpeakerStore(tmp_path / "store.sqlite3")
    store.initialize()
    try:
        speaker = store.create_candidate()
        assert store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=2.0,
            model_id=MODEL,
            embedding_version=VERSION,
            origin="bootstrap",
        ) == SAMPLE_ADDED
        assert store.add_embedding(
            speaker,
            unit(0, 1),
            quality="HIGH",
            voiced_seconds=2.0,
            model_id=MODEL,
            embedding_version=VERSION,
            origin="not-a-real-origin",
        ) == SAMPLE_ADDED
    finally:
        store.close()

    with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
        rows = connection.execute(
            "SELECT origin, quarantined FROM speaker_embeddings"
            " ORDER BY embedding_id"
        ).fetchall()
    assert rows == [("bootstrap", 0), ("auto", 0)]


def test_quarantine_excludes_a_sample_from_matching_and_duplicates(
    tmp_path,
) -> None:
    store = SpeakerStore(tmp_path / "store.sqlite3")
    store.initialize()
    try:
        speaker = store.create_candidate()
        for vector in (unit(1, 0), unit(0, 1)):
            store.add_embedding(
                speaker,
                vector,
                quality="HIGH",
                voiced_seconds=2.0,
                model_id=MODEL,
                embedding_version=VERSION,
            )
        with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
            sample_id = connection.execute(
                "SELECT embedding_id FROM speaker_embeddings"
                " WHERE vector = ?",
                (blob(1, 0),),
            ).fetchone()[0]
        assert store.quarantine_sample(sample_id) is True

        candidates, _ = store.load_candidates(
            model_id=MODEL, embedding_version=VERSION
        )
        assert [sample.sample_id for sample in candidates[0].samples] != []
        assert all(
            sample.sample_id != sample_id
            for sample in candidates[0].samples
        )

        # Were the quarantined row still a duplicate target, this would be
        # rejected as DUPLICATE. Because it is excluded, the sample is kept.
        assert store.add_embedding(
            speaker,
            unit(1, 0),
            quality="HIGH",
            voiced_seconds=2.0,
            model_id=MODEL,
            embedding_version=VERSION,
        ) == SAMPLE_ADDED
        # Resuming makes it visible again.
        assert store.quarantine_sample(sample_id, quarantined=False) is True
        candidates, _ = store.load_candidates(
            model_id=MODEL, embedding_version=VERSION
        )
        assert any(
            sample.sample_id == sample_id
            for sample in candidates[0].samples
        )
    finally:
        store.close()


def test_the_retention_trim_never_evicts_a_quarantined_row(tmp_path) -> None:
    store = SpeakerStore(
        tmp_path / "store.sqlite3", max_embeddings_per_speaker=2
    )
    store.initialize()
    try:
        speaker = store.create_candidate()
        for vector in (unit(1, 0), unit(0, 1)):
            store.add_embedding(
                speaker,
                vector,
                quality="HIGH",
                voiced_seconds=2.0,
                model_id=MODEL,
                embedding_version=VERSION,
            )
        with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
            sample_id = connection.execute(
                "SELECT embedding_id FROM speaker_embeddings"
                " WHERE vector = ?",
                (blob(0, 1),),
            ).fetchone()[0]
        assert store.quarantine_sample(sample_id) is True

        # The visible set is now (A) with one free slot; adding C must not
        # count B against the limit nor delete B's row.
        assert store.add_embedding(
            speaker,
            unit(0, 0, 1),
            quality="HIGH",
            voiced_seconds=2.0,
            model_id=MODEL,
            embedding_version=VERSION,
        ) == SAMPLE_ADDED
        with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM speaker_embeddings"
            ).fetchone()[0]
            still_there = connection.execute(
                "SELECT quarantined FROM speaker_embeddings"
                " WHERE embedding_id = ?",
                (sample_id,),
            ).fetchone()
        assert total == 3
        assert still_there == (1,)
        candidates, _ = store.load_candidates(
            model_id=MODEL, embedding_version=VERSION
        )
        assert len(candidates[0].samples) == 2
    finally:
        store.close()


def test_revoke_sample_is_idempotent_and_keeps_other_samples(tmp_path) -> None:
    store = SpeakerStore(tmp_path / "store.sqlite3")
    store.initialize()
    try:
        speaker = store.create_candidate()
        for vector in (unit(1, 0), unit(0, 1)):
            store.add_embedding(
                speaker,
                vector,
                quality="HIGH",
                voiced_seconds=2.0,
                model_id=MODEL,
                embedding_version=VERSION,
            )
        with sqlite3.connect(tmp_path / "store.sqlite3") as connection:
            sample_id = connection.execute(
                "SELECT MIN(embedding_id) FROM speaker_embeddings"
            ).fetchone()[0]
        assert store.revoke_sample(sample_id) is True
        assert store.revoke_sample(sample_id) is False
        assert store.revoke_sample(999999) is False
        candidates, _ = store.load_candidates(
            model_id=MODEL, embedding_version=VERSION
        )
        assert len(candidates[0].samples) == 1
    finally:
        store.close()


def test_backup_before_migration_can_be_restored(tmp_path) -> None:
    source = tmp_path / "v2.sqlite3"
    backup = tmp_path / "v2.backup.sqlite3"
    build_v2(source)
    shutil.copy2(source, backup)

    store = SpeakerStore(source)
    store.initialize()
    store.close()

    restored = SpeakerStore(backup)
    restored.initialize()
    try:
        candidates, _ = restored.load_candidates(
            model_id=MODEL, embedding_version=VERSION
        )
        assert len(candidates) == 1
        assert len(candidates[0].samples) == 2
        assert candidates[0].speaker_id == "TEMP-0001"
    finally:
        restored.close()
