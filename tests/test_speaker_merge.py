"""Offline duplicate-cluster review: no inferred people or silent merges."""

from __future__ import annotations

import numpy as np
import pytest

from companion.people.speaker_merge import (
    SpeakerMergeError,
    SpeakerMergeService,
)
from companion.speaker_id.config import load_speaker_config
from companion.speaker_id.store import SpeakerStore


def vec(*values: float):
    data = np.zeros(192, dtype=np.float32)
    data[:len(values)] = values
    return data / np.linalg.norm(data)


@pytest.fixture
def store(tmp_path):
    result = SpeakerStore(tmp_path / "speaker.sqlite3")
    result.initialize()
    yield result
    result.close()


def add(store, speaker, vector):
    cfg = load_speaker_config(env={})
    assert store.add_embedding(
        speaker,
        vector,
        quality="HIGH",
        voiced_seconds=2.0,
        model_id=f"{cfg.model_id}@{cfg.model_revision}",
        embedding_version=cfg.embedding_version,
        is_verified_reference=True,
    ) == "ADDED"


def pair(store, *, matching=True):
    original = store.create_candidate()
    add(store, original, vec(1, 0))
    canonical = store.promote_speaker(original)
    assert canonical is not None
    temporary = store.create_candidate()
    add(store, temporary, vec(1, 0) if matching else vec(0, 1))
    return original, canonical, temporary


async def test_local_confirmation_merges_one_person_without_losing_binding(
    store, rt
):
    legacy, canonical, temporary = pair(store)
    person = await rt.registry.register(
        "Owner", "owner", allow_processing=True, allow_persistence=True
    )
    await rt.speaker_bindings.bind_speaker(
        person.person_id, legacy, "local_cli_confirmation"
    )
    service = SpeakerMergeService(
        store, rt.speaker_bindings, audit=rt.repos.audit
    )
    preview = await service.preview(canonical, temporary)
    assert preview.can_merge
    assert preview.median_best_similarity == pytest.approx(1.0)
    # A comparison alone changes neither voiceprint nor binding.
    assert store.resolve_canonical_speaker_id(temporary) == temporary
    done = await service.confirm(canonical, temporary)
    assert done.canonical == canonical
    assert store.resolve_canonical_speaker_id(legacy) == canonical
    assert store.resolve_canonical_speaker_id(temporary) == canonical
    rows = await rt.speaker_bindings.list_bindings()
    assert len(rows) == 1
    assert rows[0]["speaker_id"] == canonical
    assert rows[0]["person_id"] == person.person_id


async def test_unrelated_candidate_cannot_be_merged(store, rt):
    _, canonical, temporary = pair(store, matching=False)
    service = SpeakerMergeService(store, rt.speaker_bindings)
    preview = await service.preview(canonical, temporary)
    assert not preview.can_merge
    assert preview.reason == "clusters_not_similar_enough"
    with pytest.raises(SpeakerMergeError):
        await service.confirm(canonical, temporary)
    assert store.resolve_canonical_speaker_id(temporary) == temporary


async def test_conflicting_human_bindings_are_never_auto_joined(store, rt):
    _, canonical, temporary = pair(store)
    first = await rt.registry.register(
        "First", "friend", allow_processing=True, allow_persistence=True
    )
    second = await rt.registry.register(
        "Second", "friend", allow_processing=True, allow_persistence=True
    )
    await rt.speaker_bindings.bind_speaker(
        first.person_id, canonical, "human"
    )
    await rt.speaker_bindings.bind_speaker(
        second.person_id, temporary, "human"
    )
    service = SpeakerMergeService(store, rt.speaker_bindings)
    preview = await service.preview(canonical, temporary)
    assert not preview.can_merge
    assert preview.reason == "binding_conflict"
    with pytest.raises(SpeakerMergeError):
        await service.confirm(canonical, temporary)
    assert len(await rt.speaker_bindings.list_bindings()) == 2
