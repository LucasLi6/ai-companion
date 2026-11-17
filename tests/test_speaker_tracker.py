"""Offline tests for cross-utterance continuity and the pending buffer.

The tracker is pure memory. These tests drive it directly with a fake clock,
so expiry and turn counting are deterministic; no model, microphone or
database is involved.
"""

from __future__ import annotations

import math

import numpy as np

from companion.speaker_id.config import EMBEDDING_DIM
from companion.speaker_id.tracker import (
    EXPIRED,
    STABLE,
    TENTATIVE,
    UNKNOWN,
    PendingSampleBuffer,
    SpeakerTrackManager,
)


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def unit_angle(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = math.cos(radians)
    vector[1] = math.sin(radians)
    return vector


def manager(clock: Clock | None = None, **overrides) -> SpeakerTrackManager:
    options = dict(
        expiry_seconds=15.0,
        recent_turns=4,
        min_turns=2,
        min_voiced_seconds=3.0,
        continuity_threshold=0.43,
        clock=clock or Clock(),
    )
    options.update(overrides)
    return SpeakerTrackManager(**options)


def new_voice(
    manager: SpeakerTrackManager,
    uid: int,
    degrees: float = 0.0,
    *,
    seconds: float = 2.0,
    final: bool = False,
):
    return manager.observe(
        uid,
        speaker_id=None,
        status="NEW_CANDIDATE",
        similarity=None,
        margin=None,
        voiced_seconds=seconds,
        quality_rank=3,
        vector=unit_angle(degrees),
        final=final,
    )


def test_first_new_voice_is_a_provisional_track_not_a_speaker_id() -> None:
    tracks = manager()
    track = new_voice(tracks, 1)
    assert track is not None
    assert track.track_id == "TRACK-0001"
    assert track.speaker_id is None
    assert track.state == TENTATIVE
    assert track.turns == 0
    assert tracks.current_track().track_id == "TRACK-0001"


def test_repeated_windows_of_one_utterance_do_not_duplicate_the_track() -> None:
    tracks = manager()
    new_voice(tracks, 1)
    new_voice(tracks, 1, 2.0)
    new_voice(tracks, 1, 4.0)
    assert len(tracks.tracks()) == 1
    assert tracks.tracks()[0].utterance_ids == (1,)


def test_turns_are_counted_only_at_finalisation() -> None:
    tracks = manager()
    first = new_voice(tracks, 1)
    assert first.turns == 0
    assert tracks.can_persist(first, extra_turns=1, extra_voiced_seconds=2.0) is False

    finalised = tracks.finalize(1, success=True)
    assert finalised.turns == 1
    assert finalised.matched_turns == 1
    assert finalised.voiced_seconds == 2.0
    # With the current turn as an extra, the two-turn/three-second gate is met.
    assert tracks.can_persist(
        finalised, extra_turns=1, extra_voiced_seconds=1.5
    ) is True


def test_the_same_voice_continues_the_same_tentative_track() -> None:
    tracks = manager()
    new_voice(tracks, 1)
    tracks.finalize(1, success=True)
    second = new_voice(tracks, 2)
    assert second.track_id == "TRACK-0001"
    assert second.state == TENTATIVE


def test_a_different_voice_starts_a_new_track_and_supersedes_the_old() -> None:
    tracks = manager()
    new_voice(tracks, 1, 0.0)
    tracks.finalize(1, success=True)
    other = new_voice(tracks, 2, 80.0)
    assert other.track_id == "TRACK-0002"
    assert tracks.stats()["superseded"] == 1
    old = tracks.tracks()[0]
    assert old.state == EXPIRED
    assert old.superseded_by == "TRACK-0002"


def test_a_known_identity_track_is_stable_and_bound() -> None:
    tracks = manager()
    track = tracks.observe(
        1,
        speaker_id="TEMP-0001",
        status="CANDIDATE_MATCH",
        similarity=0.80,
        margin=0.20,
        voiced_seconds=2.0,
        quality_rank=3,
        vector=unit_angle(0),
    )
    assert track.speaker_id == "TEMP-0001"
    assert track.state == STABLE
    assert track.turns == 0
    finalised = tracks.finalize(1, success=True)
    assert finalised.matched_turns == 1


def test_an_ambiguous_turn_without_history_is_unknown_and_not_confirmable() -> None:
    tracks = manager()
    track = tracks.observe(
        1,
        speaker_id=None,
        status="AMBIGUOUS",
        similarity=0.40,
        margin=0.01,
        voiced_seconds=2.0,
        quality_rank=3,
        vector=unit_angle(0),
    )
    assert track.state == UNKNOWN
    assert track.speaker_id is None
    tracks.finalize(1, success=True)
    assert tracks.can_persist(
        tracks.current_track(), extra_turns=5, extra_voiced_seconds=99
    ) is False


def test_an_asr_failure_removes_the_turn_evidence() -> None:
    tracks = manager()
    new_voice(tracks, 1)
    tracks.finalize(1, success=True)
    new_voice(tracks, 2)
    failed = tracks.finalize(2, success=False)
    assert failed.turns == 1
    assert failed.voiced_seconds == 2.0
    assert 2 not in failed.utterance_ids


def test_cancel_removes_evidence_but_keeps_the_track() -> None:
    tracks = manager()
    new_voice(tracks, 1)
    tracks.finalize(1, success=True)
    new_voice(tracks, 2)
    tracks.cancel(2)
    current = tracks.current_track()
    assert current is not None
    assert current.turns == 1
    assert 2 not in current.utterance_ids


def test_a_long_gap_expires_the_track_and_starts_a_new_id() -> None:
    clock = Clock()
    tracks = manager(clock)
    new_voice(tracks, 1)
    tracks.finalize(1, success=True)
    clock.advance(16.0)
    assert tracks.current_track() is None
    assert tracks.tracks()[0].state == EXPIRED
    assert tracks.stats()["expirations"] == 1
    second = new_voice(tracks, 2)
    assert second.track_id == "TRACK-0002"


def test_bind_identity_refuses_to_rebind_a_track_to_another_person() -> None:
    tracks = manager()
    track = new_voice(tracks, 1)
    assert tracks.bind_identity(track.track_id, "TEMP-0001") is True
    assert tracks.bind_identity(track.track_id, "TEMP-0002") is False
    assert tracks.current_track().speaker_id == "TEMP-0001"
    assert tracks.stats()["conflicts"] == 1


def test_confirm_promotes_a_tentative_track_but_never_an_unknown_one() -> None:
    tracks = manager()
    track = new_voice(tracks, 1)
    confirmed = tracks.confirm()
    assert confirmed is not None
    assert confirmed.track_id == track.track_id
    assert confirmed.state == STABLE
    assert tracks.stats()["transitions"] == 1
    # Idempotent: no second transition count.
    assert tracks.confirm(track.track_id).state == STABLE
    assert tracks.stats()["transitions"] == 1
    # A confirmed track still needs the turn/voiced evidence.
    assert tracks.can_persist(confirmed) is False

    # Ambiguity without any history forms an UNKNOWN track, which can never be
    # promoted by an explicit confirm.
    ambiguous = manager()
    unknown = ambiguous.observe(
        2,
        speaker_id=None,
        status="AMBIGUOUS",
        similarity=0.40,
        margin=0.01,
        voiced_seconds=2.0,
        quality_rank=3,
        vector=unit_angle(70),
    )
    assert unknown.state == UNKNOWN
    assert ambiguous.confirm(unknown.track_id) is None


def test_confirm_rejects_an_expired_or_unknown_track_id() -> None:
    clock = Clock()
    tracks = manager(clock)
    track = new_voice(tracks, 1)
    clock.advance(16.0)
    assert tracks.confirm(track.track_id) is None
    assert tracks.confirm("TRACK-9999") is None


def test_reset_session_drops_tracks_but_never_reuses_track_ids() -> None:
    tracks = manager()
    new_voice(tracks, 1)
    tracks.reset_session()
    assert tracks.tracks() == []
    assert tracks.current_track() is None
    assert new_voice(tracks, 2).track_id == "TRACK-0002"


# ----------------------------------------------------------- pending samples
def test_pending_buffer_is_bounded_and_expires() -> None:
    clock = Clock()
    pending = PendingSampleBuffer(max_count=12, expiry_seconds=60.0, clock=clock)
    for index in range(15):
        pending.add(
            "TRACK-0001",
            unit_angle(index),
            voiced_seconds=1.0,
        )
    stats = pending.stats()
    assert stats["count"] == 12
    assert stats["dropped"] == 3

    clock.advance(61.0)
    assert pending.stats()["count"] == 0


def test_pending_samples_are_never_exposed_for_matching() -> None:
    pending = PendingSampleBuffer(max_count=12, expiry_seconds=60.0)
    pending.add("TRACK-0001", unit_angle(0), voiced_seconds=2.0)
    items = pending.items()
    assert len(items) == 1
    # There is no API returning a profile/candidate for matching; the only
    # usable output is diagnostics. This pins that contract in a test.
    assert not hasattr(pending, "candidates")
    assert not hasattr(pending, "load_candidates")


def test_drop_track_removes_only_the_confirmed_tracks_staged_samples() -> None:
    pending = PendingSampleBuffer(max_count=12)
    pending.add("TRACK-0001", unit_angle(0))
    pending.add("TRACK-0002", unit_angle(30))
    assert pending.drop_track("TRACK-0001") == 1
    assert [item.track_id for item in pending.items()] == ["TRACK-0002"]
