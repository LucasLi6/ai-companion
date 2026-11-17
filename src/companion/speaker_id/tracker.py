"""Cross-utterance speaker continuity, in memory only.

The matcher answers one question for one analysed window: *does this score
belong to a stored person?* It cannot answer "is the person speaking now the
same person who spoke three seconds ago" — that is a property of the sequence
of turns, not of one embedding.

This module adds that sequence-level view with plain data structures and no
new neural network:

* :class:`SpeakerTrack` is an immutable snapshot of one continuous stretch of
  speech by (provisionally) one person;
* :class:`SpeakerTrackManager` owns the mutable state, the expiry clock and
  the bounded recent-embedding buffer used to link a tentative track across
  turns before any permanent speaker id exists;
* :class:`PendingSampleBuffer` is a bounded, memory-only buffer for samples
  that are *not yet* trustworthy. It is a diagnostic/learning staging area and
  is never handed to the matcher.

Temporary ``TRACK-0001`` ids are intentionally in a different namespace from
the store's permanent ``TEMP-0001`` ids. A track id is never persisted, never
written into ``SpeakerResult.speaker_id`` and never mapped to a ``person_id``.

States::

    TENTATIVE --(enough consistent turns)--> STABLE
        |                                      |
        +--(ambiguous only)--> UNKNOWN         +--(idle > expiry)--> EXPIRED
        +--(idle > expiry)-------------------> EXPIRED

A conflicting identity never mutates the old track into the new person: the
old track is expired and a new one starts, so history remains honest.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from companion.speaker_id.voice_profiles import (
    VoiceProfileError,
    cosine,
    normalize_embedding,
)

TENTATIVE = "TENTATIVE"
STABLE = "STABLE"
UNKNOWN = "UNKNOWN"
EXPIRED = "EXPIRED"
TRACK_STATES = (TENTATIVE, STABLE, UNKNOWN, EXPIRED)

TRACK_PREFIX = "TRACK-"
_MAX_CONFLICT_EXAMPLES = 32
_MAX_RECENT_EMBEDDINGS = 8


@dataclass(frozen=True, eq=False)
class SpeakerTrack:
    """Immutable view of one continuous speech track."""

    track_id: str
    speaker_id: Optional[str]
    state: str
    status: str
    similarity: Optional[float]
    margin: Optional[float]
    confidence: float
    created_at: float
    updated_at: float
    last_seen_at: float
    turns: int
    matched_turns: int
    ambiguous_turns: int
    voiced_seconds: float
    utterance_ids: tuple[int, ...]
    superseded_by: Optional[str] = None

    @property
    def permanent(self) -> bool:
        return self.speaker_id is not None and self.state in (
            TENTATIVE,
            STABLE,
        )

    @property
    def active(self) -> bool:
        return self.state != EXPIRED


@dataclass
class _TrackState:
    track_id: str
    speaker_id: Optional[str]
    state: str
    status: str
    similarity: Optional[float]
    margin: Optional[float]
    confidence: float
    created_at: float
    updated_at: float
    last_seen_at: float
    turns: int = 0
    matched_turns: int = 0
    ambiguous_turns: int = 0
    voiced_seconds: float = 0.0
    utterance_ids: list[int] = None  # type: ignore[assignment]
    embeddings: deque = None  # type: ignore[assignment]
    counted_utterances: set[int] = None  # type: ignore[assignment]
    superseded_by: Optional[str] = None

    def __post_init__(self) -> None:
        if self.utterance_ids is None:
            self.utterance_ids = []
        if self.embeddings is None:
            self.embeddings = deque(maxlen=_MAX_RECENT_EMBEDDINGS)
        if self.counted_utterances is None:
            self.counted_utterances = set()


@dataclass
class _Observation:
    """Per-utterance evidence stored until ASR finalisation.

    Mutable by design: the engine may analyse the same utterance more than
    once, and later windows refine the status and spoken duration.
    """

    utterance_id: int
    track_id: str
    status: str
    similarity: Optional[float]
    margin: Optional[float]
    voiced_seconds: float
    usable: bool


class SpeakerTrackManager:
    """Owns all live tracks, expiry and the tentative continuity buffer."""

    def __init__(
        self,
        *,
        expiry_seconds: float = 15.0,
        recent_turns: int = 4,
        min_turns: int = 2,
        min_voiced_seconds: float = 3.0,
        continuity_threshold: float = 0.43,
        max_tracks: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.expiry_seconds = max(1.0, float(expiry_seconds))
        self.recent_turns = max(1, int(recent_turns))
        self.min_turns = max(1, int(min_turns))
        self.min_voiced_seconds = max(0.0, float(min_voiced_seconds))
        self.continuity_threshold = float(continuity_threshold)
        self.max_tracks = max(1, int(max_tracks))
        self._clock = clock
        self._lock = threading.RLock()
        self._tracks: list[_TrackState] = []
        self._current: Optional[str] = None
        self._observations: dict[int, _Observation] = {}
        self._sequence = 0
        self.transitions = 0
        self.conflicts = 0
        self.expirations = 0
        self.superseded = 0

    # ------------------------------------------------------------- predict
    def can_persist(
        self,
        track: Optional[SpeakerTrack],
        *,
        extra_turns: int = 1,
        extra_voiced_seconds: float = 0.0,
    ) -> bool:
        """Whether this tentative track may become a permanent identity.

        The current turn is counted as ``extra_turns`` because it has not been
        finalised yet; without that, a two-turn requirement would need three
        utterances (the first one only counts after ASR finishes).
        """
        if track is None or track.speaker_id is not None:
            return False
        if track.state not in (TENTATIVE, STABLE):
            return False
        turns = track.turns + max(0, int(extra_turns))
        voiced = track.voiced_seconds + max(0.0, float(extra_voiced_seconds))
        return turns >= self.min_turns and voiced >= self.min_voiced_seconds

    # -------------------------------------------------------------- observe
    def observe(
        self,
        utterance_id: int,
        *,
        speaker_id: Optional[str],
        status: str,
        similarity: Optional[float],
        margin: Optional[float],
        voiced_seconds: float,
        quality_rank: int = 2,
        vector=None,
        final: bool = False,
    ) -> Optional[SpeakerTrack]:
        """Record one window. Repeated calls for one utterance refine it.

        ``final=False`` is an interim analysis: it may link a tentative track
        but never counts a turn. ``final=True`` (from the engine's learning
        commit) counts at most one turn per utterance.
        """
        uid = int(utterance_id)
        now = self._clock()
        normalized = None
        if vector is not None:
            try:
                normalized = normalize_embedding(vector)
            except VoiceProfileError:
                normalized = None
        with self._lock:
            self._expire_locked(now)
            observation = self._observations.get(uid)
            if observation is None or observation.track_id is None:
                track = self._select_track_locked(
                    speaker_id=speaker_id,
                    status=status,
                    vector=normalized,
                    now=now,
                )
                observation = _Observation(
                    utterance_id=uid,
                    track_id=track.track_id if track is not None else "",
                    status=status,
                    similarity=similarity,
                    margin=margin,
                    voiced_seconds=float(voiced_seconds),
                    usable=quality_rank >= 2,
                )
                self._observations[uid] = observation
            else:
                track = self._by_id_locked(observation.track_id)
                if track is None:
                    track = self._select_track_locked(
                        speaker_id=speaker_id,
                        status=status,
                        vector=normalized,
                        now=now,
                    )
                    observation.track_id = (
                        track.track_id if track is not None else ""
                    )
                observation.status = status
                observation.similarity = similarity
                observation.margin = margin
                observation.voiced_seconds = max(
                    observation.voiced_seconds, float(voiced_seconds)
                )
                observation.usable = observation.usable or quality_rank >= 2

            if track is None:
                return None
            self._refresh_locked(
                track,
                speaker_id=speaker_id,
                status=status,
                similarity=similarity,
                margin=margin,
                now=now,
                normalized=normalized,
                uid=uid,
            )
            if final:
                self._count_turn_locked(track, observation, uid, now)
            self._set_current_locked(track)
            self._prune_tracks_locked()
            return self._snapshot(track)

    def _select_track_locked(
        self,
        *,
        speaker_id: Optional[str],
        status: str,
        vector,
        now: float,
    ) -> Optional[_TrackState]:
        live = [track for track in self._tracks if track.state != EXPIRED]
        if speaker_id is not None:
            for track in reversed(live):
                if track.speaker_id == speaker_id:
                    return track
            track = self._new_track_locked(
                speaker_id=speaker_id,
                state=STABLE,
                status=status,
                now=now,
            )
            self._supersede_others_locked(track, now)
            return track

        if status == "NEW_CANDIDATE":
            if vector is not None:
                best = None
                best_score = self.continuity_threshold
                for candidate in reversed(live):
                    if candidate.speaker_id is not None:
                        continue
                    if candidate.state not in (TENTATIVE, STABLE):
                        continue
                    for stored in candidate.embeddings:
                        score = cosine(vector, stored)
                        if score >= best_score:
                            best_score = score
                            best = candidate
                if best is not None:
                    return best
            track = self._new_track_locked(
                speaker_id=None,
                state=TENTATIVE,
                status=status,
                now=now,
            )
            self._supersede_others_locked(track, now)
            return track

        # Ambiguous / insufficient / error: keep the current live track if it
        # exists, otherwise remember that this turn had no usable identity.
        if self._current is not None:
            track = self._by_id_locked(self._current)
            if track is not None and track.state != EXPIRED:
                return track
        if status == "AMBIGUOUS":
            track = self._new_track_locked(
                speaker_id=None,
                state=UNKNOWN,
                status=status,
                now=now,
            )
            self._supersede_others_locked(track, now)
            return track
        return None

    def _new_track_locked(
        self, *, speaker_id: Optional[str], state: str, status: str, now: float
    ) -> _TrackState:
        self._sequence += 1
        track = _TrackState(
            track_id=f"{TRACK_PREFIX}{self._sequence:04d}",
            speaker_id=speaker_id,
            state=state,
            status=status,
            similarity=None,
            margin=None,
            confidence=0.0,
            created_at=now,
            updated_at=now,
            last_seen_at=now,
        )
        self._tracks.append(track)
        return track

    def _refresh_locked(
        self,
        track: _TrackState,
        *,
        speaker_id: Optional[str],
        status: str,
        similarity: Optional[float],
        margin: Optional[float],
        now: float,
        normalized,
        uid: int,
    ) -> None:
        track.status = status
        track.similarity = similarity
        track.margin = margin
        track.updated_at = now
        track.last_seen_at = now
        if uid not in track.utterance_ids:
            track.utterance_ids.append(uid)
        if normalized is not None and track.speaker_id is None:
            if len(track.embeddings) == track.embeddings.maxlen:
                track.embeddings.popleft()
            track.embeddings.append(normalized)
        if speaker_id is not None and track.speaker_id is None:
            self.bind_identity(track.track_id, speaker_id)
        if similarity is not None:
            track.confidence = max(0.0, min(1.0, float(similarity)))
        if status in ("KNOWN", "CANDIDATE_MATCH"):
            track.state = STABLE
        elif status == "AMBIGUOUS" and track.state == TENTATIVE and not track.matched_turns:
            track.state = UNKNOWN

    def _count_turn_locked(
        self,
        track: _TrackState,
        observation: _Observation,
        uid: int,
        now: float,
    ) -> None:
        if uid in track.counted_utterances:
            return
        track.counted_utterances.add(uid)
        track.turns += 1
        if observation.status in ("KNOWN", "CANDIDATE_MATCH"):
            track.matched_turns += 1
            track.voiced_seconds += max(0.0, observation.voiced_seconds)
        elif observation.status == "AMBIGUOUS":
            track.ambiguous_turns += 1
        elif observation.status == "NEW_CANDIDATE" and observation.usable:
            # A tentative new voice is acoustic evidence even without a
            # permanent id; it is what allows the second turn to confirm it.
            track.matched_turns += 1
            track.voiced_seconds += max(0.0, observation.voiced_seconds)
        if track.speaker_id is not None and track.matched_turns >= 1:
            track.state = STABLE
        elif (
            track.speaker_id is None
            and track.matched_turns >= self.min_turns
            and track.voiced_seconds >= self.min_voiced_seconds
            and track.state == TENTATIVE
        ):
            track.state = STABLE
            self.transitions += 1
        track.updated_at = now

    def finalize(
        self, utterance_id: int, *, success: bool = True
    ) -> Optional[SpeakerTrack]:
        """Close one utterance: count its turn only when ASR succeeded."""
        uid = int(utterance_id)
        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            observation = self._observations.pop(uid, None)
            if observation is None:
                return None
            track = self._by_id_locked(observation.track_id)
            if track is None:
                return None
            if success:
                self._count_turn_locked(track, observation, uid, now)
            else:
                self._remove_evidence_locked(track, observation, uid)
            return self._snapshot(track)

    def cancel(self, utterance_id: int) -> None:
        """Drop one utterance's evidence (ASR failure, barge-in, cancel)."""
        uid = int(utterance_id)
        with self._lock:
            observation = self._observations.pop(uid, None)
            if observation is None:
                return
            track = self._by_id_locked(observation.track_id)
            if track is None:
                return
            self._remove_evidence_locked(track, observation, uid)

    def _remove_evidence_locked(
        self, track: _TrackState, observation: _Observation, uid: int
    ) -> None:
        if uid in track.counted_utterances:
            track.counted_utterances.discard(uid)
            track.turns = max(0, track.turns - 1)
            if observation.status in ("KNOWN", "CANDIDATE_MATCH"):
                track.matched_turns = max(0, track.matched_turns - 1)
                track.voiced_seconds = max(
                    0.0, track.voiced_seconds - observation.voiced_seconds
                )
            elif observation.status == "NEW_CANDIDATE" and observation.usable:
                track.matched_turns = max(0, track.matched_turns - 1)
                track.voiced_seconds = max(
                    0.0, track.voiced_seconds - observation.voiced_seconds
                )
        track.utterance_ids = [
            item for item in track.utterance_ids if item != uid
        ]

    def bind_identity(self, track_id: str, speaker_id: str) -> bool:
        """Attach a permanent speaker id to a track after identity creation."""
        with self._lock:
            track = self._by_id_locked(track_id)
            if track is None or track.state == EXPIRED:
                return False
            if track.speaker_id is not None and track.speaker_id != speaker_id:
                self.conflicts += 1
                return False
            track.speaker_id = str(speaker_id)
            track.state = STABLE
            track.updated_at = self._clock()
            return True

    def confirm(
        self, track_id: Optional[str] = None, *, now: Optional[float] = None
    ) -> Optional[SpeakerTrack]:
        """Explicitly confirm a tentative track as STABLE.

        This is a state transition, not new identity evidence: an ``UNKNOWN``
        track (ambiguous audio) cannot be confirmed, and a confirmed track
        still has to satisfy :meth:`can_persist`'s turn/voiced gates before
        it may become a permanent id. With no ``track_id`` the current track
        is confirmed. Confirming a STABLE track is idempotent.
        """
        moment = self._clock() if now is None else float(now)
        with self._lock:
            self._expire_locked(moment)
            selected = track_id if track_id is not None else self._current
            track = self._by_id_locked(selected) if selected else None
            if track is None or track.state in (EXPIRED, UNKNOWN):
                return None
            if track.state != STABLE:
                track.state = STABLE
                track.updated_at = moment
                self.transitions += 1
            return self._snapshot(track)

    # ------------------------------------------------------------- queries
    def current_track(self, *, now: Optional[float] = None) -> Optional[SpeakerTrack]:
        moment = self._clock() if now is None else float(now)
        with self._lock:
            self._expire_locked(moment)
            if self._current is None:
                return None
            track = self._by_id_locked(self._current)
            if track is None or track.state == EXPIRED:
                return None
            return self._snapshot(track)

    def tracks(self) -> list[SpeakerTrack]:
        with self._lock:
            return [self._snapshot(track) for track in self._tracks]

    def expire(self, *, now: Optional[float] = None) -> list[SpeakerTrack]:
        moment = self._clock() if now is None else float(now)
        with self._lock:
            expired = self._expire_locked(moment)
            return [self._snapshot(track) for track in expired]

    def reset_session(self) -> None:
        """Forget every live track. The id sequence is deliberately kept, so a
        later track never reuses a ``TRACK-`` id in the same process."""
        with self._lock:
            self._tracks.clear()
            self._observations.clear()
            self._current = None

    def stats(self) -> dict:
        with self._lock:
            counts = {state: 0 for state in TRACK_STATES}
            for track in self._tracks:
                counts[track.state] = counts.get(track.state, 0) + 1
            current = (
                self._snapshot(self._by_id_locked(self._current))
                if self._current is not None
                and self._by_id_locked(self._current) is not None
                else None
            )
            return {
                "state_counts": counts,
                "tracks": len(self._tracks),
                "current_track_id": (
                    current.track_id if current is not None else None
                ),
                "current_state": current.state if current is not None else None,
                "current_speaker_id": (
                    current.speaker_id if current is not None else None
                ),
                "transitions": self.transitions,
                "conflicts": self.conflicts,
                "expirations": self.expirations,
                "superseded": self.superseded,
                "min_turns": self.min_turns,
                "min_voiced_seconds": self.min_voiced_seconds,
                "expiry_seconds": self.expiry_seconds,
            }

    # ------------------------------------------------------------- internals
    def _supersede_others_locked(
        self, keep: _TrackState, now: float
    ) -> None:
        for track in self._tracks:
            if track is keep or track.state == EXPIRED:
                continue
            track.state = EXPIRED
            track.superseded_by = keep.track_id
            track.updated_at = now
            self.superseded += 1

    def _expire_locked(self, now: float) -> list[_TrackState]:
        expired = []
        for track in self._tracks:
            if track.state == EXPIRED:
                continue
            if now - track.last_seen_at >= self.expiry_seconds:
                track.state = EXPIRED
                track.updated_at = now
                expired.append(track)
                self.expirations += 1
                if self._current == track.track_id:
                    self._current = None
        return expired

    def _set_current_locked(self, track: _TrackState) -> None:
        if track.state != EXPIRED:
            self._current = track.track_id

    def _by_id_locked(self, track_id: str) -> Optional[_TrackState]:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def _prune_tracks_locked(self) -> None:
        if len(self._tracks) <= self.max_tracks:
            return
        for track in list(self._tracks):
            if len(self._tracks) <= self.max_tracks:
                return
            if track.state == EXPIRED:
                self._tracks.remove(track)
                if self._current == track.track_id:
                    self._current = None

    @staticmethod
    def _snapshot(track: _TrackState) -> SpeakerTrack:
        return SpeakerTrack(
            track_id=track.track_id,
            speaker_id=track.speaker_id,
            state=track.state,
            status=track.status,
            similarity=track.similarity,
            margin=track.margin,
            confidence=track.confidence,
            created_at=track.created_at,
            updated_at=track.updated_at,
            last_seen_at=track.last_seen_at,
            turns=track.turns,
            matched_turns=track.matched_turns,
            ambiguous_turns=track.ambiguous_turns,
            voiced_seconds=track.voiced_seconds,
            utterance_ids=tuple(track.utterance_ids),
            superseded_by=track.superseded_by,
        )


@dataclass(frozen=True, eq=False)
class PendingSample:
    """One unverified embedding kept in memory only."""

    pending_id: int
    track_id: str
    vector: object
    quality: str
    voiced_seconds: float
    created_at: float


class PendingSampleBuffer:
    """Bounded, expiring memory for samples that are not trustworthy yet.

    Deliberately small and deliberately not connected to the matcher: pending
    samples never influence a match, are never written to SQLite and are
    dropped on expiry or overflow. They exist so diagnostics can show *what
    was almost learnt* without letting it pollute the identity.
    """

    def __init__(
        self,
        *,
        max_count: int = 12,
        expiry_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_count = max(1, int(max_count))
        self.expiry_seconds = max(1.0, float(expiry_seconds))
        self._clock = clock
        self._lock = threading.RLock()
        self._items: deque[PendingSample] = deque()
        self._sequence = 0
        self.dropped = 0
        self.expired = 0

    def add(
        self,
        track_id: str,
        vector,
        *,
        quality: str = "MEDIUM",
        voiced_seconds: float = 0.0,
    ) -> Optional[PendingSample]:
        with self._lock:
            self._prune_locked(self._clock())
            try:
                normalized = normalize_embedding(vector)
            except VoiceProfileError:
                return None
            self._sequence += 1
            sample = PendingSample(
                pending_id=self._sequence,
                track_id=str(track_id),
                vector=normalized,
                quality=str(quality),
                voiced_seconds=float(voiced_seconds),
                created_at=self._clock(),
            )
            self._items.append(sample)
            while len(self._items) > self.max_count:
                self._items.popleft()
                self.dropped += 1
            return sample

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.expiry_seconds
        while self._items and self._items[0].created_at < cutoff:
            self._items.popleft()
            self.expired += 1

    def prune(self) -> int:
        with self._lock:
            before = len(self._items)
            self._prune_locked(self._clock())
            return before - len(self._items)

    def items(self) -> tuple[PendingSample, ...]:
        with self._lock:
            self._prune_locked(self._clock())
            return tuple(self._items)

    def drop_track(self, track_id: str) -> int:
        """Forget the staged samples of a track that became a real identity."""
        with self._lock:
            before = len(self._items)
            self._items = deque(
                item
                for item in self._items
                if item.track_id != str(track_id)
            )
            return before - len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> dict:
        with self._lock:
            self._prune_locked(self._clock())
            return {
                "count": len(self._items),
                "max_count": self.max_count,
                "dropped": self.dropped,
                "expired": self.expired,
                "expiry_seconds": self.expiry_seconds,
            }


__all__ = [
    "EXPIRED",
    "PendingSample",
    "PendingSampleBuffer",
    "STABLE",
    "TENTATIVE",
    "TRACK_PREFIX",
    "TRACK_STATES",
    "UNKNOWN",
    "SpeakerTrack",
    "SpeakerTrackManager",
]
