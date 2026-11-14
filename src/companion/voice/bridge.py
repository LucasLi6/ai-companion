"""Bounded, thread-safe delivery from ASR threads to the asyncio loop.

The ASR manager runs its own capture/VAD/session threads. It must never touch
the conversation, Router, Speaker or database directly, and it must never
``await`` anything: its callbacks are synchronous. This bridge is the only
crossing point:

* the manager callback calls :meth:`VoiceBridge.publish_final` (or one of the
  other ``publish_*`` methods) from the ASR control thread;
* the bridge hands the event to the event loop with
  ``loop.call_soon_threadsafe`` and never blocks the ASR thread;
* :meth:`next_item` is consumed by exactly one asyncio task.

The queue is bounded. Duplicate finals/turn-close notices for the same
``utterance_id`` are dropped on the loop thread, so a late retry can never
start a second conversation turn. Dropped events are counted rather than
accumulated.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Union

from companion.asr.manager import (
    FinalUtterance,
    PartialTranscript,
    SpeechActivity,
)


@dataclass(frozen=True)
class TurnClosed:
    """The ASR manager finished (or aborted) one utterance.

    ``ok`` is true only when a final text was delivered. It lets the voice
    coordinator recover from an error/empty/aborted turn instead of waiting
    forever for a final that will never arrive.
    """

    utterance_id: int
    ok: bool


@dataclass(frozen=True)
class BridgeFailure:
    """The ASR worker thread died unexpectedly (display-safe message)."""

    message: str


VoiceItem = Union[
    FinalUtterance,
    SpeechActivity,
    TurnClosed,
    BridgeFailure,
    PartialTranscript,
]

_STOP = None
_SEEN_LIMIT = 256


class VoiceBridge:
    """One bounded queue from ASR threads to one asyncio event loop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        max_pending: int = 32,
    ) -> None:
        self._loop = loop
        self._queue: "asyncio.Queue[Optional[VoiceItem]]" = asyncio.Queue(
            maxsize=max(1, int(max_pending))
        )
        self._closed = False
        self._lock = threading.Lock()
        # Dedupe runs on the loop thread (``_deliver``); the sets are only
        # touched there, never by an ASR thread.
        self._seen_finals: deque[int] = deque(maxlen=_SEEN_LIMIT)
        self._seen_final_set: set[int] = set()
        self._seen_closed: deque[int] = deque(maxlen=_SEEN_LIMIT)
        self._seen_closed_set: set[int] = set()
        self.dropped = 0
        self.dropped_finals = 0
        self.duplicate_finals = 0
        self.dropped_partials = 0
        # At most one partial per utterance is queued at a time. Provisional
        # updates arrive many times per second; without this the bounded queue
        # would fill with stale partials and delay a real final. The newest
        # revision is kept in ``_partial_latest`` and supersedes the queued one
        # (late/older sequences are dropped, never shown out of order).
        self._partial_pending: set[int] = set()
        self._partial_latest: dict[int, PartialTranscript] = {}
        # Highest revision already handed to the consumer. A late, older
        # revision that arrives after its newer sibling was displayed must be
        # dropped instead of moving the subtitle backwards.
        self._partial_delivered: dict[int, int] = {}
        self.replaced_partials = 0

    # -------------------------------------------------------------- publishers
    def publish_final(self, final: FinalUtterance) -> bool:
        """Deliver one completed user turn. Safe from the ASR thread."""
        return self._post(final)

    def publish_speech(self, activity: SpeechActivity) -> bool:
        return self._post(activity)

    def publish_partial(self, partial: PartialTranscript) -> bool:
        """Deliver one provisional transcript line (display only)."""
        return self._post(partial)

    def publish_turn_closed(self, utterance_id: int, ok: bool) -> bool:
        return self._post(TurnClosed(int(utterance_id), bool(ok)))

    def notify_failure(self, message: str) -> bool:
        """Report an ASR worker failure to the loop (called on thread death)."""
        return self._post(BridgeFailure(str(message or "语音识别已停止。")))

    def _post(self, item: VoiceItem) -> bool:
        with self._lock:
            if self._closed:
                return False
        try:
            self._loop.call_soon_threadsafe(self._deliver, item)
            return True
        except RuntimeError:
            # The loop is already closed (shutdown race); nothing to deliver.
            return False

    # ----------------------------------------------------------- loop consumer
    def _deliver(self, item: VoiceItem) -> None:
        if self._closed:
            return
        if isinstance(item, PartialTranscript):
            uid = item.utterance_id
            if item.sequence <= self._partial_delivered.get(uid, -1):
                # An older revision arrived after a newer one (possibly
                # already displayed): never move the subtitle backwards.
                self.dropped_partials += 1
                return
            latest = self._partial_latest.get(uid)
            if latest is not None and latest.sequence > item.sequence:
                # An older revision arrived after a newer one: never move the
                # subtitle backwards.
                self.dropped_partials += 1
                return
            if uid in self._partial_pending:
                # The queued text is replaced by this newer revision. Storing
                # it here (rather than dropping the newest) keeps the display
                # current without growing the bounded queue.
                self._partial_latest[uid] = item
                self.replaced_partials += 1
                return
            self._partial_latest[uid] = item
            self._partial_pending.add(uid)
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                self.dropped += 1
                self._partial_pending.discard(uid)
                self._partial_latest.pop(uid, None)
                self.dropped_partials += 1
            return
        if isinstance(item, FinalUtterance):
            if item.utterance_id in self._seen_final_set:
                self.duplicate_finals += 1
                return
        elif isinstance(item, TurnClosed):
            if item.utterance_id in self._seen_closed_set:
                return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # The consumer is stalled: drop the newest, count it, and keep the
            # queue bounded instead of growing without limit. A Final/close is
            # *not* marked seen, so a retry can still be delivered.
            self.dropped += 1
            if isinstance(item, FinalUtterance):
                self.dropped_finals += 1
            return
        # Mark reliable events seen only after the enqueue succeeded, so a
        # dropped Final is retried instead of being suppressed as a duplicate.
        if isinstance(item, FinalUtterance):
            self._remember(
                item.utterance_id, self._seen_finals, self._seen_final_set
            )
        elif isinstance(item, TurnClosed):
            self._remember(
                item.utterance_id, self._seen_closed, self._seen_closed_set
            )

    @staticmethod
    def _remember(
        uid: int, order: deque[int], seen: set[int]
    ) -> None:
        if len(order) == order.maxlen:
            oldest = order[0]
            # deque.append below evicts ``oldest``; keep the set in sync.
            order.append(uid)
            seen.discard(oldest)
            seen.add(uid)
        else:
            order.append(uid)
            seen.add(uid)

    async def next_item(self) -> Optional[VoiceItem]:
        """Wait for the next item; ``None`` means the bridge was closed."""
        item = await self._queue.get()
        if isinstance(item, PartialTranscript):
            uid = item.utterance_id
            # The slot frees as soon as the consumer has it, so the next
            # revision may be queued while this one is being displayed.
            self._partial_pending.discard(uid)
            latest = self._partial_latest.get(uid)
            self._partial_latest.pop(uid, None)
            if latest is not None and latest.sequence >= item.sequence:
                # A newer revision arrived while this one waited in the queue;
                # supersede it instead of showing stale text first.
                item = latest
            self._partial_delivered[uid] = max(
                self._partial_delivered.get(uid, -1), item.sequence
            )
        return item

    def close(self) -> None:
        """Stop accepting events and wake the consumer (loop thread)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            # Drain one slot so the sentinel is guaranteed to be visible.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(_STOP)
            except asyncio.QueueFull:
                pass

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed


__all__ = [
    "BridgeFailure",
    "PartialTranscript",
    "TurnClosed",
    "VoiceBridge",
    "VoiceItem",
]
