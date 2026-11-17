"""V0.9.9.31: asynchronous speaker updates and run-scoped voice turns.

The speaker model runs on its own worker. Results cross into the asyncio loop
through the bridge, are coalesced per utterance, and can never label a newer
microphone session. These tests use fake ASR managers only (no model, mic,
network or real audio).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from companion.asr.manager import FinalUtterance
from companion.speaker_id.models import (
    KNOWN,
    PENDING,
    QUALITY_HIGH,
    SpeakerResult,
)
from companion.voice import VoiceCoordinator
from companion.voice.bridge import SpeakerUpdate, VoiceBridge
from test_voice_integration import FakeASR


def snapshot(speaker_id="S-0001", *, status=KNOWN, uid=1) -> SpeakerResult:
    return SpeakerResult(
        utterance_id=uid,
        speaker_id=speaker_id,
        person_id=None,
        status=status,
        reid_quality=QUALITY_HIGH,
        similarity=0.7,
        margin=0.2,
        voiced_seconds=2.0,
        provisional=False,
        model_id="fake",
        inference_ms=1.0,
        completed_at=time.monotonic(),
    )


def update(run_id, uid, revision, *, status=KNOWN, speaker_id="S-0001"):
    return SpeakerUpdate(
        run_id=run_id,
        utterance_id=uid,
        revision=revision,
        result=snapshot(speaker_id, status=status, uid=uid),
    )


async def settle(rounds: int = 4) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0.01)


# --------------------------------------------------------------- bridge
async def test_latest_revision_reaches_the_loop():
    bridge = VoiceBridge(asyncio.get_running_loop(), run_id="run-a")
    try:
        bridge.publish_speaker_update(update("run-a", 7, 1))
        bridge.publish_speaker_update(update("run-a", 7, 2))
        bridge.publish_speaker_update(update("run-a", 7, 3))

        item = await asyncio.wait_for(bridge.next_item(), 1.0)
        assert isinstance(item, SpeakerUpdate)
        assert (item.run_id, item.utterance_id, item.revision) == (
            "run-a",
            7,
            3,
        )
    finally:
        bridge.close()


async def test_an_older_revision_never_overwrites_a_newer_one():
    bridge = VoiceBridge(asyncio.get_running_loop(), run_id="run-a")
    try:
        bridge.publish_speaker_update(update("run-a", 7, 2))
        first = await asyncio.wait_for(bridge.next_item(), 1.0)
        assert first.revision == 2

        bridge.publish_speaker_update(update("run-a", 7, 1))
        await settle()
        assert bridge.stale_speaker == 1
        assert bridge._queue.qsize() == 0
    finally:
        bridge.close()


async def test_a_previous_microphone_run_cannot_label_a_new_session():
    bridge = VoiceBridge(asyncio.get_running_loop(), run_id="run-b")
    try:
        # Delayed result from the session that already ended.
        bridge.publish_speaker_update(update("run-a", 7, 9))
        # The new session's own speech is already queued.
        bridge.publish_final(FinalUtterance(7, "new session", time.monotonic()))

        item = await asyncio.wait_for(bridge.next_item(), 1.0)
        assert isinstance(item, FinalUtterance)
        assert item.text == "new session"
        assert bridge.stale_speaker == 1
    finally:
        bridge.close()


async def test_identity_hints_never_crowd_out_a_final():
    bridge = VoiceBridge(
        asyncio.get_running_loop(), max_pending=1, run_id="run-a"
    )
    try:
        # The single slot is reserved for critical ASR events, so this hint is
        # dropped instead of occupying the only position.
        bridge.publish_speaker_update(update("run-a", 7, 1))
        bridge.publish_final(FinalUtterance(7, "hello", time.monotonic()))

        item = await asyncio.wait_for(bridge.next_item(), 1.0)
        assert isinstance(item, FinalUtterance)
        assert item.text == "hello"
        assert bridge.dropped_speaker == 1
    finally:
        bridge.close()


async def test_pending_identity_hints_are_bounded():
    bridge = VoiceBridge(asyncio.get_running_loop(), run_id="run-a")
    try:
        for uid in range(200):
            bridge.publish_speaker_update(update("run-a", uid, 1))
        await settle()
        assert len(bridge._speaker_pending) <= 64
        assert bridge.dropped_speaker > 0
    finally:
        bridge.close()


# ------------------------------------------------------------ coordinator
class SpeakerASR(FakeASR):
    """FakeASR that also implements the formal observer callback contract."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.repl_run_id = run_id
        self.speaker_cb = None

    def set_speaker_update_callback(self, callback) -> None:
        self.speaker_cb = callback

    def speaker(self, run_id: str, uid: int, revision: int) -> None:
        assert self.speaker_cb is not None
        self.speaker_cb(update(run_id, uid, revision))


async def make_coordinator(run_ids):
    fakes = []

    def factory():
        fake = SpeakerASR(run_ids[len(fakes)])
        fakes.append(fake)
        return fake

    coordinator = VoiceCoordinator(
        manager_factory=factory,
        tts_manager=None,
        cooldown_seconds=0.0,
    )
    return coordinator, fakes


async def test_coordinator_accepts_only_the_newest_revision():
    coordinator, fakes = await make_coordinator(["run-a"])
    assert (await coordinator.enable())[0]
    try:
        fakes[0].speaker("run-a", 7, 1)
        await settle()
        fakes[0].speaker("run-a", 7, 2)
        await settle()

        latest = coordinator.speaker_update("run-a", 7)
        assert latest is not None and latest.revision == 2

        # An old revision is ignored, not applied on top.
        fakes[0].speaker("run-a", 7, 1)
        await settle()
        assert coordinator.speaker_update("run-a", 7).revision == 2
        # The bridge filters it before it even reaches the coordinator.
        assert coordinator._bridge.stale_speaker >= 1
    finally:
        await coordinator.shutdown()


async def test_speaker_updates_never_produce_a_second_turn():
    coordinator, fakes = await make_coordinator(["run-a"])
    assert (await coordinator.enable())[0]
    try:
        fakes[0].speaker("run-a", 7, 1)
        fakes[0].speech(7)
        await settle(1)
        fakes[0].final(7, "hello")
        await settle()

        accepted = await asyncio.wait_for(
            coordinator.next_accepted_turn(), 1.0
        )
        assert accepted.run_id == "run-a"
        assert accepted.final.text == "hello"

        # A later identity revision annotates the turn; it must not enqueue a
        # second reply or a duplicate final.
        fakes[0].speaker("run-a", 7, 2)
        await settle()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coordinator.next_accepted_turn(), 0.1)
    finally:
        await coordinator.shutdown()


async def test_mic_restart_isolates_utterance_ids_by_run():
    coordinator, fakes = await make_coordinator(["run-a", "run-b"])
    assert (await coordinator.enable())[0]
    try:
        fakes[0].speaker("run-a", 7, 4)
        await settle()
        assert coordinator.speaker_update("run-a", 7).revision == 4
    finally:
        await coordinator.disable()

    assert (await coordinator.enable())[0]
    try:
        # Utterance ids restart at 7 in session B. A delayed update from A is
        # delivered to B's bridge (simulating an old worker callback) and must
        # be discarded, never applied to B's utterance 7.
        fakes[1].speaker("run-a", 7, 9)
        fakes[1].speaker("run-b", 7, 1)
        await settle()
        assert coordinator.speaker_update("run-a", 7) is None
        assert coordinator.speaker_update("run-b", 7).revision == 1
    finally:
        await coordinator.shutdown()


async def test_speaker_cache_is_bounded():
    coordinator, fakes = await make_coordinator(["run-a"])
    assert (await coordinator.enable())[0]
    try:
        for uid in range(200):
            coordinator._accept_speaker_update(update("run-a", uid, 1))
        assert len(coordinator._speaker_updates) <= 64
        assert coordinator.speaker_timings()["evicted"] > 0
    finally:
        await coordinator.shutdown()


async def test_pending_snapshot_never_falls_back_to_a_person():
    coordinator, fakes = await make_coordinator(["run-a"])
    assert (await coordinator.enable())[0]
    try:
        fakes[0].speaker("run-a", 7, 1)
        await settle()
        cached = coordinator.speaker_update("run-a", 7)
        assert cached.result.status == KNOWN
        assert cached.result.person_id is None

        # A PENDING snapshot is a valid cache entry but carries no identity.
        pending = SpeakerUpdate(
            "run-a", 8, 1, snapshot(status=PENDING, uid=8)
        )
        fakes[0].speaker_cb(pending)
        await settle()
        assert coordinator.speaker_update("run-a", 8).result.status == PENDING
        assert coordinator.speaker_update("run-a", 8).result.person_id is None
    finally:
        await coordinator.shutdown()
