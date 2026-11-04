"""V0.6 true streaming: unified lifecycle events and truthful partials.

All providers are fakes. The tests never mock the orchestrator: the same
``process_utterance_stream`` API a transport would use is driven directly.
"""

from __future__ import annotations

import asyncio

from companion.domain.enums import CognitiveAction, EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse, StreamDelta
from companion.runtime import build_runtime


class StreamingClient:
    def __init__(
        self,
        *,
        chunks=("你", "好"),
        finish_reason="stop",
        fail_after=None,
        stall_after=None,
    ):
        self.chunks = list(chunks)
        self.finish_reason = finish_reason
        self.fail_after = fail_after
        self.stall_after = stall_after
        self.calls: list[str] = []

    async def complete(self, messages, *, request_type, **kwargs):
        self.calls.append(request_type)
        return ChatResponse(
            content="".join(self.chunks),
            model="deepseek-flash",
            finish_reason=self.finish_reason,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    async def stream_completion(self, messages, *, request_type="response", **kwargs):
        self.calls.append(request_type)
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("scripted stream failure")
            if self.stall_after is not None and index >= self.stall_after:
                await asyncio.sleep(30)
            yield StreamDelta(content=chunk)
        yield StreamDelta(
            finish_reason=self.finish_reason,
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
            usage_reported=True,
        )


async def _setup(db, client):
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="你好",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    return rt, event


async def test_stream_emits_started_deltas_and_completed(db):
    client = StreamingClient(chunks=("你", "好"))
    rt, event = await _setup(db, client)

    events = [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]

    assert [e.type for e in events] == [
        "response_started",
        "response_delta",
        "response_delta",
        "response_completed",
    ]
    assert [e.sequence for e in events] == [0, 1, 2, 3]
    assert "".join(e.delta or "" for e in events) == "你好"
    completed = events[-1]
    assert completed.status == "COMPLETED"
    assert completed.text == "你好"
    assert completed.action == "RESPOND"
    assert completed.report_status == "COMPLETED"
    assert completed.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    # Every event carries stable identity for reassembly.
    assert {e.event_id for e in events} == {event.event_id}
    assert {e.conversation_id for e in events} == {event.conversation_id}
    assert len({e.response_id for e in events}) == 1

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None and action.status == "COMPLETED"
    assert action.response_text == "你好"

    # Latency metrics are content-free and expose P50/P95 per stage.
    snapshot = rt.metrics.snapshot()
    assert snapshot["time_to_first_token_ms"]["count"] == 1
    ttft = snapshot["time_to_first_token_ms"]
    assert 0.0 <= ttft["p50_ms"] <= ttft["p95_ms"] <= ttft["max_ms"]


async def test_token_limit_is_reported_as_truncated(db):
    client = StreamingClient(chunks=("半句",), finish_reason="length")
    rt, event = await _setup(db, client)

    events = [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]

    assert events[-1].type == "response_completed"
    assert events[-1].status == "TRUNCATED"
    assert events[-1].finish_reason == "length"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None and action.status == "TRUNCATED"


async def test_mid_stream_failure_is_failed_not_completed(db):
    client = StreamingClient(chunks=("你", "好", "呀"), fail_after=1)
    rt, event = await _setup(db, client)

    events = [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]

    assert [e.type for e in events] == [
        "response_started",
        "response_delta",
        "response_failed",
    ]
    failed = events[-1]
    assert failed.status == "FAILED"
    assert failed.text == "你"  # exactly what was shown
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None and action.status == "FAILED"
    assert action.response_text == "你"
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status.value == "FAILED"


async def test_cancellation_persists_a_partial_and_never_completes(db):
    client = StreamingClient(chunks=("你", "好"), stall_after=1)
    rt, event = await _setup(db, client)

    stream = rt.orchestrator.process_utterance_stream(event)
    first = await stream.__anext__()
    second = await stream.__anext__()
    assert first.type == "response_started"
    assert second.type == "response_delta"
    await stream.aclose()  # consumer disconnects

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None
    assert action.status == "CANCELLED"
    assert action.response_text == "你"
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status.value == "FAILED"


async def test_interrupted_partial_is_not_reused_by_a_retry(db):
    failing = StreamingClient(chunks=("你", "好"), fail_after=1)
    rt, event = await _setup(db, failing)
    events = [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]
    assert events[-1].type == "response_failed"

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None and action.status == "FAILED"

    # A fresh turn on the same event id now succeeds: the partial was
    # discarded, not mistaken for a finished reply.
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    assert report.response_text == "你好"
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None and action.status == "COMPLETED"


async def test_non_stream_call_still_uses_the_blocking_complete(db):
    client = StreamingClient(chunks=("好",))
    rt, event = await _setup(db, client)

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.RESPOND
    assert report.response_text == "好"
    assert client.calls == ["response"]


async def test_non_stream_token_limit_is_persisted_as_truncated(db):
    # The blocking path must not claim a cut-off reply is complete either.
    client = StreamingClient(chunks=("半句",), finish_reason="length")
    rt, event = await _setup(db, client)

    await rt.orchestrator.process_utterance(event)

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None
    assert action.status == "TRUNCATED"
    assert action.response_text == "半句"
