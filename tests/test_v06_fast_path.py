"""V0.6 Fast Path: one model call for a clearly addressed direct turn.

The tests use a fake DeepSeek client (no network, no key). They pin down the
routing rule itself: the decision comes from trusted metadata (addressee),
never from message text, and every local gate still runs.
"""

from __future__ import annotations

import json

from companion.domain.enums import CognitiveAction, EventSource, JobType, ReasonCode
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

ANALYSIS_JSON = json.dumps(
    {
        "meaning": "用户在直接提问",
        "action": "RESPOND",
        "reason_code": "direct_agent_address",
        "referenced_memory_ids": [],
        "memory_candidates": [{"content": "Alice 喜欢猫"}],
        "response_style": "warm",
        "reply_necessity": "REQUIRED",
    },
    ensure_ascii=False,
)


class FakeClient:
    def __init__(self, *, analysis_failures=0):
        self.calls: list[str] = []
        self.analysis_failures = analysis_failures

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        if request_type == "analysis":
            if self.analysis_failures:
                self.analysis_failures -= 1
                raise RuntimeError("scripted analysis failure")
            content = ANALYSIS_JSON
            finish = "stop"
        else:
            content = "我在。"
            finish = "stop"
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason=finish,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _person(rt, *, external_ai=True):
    return await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
    )


def _event(person_id, text="你好", *, to="agent"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def test_direct_turn_makes_one_response_call(db):
    client = FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(_event(person.person_id))

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    assert report.reason_code == ReasonCode.DIRECT_AGENT_ADDRESS
    assert client.calls == ["response"]
    analysis = await rt.repos.analysis.get_analysis(report.event_id)
    assert analysis is not None and analysis.provider == "local_fast_path"
    # Stage metrics are recorded without any content.
    assert {"context_ms", "analysis_ms", "generation_ms", "total_ms"} <= set(
        report.timings
    )


async def test_latency_metrics_expose_p50_and_p95(db):
    client = FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)

    await rt.orchestrator.process_utterance(_event(person.person_id))
    await rt.orchestrator.process_utterance(_event(person.person_id, "又一个问题"))

    snapshot = rt.metrics.snapshot()
    # A stage entry has count and percentiles, and nothing else.
    entry = snapshot["total_ms"]
    assert set(entry) == {"count", "p50_ms", "p95_ms", "max_ms"}
    assert entry["count"] == 2
    assert 0.0 <= entry["p50_ms"] <= entry["p95_ms"] <= entry["max_ms"]


async def test_text_mentioning_ava_does_not_force_the_fast_path(db):
    client = FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)

    # The name is not a routing signal: an unclear addressee still deliberates
    # even though the text literally contains "Ava".
    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "Ava，你觉得呢？", to="unknown")
    )

    assert report.status == "COMPLETED"
    assert client.calls == ["analysis", "response"]


async def test_explicit_remember_uses_deliberation_and_waits_for_the_write(db):
    client = FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(
        _event(person.person_id, "记住我喜欢猫")
    )

    assert report.status == "COMPLETED"
    assert client.calls == ["analysis", "response"]
    # The explicit write finished before the reply, and no deferred analysis
    # job was queued for a turn that already had a real analysis.
    assert await rt.repos.job.get_by_key(
        f"turn_analysis:{report.event_id}"
    ) is None
    outcomes = report.memory_write_outcomes
    assert outcomes and outcomes[0]["outcome"] in (
        "SAVED",
        "CREATED",
        "REINFORCED",
    )


async def test_fast_path_is_not_taken_without_external_ai_consent(db):
    client = FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt, external_ai=False)

    report = await rt.orchestrator.process_utterance(_event(person.person_id))

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.WAIT
    assert report.reason_code == ReasonCode.PERMISSION_DENIED
    assert client.calls == []  # nothing left the machine


async def test_fast_path_queues_a_durable_analysis_job(db):
    client = FakeClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(_event(person.person_id))

    job = await rt.repos.job.get_by_key(f"turn_analysis:{report.event_id}")
    assert job is not None
    assert job.job_type == JobType.TURN_ANALYSIS
    assert job.payload == {"event_id": report.event_id}
