"""V0.6 background analysis: durable, idempotent, ordered and retryable."""

from __future__ import annotations

import json

from companion.domain.enums import EventSource, JobStatus, JobType
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

ANALYSIS_JSON = json.dumps(
    {
        "meaning": "用户在说自己喜欢猫",
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
        self.analysis_messages: list[tuple[str | None, list[dict]]] = []
        self.analysis_failures = analysis_failures

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        if request_type == "analysis":
            self.analysis_messages.append((event_id, messages))
            if self.analysis_failures:
                self.analysis_failures -= 1
                raise RuntimeError("scripted analysis failure")
            content = ANALYSIS_JSON
        else:
            content = "我也喜欢猫。"
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _runtime(db, client):
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
        text="我喜欢猫",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    return rt, event


async def test_fast_path_learning_runs_as_a_durable_job(db):
    client = FakeClient()
    rt, event = await _runtime(db, client)
    report = await rt.orchestrator.process_utterance(event)

    job = await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    assert job is not None
    assert job.status == JobStatus.PENDING
    assert job.job_type == JobType.TURN_ANALYSIS

    await rt.worker.run_until_idle()

    job = await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    assert job is not None and job.status == JobStatus.COMPLETED
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.provider == "deepseek"
    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert len(candidates) == 1
    # The reply happened before the analysis, and the analysis is the only
    # extra model call.
    assert client.calls == ["response", "analysis"]


async def test_re_running_the_job_is_idempotent(db):
    client = FakeClient()
    rt, event = await _runtime(db, client)
    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_until_idle()
    calls_after_first = list(client.calls)

    # A duplicate enqueue keeps the same key and does not create new work.
    await rt.repos.job.enqueue(
        JobType.TURN_ANALYSIS,
        f"turn_analysis:{event.event_id}",
        {"event_id": event.event_id},
    )
    await rt.worker.run_until_idle()

    assert client.calls == calls_after_first
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.provider == "deepseek"


async def test_job_is_retried_after_a_transient_failure(db):
    client = FakeClient(analysis_failures=1)
    rt, event = await _runtime(db, client)
    await rt.orchestrator.process_utterance(event)

    await rt.worker.run_until_idle()

    job = await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    assert job is not None and job.status == JobStatus.COMPLETED
    # One failed attempt, one successful attempt, plus the Fast Path response.
    assert client.calls == ["response", "analysis", "analysis"]
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.provider == "deepseek"


async def test_job_uses_history_only_before_the_event(db):
    client = FakeClient()
    rt, event = await _runtime(db, client)
    await rt.orchestrator.process_utterance(event)

    later = make_utterance_event(
        conversation_id="c1",
        text="后来说的话",
        addressee_hint="agent",
        speaker_person_id=event.speaker_person_id,
        source=EventSource.TEST,
    )
    await rt.orchestrator.process_utterance(later)

    await rt.worker.run_until_idle()
    # The deferred analysis of the first event sees only what came before it:
    # the later turn is not part of that event's history.
    first_event_prompts = [
        messages
        for call_id, messages in client.analysis_messages
        if call_id == event.event_id
    ]
    assert first_event_prompts
    for messages in first_event_prompts:
        assert "后来说的话" not in json.dumps(messages, ensure_ascii=False)
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.event_id == event.event_id
    assert analysis.provider == "deepseek"


async def test_consent_revocation_cancels_the_background_analysis(db):
    client = FakeClient()
    rt, event = await _runtime(db, client)
    await rt.orchestrator.process_utterance(event)
    await rt.registry.set_consent(
        Consent(
            person_id=event.speaker_person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=True,
            updated_at=utcnow(),
        )
    )

    await rt.worker.run_until_idle()

    job = await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    assert job is not None and job.status == JobStatus.COMPLETED
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.provider == "local_fast_path"
    assert client.calls == ["response"]
