"""V0.6.1 background extraction: durable, batched, idempotent and retryable."""

from __future__ import annotations

import json
import re

from companion.domain.enums import EventSource, JobStatus, JobType
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

_EVENT_ID = re.compile(r"event_id=(\S+)")


def _extraction_json(messages: list[dict]) -> str:
    """Answer an extraction prompt with one candidate per event id."""
    text = "\n".join(str(m.get("content") or "") for m in messages)
    event_ids = _EVENT_ID.findall(text)
    return json.dumps(
        {
            "results": [
                {
                    "event_id": event_id,
                    "memory_candidates": [
                        {
                            "content": "Alice 喜欢猫",
                            "operation": "ADD",
                            "assertion_type": "SELF_REPORT",
                            "certainty": "DIRECT",
                            "topic": "preference",
                            "kind": "PERSONAL_FACT",
                            "evidence_span": "猫",
                        }
                    ],
                }
                for event_id in event_ids
            ]
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
            content = _extraction_json(messages)
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
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        # Tests exercise the batched job without waiting a real minute.
        turn_analysis_delay_seconds=0,
    )
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
    # The truthful local decision record is never replaced by a model claim.
    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis is not None and analysis.provider == "local_fast_path"
    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert len(candidates) == 1
    marked = await rt.repos.extraction.get(event.event_id)
    assert marked is not None and marked["status"] == "processed"
    # The reply happened before the extraction, which is the only extra call.
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
    assert analysis is not None and analysis.provider == "local_fast_path"


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
    assert analysis is not None and analysis.provider == "local_fast_path"


async def test_job_uses_history_only_before_the_event(db):
    client = FakeClient()
    rt, event = await _runtime(db, client)
    await rt.orchestrator.process_utterance(event)
    # Run the batch before the later turn exists, so the first event's request
    # cannot contain words that were spoken after it.
    await rt.worker.run_until_idle()

    later = make_utterance_event(
        conversation_id="c1",
        text="后来说的话",
        addressee_hint="agent",
        speaker_person_id=event.speaker_person_id,
        source=EventSource.TEST,
    )
    await rt.orchestrator.process_utterance(later)

    await rt.worker.run_until_idle()
    # The deferred extraction of the first event sees only what came before it:
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
    assert analysis.provider == "local_fast_path"


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


async def test_five_waiting_events_share_one_extraction_request(db):
    client = FakeClient()
    rt, first = await _runtime(db, client)
    events = [first]
    for index in range(1, 5):
        events.append(
            make_utterance_event(
                conversation_id="c1",
                text=f"我还喜欢第{index}只猫",
                addressee_hint="agent",
                speaker_person_id=first.speaker_person_id,
                source=EventSource.TEST,
            )
        )
    for event in events:
        assert (await rt.orchestrator.process_utterance(event)).status == "COMPLETED"
    assert client.calls == ["response"] * 5

    await rt.worker.run_until_idle()

    analysis_calls = [
        messages for _, messages in client.analysis_messages
    ]
    assert len(analysis_calls) == 1
    prompt = json.dumps(analysis_calls[0], ensure_ascii=False)
    for event in events:
        assert f"event_id={event.event_id}" in prompt
    for event in events:
        marked = await rt.repos.extraction.get(event.event_id)
        assert marked is not None and marked["status"] == "processed"


async def test_small_batch_is_delayed_until_it_fills_or_times_out(db):
    client = FakeClient()
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        turn_analysis_delay_seconds=60,
    )
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
    await rt.orchestrator.process_utterance(event)

    job = await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    assert job is not None and job.status == JobStatus.PENDING
    assert job.next_run_at is not None

    # A delayed job is invisible to the worker until its due time.
    await rt.worker.run_until_idle()
    assert "analysis" not in client.calls

    await db.conn.execute(
        "UPDATE processing_jobs SET next_run_at = NULL WHERE job_id = ?",
        (job.job_id,),
    )
    await db.conn.commit()
    await rt.worker.run_until_idle()
    assert client.calls.count("analysis") == 1
