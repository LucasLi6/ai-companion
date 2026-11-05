"""Phase 9 — failure, recovery and idempotency with the real-provider pipeline.

All clients here are fakes. The tests cover the operational guarantees that
matter once real network calls are in the loop:

- a cognitive failure leaves the event retryable and re-runs cognition;
- a response failure keeps the valid analysis, candidates and job, and a retry
  reuses that analysis (no second analysis request);
- duplicate delivery is detected without any further API call;
- RecoveryManager.retry drives the same resumable pipeline;
- neither the API key nor the system prompt is ever written to SQLite.
"""

from __future__ import annotations

import json

from companion.domain.enums import CognitiveAction, EventSource, ProcessingStatus
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import (
    ChatResponse,
    DeepSeekConfig,
    DeepSeekServerError,
)
from companion.prompts.cognitive import SYSTEM_PROMPT
from companion.recovery import RecoveryManager
from companion.runtime import build_runtime

SENTINEL_KEY = "sk-test-sentinel-never-persist"

ANALYSIS_JSON = json.dumps(
    {
        "meaning": "用户在问好",
        "action": "RESPOND",
        "reason_code": "direct_agent_address",
        "referenced_memory_ids": [],
        "memory_candidates": [{"content": "Alice 喜欢川菜"}],
        "response_style": "warm",
        "reply_necessity": "REQUIRED",
    },
    ensure_ascii=False,
)


def _chat(content: str) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="deepseek-flash",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )


class ScriptedClient:
    """Raise a scripted number of failures for analysis and/or response."""

    def __init__(self, *, analysis_failures=0, response_failures=0):
        self.analysis_failures = analysis_failures
        self.response_failures = response_failures
        self.calls: list[str] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        if request_type == "analysis":
            if self.analysis_failures:
                self.analysis_failures -= 1
                raise DeepSeekServerError("scripted analysis failure")
            return _chat(ANALYSIS_JSON)
        if self.response_failures:
            self.response_failures -= 1
            raise DeepSeekServerError("scripted response failure")
        return _chat("你好！很高兴见到你。")


async def _person(rt):
    return await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _event(person_id, text="你好"):
    # ``unknown`` keeps these recovery tests on the Deliberation Path: they
    # exercise the two-call analysis+response pipeline (Fast Path coverage
    # lives in test_v06_fast_path.py).
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="unknown",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def test_analysis_failure_is_retryable(db):
    client = ScriptedClient(analysis_failures=1)
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    event = _event(person.person_id)

    failed = await rt.orchestrator.process_utterance(event)
    assert failed.status == "FAILED"
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status == ProcessingStatus.FAILED
    assert await rt.repos.analysis.get_analysis(event.event_id) is None

    recovery = RecoveryManager(rt.repos, rt.orchestrator)
    assert [e.event_id for e in await recovery.list_unfinished()] == [event.event_id]

    retried = await recovery.retry(event.event_id)
    assert retried.status == "COMPLETED"
    assert retried.action == CognitiveAction.RESPOND
    assert retried.response_text == "你好！很高兴见到你。"


async def test_response_failure_keeps_analysis_and_reuses_it_on_retry(db):
    client = ScriptedClient(response_failures=1)
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    event = _event(person.person_id)

    failed = await rt.orchestrator.process_utterance(event)
    assert failed.status == "FAILED"
    # The analysis and its candidates survived; only the reply failed.
    assert await rt.repos.analysis.get_analysis(event.event_id) is not None
    assert len(await rt.repos.analysis.get_candidates(event.event_id)) == 1
    assert await rt.repos.analysis.get_action(event.event_id) is None
    assert client.calls == ["analysis", "response"]

    retried = await rt.orchestrator.process_utterance(event)
    assert retried.status == "COMPLETED"
    assert retried.response_text == "你好！很高兴见到你。"
    # Exactly one analysis request across both attempts.
    assert client.calls == ["analysis", "response", "response"]


async def test_duplicate_delivery_makes_no_further_call(db):
    client = ScriptedClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    event = _event(person.person_id)

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "COMPLETED"
    calls_after_first = list(client.calls)

    duplicate = await rt.orchestrator.process_utterance(event)
    assert duplicate.status == "DUPLICATE"
    assert duplicate.response_text == "你好！很高兴见到你。"
    assert client.calls == calls_after_first


async def test_quiet_mode_withholds_response_but_still_learns(db):
    """V0.7: quiet mode is a local WAIT, not a synchronous model turn.

    No response call is made and no reply text is produced, but the event is
    recorded, the turn is persisted and its memory learning is queued as a
    background job so the conversation is not silently lost.
    """
    client = ScriptedClient()
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await _person(rt)
    await rt.conversations.set_quiet_mode("c1", True)
    event = _event(person.person_id)

    quiet = await rt.orchestrator.process_utterance(event)
    assert quiet.status == "COMPLETED"
    assert quiet.action == CognitiveAction.WAIT
    assert quiet.response_text is None
    # A local decision makes no provider call at all.
    assert client.calls == []
    # The event is still recorded and background learning is queued.
    assert await rt.repos.utterance.get(event.event_id) is not None
    job = await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    assert job is not None

    # A second *different* event after quiet mode is turned off does get a
    # response, using the normal pipeline.
    await rt.conversations.set_quiet_mode("c1", False)
    second = await rt.orchestrator.process_utterance(_event(person.person_id, "再说一次"))
    assert second.action == CognitiveAction.RESPOND
    assert client.calls == ["analysis", "response"]


async def _dump_all_text(db) -> str:
    cur = await db.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type = 'table'"
        " AND name NOT LIKE 'sqlite_%'"
        " AND name NOT LIKE '%_fts%'"
    )
    tables = [row["name"] for row in await cur.fetchall()]
    chunks: list[str] = []
    for table in tables:
        cur = await db.execute(f'SELECT * FROM "{table}"')
        for row in await cur.fetchall():
            for value in tuple(row):
                if isinstance(value, str):
                    chunks.append(value)
    return "\n".join(chunks)


async def test_key_and_prompt_are_never_written_to_sqlite(db):
    client = ScriptedClient()
    runtime = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        deepseek_config=DeepSeekConfig(api_key=SENTINEL_KEY),
    )
    person = await _person(runtime)
    await runtime.orchestrator.process_utterance(_event(person.person_id))

    blob = await _dump_all_text(db)
    assert SENTINEL_KEY not in blob
    assert SYSTEM_PROMPT[:30] not in blob
    # The legitimate result *is* stored.
    assert "用户在问好" in blob
