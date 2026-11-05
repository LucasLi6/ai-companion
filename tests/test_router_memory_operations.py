"""Explicit memory operations requested by the semantic router.

These tests cover the repaired write chain: a ``MEMORY_SAVE`` /
``MEMORY_CORRECTION`` intent must reach the existing candidate pipeline, must
respect stored consent, and must report what actually happened -- never a
fabricated success. WAIT/IGNORE may still save without producing a reply.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import (
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    EventSource,
    MemoryKind,
    MemoryOperation,
    ReplyNecessity,
)
from companion.domain.models import Consent, MemoryCandidate, make_utterance_event, utcnow
from companion.memory.consolidation import MemoryWriter
from companion.memory.explicit_operations import ExplicitMemoryOperationService
from companion.routing.models import (
    ProcessingIntent,
    RawRouteDecision,
    RouteAction,
    RouteAddressee,
    ValidatedRouteDecision,
)
from companion.runtime import build_runtime


class ExtractionCognitive:
    """A cognitive stand-in that only performs explicit extraction."""

    def __init__(self, candidates=None, *, fail=False):
        self.candidates = list(candidates or [])
        self.fail = fail
        self.extractions = 0
        self.analyze_calls = 0

    async def extract_events(self, events, *, speaker_label, prior_turns=None):
        self.extractions += 1
        if self.fail:
            raise RuntimeError("extraction exploded")
        return {event.event_id: list(self.candidates) for event in events}

    async def analyze(self, *args, **kwargs):
        self.analyze_calls += 1
        raise AssertionError("semantic mode must not call analyze")


class RecordingResponse:
    def __init__(self):
        self.calls = []

    async def generate(self, event, analysis, context):
        self.calls.append((event, analysis, context))
        return "好的，记住了。"


class FakeRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self, *, action="RESPOND", intents=None, necessity=None):
        self.calls = 0
        self._action = action
        self._intents = intents or []
        self._necessity = necessity or (
            "REQUIRED" if action == "RESPOND" else "OPTIONAL"
        )

    def _raw(self):
        return {
            "schema_version": 1,
            "action": self._action,
            "reply_necessity": self._necessity,
            "addressee_assessment": "AGENT",
            "input_semantics": "COMMUNICATIVE",
            "turn_state": "CONTINUING",
            "processing_intents": self._intents,
            "tool_requests": [],
            "reason": "explicit memory test decision",
        }

    async def route(self, context):
        self.calls += 1
        return RawRouteDecision.model_validate(self._raw())

    async def replan(self, context, previous, results):
        raise AssertionError("no follow-up planning expected")


def _candidate(event, content, *, topic="name", explicit=False):
    return MemoryCandidate(
        candidate_id="memc_model_proposal",
        source_event_id=event.event_id,
        subject_person_id=None,
        content=content,
        operation=MemoryOperation.ADD,
        assertion_type=AssertionType.SELF_REPORT,
        certainty_category=CertaintyCategory.DIRECT,
        topic=topic,
        kind=MemoryKind.PERSONAL_FACT,
        source_evidence_span=content,
        explicit_remember_request=explicit,
    )


def _route(intents):
    return ValidatedRouteDecision(
        action=RouteAction.RESPOND,
        reply_necessity=ReplyNecessity.REQUIRED,
        addressee_assessment=RouteAddressee.AGENT,
        processing_intents=list(intents),
    )


def _person_event(person_id, text, *, event_id=None):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


def _service(rt, cognitive):
    return ExplicitMemoryOperationService(
        repos=rt.repos,
        cognitive=cognitive,
        memory_writer=MemoryWriter(rt.repos, rt.memory_service),
    )


async def _save_event(rt, event):
    await rt.repos.conversation.get_or_create(event.conversation_id)
    await rt.repos.utterance.save(event)


def _semantic_runtime(db, router, cognitive, response=None):
    return build_runtime(
        db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=router,
        deepseek_client=object(),
        cognitive=cognitive,
        response=response or RecordingResponse(),
        allow_mixed_providers=True,
        fast_path_enabled=False,
    )


# -- direct service semantics ------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_save_reports_saved_and_writes_one_memory(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    event = _person_event(person.person_id, "我叫 Alice。")
    await _save_event(rt, event)

    cognitive = ExtractionCognitive([_candidate(event, "我叫 Alice", topic="name")])
    service = _service(rt, cognitive)

    result = await service.execute(
        event, _route([ProcessingIntent.MEMORY_SAVE]), None
    )

    assert result.status == "SAVED"
    assert result.intent == "MEMORY_SAVE"
    assert result.completed is True
    assert result.candidate_count == 1
    assert cognitive.extractions == 1

    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert len(candidates) == 1
    assert candidates[0].explicit_remember_request is True
    write = await rt.repos.analysis.get_candidate_write(candidates[0].candidate_id)
    assert write["outcome"] == "SAVED"
    memory = await rt.memory_service.repo.get(write["memory_id"])
    assert memory is not None
    assert memory.content == "我叫 Alice"


@pytest.mark.asyncio
async def test_retry_reuses_candidates_and_never_writes_twice(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    event = _person_event(person.person_id, "我叫 Alice。")
    await _save_event(rt, event)

    cognitive = ExtractionCognitive([_candidate(event, "我叫 Alice", topic="name")])
    service = _service(rt, cognitive)
    route = _route([ProcessingIntent.MEMORY_SAVE])

    first = await service.execute(event, route, None)
    second = await service.execute(event, route, None)

    assert first.status == "SAVED"
    assert second.status == "SAVED"
    assert cognitive.extractions == 1  # no second model call on retry
    assert second.candidate_count == 1

    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS n FROM memory_candidate_writes"
    )
    assert (await cur.fetchone())["n"] == 1


@pytest.mark.asyncio
async def test_withdrawn_learning_consent_denies_the_write(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    event = _person_event(person.person_id, "我叫 Alice。")
    await _save_event(rt, event)

    # Consent is re-read from storage at execution time; the route's intent is
    # not permission.
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )

    cognitive = ExtractionCognitive([_candidate(event, "我叫 Alice", topic="name")])
    result = await _service(rt, cognitive).execute(
        event, _route([ProcessingIntent.MEMORY_SAVE]), None
    )

    assert result.status == "DENIED"
    assert result.error_code == "memory_learning_not_granted"
    assert cognitive.extractions == 0
    assert await rt.repos.analysis.get_candidates(event.event_id) == []
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0


@pytest.mark.asyncio
async def test_no_intent_changes_nothing_and_reports_no_memory(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    event = _person_event(person.person_id, "今天天气不错。")
    await _save_event(rt, event)

    cognitive = ExtractionCognitive([])
    result = await _service(rt, cognitive).execute(event, _route([]), None)

    assert result.status == "NO_MEMORY"
    assert result.intent is None
    assert result.completed is True
    assert cognitive.extractions == 0
    assert await rt.repos.analysis.get_candidates(event.event_id) == []


@pytest.mark.asyncio
async def test_extraction_failure_is_reported_truthfully(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    event = _person_event(person.person_id, "记住我叫 Alice。")
    await _save_event(rt, event)

    cognitive = ExtractionCognitive([], fail=True)
    result = await _service(rt, cognitive).execute(
        event, _route([ProcessingIntent.MEMORY_SAVE]), None
    )

    assert result.status == "FAILED"
    assert result.completed is False
    assert result.error_code == "extraction_failed"
    assert await rt.repos.analysis.get_candidates(event.event_id) == []


@pytest.mark.asyncio
async def test_correction_intent_wins_over_save(rt):
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )
    event = _person_event(person.person_id, "不，我叫 Alicia。")
    await _save_event(rt, event)

    cognitive = ExtractionCognitive(
        [_candidate(event, "我叫 Alicia", topic="name")]
    )
    result = await _service(rt, cognitive).execute(
        event,
        _route(
            [ProcessingIntent.MEMORY_SAVE, ProcessingIntent.MEMORY_CORRECTION]
        ),
        None,
    )

    assert result.intent == "MEMORY_CORRECTION"
    assert result.status == "SAVED"


# -- end-to-end semantic pipeline -------------------------------------------


@pytest.mark.asyncio
async def test_router_save_reaches_the_speaker_context(db):
    router = FakeRouter(intents=["MEMORY_SAVE"])
    response = RecordingResponse()
    rt = _semantic_runtime(db, router, ExtractionCognitive([]), response)
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    event = _person_event(person.person_id, "我叫 Alice。")
    rt.cognitive.candidates = [_candidate(event, "我叫 Alice", topic="name")]

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.RESPOND
    assert len(response.calls) == 1
    context = response.calls[0][2]
    assert context.memory_write_outcomes
    assert context.memory_write_outcomes[0]["outcome"] == "SAVED"
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1


@pytest.mark.asyncio
async def test_wait_turn_can_save_without_a_reply(db):
    router = FakeRouter(action="WAIT", intents=["MEMORY_SAVE"])
    response = RecordingResponse()
    rt = _semantic_runtime(db, router, ExtractionCognitive([]), response)
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    event = _person_event(person.person_id, "记住我叫 Alice。")
    rt.cognitive.candidates = [_candidate(event, "我叫 Alice", topic="name")]

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.WAIT
    assert response.calls == []
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1


@pytest.mark.asyncio
async def test_non_persistence_turn_never_writes_memory(db):
    router = FakeRouter(intents=["MEMORY_SAVE"])
    response = RecordingResponse()
    rt = _semantic_runtime(db, router, ExtractionCognitive([]), response)
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=False,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    event = _person_event(person.person_id, "我叫 Alice。")
    rt.cognitive.candidates = [_candidate(event, "我叫 Alice", topic="name")]

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_candidates")
    assert (await cur.fetchone())["n"] == 0
    assert await rt.repos.utterance.get(event.event_id) is None


@pytest.mark.asyncio
async def test_memory_saved_before_a_cancelled_reply_stays_saved(db):
    import asyncio

    from companion.providers.deepseek_client import StreamDelta

    class StreamingClient:
        async def complete(self, *args, **kwargs):
            raise AssertionError("analysis must not run in semantic mode")

        async def stream_completion(
            self, messages, *, request_type="response", **kwargs
        ):
            yield StreamDelta(content="你")
            await asyncio.sleep(30)
            yield StreamDelta(content="好")

    router = FakeRouter(intents=["MEMORY_SAVE"])
    cognitive = ExtractionCognitive([])
    rt = build_runtime(
        db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=router,
        deepseek_client=StreamingClient(),
        cognitive=cognitive,
        allow_mixed_providers=True,
        fast_path_enabled=False,
    )
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    event = _person_event(person.person_id, "我叫 Alice。")
    cognitive.candidates = [_candidate(event, "我叫 Alice", topic="name")]

    stream = rt.orchestrator.process_utterance_stream(event)
    first = await stream.__anext__()
    second = await stream.__anext__()
    assert first.type == "response_started"
    assert second.type == "response_delta"
    await stream.aclose()  # the consumer disconnects mid-reply

    # The explicit write was committed before generation; cancelling the reply
    # neither invents a success nor removes the committed memory.
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 1
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None
    assert action.status == "CANCELLED"
    assert action.response_text == "你"
