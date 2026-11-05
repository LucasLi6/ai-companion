"""Privacy requests: structured resolution, confirmation and real deletion.

A router ``PRIVACY_REQUEST`` is a *request*. It can never delete anything by
itself, it is never parsed from free text, and a pending operation is not a
completed one. These tests check the whole trusted flow, including ownership,
confirmation, cancellation, idempotency and the event cascade.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import (
    AssertionType,
    EventSource,
    CertaintyCategory,
    MemoryKind,
    MemoryOperation,
)
from companion.domain.models import MemoryCandidate, make_utterance_event
from companion.memory.consolidation import MemoryWriter
from companion.memory.explicit_operations import ExplicitMemoryOperationService
from companion.memory.models import AccessContext
from companion.routing.models import (
    DataControlRequest,
    ProcessingIntent,
    RawRouteDecision,
)
from companion.runtime import build_runtime


class RecordingResponse:
    def __init__(self):
        self.calls = []

    async def generate(self, event, analysis, context):
        self.calls.append((event, analysis, context))
        return "好的。"


class ExtractionCognitive:
    def __init__(self, candidates=None):
        self.candidates = list(candidates or [])
        self.extractions = 0

    async def extract_events(self, events, *, speaker_label, prior_turns=None):
        self.extractions += 1
        return {event.event_id: list(self.candidates) for event in events}

    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call analyze")


class FakeRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self, *, request: dict | None):
        self._request = request
        self.calls = 0

    async def route(self, context):
        self.calls += 1
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": "RESPOND",
                "reply_necessity": "REQUIRED",
                "addressee_assessment": "AGENT",
                "turn_state": "CONTINUING",
                "processing_intents": ["PRIVACY_REQUEST"],
                "tool_requests": [],
                "data_control_request": self._request,
                "reason": "privacy request test decision",
            }
        )

    async def replan(self, context, previous, results):
        raise AssertionError("no follow-up planning expected")


def _event(person_id, text="please forget this", *, event_id=None, conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


async def _save_event(rt, event):
    await rt.repos.conversation.get_or_create(event.conversation_id)
    await rt.repos.utterance.save(event)


async def _person(rt, name="Alice", *, learning=True):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=learning,
        allow_external_ai=True,
    )


async def _memory(rt, person, content="Alice 的私密事实"):
    return await rt.memory_service.create(
        kind="EPISODE",
        content=content,
        access=AccessContext("admin", is_admin=True),
        visibility="PRIVATE",
        subjects=[("PERSON", person.person_id)],
    )


def _request(
    *, operation="DELETE", target_type="MEMORY", target_id=None, scope="SELF"
):
    return DataControlRequest(
        operation=operation,
        target_type=target_type,
        target_id=target_id,
        scope=scope,
    )


async def _pending(rt, event, request, consent=None):
    return await rt.data_control.resolve_request(request, event, consent)


# -- resolution never deletes -------------------------------------------------


@pytest.mark.asyncio
async def test_free_text_privacy_intent_without_a_structured_request_needs_clarification(rt):
    person = await _person(rt)
    event = _event(person.person_id, "把我说过的话都删掉")
    await _save_event(rt, event)

    result = await _pending(rt, event, None)

    assert result.status == "NEEDS_CLARIFICATION"
    assert result.detail == "no_structured_request"
    assert result.operation_id is None
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS n FROM data_control_operations"
    )
    assert (await cur.fetchone())["n"] == 0


@pytest.mark.asyncio
async def test_vague_target_is_not_guessed(rt):
    person = await _person(rt)
    event = _event(person.person_id)
    await _save_event(rt, event)

    result = await _pending(rt, event, _request(target_type="UNKNOWN"))

    assert result.status == "UNSUPPORTED"
    assert "unsupported_target" in result.detail

    vague = await _pending(rt, event, _request(target_type="MEMORY"))
    assert vague.status == "NEEDS_CLARIFICATION"
    assert vague.detail == "target_id_required:MEMORY"


@pytest.mark.asyncio
async def test_unsupported_operation_and_scope_are_refused(rt):
    person = await _person(rt)
    event = _event(person.person_id)
    await _save_event(rt, event)

    export = await _pending(
        rt, event, _request(operation="EXPORT", target_type="PROFILE")
    )
    assert export.status == "UNSUPPORTED"
    assert "unsupported_operation" in export.detail

    everyone = await _pending(
        rt, event, _request(operation="DELETE", target_type="PROFILE", scope="ALL")
    )
    assert everyone.status == "UNSUPPORTED"
    assert "unsupported_scope" in everyone.detail


@pytest.mark.asyncio
async def test_pending_operation_and_audit_are_not_a_deletion(rt):
    person = await _person(rt)
    event = _event(person.person_id)
    await _save_event(rt, event)
    memory = await _memory(rt, person)

    result = await _pending(
        rt, event, _request(target_type="MEMORY", target_id=memory.memory_id)
    )

    assert result.status == "AWAITING_CONFIRMATION"
    assert result.operation_id is not None
    # The audit trail and the pending row exist, but the data is untouched.
    stored = await rt.data_control.get_operation_result(result.operation_id)
    assert stored["status"] == "AWAITING_CONFIRMATION"
    assert await rt.memory_service.repo.get(memory.memory_id) is not None
    assert await rt.data_control.request_confirmation(result.operation_id) is not None

    cur = await rt.db.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE entity_id = ?",
        (result.operation_id,),
    )
    assert (await cur.fetchone())["n"] >= 1


# -- confirmation ------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmed_delete_removes_the_memory_and_reports_completed(rt):
    person = await _person(rt)
    event = _event(person.person_id)
    await _save_event(rt, event)
    memory = await _memory(rt, person)

    pending = await _pending(
        rt, event, _request(target_type="MEMORY", target_id=memory.memory_id)
    )
    result = await rt.data_control.execute_confirmed(pending.operation_id)

    assert result.status == "COMPLETED"
    assert result.result["deleted"]["memory"] == 1
    assert await rt.memory_service.repo.get(memory.memory_id) is None

    second = await rt.data_control.execute_confirmed(pending.operation_id)
    assert second.status == "COMPLETED"
    assert second.result == result.result


@pytest.mark.asyncio
async def test_cancel_and_wrong_actor_never_delete(rt):
    person = await _person(rt)
    event = _event(person.person_id)
    await _save_event(rt, event)
    memory = await _memory(rt, person)

    pending = await _pending(
        rt, event, _request(target_type="MEMORY", target_id=memory.memory_id)
    )
    denied = await rt.data_control.execute_confirmed(
        pending.operation_id, actor_id="person_other"
    )
    assert denied.status == "DENIED"
    assert await rt.memory_service.repo.get(memory.memory_id) is not None

    cancelled = await rt.data_control.execute_confirmed(
        pending.operation_id, confirmed=False
    )
    assert cancelled.status == "CANCELLED"
    assert await rt.memory_service.repo.get(memory.memory_id) is not None

    # A cancelled operation is terminal: it cannot be resumed later.
    after = await rt.data_control.execute_confirmed(pending.operation_id)
    assert after.status == "CANCELLED"


@pytest.mark.asyncio
async def test_another_persons_memory_is_denied_at_resolution(rt):
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    event = _event(alice.person_id)
    await _save_event(rt, event)
    bobs_memory = await _memory(rt, bob, "Bob 的私密事实")

    result = await _pending(
        rt, event, _request(target_type="MEMORY", target_id=bobs_memory.memory_id)
    )

    assert result.status == "DENIED"
    assert result.detail == "memory_not_owned"
    assert await rt.memory_service.repo.get(bobs_memory.memory_id) is not None


@pytest.mark.asyncio
async def test_profile_purge_runs_only_after_confirmation(rt):
    person = await _person(rt)
    event = _event(person.person_id)
    await _save_event(rt, event)
    await rt.profile_service.repo.ensure(person.person_id)
    await rt.profile_service.repo.upsert_field(
        person.person_id,
        "communication_style",
        value={"value": "direct"},
        source=EventSource.TEST,
        confidence=0.9,
        evidence_memory_ids=[],
        evidence_event_ids=[],
        evidence_candidate_ids=[],
        revision=1,
    )

    pending = await _pending(rt, event, _request(target_type="PROFILE"))
    assert pending.status == "AWAITING_CONFIRMATION"
    assert await rt.profile_service.repo.list_fields(person.person_id)

    result = await rt.data_control.execute_confirmed(pending.operation_id)

    assert result.status == "COMPLETED"
    assert result.result["deleted"]["profile"] == 1
    assert await rt.profile_service.repo.list_fields(person.person_id) == []


# -- event lifecycle ---------------------------------------------------------


@pytest.mark.asyncio
async def test_event_delete_cascades_route_and_tool_rows_and_derived_memory(rt):
    person = await _person(rt)
    event = _event(person.person_id, "remember my name is Alice")
    await _save_event(rt, event)

    writer = MemoryWriter(rt.repos, rt.memory_service)
    candidate = MemoryCandidate(
        candidate_id="memc_event_delete",
        source_event_id=event.event_id,
        subject_person_id=None,
        content="remember my name is Alice",
        operation=MemoryOperation.ADD,
        assertion_type=AssertionType.SELF_REPORT,
        certainty_category=CertaintyCategory.DIRECT,
        topic="name",
        kind=MemoryKind.PERSONAL_FACT,
        source_evidence_span="remember my name is Alice",
    )
    await rt.repos.analysis.save_candidate(candidate)
    outcomes = await writer.consolidate_event(event.event_id, only_explicit=False)
    memory_id = next(o["memory_id"] for o in outcomes if o["memory_id"])
    assert await rt.memory_service.repo.get(memory_id) is not None

    from companion.routing.models import RoutePhase, ValidatedRouteDecision

    route = ValidatedRouteDecision(
        action="RESPOND",
        reply_necessity="REQUIRED",
        addressee_assessment="AGENT",
    )
    await rt.repos.routing.save_decision(
        event_id=event.event_id,
        phase=RoutePhase.INITIAL,
        step_index=0,
        route=route,
    )
    await rt.repos.routing.save_tool_call(
        tool_call_id="tc_event_delete",
        event_id=event.event_id,
        route_step_index=0,
        tool_name="memory.search",
        arguments={"query": "name"},
        arguments_hash="hash",
    )
    assert await rt.repos.routing.list_decisions(event.event_id)
    assert await rt.repos.routing.list_tool_calls(event.event_id)

    pending = await _pending(
        rt, event, _request(target_type="EVENT", target_id=event.event_id)
    )
    result = await rt.data_control.execute_confirmed(pending.operation_id)

    assert result.status == "COMPLETED"
    assert result.result["deleted"]["event"] == 1
    assert result.result["deleted"]["memory"] == 1
    assert await rt.repos.utterance.get(event.event_id) is None
    assert await rt.memory_service.repo.get(memory_id) is None
    # SQLite ON DELETE CASCADE removed the derived route/tool rows too.
    assert await rt.repos.routing.list_decisions(event.event_id) == []
    assert await rt.repos.routing.list_tool_calls(event.event_id) == []


@pytest.mark.asyncio
async def test_another_persons_event_cannot_be_deleted(rt):
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    bobs_event = _event(bob.person_id, "Bob's utterance", conversation="c2")
    await _save_event(rt, bobs_event)
    alice_event = _event(alice.person_id)
    await _save_event(rt, alice_event)

    result = await _pending(
        rt, alice_event, _request(target_type="EVENT", target_id=bobs_event.event_id)
    )

    assert result.status == "DENIED"
    assert result.detail == "event_not_owned"
    assert await rt.repos.utterance.get(bobs_event.event_id) is not None


# -- end-to-end router connection -------------------------------------------


@pytest.mark.asyncio
async def test_router_privacy_request_records_pending_and_does_not_delete(db):
    response = RecordingResponse()
    rt = build_runtime(
        db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=FakeRouter(
            request={
                "operation": "DELETE",
                "target_type": "MEMORY",
                "target_id": "mem_later",
                "scope": "SELF",
                "requires_confirmation": True,
            }
        ),
        deepseek_client=object(),
        cognitive=ExtractionCognitive([]),
        response=response,
        allow_mixed_providers=True,
        fast_path_enabled=False,
    )
    person = await _person(rt)
    event = _event(person.person_id, "forget that memory")
    memory = await rt.memory_service.create(
        kind="EPISODE",
        content="a kept memory",
        access=AccessContext("admin", is_admin=True),
        visibility="PRIVATE",
        subjects=[("PERSON", person.person_id)],
        memory_id="mem_later",
    )

    report = await rt.orchestrator.process_utterance(event)

    assert report.status == "COMPLETED"
    assert await rt.memory_service.repo.get(memory.memory_id) is not None
    cur = await rt.db.execute(
        "SELECT status FROM data_control_operations WHERE person_id = ?",
        (person.person_id,),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row["status"] == "AWAITING_CONFIRMATION"

    cur = await rt.db.execute(
        "SELECT operation_id FROM data_control_operations WHERE person_id = ?",
        (person.person_id,),
    )
    operation_id = (await cur.fetchone())["operation_id"]
    completed = await rt.data_control.execute_confirmed(operation_id)
    assert completed.status == "COMPLETED"
    assert await rt.memory_service.repo.get(memory.memory_id) is None


@pytest.mark.asyncio
async def test_a_deleted_event_is_never_relearned_by_a_pending_job(db):
    from companion.domain.enums import JobType

    cognitive = ExtractionCognitive([])
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=object(),
        cognitive=cognitive,
        response=RecordingResponse(),
        allow_mixed_providers=True,
        router_mode="legacy",
        fast_path_enabled=False,
    )
    person = await _person(rt)
    event = _event(person.person_id, "remember this detail")
    await _save_event(rt, event)
    await rt.repos.job.enqueue(
        JobType.TURN_ANALYSIS,
        f"turn-analysis:{event.event_id}",
        {"event_id": event.event_id},
    )

    pending = await _pending(
        rt, event, _request(target_type="EVENT", target_id=event.event_id)
    )
    result = await rt.data_control.execute_confirmed(pending.operation_id)
    assert result.status == "COMPLETED"

    await rt.worker.run_once()

    assert cognitive.extractions == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_candidates")
    assert (await cur.fetchone())["n"] == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS n FROM memory_items")
    assert (await cur.fetchone())["n"] == 0
    assert await rt.repos.utterance.get(event.event_id) is None
