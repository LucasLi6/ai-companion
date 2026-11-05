"""Routing persistence and recovery: idempotent, append-fixed, private."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from companion.routing.models import RawRouteDecision, RoutingContext
from companion.routing.repository import RoutingRepository
from companion.routing.service import RoutingService
from companion.routing.validator import RoutingValidator
from companion.tools.executor import ToolExecutor
from companion.tools.models import ToolExecutionContext
from companion.tools.registry import build_default_tool_registry
from companion.storage.schema import SCHEMA_VERSION

class CountingRouter:
    provider_name = "counting_router"
    model = "counting-model"

    def __init__(self):
        self.calls = 0

    async def route(self, context):
        self.calls += 1
        return RawRouteDecision.model_validate(
            {
                "action": "WAIT",
                "reply_necessity": "OPTIONAL",
                "addressee_assessment": "UNKNOWN",
                "reason": "no visible reply needed",
            }
        )

    async def replan(self, context, previous, results):
        raise AssertionError("no follow-up was requested")


def routing_context():
    return RoutingContext(
        conversation_id="conv",
        current_event_id="evt",
        current_speaker_id="p1",
        current_speaker_label="Lucas",
        current_text="hm",
    )


def execution_context(memory=None):
    return ToolExecutionContext(
        event=SimpleNamespace(event_id="evt", conversation_id="conv"),
        person_id="p1",
        repos=SimpleNamespace(),
        memory=memory or SimpleNamespace(),
    )


def build_service(repo, router):
    registry = build_default_tool_registry(include_web=False)
    return RoutingService(
        provider=router,
        validator=RoutingValidator(),
        registry=registry,
        executor=ToolExecutor(registry),
        repository=repo,
    )


async def _seed_event(db, event_id: str, conversation_id: str = "conv") -> None:
    """Route rows now have a real FK to the source utterance."""
    now = "2026-01-01T00:00:00+00:00"
    await db.execute(
        "INSERT OR IGNORE INTO conversations(conversation_id, quiet_mode,"
        " created_at, updated_at) VALUES (?, 0, ?, ?)",
        (conversation_id, now, now),
    )
    await db.execute(
        "INSERT OR IGNORE INTO utterances(event_id, conversation_id,"
        " speaker_person_id, speaker_label, text, source, addressee_hint,"
        " occurred_at, processing_status, created_at)"
        " VALUES (?, ?, NULL, 'Lucas', 'hm', 'test', 'unknown', ?,"
        " 'PROCESSING', ?)",
        (event_id, conversation_id, now, now),
    )


@pytest.mark.asyncio
async def test_committed_decision_is_reused_without_a_second_router_call(db):
    repo = RoutingRepository(db)
    router = CountingRouter()
    service = build_service(repo, router)
    await _seed_event(db, "evt")

    first = await service.execute(
        SimpleNamespace(event_id="evt"), routing_context(),
        execution_context(), persist=True,
    )
    second = await service.execute(
        SimpleNamespace(event_id="evt"), routing_context(),
        execution_context(), persist=True,
    )

    assert router.calls == 1
    assert first.route.action == second.route.action
    rows = await repo.list_decisions("evt")
    assert len(rows) == 1
    assert rows[0]["phase"] == "INITIAL"
    assert rows[0]["schema_version"] == 1


@pytest.mark.asyncio
async def test_no_persistence_writes_no_route_or_tool_rows(db):
    repo = RoutingRepository(db)
    router = CountingRouter()
    service = build_service(repo, router)

    await service.execute(
        SimpleNamespace(event_id="evt2"), routing_context(),
        execution_context(), persist=False,
    )
    assert await repo.list_decisions("evt2") == []
    assert await repo.list_tool_calls("evt2") == []


class _Memory:
    async def retrieve(self, query, person_id):
        return [
            {
                "memory_id": "mem1",
                "kind": "FACT",
                "content": "sensitive private content",
                "updated_at": "2026-01-01",
            }
        ]

    async def browse(self, person_id, limit=3):
        return []


class _ToolRouter(CountingRouter):
    async def route(self, context):
        self.calls += 1
        return RawRouteDecision.model_validate(
            {
                "action": "RESPOND",
                "reply_necessity": "USEFUL",
                "addressee_assessment": "AGENT",
                "reason": "use memory",
                "tool_requests": [
                    {"tool_name": "memory.search", "arguments": {"query": "x"}}
                ],
            }
        )


@pytest.mark.asyncio
async def test_tool_rows_store_status_refs_not_private_content(db):
    repo = RoutingRepository(db)
    router = _ToolRouter()
    service = build_service(repo, router)
    await _seed_event(db, "evt3")

    outcome = await service.execute(
        SimpleNamespace(event_id="evt3"), routing_context(),
        execution_context(memory=_Memory()), persist=True,
    )
    assert outcome.tool_results[0].status.value == "SUCCESS"
    rows = await repo.list_tool_calls("evt3")
    assert len(rows) == 1
    row = rows[0]
    assert row["tool_name"] == "memory.search"
    assert row["result_reference"]["status"] == "SUCCESS"
    assert row["result_reference"]["source_refs"] == ["mem1"]
    stored = json.dumps(row, ensure_ascii=False)
    assert "sensitive private content" not in stored


@pytest.mark.asyncio
async def test_migration_keeps_existing_data_and_adds_route_tables(db):
    assert SCHEMA_VERSION >= 21
    await db.initialize()  # idempotent re-run
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    assert (await cur.fetchone())["v"] == SCHEMA_VERSION
    for table in ("route_decisions", "route_tool_calls"):
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        assert await cur.fetchone() is not None


# -- multi-stage recovery, total budget and stable ids ----------------------


class _MultiToolRouter:
    provider_name = "multi_router"
    model = "multi-model"

    def __init__(self, initial, followup=None, *, needs_followup=True):
        self.initial = list(initial)
        self.followup = list(followup or [])
        self.needs_followup = needs_followup
        self.calls = 0
        self.replans = 0

    def _raw(self, queries):
        return RawRouteDecision.model_validate(
            {
                "action": "RESPOND",
                "reply_necessity": "REQUIRED",
                "addressee_assessment": "AGENT",
                "reason": "bounded multi-stage test",
                "tool_requests": [
                    {"tool_name": "memory.search", "arguments": {"query": q}}
                    for q in queries
                ],
                "needs_followup_planning": self.needs_followup,
            }
        )

    async def route(self, context):
        self.calls += 1
        return self._raw(self.initial)

    async def replan(self, context, previous, results):
        self.replans += 1
        return self._raw(self.followup)


@pytest.mark.asyncio
async def test_the_tool_budget_is_shared_across_both_planning_phases(db):
    repo = RoutingRepository(db)
    router = _MultiToolRouter(
        ["a", "b", "c"], followup=["d", "e", "f"]
    )
    service = build_service(repo, router)
    await _seed_event(db, "evt_budget")

    outcome = await service.execute(
        SimpleNamespace(event_id="evt_budget"),
        routing_context(),
        execution_context(memory=_Memory()),
        persist=True,
    )

    assert router.calls == 1
    assert router.replans == 1
    # 3 from phase one + at most one from phase two = the shared cap of 4.
    assert len(outcome.tool_results) == 4
    assert [result.data[0]["memory_id"] for result in outcome.tool_results] == [
        "mem1"
    ] * 4
    rows = await repo.list_tool_calls("evt_budget")
    assert len(rows) == 4
    assert len({row["tool_call_id"] for row in rows}) == 4
    assert {row["route_step_index"] for row in rows} == {0, 1}


@pytest.mark.asyncio
async def test_a_full_first_phase_leaves_no_followup_budget(db):
    repo = RoutingRepository(db)
    router = _MultiToolRouter(["a", "b", "c", "d"], followup=["e"])
    service = build_service(repo, router)
    await _seed_event(db, "evt_full")

    outcome = await service.execute(
        SimpleNamespace(event_id="evt_full"),
        routing_context(),
        execution_context(memory=_Memory()),
        persist=True,
    )

    assert router.calls == 1
    assert router.replans == 0
    assert len(outcome.tool_results) == 4


@pytest.mark.asyncio
async def test_followup_does_not_repeat_a_phase_one_request(db):
    repo = RoutingRepository(db)
    router = _MultiToolRouter(["a", "b"], followup=["b", "c"])
    service = build_service(repo, router)
    await _seed_event(db, "evt_dedupe")

    outcome = await service.execute(
        SimpleNamespace(event_id="evt_dedupe"),
        routing_context(),
        execution_context(memory=_Memory()),
        persist=True,
    )

    rows = await repo.list_tool_calls("evt_dedupe")
    assert len(rows) == 3
    queries = sorted(row["arguments"]["query"] for row in rows)
    assert queries == ["a", "b", "c"]
    assert len(outcome.tool_results) == 3


@pytest.mark.asyncio
async def test_a_retry_restores_tool_calls_without_duplicating_rows(db):
    repo = RoutingRepository(db)
    router = _MultiToolRouter(["a"], needs_followup=False)
    service = build_service(repo, router)
    await _seed_event(db, "evt_retry")

    first = await service.execute(
        SimpleNamespace(event_id="evt_retry"),
        routing_context(),
        execution_context(memory=_Memory()),
        persist=True,
    )
    second = await service.execute(
        SimpleNamespace(event_id="evt_retry"),
        routing_context(),
        execution_context(memory=_Memory()),
        persist=True,
    )

    assert router.calls == 1  # the committed decision is reused
    assert first.tool_results[0].restored is False
    assert second.tool_results[0].restored is True
    assert second.tool_results[0].tool_call_id == first.tool_results[0].tool_call_id

    rows = await repo.list_tool_calls("evt_retry")
    assert len(rows) == 1
    assert rows[0]["tool_call_id"] == first.tool_results[0].tool_call_id
    assert rows[0]["execution_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_tool_call_upsert_preserves_created_at_and_history(db):
    repo = RoutingRepository(db)
    await _seed_event(db, "evt_upsert")
    await repo.save_tool_call(
        tool_call_id="tc_stable",
        event_id="evt_upsert",
        route_step_index=0,
        tool_name="memory.search",
        arguments={"query": "x"},
        arguments_hash="hash",
        execution_status="RUNNING",
    )
    cur = await db.execute(
        "SELECT created_at, completed_at FROM route_tool_calls"
        " WHERE tool_call_id = 'tc_stable'"
    )
    first = dict(await cur.fetchone())
    assert first["completed_at"] is None

    await repo.save_tool_call(
        tool_call_id="tc_stable",
        event_id="evt_upsert",
        route_step_index=0,
        tool_name="memory.search",
        arguments={"query": "x"},
        arguments_hash="hash",
        execution_status="COMPLETED",
    )
    cur = await db.execute(
        "SELECT created_at, completed_at, execution_status"
        " FROM route_tool_calls WHERE tool_call_id = 'tc_stable'"
    )
    second = dict(await cur.fetchone())
    assert second["created_at"] == first["created_at"]
    assert second["completed_at"] is not None
    assert second["execution_status"] == "COMPLETED"

    cur = await db.execute(
        "SELECT COUNT(*) AS n FROM route_tool_calls WHERE event_id = 'evt_upsert'"
    )
    assert (await cur.fetchone())["n"] == 1
