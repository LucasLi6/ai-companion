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


@pytest.mark.asyncio
async def test_committed_decision_is_reused_without_a_second_router_call(db):
    repo = RoutingRepository(db)
    router = CountingRouter()
    service = build_service(repo, router)

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
