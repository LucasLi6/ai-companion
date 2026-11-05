"""Persistence for validated route decisions and tool-call audit rows.

Only *validated* decisions are stored. Raw invalid model output is never
promoted to a successful decision. Decisions are append-fixed per
``(event_id, step_index)``: a retry reuses the first committed decision rather
than re-interpreting a completed turn.
"""

from __future__ import annotations

import json
from typing import Optional

from companion.domain.models import utcnow
from companion.routing.models import (
    RoutePhase,
    ToolRequest,
    ToolStatus,
    ValidatedRouteDecision,
)
from companion.storage.database import Database


def _iso(value) -> str:
    return value.isoformat()


class RoutingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save_decision(
        self,
        *,
        event_id: str,
        phase: RoutePhase,
        step_index: int,
        route: ValidatedRouteDecision,
        validation_status: str = "VALID",
    ) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO route_decisions("
            "event_id, phase, step_index, schema_version, router_version,"
            " provider, model, decision_json, validation_status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                phase.value,
                int(step_index),
                int(route.schema_version),
                route.router_version,
                route.provider,
                route.model,
                json.dumps(route.audit_dict(), ensure_ascii=False),
                validation_status,
                _iso(utcnow()),
            ),
        )

    async def get_decision(
        self, event_id: str, step_index: int = 0
    ) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT event_id, phase, step_index, schema_version, router_version,"
            " provider, model, decision_json, validation_status, created_at"
            " FROM route_decisions WHERE event_id = ? AND step_index = ?",
            (event_id, int(step_index)),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._decision_row(row)

    async def list_decisions(self, event_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT event_id, phase, step_index, schema_version, router_version,"
            " provider, model, decision_json, validation_status, created_at"
            " FROM route_decisions WHERE event_id = ? ORDER BY step_index ASC",
            (event_id,),
        )
        return [self._decision_row(row) for row in await cur.fetchall()]

    async def load_validated(
        self, event_id: str, step_index: int = 0
    ) -> Optional[ValidatedRouteDecision]:
        row = await self.get_decision(event_id, step_index)
        if row is None or row.get("validation_status") != "VALID":
            return None
        try:
            return ValidatedRouteDecision.model_validate(row["decision"])
        except (TypeError, ValueError):
            return None

    async def save_tool_call(
        self,
        *,
        tool_call_id: str,
        event_id: str,
        route_step_index: int,
        tool_name: str,
        arguments: dict,
        arguments_hash: str,
        status: ToolStatus,
        error_code: Optional[str] = None,
        source_refs: Optional[list[str]] = None,
    ) -> None:
        reference = {
            "status": status.value,
            "source_refs": list(source_refs or [])[:20],
        }
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT OR REPLACE INTO route_tool_calls("
            "tool_call_id, event_id, route_step_index, tool_name,"
            " arguments_json, arguments_hash, status, result_reference_json,"
            " error_code, created_at, completed_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tool_call_id,
                event_id,
                int(route_step_index),
                tool_name,
                json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                arguments_hash,
                status.value,
                json.dumps(reference, ensure_ascii=False),
                error_code,
                now,
                now,
            ),
        )

    async def list_tool_calls(self, event_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT tool_call_id, event_id, route_step_index, tool_name,"
            " arguments_json, arguments_hash, status, result_reference_json,"
            " error_code, created_at, completed_at"
            " FROM route_tool_calls WHERE event_id = ? ORDER BY created_at ASC",
            (event_id,),
        )
        rows = await cur.fetchall()
        return [
            {
                "tool_call_id": row["tool_call_id"],
                "event_id": row["event_id"],
                "route_step_index": row["route_step_index"],
                "tool_name": row["tool_name"],
                "arguments": json.loads(row["arguments_json"] or "{}"),
                "arguments_hash": row["arguments_hash"],
                "status": row["status"],
                "result_reference": json.loads(
                    row["result_reference_json"] or "{}"
                ),
                "error_code": row["error_code"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _decision_row(row) -> dict:
        try:
            decision = json.loads(row["decision_json"])
        except (TypeError, ValueError):
            decision = None
        return {
            "event_id": row["event_id"],
            "phase": row["phase"],
            "step_index": row["step_index"],
            "schema_version": row["schema_version"],
            "router_version": row["router_version"],
            "provider": row["provider"],
            "model": row["model"],
            "decision": decision,
            "validation_status": row["validation_status"],
            "created_at": row["created_at"],
        }


def request_signature(request: ToolRequest) -> str:
    return json.dumps(
        {"tool": request.tool_name, "arguments": request.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
