"""Tool contracts, execution context and results.

A tool is a *read-only, registered capability*. The router may request one,
but only :class:`ToolExecutor` constructs the execution context (trusted
identity, permissions, event scope) and only the executor decides whether the
request runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, Type

from pydantic import BaseModel, ConfigDict, Field

from companion.routing.models import ToolStatus

if TYPE_CHECKING:  # pragma: no cover - imports used for typing only
    from companion.domain.models import UtteranceEvent
    from companion.memory.store import SqliteMemoryStore
    from companion.people.registry import PersonRegistry
    from companion.storage.repositories import Repositories


class ToolResult(BaseModel):
    """A truthful execution result. The router can never fill these fields."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    status: ToolStatus
    data: Any = None
    error_code: Optional[str] = None
    source_refs: list[str] = Field(default_factory=list)
    # External/web content is untrusted and must be rendered as data, never as
    # instructions.
    untrusted: bool = False
    # True when this row was produced by re-reading a read-only tool during
    # recovery, not by a newly planned request. It uses the same stable call id.
    restored: bool = False
    # Bounded, non-content bookkeeping (e.g. returned/truncated counts). Never
    # carries page text or instructions.
    meta: dict = Field(default_factory=dict)


class ToolHandlerResult(BaseModel):
    """What a handler returns; the executor adds the trusted identifiers."""

    model_config = ConfigDict(extra="forbid")

    status: ToolStatus = ToolStatus.SUCCESS
    data: Any = None
    error_code: Optional[str] = None
    source_refs: list[str] = Field(default_factory=list)
    untrusted: bool = False
    meta: dict = Field(default_factory=dict)


class ToolHandler(Protocol):
    async def execute(
        self,
        arguments: BaseModel,
        context: "ToolExecutionContext",
    ) -> ToolHandlerResult: ...


@dataclass(frozen=True)
class ToolSpec:
    """Static declaration of one tool."""

    name: str
    description: str
    argument_model: Type[BaseModel]
    permission: str = "read"
    has_side_effects: bool = False
    timeout_seconds: float = 5.0
    max_result_chars: int = 6000

    def prompt_spec(self) -> dict:
        schema = self.argument_model.model_json_schema()
        # The router only needs the argument contract. Local validation already
        # enforces the pydantic model, so per-field titles and the redundant
        # ``additionalProperties: false`` are removed to keep the prompt small.
        schema = _compact_argument_schema(schema)
        schema.pop("$defs", None)
        return {
            "name": self.name,
            "description": self.description,
            "arguments": schema,
            "required_permission": self.permission,
            "read_only": not self.has_side_effects,
        }


def _compact_argument_schema(schema: dict) -> dict:
    """Strip JSON-Schema metadata that does not help the model fill arguments."""
    if not isinstance(schema, dict):
        return schema
    compact: dict = {}
    for key, value in schema.items():
        if key in ("title", "additionalProperties"):
            continue
        if isinstance(value, dict):
            compact[key] = _compact_argument_schema(value)
        elif isinstance(value, list):
            compact[key] = [
                _compact_argument_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            compact[key] = value
    return compact


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler


@dataclass
class ToolExecutionContext:
    """Trusted, program-created scope for one tool execution."""

    event: "UtteranceEvent"
    person_id: Optional[str]
    repos: "Repositories"
    memory: "SqliteMemoryStore"
    registry: Optional["PersonRegistry"] = None
    privacy_gateway: Any = None
    runtime_snapshot: Any = None
    allow_external_ai: bool = False
    web_search: Any = None
    logger: Any = None
    extras: dict = field(default_factory=dict)

    @property
    def conversation_id(self) -> str:
        return self.event.conversation_id

    @property
    def event_id(self) -> str:
        return self.event.event_id
