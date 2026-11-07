"""NiceGUI application for the Visual Memory Manager.

Every management page opens its own short-lived SQLite connection through
:func:`open_runtime`, performs its reads/writes through the ordinary
repositories and :class:`~companion.memory.service.MemoryService`, and closes
it again. This keeps the async connection bound to the request's own event
loop (no cross-loop sharing) and means the UI obeys exactly the same
visibility, subject-privacy and optimistic-locking rules as the CLI.

``/chat`` is the one exception. Proactive speech has to keep running after the
reply that triggered it has finished, so the chat surface uses a single
server-lifetime runtime owned by :class:`~companion.ui.chat_coordinator.ChatCoordinator`
instead of a request-scoped one. It is the same object graph the CLI builds —
same orchestrator, persona, memory, proactive manager and judge — and the same
``proactive_messages`` rows, so a message generated for the web is also visible
to the CLI and survives a page reload.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import timezone
from pathlib import Path
from typing import AsyncIterator

from nicegui import app, ui

from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
    PersonRole,
    RegistrationStatus,
)
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.memory.models import AccessContext
from companion.runtime import Runtime, build_runtime
from companion.storage.database import Database
from companion.ui.chat_coordinator import ChatCoordinator

# The management UI acts as the trusted local admin, never as a person.
ADMIN = AccessContext("admin", is_admin=True, role="ui")

_KINDS = [k.value for k in MemoryKind]
_STATUSES = [s.value for s in MemoryStatus]
_VISIBILITIES = [v.value for v in MemoryVisibility]
_ROLES = [r.value for r in PersonRole]

_DB_PATH: Path | None = None
_PROVIDER = "mock"
_DEEPSEEK_CLIENT = None
_COORDINATOR: ChatCoordinator | None = None
_RETIRED_COORDINATORS: list[ChatCoordinator] = []

NAV = [
    ("/", "Dashboard"),
    ("/chat", "Chat"),
    ("/monitor", "Monitor"),
    ("/memory", "Memory Explorer"),
    ("/people", "People"),
    ("/research", "Research"),
    ("/audit", "Activity"),
    ("/settings", "Settings"),
]


def _db() -> Path:
    if _DB_PATH is None:
        raise RuntimeError("UI app is not configured; call create_app(db_path) first")
    return _DB_PATH


def _shared_deepseek_client():
    """One lazily-built client per server process.

    Per-request runtimes reuse it, so a long chat does not create (and leak) one
    transport client per message. The client is created on the server's own
    event loop, the first time a real-provider turn runs.
    """
    global _DEEPSEEK_CLIENT
    if _PROVIDER != "deepseek":
        return None
    if _DEEPSEEK_CLIENT is None:
        from companion.providers.deepseek_client import (
            build_deepseek_client,
            load_deepseek_config,
        )

        _DEEPSEEK_CLIENT = build_deepseek_client(load_deepseek_config())
    return _DEEPSEEK_CLIENT


@asynccontextmanager
async def open_runtime() -> AsyncIterator[Runtime]:
    """Open a request-scoped database connection and a fully wired runtime."""
    db = await Database(_db()).connect()
    try:
        await db.initialize()
        yield build_runtime(
            db,
            with_worker=False,
            provider=_PROVIDER,
            deepseek_client=_shared_deepseek_client(),
        )
    finally:
        await db.close()


def _chat_coordinator() -> ChatCoordinator:
    """Return the server-lifetime chat runtime, creating it on first use."""
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = ChatCoordinator(
            _db(),
            provider=_PROVIDER,
            deepseek_client_factory=_shared_deepseek_client,
        )
    return _COORDINATOR


async def _start_chat_coordinator() -> None:
    """Open the shared chat runtime before the first browser connects.

    Startup only opens the database and starts the proactive dispatcher. It
    never generates a proactive message: generation is still gated on a real
    delivered reply and on an attached client.
    """
    await _chat_coordinator().start()


async def _stop_chat_coordinator() -> None:
    """Release every timer, task and connection owned by the chat surface."""
    global _COORDINATOR
    coordinators = []
    if _COORDINATOR is not None:
        coordinators.append(_COORDINATOR)
        _COORDINATOR = None
    coordinators.extend(_RETIRED_COORDINATORS)
    _RETIRED_COORDINATORS.clear()
    for coordinator in coordinators:
        try:
            await coordinator.close()
        except Exception:  # shutdown must not raise into the server
            pass


class _Activity:
    """Content-free, in-process processing state for the monitor view.

    Only a masked speaker id, the conversation id and a monotonic start time are
    held. No utterance text, prompt, token count or address ever enters it.
    """

    def __init__(self) -> None:
        self.current: dict | None = None
        self.last_finished: float | None = None

    def begin(self, *, conversation_id: str, person_id: str | None) -> None:
        self.current = {
            "conversation": conversation_id,
            "speaker": person_id,
            "started": time.monotonic(),
        }

    def end(self) -> None:
        self.current = None
        self.last_finished = time.monotonic()


_ACTIVITY = _Activity()


def _mask(value: object, keep: int = 4) -> str:
    """Keep only the last few characters of an identifier; never show content."""
    text = str(value or "")
    if not text:
        return "—"
    return f"…{text[-keep:]}" if len(text) > keep else "…"


def _ava_name(data: dict | None) -> str:
    return str((data or {}).get("display_name") or "Ava")


def _fmt_ts(iso: str | None) -> str:
    return iso[:19].replace("T", " ") if iso else ""


def header(active: str) -> None:
    with ui.row().classes("w-full items-center gap-4 py-2 no-wrap"):
        ui.label("AI Companion").classes("text-lg font-bold")
        for path, label in NAV:
            cls = "font-bold" if path == active else ""
            ui.link(label, path).classes(cls)


def _provenance_badges(item: dict) -> None:
    flags = [
        ("human", item.get("human_confirmed"), "green"),
        ("source", item.get("source_supported"), "blue"),
        ("self", item.get("self_reported"), "teal"),
        ("3rd-party", item.get("third_party_reported"), "orange"),
        ("model", item.get("model_inferred"), "purple"),
    ]
    for name, on, color in flags:
        if on:
            ui.badge(name, color=color, text_color="white")


# -- pages ---------------------------------------------------------------


@ui.page("/")
async def dashboard() -> None:
    header("/")
    async with open_runtime() as rt:
        counts = await rt.memory_service.repo.counts()
        people = await rt.registry.list()
        jobs = await rt.repos.job.list_jobs()
        audit = await rt.repos.audit.list(limit=10)

    with ui.row().classes("gap-4"):
        for label, value in [
            ("Memories", counts["total"]),
            ("Active", counts["by_status"].get("ACTIVE", 0)),
            ("Candidates", counts["candidates"]),
            ("Needs review", counts["needs_review"]),
            ("People", len(people)),
            ("Jobs", len(jobs)),
        ]:
            with ui.card().classes("items-center"):
                ui.label(str(value)).classes("text-h4")
                ui.label(label).classes("text-caption")

    ui.label("Memory by kind").classes("text-subtitle1 mt-4")
    for kind, n in sorted(counts["by_kind"].items()):
        ui.label(f"{kind}: {n}")

    ui.label("Recent activity").classes("text-subtitle1 mt-4")
    ui.table(
        columns=[
            {"name": "created_at", "label": "When", "field": "created_at", "align": "left"},
            {"name": "actor", "label": "Actor", "field": "actor", "align": "left"},
            {"name": "action", "label": "Action", "field": "action", "align": "left"},
            {"name": "entity", "label": "Entity", "field": "entity", "align": "left"},
        ],
        rows=[
            {
                "created_at": _fmt_ts(r["created_at"]),
                "actor": r["actor"],
                "action": r["action"],
                "entity": f"{r['entity_type']}:{r['entity_id'] or ''}",
            }
            for r in audit
        ],
    )


@ui.page("/memory")
async def memory_explorer() -> None:
    header("/memory")

    query = ui.input("Search", placeholder="text…").classes("w-64")
    kind = ui.select("Kind", options=["All", *_KINDS], value="All").classes("w-48")
    status = ui.select("Status", options=["All", *_STATUSES], value="All").classes("w-48")
    include_retracted = ui.switch("Include retracted", value=False)

    table = ui.table(
        columns=[
            {"name": "memory_id", "label": "ID", "field": "memory_id", "align": "left"},
            {"name": "kind", "label": "Kind", "field": "kind", "align": "left"},
            {"name": "status", "label": "Status", "field": "status", "align": "left"},
            {"name": "visibility", "label": "Visibility", "field": "visibility", "align": "left"},
            {"name": "content", "label": "Content", "field": "content", "align": "left"},
            {"name": "updated_at", "label": "Updated", "field": "updated_at", "align": "left"},
        ],
        rows=[],
    )

    async def refresh() -> None:
        async with open_runtime() as rt:
            rows = await rt.memory_service.search(
                access=ADMIN,
                query=query.value or None,
                kinds=None if kind.value == "All" else [kind.value],
                statuses=None if status.value == "All" else [status.value],
                include_retracted=include_retracted.value,
                limit=200,
            )
        table.rows = [
            {
                "memory_id": r["memory_id"],
                "kind": r["kind"],
                "status": r["status"],
                "visibility": r["visibility"],
                "content": (r["content"][:80] + "…") if len(r["content"]) > 80 else r["content"],
                "updated_at": _fmt_ts(r["updated_at"]),
            }
            for r in rows
        ]
        table.update()

    ui.button("Search", icon="search", on_click=refresh).props("outline")
    ui.space()
    open_id = ui.input("Open by id", placeholder="mem_…").classes("w-72")

    def open_by_id() -> None:
        if open_id.value:
            ui.navigate.to(f"/memory/{open_id.value}")

    ui.button("Open", on_click=open_by_id).props("outline")

    with ui.dialog() as create_dialog, ui.card():
        ui.label("New memory").classes("text-h6")
        new_kind = ui.select("Kind", options=_KINDS, value="WORLD_KNOWLEDGE")
        new_status = ui.select("Status", options=_STATUSES, value="ACTIVE")
        new_visibility = ui.select("Visibility", options=_VISIBILITIES, value="PRIVATE")
        new_content = ui.textarea("Content").classes("w-full")
        new_human = ui.switch("Human-confirmed", value=False)

        async def do_create() -> None:
            async with open_runtime() as rt:
                item = await rt.memory_service.create(
                    kind=new_kind.value,
                    content=new_content.value,
                    access=ADMIN,
                    status=new_status.value,
                    visibility=new_visibility.value,
                    human_confirmed=new_human.value,
                )
            create_dialog.close()
            ui.notify(f"Created {item.memory_id}", type="positive")
            await refresh()

        ui.button("Create", on_click=do_create, color="primary")
        ui.button("Cancel", on_click=create_dialog.close).props("flat")

    ui.button("New memory", icon="add", on_click=create_dialog.open, color="primary")

    await refresh()


@ui.page("/memory/{memory_id}")
async def memory_detail(memory_id: str) -> None:
    header("/memory")
    async with open_runtime() as rt:
        view = await rt.memory_service.get_view(memory_id, ADMIN)

    if view is None:
        ui.label(f"Memory not found or not visible: {memory_id}").classes("text-negative")
        ui.link("Back to Explorer", "/memory")
        return

    item = view["item"]
    ui.link("← Memory Explorer", "/memory")
    ui.label(f"Kind: {item['kind']} · Status: {item['status']} · Visibility: {item['visibility']}").classes("text-caption")
    _provenance_badges(item)
    ui.markdown(item["content"]).classes("text-body1 my-4")

    with ui.row().classes("gap-4"):
        ui.label(f"id: {item['memory_id']}")
        ui.label(f"version: {item['version']}")
        ui.label(f"created by: {item['created_by']}")
        ui.label(f"updated: {_fmt_ts(item['updated_at'])}")

    async def reload() -> None:
        ui.navigate.to(f"/memory/{memory_id}")

    async def do_confirm() -> None:
        async with open_runtime() as rt:
            await rt.memory_service.confirm(memory_id, access=ADMIN)
        ui.notify("Confirmed", type="positive")
        await reload()

    async def do_retract() -> None:
        async with open_runtime() as rt:
            await rt.memory_service.retract(memory_id, access=ADMIN)
        ui.notify("Retracted", type="warning")
        await reload()

    async def do_delete() -> None:
        async with open_runtime() as rt:
            await rt.memory_service.delete(memory_id, access=ADMIN)
        ui.notify("Deleted", type="negative")
        ui.navigate.to("/memory")

    with ui.row().classes("gap-2 my-4"):
        ui.button("Edit", icon="edit", on_click=lambda: edit_dialog.open()).props("outline")
        if not item["human_confirmed"]:
            ui.button("Confirm", icon="check", color="green", on_click=do_confirm)
        ui.button("Retract", icon="block", color="orange", on_click=do_retract).props("outline")
        ui.button("Delete", icon="delete", color="red", on_click=do_delete).props("outline")

    with ui.dialog() as edit_dialog, ui.card():
        ui.label("Edit content").classes("text-h6")
        edit_content = ui.textarea("Content", value=item["content"]).classes("w-full")

        async def do_edit() -> None:
            async with open_runtime() as rt:
                try:
                    await rt.memory_service.update(
                        memory_id,
                        access=ADMIN,
                        expected_version=item["version"],
                        content=edit_content.value,
                        reason="edited in UI",
                    )
                except Exception as exc:  # optimistic-lock / permission
                    ui.notify(f"Update failed: {exc}", type="negative")
                    return
            edit_dialog.close()
            ui.notify("Saved", type="positive")
            await reload()

        ui.button("Save", on_click=do_edit, color="primary")
        ui.button("Cancel", on_click=edit_dialog.close).props("flat")

    if view["subjects"]:
        ui.label("Subjects").classes("text-subtitle1 mt-4")
        ui.table(
            columns=[
                {"name": "subject_type", "label": "Type", "field": "subject_type"},
                {"name": "subject_id", "label": "ID", "field": "subject_id"},
            ],
            rows=[{"subject_type": s["subject_type"], "subject_id": s["subject_id"]} for s in view["subjects"]],
        )

    if view["sources"]:
        ui.label("Sources").classes("text-subtitle1 mt-4")
        ui.table(
            columns=[
                {"name": "source_id", "label": "ID", "field": "source_id"},
                {"name": "source_type", "label": "Type", "field": "source_type"},
                {"name": "title", "label": "Title", "field": "title"},
                {"name": "verification_level", "label": "Verification", "field": "verification_level"},
            ],
            rows=[
                {
                    "source_id": s["source_id"],
                    "source_type": s["source_type"],
                    "title": s["title"] or "",
                    "verification_level": s.get("verification_level", ""),
                }
                for s in view["sources"]
            ],
        )

    if view["links"]:
        ui.label("Links").classes("text-subtitle1 mt-4")
        for link in view["links"]:
            other = link["linked_memory_id"] if link["memory_id"] == memory_id else link["memory_id"]
            ui.link(f"{link['link_type']}: {other}", f"/memory/{other}")

    ui.label("Revision history").classes("text-subtitle1 mt-4")
    ui.table(
        columns=[
            {"name": "version", "label": "Version", "field": "version"},
            {"name": "status", "label": "Status", "field": "status"},
            {"name": "changed_by", "label": "By", "field": "changed_by"},
            {"name": "changed_at", "label": "When", "field": "changed_at"},
            {"name": "reason", "label": "Reason", "field": "reason"},
            {"name": "content", "label": "Content", "field": "content"},
        ],
        rows=[
            {
                "version": r["version"],
                "status": r["status"],
                "changed_by": r["changed_by"],
                "changed_at": _fmt_ts(r["changed_at"]),
                "reason": r["reason"] or "",
                "content": (r["content"][:60] + "…") if len(r["content"]) > 60 else r["content"],
            }
            for r in view["revisions"]
        ],
    )


@ui.page("/people")
async def person_manager() -> None:
    header("/people")

    async def load_people() -> list[dict]:
        async with open_runtime() as rt:
            people = await rt.registry.list()
            out = []
            for p in people:
                c = await rt.registry.get_consent(p.person_id)
                out.append(
                    {
                        "person_id": p.person_id,
                        "display_name": p.display_name,
                        "role": p.role.value,
                        "status": p.registration_status.value,
                        "processing": bool(c.allow_processing) if c else False,
                        "persistence": bool(c.allow_persistence) if c else False,
                        "learning": bool(c.allow_memory_learning) if c else False,
                        "external_ai": bool(c.allow_external_ai) if c else False,
                    }
                )
            return out

    table = ui.table(
        columns=[
            {"name": "person_id", "label": "ID", "field": "person_id", "align": "left"},
            {"name": "display_name", "label": "Name", "field": "display_name", "align": "left"},
            {"name": "role", "label": "Role", "field": "role", "align": "left"},
            {"name": "status", "label": "Status", "field": "status", "align": "left"},
            {"name": "processing", "label": "Processing", "field": "processing", "align": "left"},
            {"name": "persistence", "label": "Persistence", "field": "persistence", "align": "left"},
            {"name": "learning", "label": "Learning", "field": "learning", "align": "left"},
            {"name": "external_ai", "label": "External AI", "field": "external_ai", "align": "left"},
        ],
        rows=[],
    )

    async def refresh() -> None:
        table.rows = await load_people()
        table.update()

    # -- add person -----------------------------------------------------
    with ui.dialog() as add_dialog, ui.card():
        ui.label("Register person").classes("text-h6")
        add_name = ui.input("Name")
        add_role = ui.select("Role", options=_ROLES, value="friend")
        add_processing = ui.switch("Allow processing", value=True)
        add_persistence = ui.switch("Allow persistence", value=True)
        add_learning = ui.switch("Allow memory learning", value=True)
        add_external_ai = ui.switch("Allow external AI", value=False)

        async def do_add() -> None:
            async with open_runtime() as rt:
                await rt.registry.register(
                    add_name.value,
                    add_role.value,
                    allow_processing=add_processing.value,
                    allow_persistence=add_persistence.value,
                    allow_memory_learning=add_learning.value,
                    allow_external_ai=add_external_ai.value,
                )
            add_dialog.close()
            ui.notify(f"Registered {add_name.value}", type="positive")
            await refresh()

        ui.button("Register", on_click=do_add, color="primary")
        ui.button("Cancel", on_click=add_dialog.close).props("flat")

    ui.button("Add person", icon="person_add", on_click=add_dialog.open, color="primary")

    # -- edit consent / status -----------------------------------------
    ui.label("Update consent or status").classes("text-subtitle1 mt-4")
    person_select = ui.select("Person", options=[], with_input=True).classes("w-72")
    status_select = ui.select("Status", options=["active", "suspended", "revoked"], value="active")
    c_processing = ui.switch("Allow processing")
    c_persistence = ui.switch("Allow persistence")
    c_learning = ui.switch("Allow memory learning")
    c_external_ai = ui.switch("Allow external AI")

    async def populate_select() -> None:
        people = await load_people()
        person_select.options = [p["person_id"] for p in people]
        person_select.update()

    async def do_set_status() -> None:
        if not person_select.value:
            ui.notify("Pick a person first", type="warning")
            return
        async with open_runtime() as rt:
            await rt.registry.set_status(
                person_select.value, RegistrationStatus(status_select.value)
            )
        ui.notify("Status updated", type="positive")

    async def do_set_consent() -> None:
        if not person_select.value:
            ui.notify("Pick a person first", type="warning")
            return
        async with open_runtime() as rt:
            await rt.registry.set_consent(
                Consent(
                    person_id=person_select.value,
                    allow_processing=c_processing.value,
                    allow_persistence=c_persistence.value,
                    allow_memory_learning=c_learning.value,
                    allow_external_ai=c_external_ai.value,
                    updated_at=utcnow(),
                )
            )
        ui.notify("Consent updated", type="positive")
        await refresh()

    ui.button("Apply status", on_click=do_set_status).props("outline")
    ui.button("Apply consent", on_click=do_set_consent, color="primary")

    await populate_select()
    await refresh()


@ui.page("/research")
async def research_manager() -> None:
    header("/research")
    table = ui.table(
        columns=[
            {"name": "task_id", "label": "ID", "field": "task_id", "align": "left"},
            {"name": "question", "label": "Question", "field": "question", "align": "left"},
            {"name": "status", "label": "Status", "field": "status", "align": "left"},
            {"name": "priority", "label": "Priority", "field": "priority", "align": "left"},
            {"name": "source_kind", "label": "Source", "field": "source_kind", "align": "left"},
            {"name": "result", "label": "Result", "field": "result", "align": "left"},
        ],
        rows=[],
    )

    async def refresh() -> None:
        async with open_runtime() as rt:
            tasks = await rt.repos.research.list(limit=200)
        table.rows = [
            {
                "task_id": t["task_id"],
                "question": t["question"],
                "status": t["status"],
                "priority": t["priority"],
                "source_kind": t["source_kind"],
                "result": (t["result"] or "")[:80] if t["result"] else "",
            }
            for t in tasks
        ]
        table.update()

    with ui.dialog() as add_dialog, ui.card():
        ui.label("Queue research question").classes("text-h6")
        q = ui.input("Question").classes("w-full")
        source_kind = ui.select("Source kind", options=["curiosity", "manual", "reflection"], value="manual")

        async def do_add() -> None:
            async with open_runtime() as rt:
                await rt.repos.research.create(q.value, source_kind=source_kind.value)
            add_dialog.close()
            ui.notify("Queued", type="positive")
            await refresh()

        ui.button("Queue", on_click=do_add, color="primary")
        ui.button("Cancel", on_click=add_dialog.close).props("flat")

    ui.button("New question", icon="add", on_click=add_dialog.open, color="primary")
    await refresh()


@ui.page("/audit")
async def audit_log() -> None:
    header("/audit")
    entity_filter = ui.select(
        "Entity type", options=["All", "memory", "person"], value="All"
    ).classes("w-48")

    table = ui.table(
        columns=[
            {"name": "created_at", "label": "When", "field": "created_at", "align": "left"},
            {"name": "actor", "label": "Actor", "field": "actor", "align": "left"},
            {"name": "action", "label": "Action", "field": "action", "align": "left"},
            {"name": "entity_type", "label": "Type", "field": "entity_type", "align": "left"},
            {"name": "entity_id", "label": "Entity", "field": "entity_id", "align": "left"},
            {"name": "detail", "label": "Detail", "field": "detail", "align": "left"},
        ],
        rows=[],
    )

    async def refresh() -> None:
        async with open_runtime() as rt:
            rows = await rt.repos.audit.list(
                limit=500,
                entity_type=None if entity_filter.value == "All" else entity_filter.value,
            )
        table.rows = [
            {
                "created_at": _fmt_ts(r["created_at"]),
                "actor": r["actor"],
                "action": r["action"],
                "entity_type": r["entity_type"],
                "entity_id": r["entity_id"] or "",
                "detail": r["detail"] or "",
            }
            for r in rows
        ]
        table.update()

    ui.button("Refresh", icon="refresh", on_click=refresh).props("outline")
    await refresh()


@ui.page("/settings")
async def settings_page() -> None:
    header("/settings")
    ui.label(
        "Runtime settings. Real web search and remote providers stay OFF until "
        "you enable them here."
    ).classes("text-caption")

    keys = [
        ("idle.enabled", "Idle learning enabled"),
        ("research.enabled", "Autonomous research enabled"),
        ("web_search.enabled", "Real web search (network) enabled"),
        ("deepseek.enabled", "DeepSeek provider enabled"),
        ("cognitive.provider", "Cognitive provider"),
    ]

    async def load_settings() -> dict[str, str]:
        async with open_runtime() as rt:
            return await rt.repos.settings.all()

    async def ensure_defaults() -> None:
        async with open_runtime() as rt:
            defaults = {
                "idle.enabled": "false",
                "research.enabled": "false",
                "web_search.enabled": "false",
                "deepseek.enabled": "false",
                "cognitive.provider": "mock",
            }
            for k, v in defaults.items():
                if await rt.repos.settings.get(k) is None:
                    await rt.repos.settings.set(k, v)

    await ensure_defaults()
    values = await load_settings()

    with ui.card():
        for key, label in keys:
            with ui.row().classes("items-center gap-4"):
                value = values.get(key, "")
                if value in ("true", "false"):
                    switch = ui.switch(label, value=(value == "true"))

                    async def save_bool(k=key, s=switch) -> None:
                        async with open_runtime() as rt:
                            await rt.repos.settings.set(k, "true" if s.value else "false")
                        ui.notify(f"Saved {k}", type="positive")

                    switch.on_value_change(save_bool)
                else:
                    ui.label(label)
                    inp = ui.input("Value", value=value)

                    async def save_text(k=key, i=inp) -> None:
                        async with open_runtime() as rt:
                            await rt.repos.settings.set(k, i.value)
                        ui.notify(f"Saved {k}", type="positive")

                    inp.on_value_change(save_text)


# -- chat and monitor (two views of one backend) ---------------------------
#
# ``/chat`` is the human-facing conversation surface: a speaker name, that
# person's messages, the companion's name and replies, and an input box. It
# deliberately never renders event ids, tokens, latency, HTTP status, emotion
# numbers, jobs or prompts.
#
# ``/monitor`` is the operator-facing processing view. It shows status and
# masked identifiers only (last few characters of an id), never utterance
# content, prompts or a raw IP. Both pages subscribe to the exact same backend:
# ``/chat`` consumes ``Orchestrator.process_utterance_stream`` (the same
# ResponseEvent stream the CLI uses), while ``/monitor`` observes the shared
# activity/event tables. No second agent or reply generator is created.


def _chat_speaker(people, consents: dict):
    """Prefer the owner, otherwise the first active person who may be processed."""
    eligible = [
        person
        for person in people
        if person.registration_status == RegistrationStatus.ACTIVE
        and consents.get(person.person_id) is not None
        and consents[person.person_id].allow_processing
    ]
    for person in eligible:
        if person.role == PersonRole.OWNER:
            return person
    return eligible[0] if eligible else None


def _history_timestamp(value) -> float:
    """Comparable timestamp for stored datetimes (naive rows are UTC)."""
    if value is None:
        return 0.0
    if getattr(value, "tzinfo", None) is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


async def load_chat_history(
    rt: Runtime, conversation_id: str, *, limit: int = 20
) -> list[dict]:
    """Read history from both normal replies and proactive messages.

    Normal replies live in ``agent_actions``; Ava-initiated messages live in
    ``proactive_messages`` linked by ``source_event_id``. A page reload must
    merge them in real chronological order, keeping the real message id so a
    live event that history already rendered can be skipped.
    """
    events = await rt.repos.utterance.recent_context(
        conversation_id, exclude_event_id=None, limit=limit
    )
    proactive = await rt.repos.proactive.list_for_conversation(
        conversation_id, limit=max(limit * 3, limit)
    )

    entries: list[dict] = []
    seen_ids: set[str] = set()
    for event in events:
        entries.append(
            {
                "kind": "user",
                "message_id": event.event_id,
                "text": event.text,
                "order": 0,
                "proactive": False,
                "at": _history_timestamp(event.occurred_at),
            }
        )
        seen_ids.add(event.event_id)
        action = await rt.repos.analysis.get_action(event.event_id)
        if (
            action is not None
            and action.action_type == CognitiveAction.RESPOND
            and action.response_text
        ):
            entries.append(
                {
                    "kind": "ava",
                    "message_id": action.action_id,
                    "text": action.response_text,
                    "order": 1,
                    "proactive": False,
                    "at": _history_timestamp(action.created_at),
                }
            )
            seen_ids.add(action.action_id)
    # Proactive rows are read independently of the utterance window: a message
    # may have been generated from a turn that is no longer among the last
    # ``limit`` utterances. Their real ``created_at`` puts them back in place.
    for message in proactive:
        if message.status != "COMPLETED" or not (message.text or "").strip():
            continue
        if message.message_id in seen_ids:
            continue
        entries.append(
            {
                "kind": "ava",
                "message_id": message.message_id,
                "text": message.text,
                "order": 2,
                "proactive": True,
                "at": _history_timestamp(message.created_at),
            }
        )
    entries.sort(key=lambda entry: (entry["at"], entry["order"]))
    return entries


@ui.page("/chat")
async def chat_page() -> None:
    # The chat surface is the one page that must outlive a single request:
    # the proactive cycles it starts still run while no reply is streaming.
    coordinator = _chat_coordinator()
    rt = await coordinator.start()

    people = await rt.registry.list()
    consents = {
        person.person_id: await rt.registry.get_consent(person.person_id)
        for person in people
    }
    agent = await rt.agent_service.show()
    ava = _ava_name(agent)
    speaker = _chat_speaker(people, consents)

    with ui.column().classes("w-full max-w-2xl mx-auto p-4 gap-4"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(ava).classes("text-h6")
            ui.link("Monitor", "/monitor").classes("text-caption")
        if speaker is None:
            ui.label(
                "还没有可用的人。请先在终端运行 `person add` 注册一个允许处理消息的人。"
            ).classes("text-caption text-gray-500")
            return

        conversation_id = f"chat:{speaker.person_id}"
        body = ui.column().classes("w-full gap-3")

        def add_bubble(name: str, text: str, *, mine: bool) -> None:
            align = "items-end" if mine else "items-start"
            tint = "bg-blue-50" if mine else "bg-grey-2"
            with body:
                with ui.column().classes(f"w-full {align}"):
                    with ui.card().classes(f"max-w-[85%] {tint} shadow-none"):
                        ui.label(name).classes("text-caption text-grey-6")
                        ui.label(text).classes("whitespace-pre-wrap text-body2")

        box = (
            ui.textarea(placeholder="说点什么…")
            .props("autogrow outlined")
            .classes("w-full")
        )
        with ui.row().classes("w-full justify-end"):
            send = ui.button("发送", icon="send")

        # -- live proactive delivery ------------------------------------
        # One subscription per browser tab. The coordinator fans a delivered
        # message out to every tab; none of them consumes the manager's queue
        # directly, so a second tab cannot steal the message from the first.
        client = ui.context.client
        history_proactive_ids: set[str] = set()
        live_seen: set[str] = set()
        pending_live: list = []
        history_ready = False

        def render_proactive(event) -> None:
            if getattr(client, "_deleted", False):
                return
            with client:
                text = (
                    event.display_text
                    if getattr(event, "display_text", None) is not None
                    else event.text
                )
                add_bubble(ava, text, mine=False)

        async def on_proactive(event) -> None:
            if (
                event.response_id in history_proactive_ids
                or event.response_id in live_seen
            ):
                return
            live_seen.add(event.response_id)
            if not history_ready:
                # A message committed while history was loading is replayed
                # after it, unless history already contains the same id.
                pending_live.append(event)
                return
            render_proactive(event)

        async def on_disconnect() -> None:
            await coordinator.unsubscribe(conversation_id, client.id)

        client.on_disconnect(on_disconnect)
        await coordinator.subscribe(conversation_id, client.id, on_proactive)

        async def load_history() -> None:
            entries = await load_chat_history(rt, conversation_id)
            for entry in entries:
                add_bubble(
                    speaker.display_name if entry["kind"] == "user" else ava,
                    entry["text"],
                    mine=entry["kind"] == "user",
                )
                if entry["proactive"]:
                    history_proactive_ids.add(entry["message_id"])

        async def do_send() -> None:
            value = (box.value or "").strip()
            if not value:
                return
            box.value = ""
            add_bubble(speaker.display_name, value, mine=True)

            # A live reply bubble, filled from the same ResponseEvent stream
            # the CLI consumes. Emitted deltas are already normalised, and the
            # completed event carries exactly the persisted text, so what the
            # screen shows and what history stores are the same string.
            with body:
                with ui.column().classes("w-full items-start") as reply_row:
                    with ui.card().classes("max-w-[85%] bg-grey-2 shadow-none"):
                        ui.label(ava).classes("text-caption text-grey-6")
                        live = ui.label("").classes(
                            "whitespace-pre-wrap text-body2"
                        )

            shown = ""
            finished = False
            delivered = False
            completed = None
            event = make_utterance_event(
                conversation_id=conversation_id,
                text=value,
                addressee_hint="agent",
                speaker_person_id=speaker.person_id,
                source=EventSource.TEXT,
            )
            _ACTIVITY.begin(
                conversation_id=conversation_id, person_id=speaker.person_id
            )
            try:
                # Before this turn begins: any pending IMMEDIATE/LONG_IDLE
                # work from the previous turn is stale and must be cancelled.
                coordinator.note_user_input(conversation_id, event.event_id)
                async for item in rt.orchestrator.process_utterance_stream(
                    event
                ):
                    if item.type == "response_delta":
                        shown += (
                            item.display_delta
                            if item.display_delta is not None
                            else (item.delta or "")
                        )
                        live.text = shown
                        ui.update()
                    elif item.type == "response_completed":
                        finished = True
                        completed = item
                        final = (
                            item.display_text
                            if item.display_text is not None
                            else (item.text if item.text is not None else shown)
                        )
                        if final:
                            live.text = final
                            ui.update()
                            delivered = bool(final.strip())
                        else:
                            reply_row.delete()
                    elif item.type == "response_failed":
                        finished = True
                        live.text = "（这次没有回应，请稍后重试）"
                        ui.update()
                    elif item.type == "response_cancelled":
                        finished = True
                        if not shown:
                            live.text = "（这次没有回应，请稍后重试）"
                            ui.update()
            except Exception:  # provider/transport failure: neutral, no internals
                if not finished:
                    live.text = "（这次没有回应，请稍后重试）"
                    ui.update()
                return
            finally:
                _ACTIVITY.end()

            if not finished:
                # WAIT/IGNORE leaves no reply: silence is a valid response and
                # no internal action label is ever shown on the chat surface.
                reply_row.delete()
                return

            if (
                delivered
                and completed is not None
                and completed.status == "COMPLETED"
                and completed.action == "RESPOND"
            ):
                # Only a real user turn that produced a complete, delivered
                # normal reply starts a proactive cycle. WAIT/IGNORE, failed,
                # cancelled and truncated output never do.
                coordinator.on_normal_response_delivered(
                    conversation_id=conversation_id,
                    person_id=speaker.person_id,
                    source_event_id=event.event_id,
                )

        send.on_click(do_send)
        await load_history()
        history_ready = True
        for event in pending_live:
            if event.response_id not in history_proactive_ids:
                render_proactive(event)
        pending_live.clear()


@ui.page("/monitor")
async def monitor_page() -> None:
    with ui.column().classes("w-full max-w-5xl mx-auto p-4 gap-4"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Monitor").classes("text-h6")
            ui.link("Chat", "/chat").classes("text-caption")
        ui.label(
            "处理状态与脱敏元数据。这里不显示对话正文、完整提示词或原始 IP。"
        ).classes("text-caption text-grey-6")
        state_label = ui.label("").classes("text-body2")

        table = ui.table(
            columns=[
                {"name": "time", "label": "Time", "field": "time", "align": "left"},
                {"name": "event", "label": "Event", "field": "event", "align": "left"},
                {"name": "speaker", "label": "Speaker", "field": "speaker", "align": "left"},
                {"name": "status", "label": "Status", "field": "status", "align": "left"},
                {"name": "action", "label": "Action", "field": "action", "align": "left"},
                {"name": "reason", "label": "Reason", "field": "reason", "align": "left"},
                {"name": "source", "label": "Source", "field": "source", "align": "left"},
            ],
            rows=[],
        ).classes("w-full")

        async def refresh() -> None:
            current = _ACTIVITY.current
            if current is not None:
                elapsed = time.monotonic() - current["started"]
                state_label.text = (
                    f"正在处理…  speaker={_mask(current['speaker'])}"
                    f"  已用 {elapsed:.1f}s"
                )
                state_label.classes(replace="text-body2 text-primary")
            else:
                state_label.text = "空闲"
                state_label.classes(replace="text-body2 text-grey-6")

            async with open_runtime() as rt:
                events = await rt.repos.utterance.list_events(None, limit=25)
                rows = []
                for event in events:
                    action = await rt.repos.analysis.get_action(event.event_id)
                    rows.append(
                        {
                            "time": _fmt_ts(event.occurred_at.isoformat())[11:],
                            "event": _mask(event.event_id),
                            "speaker": _mask(event.speaker_person_id),
                            "status": event.processing_status.value,
                            "action": action.action_type.value if action else "—",
                            "reason": action.reason_code.value if action else "—",
                            "source": event.source.modality,
                        }
                    )
                table.rows = rows
                table.update()

        ui.timer(1.0, refresh)
        await refresh()


def create_app(db_path: Path, *, provider: str = "mock"):
    """Register every page against the given database path.

    Returns the NiceGUI :class:`~nicegui.app.App` so callers (and tests) can
    inspect the registered routes before the server starts.
    """
    global _DB_PATH, _PROVIDER, _COORDINATOR
    new_db_path = Path(db_path)
    new_provider = (provider or "mock").lower()
    if _DB_PATH != new_db_path or _PROVIDER != new_provider:
        # A different database/provider means the previously built shared
        # runtime no longer describes this app. Retire it so shutdown still
        # closes its connection and timers.
        if _COORDINATOR is not None:
            _RETIRED_COORDINATORS.append(_COORDINATOR)
            _COORDINATOR = None
    _DB_PATH = new_db_path
    _PROVIDER = new_provider

    # Register once per application instance. NiceGUI's test reset clears the
    # handler lists, so membership (not a module flag) is what makes this
    # idempotent across create_app calls.
    startup_handlers = getattr(app, "_startup_handlers", [])
    if _start_chat_coordinator not in startup_handlers:
        app.on_startup(_start_chat_coordinator)
    shutdown_handlers = getattr(app, "_shutdown_handlers", [])
    if _stop_chat_coordinator not in shutdown_handlers:
        app.on_shutdown(_stop_chat_coordinator)

    ui.page("/")(dashboard)
    ui.page("/chat")(chat_page)
    ui.page("/monitor")(monitor_page)
    ui.page("/memory")(memory_explorer)
    ui.page("/memory/{memory_id}")(memory_detail)
    ui.page("/people")(person_manager)
    ui.page("/research")(research_manager)
    ui.page("/audit")(audit_log)
    ui.page("/settings")(settings_page)

    return app


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def serve(
    db_path: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    provider: str = "mock",
) -> None:
    """Build the app and run the NiceGUI server (blocking).

    Binding is loopback-only by default and refused on any non-loopback host,
    because the management UI has no authentication layer yet: exposing it on
    a network interface would publish every person's private memories. A real
    auth layer (not implemented in V0.0) is the precondition for lifting this.
    """
    create_app(db_path, provider=provider)
    if host not in _LOOPBACK_HOSTS:
        raise RuntimeError(
            f"refusing to bind to non-loopback host {host!r}: the management UI "
            "has no authentication and must stay loopback-only"
        )
    print(f"AI Companion UI running at http://{host}:{port}")
    ui.run(host=host, port=port, title="AI Companion", reload=False, show=False)
