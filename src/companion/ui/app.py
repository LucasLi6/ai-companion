"""NiceGUI application for the Visual Memory Manager.

Every page opens its own short-lived SQLite connection through
:func:`open_runtime`, performs its reads/writes through the ordinary
repositories and :class:`~companion.memory.service.MemoryService`, and closes
it again. This keeps the async connection bound to the request's own event
loop (no cross-loop sharing) and means the UI obeys exactly the same
visibility, subject-privacy and optimistic-locking rules as the CLI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from nicegui import app, ui

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
    PersonRole,
    RegistrationStatus,
)
from companion.domain.models import Consent, utcnow
from companion.memory.models import AccessContext
from companion.runtime import Runtime, build_runtime
from companion.storage.database import Database

# The management UI acts as the trusted local admin, never as a person.
ADMIN = AccessContext("admin", is_admin=True, role="ui")

_KINDS = [k.value for k in MemoryKind]
_STATUSES = [s.value for s in MemoryStatus]
_VISIBILITIES = [v.value for v in MemoryVisibility]
_ROLES = [r.value for r in PersonRole]

_DB_PATH: Path | None = None

NAV = [
    ("/", "Dashboard"),
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


@asynccontextmanager
async def open_runtime() -> AsyncIterator[Runtime]:
    """Open a request-scoped database connection and a fully wired runtime."""
    db = await Database(_db()).connect()
    try:
        await db.initialize()
        yield build_runtime(db, with_worker=False)
    finally:
        await db.close()


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

        async def do_add() -> None:
            async with open_runtime() as rt:
                await rt.registry.register(
                    add_name.value,
                    add_role.value,
                    allow_processing=add_processing.value,
                    allow_persistence=add_persistence.value,
                    allow_memory_learning=add_learning.value,
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


def create_app(db_path: Path):
    """Register every page against the given database path.

    Returns the NiceGUI :class:`~nicegui.app.App` so callers (and tests) can
    inspect the registered routes before the server starts.
    """
    global _DB_PATH
    _DB_PATH = Path(db_path)

    ui.page("/")(dashboard)
    ui.page("/memory")(memory_explorer)
    ui.page("/memory/{memory_id}")(memory_detail)
    ui.page("/people")(person_manager)
    ui.page("/research")(research_manager)
    ui.page("/audit")(audit_log)
    ui.page("/settings")(settings_page)

    return app


def serve(db_path: Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Build the app and run the NiceGUI server (blocking)."""
    create_app(db_path)
    print(f"AI Companion UI running at http://{host}:{port}")
    ui.run(host=host, port=port, title="AI Companion", reload=False, show=False)
