"""Phase 6: UI data-layer behaviour and loopback-only binding.

These exercise the exact MemoryService / registry / research operations the
NiceGUI buttons call (through the request-scoped :func:`open_runtime`), plus
the loopback-only guarantee of :func:`serve`.
"""

from __future__ import annotations

import pytest

nicegui = pytest.importorskip("nicegui")

from companion.domain.enums import MemoryKind, MemoryStatus, MemoryVisibility  # noqa: E402
from companion.idle import build_idle_scheduler  # noqa: E402
from companion.ui import app as ui_app  # noqa: E402
from companion.ui.app import ADMIN, create_app, open_runtime, serve  # noqa: E402


@pytest.fixture
def ui_db(tmp_path):
    create_app(tmp_path / "ui.db")
    return tmp_path / "ui.db"


async def test_ui_create_and_read(ui_db):
    async with open_runtime() as rt:
        item = await rt.memory_service.create(
            kind=MemoryKind.WORLD_KNOWLEDGE,
            content="烧录使用 STM32CubeProgrammer",
            access=ADMIN,
            status=MemoryStatus.ACTIVE,
            visibility=MemoryVisibility.PUBLIC,
        )
    async with open_runtime() as rt:
        view = await rt.memory_service.get_view(item.memory_id, ADMIN)
    assert view is not None
    assert view["item"]["content"] == "烧录使用 STM32CubeProgrammer"
    assert view["item"]["status"] == "ACTIVE"
    assert view["item"]["visibility"] == "PUBLIC"


async def test_ui_edit_and_history(ui_db):
    async with open_runtime() as rt:
        item = await rt.memory_service.create(
            kind=MemoryKind.PROJECT_STATE, content="v1", access=ADMIN
        )
        mid = item.memory_id
        updated = await rt.memory_service.update(
            mid, access=ADMIN, expected_version=1, content="v2", reason="edited in UI"
        )
        assert updated.version == 2
    async with open_runtime() as rt:
        view = await rt.memory_service.get_view(mid, ADMIN)
    assert view["item"]["content"] == "v2"
    assert [r["version"] for r in view["revisions"]] == [2, 1]


async def test_ui_retract_excludes_from_default_search(ui_db):
    async with open_runtime() as rt:
        item = await rt.memory_service.create(
            kind=MemoryKind.EPISODE, content="将被撤销的记忆", access=ADMIN
        )
        mid = item.memory_id
        await rt.memory_service.retract(mid, access=ADMIN)
    async with open_runtime() as rt:
        default = await rt.memory_service.search(access=ADMIN, query="将被撤销")
        with_retracted = await rt.memory_service.search(
            access=ADMIN, query="将被撤销", include_retracted=True
        )
        item = await rt.memory_service.get_item(mid, ADMIN)
    assert all(r["memory_id"] != mid for r in default)
    assert any(r["memory_id"] == mid for r in with_retracted)
    assert item.status == MemoryStatus.RETRACTED


async def test_ui_confirm_candidate(ui_db):
    async with open_runtime() as rt:
        item = await rt.memory_service.create(
            kind=MemoryKind.WORLD_KNOWLEDGE,
            content="待确认的候选",
            access=ADMIN,
            status=MemoryStatus.CANDIDATE,
        )
        confirmed = await rt.memory_service.confirm(item.memory_id, access=ADMIN)
    assert confirmed.status == MemoryStatus.ACTIVE
    assert confirmed.human_confirmed is True


async def test_ui_research_status_flow(ui_db):
    async with open_runtime() as rt:
        await rt.repos.settings.set("research.enabled", "true")
        task_id = await rt.repos.research.create("OTA 为什么不稳定？", source_kind="manual")
        scheduler = build_idle_scheduler(rt)
        await scheduler.research.run_all(limit=1)
    async with open_runtime() as rt:
        tasks = await rt.repos.research.list()
        task = next(t for t in tasks if t["task_id"] == task_id)
    assert task["status"] == "COMPLETED"
    assert task["result"]


def test_serve_refuses_non_loopback(tmp_path):
    with pytest.raises(RuntimeError, match="loopback-only"):
        serve(tmp_path / "ui.db", host="0.0.0.0", port=8080)


def test_serve_allows_loopback(tmp_path, monkeypatch):
    calls: dict = {}

    def fake_run(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(ui_app.ui, "run", fake_run)
    serve(tmp_path / "ui.db", host="127.0.0.1", port=8080)
    assert calls["host"] == "127.0.0.1"
