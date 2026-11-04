"""V0.5 — importing the reviewed Ava style-example batch.

The 50-example JSONL is style reference material, never conversation history.
Import is an explicit admin action, idempotent, and reversible by batch marker
without a schema change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from companion.agent.style_examples import StyleExampleRepository
from companion.agent.style_seed import (
    DEFAULT_BATCH,
    load_style_examples,
    seed_style_examples,
)

ASSET = Path(__file__).resolve().parents[1] / "assets" / "ava_style_examples_v1.jsonl"
AGENT = "agent_v05"
SOCIAL_CATEGORIES = {
    "daily",
    "playful",
    "relationship",
    "emotion",
    "debate",
    "identity",
    "preferences",
    "continuity",
    "social",
    "other",
}


async def _count(db, where="1=1", params=()):
    cur = await db.execute(
        f"SELECT COUNT(*) AS n FROM dialogue_style_examples WHERE {where}", params
    )
    return (await cur.fetchone())["n"]


def test_batch_file_has_fifty_unique_complete_examples():
    records = load_style_examples(ASSET)

    assert len(records) == 50
    assert len({record["id"] for record in records}) == 50
    for record in records:
        assert record["scenario"].strip()
        assert record["user_text"].strip()
        assert record["ideal_response"].strip()
        assert record["category"] == record["category"].strip().lower()
        assert record["category"] in SOCIAL_CATEGORIES | {"task", "engineering"}


async def test_import_saves_all_fifty_as_global_examples(db):
    result = await seed_style_examples(db, agent_id=AGENT, path=ASSET)

    assert result == {
        "total": 50,
        "created": 50,
        "skipped": 0,
        "dry_run": False,
        "batch": DEFAULT_BATCH,
        "agent_id": AGENT,
    }
    assert await _count(db) == 50
    assert await _count(db, "person_id IS NULL") == 50
    assert await _count(db, "created_by = ?", (DEFAULT_BATCH,)) == 50
    assert await _count(db, "status = 'ACTIVE'") == 50

    repo = StyleExampleRepository(db)
    rows = await repo.list(agent_id=AGENT, limit=100)
    scenarios = [row["scenario"] for row in rows]
    assert all(scenario.startswith("[") for scenario in scenarios)
    assert any(scenario.startswith("[engineering]") for scenario in scenarios)
    assert any(scenario.startswith("[daily]") for scenario in scenarios)


async def test_import_is_idempotent(db):
    first = await seed_style_examples(db, agent_id=AGENT, path=ASSET)
    second = await seed_style_examples(db, agent_id=AGENT, path=ASSET)

    assert first["created"] == 50
    assert second["created"] == 0
    assert second["skipped"] == 50
    assert await _count(db) == 50


async def test_dry_run_writes_nothing_and_reports_the_plan(db):
    result = await seed_style_examples(
        db, agent_id=AGENT, path=ASSET, dry_run=True
    )

    assert result["created"] == 50
    assert result["dry_run"] is True
    assert await _count(db) == 0

    # A dry run does not make the next real import skip anything.
    real = await seed_style_examples(db, agent_id=AGENT, path=ASSET)
    assert real["created"] == 50


async def test_disable_batch_uses_created_by_and_leaves_other_rows(db):
    await seed_style_examples(db, agent_id=AGENT, path=ASSET)
    repo = StyleExampleRepository(db)
    manual = await repo.add(
        agent_id=AGENT,
        category="daily",
        scenario="管理员手写参考",
        user_text="今天还行。",
        ideal_response="那就好。",
        created_by="admin",
    )

    disabled = await repo.disable_batch(
        agent_id=AGENT, created_by=DEFAULT_BATCH
    )

    assert disabled == 50
    assert await _count(db, "status = 'ACTIVE'") == 1
    assert await _count(db, "status = 'DISABLED'") == 50
    kept = await repo.get(manual["example_id"])
    assert kept["status"] == "ACTIVE"

    active = await repo.list(agent_id=AGENT)
    assert [row["example_id"] for row in active] == [manual["example_id"]]


async def test_reimport_after_disable_does_not_silently_reenable(db):
    await seed_style_examples(db, agent_id=AGENT, path=ASSET)
    repo = StyleExampleRepository(db)
    await repo.disable_batch(agent_id=AGENT, created_by=DEFAULT_BATCH)

    result = await seed_style_examples(db, agent_id=AGENT, path=ASSET)

    assert result["created"] == 0
    assert result["skipped"] == 50
    assert await _count(db, "status = 'ACTIVE'") == 0


async def test_import_is_scoped_to_the_target_agent(db):
    await seed_style_examples(db, agent_id=AGENT, path=ASSET)
    await seed_style_examples(db, agent_id="agent_other", path=ASSET)

    assert await _count(db, "agent_id = ?", (AGENT,)) == 50
    assert await _count(db, "agent_id = ?", ("agent_other",)) == 50


async def test_missing_agent_id_is_rejected(db):
    with pytest.raises(ValueError):
        await seed_style_examples(db, agent_id="", path=ASSET)


def test_invalid_jsonl_fails_loudly(tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        json.dumps(
            {
                "id": "E01",
                "category": "daily",
                "scenario": "缺字段",
                "user_text": "你好。",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ideal_response"):
        load_style_examples(broken)


def test_malformed_json_line_names_the_line(tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=":1: invalid JSON"):
        load_style_examples(broken)
