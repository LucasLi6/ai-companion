"""V0.5 — deterministic, context-relevant style-example retrieval.

Retrieval is local and bounded: lexical/character-bigram overlap, scenario/act
match and recent-context match. It is not a model call and not a vector index.
An empty result is a valid result: no irrelevant reference is better than a
wrong one.
"""

from __future__ import annotations

from pathlib import Path

from companion.agent.style_examples import StyleExampleRepository
from companion.agent.style_seed import seed_style_examples
from companion.cognition.models import CognitiveTurn
from companion.domain.models import PragmaticAnalysis

ASSET = Path(__file__).resolve().parents[1] / "assets" / "ava_style_examples_v1.jsonl"
AGENT = "agent_v05"
TASK_CATEGORIES = {"task", "engineering"}


async def _seed(db):
    return await seed_style_examples(db, agent_id=AGENT, path=ASSET)


async def _select(
    db,
    *,
    text,
    mode="SOCIAL",
    person_id=None,
    turns=None,
    pragmatic=None,
    limit=3,
):
    repo = StyleExampleRepository(db)
    return await repo.list_relevant_for_prompt(
        agent_id=AGENT,
        person_id=person_id,
        current_text=text,
        recent_turns=turns or [],
        mode=mode,
        pragmatic=pragmatic,
        limit=limit,
    )


def _human(event_id, text):
    return CognitiveTurn(
        event_id=event_id,
        role="human",
        speaker_person_id="person_a",
        speaker_label="Alice",
        text=text,
    )


async def test_empty_table_returns_nothing(db):
    assert await _select(db, text="今天好累", mode="SOCIAL") == []


async def test_zero_limit_returns_nothing(db):
    await _seed(db)
    assert await _select(db, text="今天好累", limit=0) == []


async def test_task_selects_relevant_engineering_examples(db):
    await _seed(db)

    rows = await _select(db, text="帮我解释一下 PID 控制", mode="TASK")

    assert rows
    assert {row["category"] for row in rows} <= TASK_CATEGORIES
    assert any("PID" in row["user_text"] or "PID" in row["scenario"] for row in rows)
    assert len(rows) <= 3


async def test_non_technical_task_uses_task_examples_not_social(db):
    await _seed(db)

    rows = await _select(db, text="帮我写一封道歉邮件", mode="TASK")

    assert rows
    assert {row["category"] for row in rows} <= TASK_CATEGORIES
    assert any(row["category"] == "task" for row in rows)


async def test_task_never_returns_unrelated_social_examples(db):
    await _seed(db)

    rows = await _select(db, text="帮我设计一个登录流程", mode="TASK")

    assert all(row["category"] in TASK_CATEGORIES for row in rows)


async def test_social_turn_does_not_drag_in_task_material(db):
    await _seed(db)

    rows = await _select(db, text="我今天好累，什么都不想干", mode="SOCIAL")

    assert rows
    assert all(row["category"] not in TASK_CATEGORIES for row in rows)


async def test_social_pragmatics_favour_the_matching_category(db):
    await _seed(db)
    pragmatic = PragmaticAnalysis(
        primary_act="COMPLAINT",
        interaction_frame="SUPPORT",
        confidence=0.9,
    )

    rows = await _select(
        db, text="今天真的烦死了", mode="SOCIAL", pragmatic=pragmatic
    )

    assert rows
    assert rows[0]["category"] in {"emotion", "social"}


async def test_results_are_deterministic(db):
    await _seed(db)

    first = await _select(db, text="帮我解释一下 PID", mode="TASK")
    second = await _select(db, text="帮我解释一下 PID", mode="TASK")

    assert [row["example_id"] for row in first] == [
        row["example_id"] for row in second
    ]


async def test_limit_is_respected(db):
    await _seed(db)

    rows = await _select(db, text="今天", mode="SOCIAL", limit=1)

    assert len(rows) == 1


async def test_near_duplicate_examples_are_not_repeated(db):
    repo = StyleExampleRepository(db)
    for index in range(3):
        await repo.add(
            agent_id=AGENT,
            category="daily",
            scenario="几乎一样的情景",
            user_text="我今天差点迟到。",
            ideal_response=f"回应{index}",
            created_by="admin",
        )

    rows = await _select(db, text="我今天差点迟到。", mode="SOCIAL")

    assert len(rows) == 1


async def test_disabled_examples_are_excluded(db):
    await _seed(db)
    repo = StyleExampleRepository(db)
    rows = await repo.list(agent_id=AGENT, limit=100)
    pid = next(row for row in rows if "PID" in row["user_text"])
    await repo.disable(pid["example_id"])

    selected = await _select(db, text="帮我解释一下 PID 控制", mode="TASK")

    assert pid["example_id"] not in {
        row["example_id"] for row in selected
    }


async def test_person_scoped_example_is_visible_only_to_that_person(
    db, register
):
    alice = await register("Alice")
    bob = await register("Bob")
    repo = StyleExampleRepository(db)
    private = await repo.add(
        agent_id=AGENT,
        person_id=alice.person_id,
        category="relationship",
        scenario="只有 Alice 相关的私密对话",
        user_text="我家的猫叫年糕，今天又把杯子打翻了。",
        ideal_response="年糕这名字听着就不太安分。",
        created_by="admin",
    )
    await _seed(db)

    for_alice = await _select(
        db,
        text="我家的猫叫年糕，今天又把杯子打翻了。",
        person_id=alice.person_id,
    )
    for_bob = await _select(
        db,
        text="我家的猫叫年糕，今天又把杯子打翻了。",
        person_id=bob.person_id,
    )

    assert private["example_id"] in {row["example_id"] for row in for_alice}
    assert private["example_id"] not in {row["example_id"] for row in for_bob}


def _has_antecedent_example(rows):
    return any("机器狗转弯时后腿打滑" in row["user_text"] for row in rows)


async def test_antecedent_example_requires_visible_history(db):
    await _seed(db)

    without_history = await _select(
        db, text="我刚才说的那个问题呢？", mode="SOCIAL"
    )
    assert not _has_antecedent_example(without_history)

    with_history = await _select(
        db,
        text="我刚才说的那个问题呢？",
        mode="SOCIAL",
        turns=[_human("evt_h1", "机器狗转弯的时候后腿老是打滑。")],
    )
    assert _has_antecedent_example(with_history)
