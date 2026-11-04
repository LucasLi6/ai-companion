"""Trusted, admin-authored dialogue style examples (V0.4).

A style example says "a response in this situation should meet this standard".
It is never a real conversation turn and never a fixed reply. Only the trusted
CLI/admin writes here: the cognitive/response providers have no write path to
this table.

Scoping rules:

- ``person_id IS NULL`` is a global example for the agent.
- a person-scoped example is returned only for that person;
- at most three are ever handed to the prompt, and disabled rows are skipped.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from companion.agent.reply_mode import has_task_signal
from companion.domain.models import new_id, utcnow
from companion.storage.database import Database
from companion.storage.repositories import _iso


class StyleExampleError(ValueError):
    pass


class StyleExampleRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def add(
        self,
        *,
        agent_id: str,
        scenario: str,
        user_text: str,
        ideal_response: str,
        person_id: Optional[str] = None,
        created_by: str = "admin",
        category: Optional[str] = None,
    ) -> dict:
        if category:
            category = _clean(category, 40, "category").lower()
            scenario = f"[{category}] {_clean(scenario, 150, 'scenario')}"
        scenario = _clean(scenario, 200, "scenario")
        user_text = _clean(user_text, 4000, "user_text")
        ideal_response = _clean(ideal_response, 8000, "ideal_response")
        if not agent_id:
            raise StyleExampleError("agent_id is required")
        if person_id is not None:
            cur = await self.db.execute(
                "SELECT 1 FROM persons WHERE person_id = ?", (person_id,)
            )
            if await cur.fetchone() is None:
                raise StyleExampleError(f"unknown person: {person_id}")
        now = _iso(utcnow())
        example_id = new_id("style")
        await self.db.execute(
            "INSERT INTO dialogue_style_examples(example_id, agent_id, person_id,"
            " scenario, user_text, ideal_response, status, created_by, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)",
            (
                example_id,
                agent_id,
                person_id,
                scenario,
                user_text,
                ideal_response,
                created_by,
                now,
                now,
            ),
        )
        return await self.get(example_id)

    async def get(self, example_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT * FROM dialogue_style_examples WHERE example_id = ?",
            (example_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list(
        self,
        *,
        agent_id: Optional[str] = None,
        person_id: Optional[str] = None,
        include_disabled: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        where = []
        params: list = []
        if agent_id is not None:
            where.append("agent_id = ?")
            params.append(agent_id)
        if person_id is not None:
            where.append("(person_id IS NULL OR person_id = ?)")
            params.append(person_id)
        if not include_disabled:
            where.append("status = 'ACTIVE'")
        sql = "SELECT * FROM dialogue_style_examples"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        cur = await self.db.execute(sql, tuple(params))
        return [dict(r) for r in await cur.fetchall()]

    async def list_for_prompt(
        self,
        *,
        agent_id: str,
        person_id: Optional[str],
        scenario: Optional[str] = None,
        limit: int = 3,
    ) -> list[dict]:
        """Return at most ``limit`` active examples visible to this person.

        A person-scoped row is preferred over a global row when both exist for
        the same scenario, but a global row is never person-specific evidence.
        """
        if not agent_id:
            return []
        limit = max(0, min(int(limit), 3))
        if limit == 0:
            return []
        params: list = [agent_id]
        scope = "(person_id IS NULL"
        if person_id is not None:
            scope += " OR person_id = ?"
            params.append(person_id)
        scope += ")"
        sql = (
            "SELECT example_id, agent_id, person_id, scenario, user_text,"
            " ideal_response, status FROM dialogue_style_examples"
            f" WHERE agent_id = ? AND status = 'ACTIVE' AND {scope}"
        )
        if scenario:
            sql += " ORDER BY CASE WHEN scenario = ? THEN 0 ELSE 1 END,"
            sql += " CASE WHEN person_id IS NULL THEN 1 ELSE 0 END, created_at DESC"
            params.append(scenario)
        else:
            sql += " ORDER BY CASE WHEN person_id IS NULL THEN 1 ELSE 0 END, created_at DESC"
        sql += " LIMIT ?"
        params.append(limit)
        cur = await self.db.execute(sql, tuple(params))
        return [
            {
                "example_id": row["example_id"],
                "scenario": row["scenario"],
                "user_text": row["user_text"],
                "ideal_response": row["ideal_response"],
            }
            for row in await cur.fetchall()
        ]

    async def list_relevant_for_prompt(
        self,
        *,
        agent_id: str,
        person_id: Optional[str],
        current_text: str,
        recent_turns: Optional[list] = None,
        mode: Any = "SOCIAL",
        pragmatic: Any = None,
        scene_state: Any = None,
        limit: int = 3,
    ) -> list[dict]:
        """Return up to ``limit`` relevant active examples, or none.

        Retrieval is deterministic and local: lexical/character-bigram overlap,
        the example's category versus the turn's pragmatic shape, and overlap
        with a few recent turns. It is not a model call and not a vector index.

        Seed examples whose text begins with ``前情：`` are only eligible when
        the required antecedent actually appears in the visible human history.
        This prevents a style example from inventing a shared memory.
        """
        if not agent_id:
            return []
        limit = max(0, min(int(limit), 3))
        if limit == 0:
            return []

        rows = await self._prompt_candidates(agent_id=agent_id, person_id=person_id)
        if not rows:
            return []

        mode_value = str(getattr(mode, "value", mode) or "SOCIAL").upper()
        recent_texts = _turn_texts(recent_turns or [])
        human_recent = _human_turn_texts(recent_turns or [])
        query_tokens = _tokens(" ".join([current_text or "", *recent_texts]))
        recent_tokens = _tokens(" ".join(recent_texts))
        scene_topic = str(_field(scene_state, "topic", "") or "")

        mode_categories = _mode_categories(mode_value, pragmatic)
        allowed = (
            _TASK_CATEGORIES
            if mode_value == "TASK"
            else _SOCIAL_CATEGORIES
        )
        favoured = _favoured_categories(mode_value, pragmatic)
        technical_turn = _has_technical_hint(" ".join([current_text or "", *recent_texts]))

        scored: list[dict] = []
        for row in rows:
            category, scenario_text = _split_scenario(row.get("scenario", ""))
            if category not in allowed:
                continue

            example_text = " ".join(
                [
                    scenario_text,
                    str(row.get("user_text") or ""),
                ]
            )
            semantic = _dice(query_tokens, _tokens(example_text))
            if category == "engineering" and mode_value != "TASK" and semantic < 0.18:
                # A technical example can be relevant to a social turn about
                # engineering, but it must actually overlap. Otherwise a mere
                # technical noun would drag task material into small talk.
                continue

            antecedent = _antecedent_tokens(
                str(row.get("user_text") or ""),
                str(row.get("scenario") or ""),
            )
            if antecedent and not _antecedent_supported(antecedent, human_recent):
                continue

            if mode_value == "TASK":
                preferred = "engineering" if technical_turn else "task"
                scenario_match = 1.0 if category == preferred else 0.75
            elif category in favoured:
                scenario_match = 1.0
            elif category in mode_categories:
                scenario_match = 0.7
            else:
                scenario_match = 0.3

            context_match = max(
                _dice(recent_tokens, _tokens(example_text)),
                _dice(_tokens(scene_topic), _tokens(example_text)),
            )
            score = semantic * 0.40 + scenario_match * 0.40 + context_match * 0.20
            if row.get("person_id") is not None:
                score += 0.02
            if score < 0.15:
                continue

            scored.append(
                {
                    "example_id": row["example_id"],
                    "category": category,
                    "scenario": scenario_text,
                    "user_text": str(row.get("user_text") or ""),
                    "ideal_response": str(row.get("ideal_response") or ""),
                    "_score": round(score, 6),
                    "_created_at": str(row.get("created_at") or ""),
                    "_person_scoped": row.get("person_id") is not None,
                }
            )

        scored.sort(
            key=lambda item: (
                -item["_score"],
                0 if item["_person_scoped"] else 1,
                item["_created_at"],
                item["example_id"],
            )
        )

        selected: list[dict] = []
        for item in scored:
            if any(
                _dice(
                    _tokens(item["scenario"] + " " + item["user_text"]),
                    _tokens(other["scenario"] + " " + other["user_text"]),
                )
                > 0.72
                for other in selected
            ):
                continue
            if any(
                item["category"] == other["category"]
                and item["scenario"] == other["scenario"]
                for other in selected
            ):
                continue
            selected.append(item)
            if len(selected) >= limit:
                break

        return [
            {
                "example_id": item["example_id"],
                "category": item["category"],
                "scenario": item["scenario"],
                "user_text": item["user_text"],
                "ideal_response": item["ideal_response"],
            }
            for item in selected
        ]

    async def _prompt_candidates(
        self, *, agent_id: str, person_id: Optional[str]
    ) -> list[dict]:
        params: list = [agent_id]
        scope = "(person_id IS NULL"
        if person_id is not None:
            scope += " OR person_id = ?"
            params.append(person_id)
        scope += ")"
        sql = (
            "SELECT example_id, agent_id, person_id, scenario, user_text,"
            " ideal_response, status, created_at FROM dialogue_style_examples"
            f" WHERE agent_id = ? AND status = 'ACTIVE' AND {scope}"
            " ORDER BY created_at DESC LIMIT 500"
        )
        cur = await self.db.execute(sql, tuple(params))
        return [dict(row) for row in await cur.fetchall()]

    async def disable_batch(
        self, *, agent_id: str, created_by: str, actor: str = "admin"
    ) -> int:
        """Disable every ACTIVE example from one seeded batch.

        No new table is needed: the existing ``created_by`` column is the batch
        marker. Other examples, including user-authored ones, are untouched.
        """
        if not agent_id or not created_by:
            return 0
        now = _iso(utcnow())
        cur = await self.db.execute(
            "UPDATE dialogue_style_examples SET status = 'DISABLED', updated_at = ?"
            " WHERE agent_id = ? AND created_by = ? AND status = 'ACTIVE'",
            (now, agent_id, created_by),
        )
        return int(cur.rowcount or 0)

    async def disable(self, example_id: str, *, actor: str = "admin") -> dict:
        now = _iso(utcnow())
        await self.db.execute(
            "UPDATE dialogue_style_examples SET status = 'DISABLED', updated_at = ?"
            " WHERE example_id = ?",
            (now, example_id),
        )
        row = await self.get(example_id)
        if row is None:
            raise StyleExampleError(f"style example not found: {example_id}")
        return row


def _clean(value: object, limit: int, name: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise StyleExampleError(f"{name} must not be empty")
    if len(text) > limit:
        raise StyleExampleError(f"{name} must be at most {limit} characters")
    return text


_CATEGORY_RE = re.compile(r"^\[([a-z_]+)\]\s*(.*)$", re.IGNORECASE)
_TASK_CATEGORIES = frozenset({"task", "engineering"})
_SOCIAL_CATEGORIES = frozenset(
    {
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
)

_TECHNICAL_HINTS = (
    "pid",
    "uart",
    "dma",
    "stm32",
    "python",
    "c++",
    "java",
    "sql",
    "代码",
    "程序",
    "算法",
    "电路",
    "机器人",
    "imu",
    "mosfet",
    "寄存器",
    "中断",
    "协议",
    "原理",
    "机制",
    "报错",
    "bug",
    "调试",
    "数据",
    "论文",
    "报告",
    "数学",
    "物理",
    "工程",
)

_PRAGMATIC_CATEGORIES = {    "TEASING": {"playful"},
    "SMALL_TALK": {"social", "daily"},
    "GREETING": {"social", "daily"},
    "GRATITUDE": {"social", "relationship"},
    "SELF_DISCLOSURE": {"emotion", "social"},
    "COMPLAINT": {"emotion", "social"},
    "APOLOGY": {"relationship", "continuity"},
    "INVITATION": {"social", "playful"},
    "REPAIR": {"relationship", "continuity"},
}
_FRAME_CATEGORIES = {
    "PLAYFUL": {"playful"},
    "CASUAL": {"daily", "social"},
    "SUPPORT": {"emotion", "social"},
    "REPAIR": {"relationship", "continuity"},
    "CONFLICT": {"debate", "relationship"},
    "TASK": {"task", "engineering"},
}


def _split_scenario(value: Any) -> tuple[str, str]:
    text = " ".join(str(value or "").split()).strip()
    match = _CATEGORY_RE.match(text)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    # Legacy/unlabelled rows still participate; infer only a coarse category.
    if has_task_signal(text):
        return "engineering", text
    return "other", text


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _turn_texts(turns: list) -> list[str]:
    texts: list[str] = []
    for turn in turns:
        text = " ".join(str(_field(turn, "text", "") or "").split())
        if text:
            texts.append(text)
    return texts


def _human_turn_texts(turns: list) -> list[str]:
    texts: list[str] = []
    for turn in turns:
        if str(_field(turn, "role", "")).lower() == "assistant":
            continue
        text = " ".join(str(_field(turn, "text", "") or "").split())
        if text:
            texts.append(text)
    return texts


def _has_technical_hint(value: Any) -> bool:
    text = str(value or "").lower()
    return any(hint in text for hint in _TECHNICAL_HINTS)


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    tokens: set[str] = set()
    for run in re.findall(r"[一-鿿]+", text):
        if len(run) == 1:
            tokens.add(run)
        for index in range(len(run) - 1):
            tokens.add(run[index : index + 2])
    for word in re.findall(r"[a-z0-9_+#.]+", text):
        if len(word) > 1:
            tokens.add(word)
    return tokens


def _dice(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return 2.0 * len(left & right) / (len(left) + len(right))


def _antecedent_tokens(user_text: str, scenario: str) -> set[str]:
    for value in (user_text, scenario):
        text = str(value or "")
        match = re.search(r"前情[：:]\s*(.*?)(?:当前[：:]|$)", text, re.S)
        if match:
            return _tokens(match.group(1))
    return set()


def _antecedent_supported(antecedent: set[str], human_recent: list[str]) -> bool:
    if not antecedent:
        return True
    visible = _tokens(" ".join(human_recent))
    return _dice(antecedent, visible) >= 0.2


def _mode_categories(mode: str, pragmatic: Any) -> set[str]:
    if mode == "TASK":
        return set(_TASK_CATEGORIES)
    categories = set(_SOCIAL_CATEGORIES)
    act = str(_field(pragmatic, "primary_act", "") or "").upper()
    frame = str(_field(pragmatic, "interaction_frame", "") or "").upper()
    categories.update(_PRAGMATIC_CATEGORIES.get(act, set()))
    categories.update(_FRAME_CATEGORIES.get(frame, set()))
    return categories


def _favoured_categories(mode: str, pragmatic: Any) -> set[str]:
    if mode == "TASK":
        return set(_TASK_CATEGORIES)
    categories: set[str] = set()
    act = str(_field(pragmatic, "primary_act", "") or "").upper()
    frame = str(_field(pragmatic, "interaction_frame", "") or "").upper()
    role = str(_field(pragmatic, "setup_payoff_role", "") or "").upper()
    categories.update(_PRAGMATIC_CATEGORIES.get(act, set()))
    categories.update(_FRAME_CATEGORIES.get(frame, set()))
    if role in {"SETUP", "PAYOFF"}:
        categories.update({"playful", "continuity"})
    return categories & set(_SOCIAL_CATEGORIES)
