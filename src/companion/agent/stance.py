"""Trusted stance continuity (V0.4.2).

A companion with a stable character should not reverse an opinion it actually
expressed two turns ago because the next sentence was friendly. This module
keeps a small, bounded record of *public* stances:

- a stance may come from an admin configuration (``ADMIN``), or
- from a sentence the companion actually emitted in a response (``RESPONSE``).

There is deliberately no ``THOUGHT`` source. A model's private analysis is not a
statement the companion made, so it must never become a continuity claim. The
extractor below is local and conservative: it records only explicit first-person
stance sentences, verbatim, and drops questions, very short fragments and
anything that resembles a memory acknowledgement.

A stance is not a fact about a person and not a durable memory. It is scoped to
the agent and, when known, the person the response was addressed to, and it is
served to the response prompt only as "what you said before — stay consistent or
explain the change naturally".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from companion.domain.models import new_id
from companion.storage.database import Database

MAX_STANCE_LENGTH = 240
MAX_STANCES_PER_RESPONSE = 2
DEFAULT_PROMPT_STANCES = 6
MAX_PROMPT_STANCES = 8

# Only explicit first-person stance constructions. "我觉得" alone is too broad
# ("我觉得你说得对" is agreement, not a stance) but recording it is harmless as
# long as it is bounded and short-lived in the prompt. The markers below favour
# preference/value statements over momentary reactions.
_STANCE_MARKERS = (
    "我觉得",
    "我认为",
    "对我来说",
    "我更喜欢",
    "我比较喜欢",
    "我不喜欢",
    "我不太喜欢",
    "我最喜欢",
    "我受不了",
    "我偏向",
    "我倾向于",
    "我宁愿",
    "我更看重",
    "我看重",
    "我在意",
)

# Statements that are just conversational maintenance or memory bookkeeping,
# not a position worth carrying forward.
_NON_STANCE_PREFIXES = (
    "我觉得你说得对",
    "我觉得可以",
    "我觉得没问题",
    "我觉得还好",
    "我记得",
    "我记住了",
    "我明白",
    "我知道了",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_QUESTION = re.compile(r"[?？]|吗|呢")
# Bare agreement is not a stance worth carrying across turns.
_TRIVIAL_AGREEMENT = re.compile(
    r"^(我觉得|我认为)(这|那)?(件|个)?(事|事情)?(还|也)?(可以|行|没问题|挺好|不错)"
    r"[。！!]?$"
)


@dataclass(frozen=True)
class AgentStance:
    stance_id: str
    agent_id: str
    person_id: Optional[str]
    conversation_id: Optional[str]
    topic: str
    stance: str
    source: str
    source_event_id: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime

    def prompt_dict(self) -> dict:
        return {
            "topic": self.topic,
            "stance": self.stance,
            "source": self.source,
        }


def normalize_stance(text: str) -> Optional[str]:
    """Return a bounded, single-line stance or ``None``."""
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_STANCE_LENGTH:
        return None
    return cleaned


def extract_stance_candidates(
    response_text: str, *, limit: int = MAX_STANCES_PER_RESPONSE
) -> list[str]:
    """Extract at most ``limit`` explicit first-person stances.

    This is intentionally conservative. It is not a semantic parser; it only
    preserves sentences that already read as a stated preference or value, so
    the prompt can remind the companion what it said without inventing one.
    """
    if not response_text:
        return []
    found: list[str] = []
    for raw in _SENTENCE_SPLIT.split(str(response_text)):
        sentence = " ".join(raw.split()).strip()
        if not sentence or len(sentence) < 6:
            continue
        if _QUESTION.search(sentence):
            continue
        if any(sentence.startswith(prefix) for prefix in _NON_STANCE_PREFIXES):
            continue
        if _TRIVIAL_AGREEMENT.match(sentence):
            continue
        if not any(marker in sentence for marker in _STANCE_MARKERS):
            continue
        stance = normalize_stance(sentence)
        if stance is None or stance in found:
            continue
        found.append(stance)
        if len(found) >= limit:
            break
    return found


class AgentStanceRepository:
    """Bounded persistence for stances the companion actually expressed."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def insert(
        self,
        *,
        agent_id: str,
        person_id: Optional[str],
        conversation_id: Optional[str],
        topic: str,
        stance: str,
        source: str,
        source_event_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        if source not in ("ADMIN", "RESPONSE"):
            raise ValueError(f"unsupported stance source {source!r}")
        cleaned = normalize_stance(stance)
        if cleaned is None:
            return None
        moment = now or datetime.now(timezone.utc)
        stance_id = new_id("stance")
        await self.db.execute(
            "INSERT OR IGNORE INTO agent_stances(stance_id, agent_id,"
            " person_id, conversation_id, topic, stance, source,"
            " source_event_id, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)",
            (
                stance_id,
                agent_id,
                person_id,
                conversation_id,
                (topic or "UNKNOWN")[:120],
                cleaned,
                source,
                source_event_id,
                moment.isoformat(),
                moment.isoformat(),
            ),
        )
        # INSERT OR IGNORE returns no usable rowcount here; the unique index on
        # (source_event_id, stance) makes retries idempotent.
        return stance_id

    async def list_recent(
        self,
        agent_id: str,
        *,
        person_id: Optional[str] = None,
        limit: int = DEFAULT_PROMPT_STANCES,
    ) -> list[AgentStance]:
        bound = max(1, min(int(limit), MAX_PROMPT_STANCES))
        if person_id is None:
            # Admin/audit view: all scopes for this agent.
            cur = await self.db.execute(
                "SELECT stance_id, agent_id, person_id, conversation_id, topic,"
                " stance, source, source_event_id, status, created_at,"
                " updated_at FROM agent_stances"
                " WHERE agent_id = ? AND status = 'ACTIVE'"
                " ORDER BY created_at DESC LIMIT ?",
                (agent_id, bound),
            )
        else:
            # Prompt view: this person's stances plus agent-global ones only.
            cur = await self.db.execute(
                "SELECT stance_id, agent_id, person_id, conversation_id, topic,"
                " stance, source, source_event_id, status, created_at,"
                " updated_at FROM agent_stances"
                " WHERE agent_id = ? AND status = 'ACTIVE'"
                " AND (person_id = ? OR person_id IS NULL)"
                " ORDER BY created_at DESC LIMIT ?",
                (agent_id, person_id, bound),
            )
        rows = await cur.fetchall()
        return [
            AgentStance(
                stance_id=row["stance_id"],
                agent_id=row["agent_id"],
                person_id=row["person_id"],
                conversation_id=row["conversation_id"],
                topic=row["topic"],
                stance=row["stance"],
                source=row["source"],
                source_event_id=row["source_event_id"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    async def retire(self, agent_id: str, stance_id: str) -> bool:
        cur = await self.db.execute(
            "UPDATE agent_stances SET status = 'RETIRED', updated_at = ?"
            " WHERE agent_id = ? AND stance_id = ? AND status = 'ACTIVE'",
            (datetime.now(timezone.utc).isoformat(), agent_id, stance_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1
