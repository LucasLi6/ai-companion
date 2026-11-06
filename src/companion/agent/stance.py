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
from typing import Mapping, Optional

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
# A passing mood is not a stable position. V0.4.3 keeps the stance extractor
# deliberately conservative so "我觉得今天真烦" never becomes character.
_TRANSIENT = re.compile(
    r"(现在|目前|今天|今晚|刚才|刚刚|最近|这两天|这几天|这会儿|这一刻)"
    r"[^。！？!?]{0,16}"
    r"(烦|累|难过|伤心|生气|焦虑|紧张|压力|心情|崩|沮丧|郁闷)"
)
_HYPOTHETICAL_OR_QUOTED = re.compile(
    r"(如果|假如|要是|若是|万一|假设|哪天|"
    r"你说|他说|她说|别人说|听说|据说|有人说|台词|歌词|引用|转述|"
    r"\bif\b|\bwould\b|\bsuppose\b|\bquote\b)",
    re.IGNORECASE,
)
# V0.9.6: the speaker is now encouraged to react casually, joke and use
# half-sentences. A reaction to *this* moment must not become permanent
# character, even when it happens to start with "我觉得". These markers are
# deliberately about register (laughter, surprise, unfinished speech) rather
# than about any particular opinion.
_CASUAL_REACTION = re.compile(
    r"(哈哈|嘿+|诶|欸|哎|哟|唔|嗯|呃|嗐|"
    r"笑死|好笑|好玩|离谱|绝了|服了|太真实|"
    r"居然|竟然|没想到|吓一跳|"
    r"……|~|～|!{2,}|！{2,})"
)
# A comment on the news result / event in front of us is a remark, not a
# standing position. Only deictic references to the current item count; a
# general preference ("我更看重可验证的证据") is unaffected.
_EVENT_SPECIFIC_REMARK = re.compile(
    r"(这次|这回|这波|这条|这个新闻|这条新闻|这条消息|这个结果|这局|"
    r"刚才那|刚刚那|今天这)"
)
# "我觉得" / "我认为" / "对我来说" introduce any opinion, including a
# prediction about tomorrow. On their own they are not a stated preference;
# with them present, the sentence must also carry a value judgement.
_GENERIC_OPINION_MARKERS = ("我觉得", "我认为", "对我来说")
_VALUE_JUDGEMENT = re.compile(
    r"(喜欢|讨厌|受不了|宁愿|更愿意|愿意|偏向|倾向|看重|在意|"
    r"赞成|反对|值得|重要|更好|最好|适合|舍不得|热爱)"
)

_CJK_RUN = re.compile(r"[一-鿿]{2,}")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def _terms(text: str) -> set[str]:
    """Small, deterministic relevance vocabulary (CJK bigrams + latin words)."""
    value = str(text or "")
    terms: set[str] = set()
    for run in _CJK_RUN.findall(value):
        for index in range(len(run) - 1):
            terms.add(run[index : index + 2])
    terms.update(word.lower() for word in _LATIN_WORD.findall(value))
    return terms


def filter_relevant_stances(
    stances,
    *,
    topic: str | None = None,
    query_text: str = "",
    limit: int = DEFAULT_PROMPT_STANCES,
) -> list:
    """Keep only stances that plausibly bear on the current topic.

    A stable opinion about music must not follow the companion into a
    discussion about a debugger. Relevance is a local, conservative token
    overlap between the stance (topic + text) and the current topic/utterance.
    An unmatched *response-derived* stance is dropped rather than injected as
    generic character dressing; the persona can still form a fresh view from
    its own values. Admin-configured stances are curated character, so they
    stay available (within the bounded limit) even without a token overlap.
    """
    candidates = list(stances or [])
    if not candidates:
        return []

    def field(record, name):
        return (
            record.get(name)
            if isinstance(record, Mapping)
            else getattr(record, name, None)
        )

    bound = max(1, min(int(limit), MAX_PROMPT_STANCES))
    # Admin-configured stances are curated character, not conversational
    # history: keep them available (bounded) so the persona does not lose its
    # principles just because the current wording shares no tokens.
    selected: list = [r for r in candidates if field(r, "source") == "ADMIN"][:bound]
    remaining = bound - len(selected)
    if remaining <= 0:
        return selected

    topic_terms = _terms(topic or "")
    query_terms = _terms(query_text or "") | topic_terms
    if not query_terms:
        return selected

    scored: list[tuple[int, object]] = []
    for record in candidates:
        if field(record, "source") == "ADMIN":
            continue
        stance_text = field(record, "stance")
        if not stance_text:
            continue
        stance_terms = _terms(f"{field(record, 'topic') or ''} {stance_text}")
        overlap = len(stance_terms & query_terms)
        if overlap == 0:
            continue
        score = overlap
        if topic_terms and stance_terms & topic_terms:
            score += 3
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return selected + [record for _score, record in scored[:remaining]]


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
        if _TRANSIENT.search(sentence):
            # A momentary mood ("我觉得今天真烦") is not a durable position.
            continue
        if _HYPOTHETICAL_OR_QUOTED.search(sentence):
            # Hypotheticals, reported speech and quotations belong to someone
            # else's view, not to the companion's own stance.
            continue
        if _CASUAL_REACTION.search(sentence):
            # A joke, interjection, surprise or unfinished colloquial line is
            # momentary wording, not a stable position.
            continue
        if _EVENT_SPECIFIC_REMARK.search(sentence):
            # A passing comment on this turn's news/event is not character.
            continue
        if any(sentence.startswith(prefix) for prefix in _NON_STANCE_PREFIXES):
            continue
        if _TRIVIAL_AGREEMENT.match(sentence):
            continue
        markers = [marker for marker in _STANCE_MARKERS if marker in sentence]
        if not markers:
            continue
        if all(
            marker in _GENERIC_OPINION_MARKERS for marker in markers
        ) and not _VALUE_JUDGEMENT.search(sentence):
            # "我觉得明天会下雨" is a prediction; "对我来说，只能靠语言了解"
            # is an explanation of this moment. Neither is a stable position.
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
