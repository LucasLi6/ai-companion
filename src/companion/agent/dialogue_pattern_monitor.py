"""Detect repetitive reply patterns and turn them into prompt-only feedback.

The monitor looks at the companion's last few *actual* assistant replies and
counts mechanical habits that make a conversation feel generated:

- the same comfort opening ("我理解…", "听起来…", "别担心…");
- ending nearly every reply with a question;
- over-using the user's or the agent's name;
- repeating a joke/formula or the same sentence shape;
- mirroring the user's emotional label instead of responding to it;
- repeated companionship assurances ("我会一直陪着你").

It **never rewrites a reply**, never calls a model, and never changes the
Policy action. Its output is a short list of reminders attached to the single
response prompt, so the next reply can vary naturally where variation is
appropriate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

MAX_ANALYSED_REPLIES = 5
MIN_REPEATS = 2

_COMFORT_OPENINGS = (
    "我理解",
    "我能理解",
    "听起来",
    "别担心",
    "不要担心",
    "没事的",
    "会没事",
    "辛苦了",
    "我明白",
    "抱抱",
    "先别急",
    "慢慢来",
)
_ASSURANCES = (
    "我会一直",
    "我一直都在",
    "我在这里",
    "我陪着你",
    "我会陪",
    "放心吧",
    "你可以依靠我",
)
_EMOTION_WORDS = (
    "难过",
    "伤心",
    "焦虑",
    "压力",
    "开心",
    "生气",
    "委屈",
    "孤独",
    "害怕",
    "紧张",
)
_QUESTION_RE = re.compile(r"[?？]\s*$|(吗|呢|嘛)[。.!！~～]*\s*$")


@dataclass(frozen=True)
class DialoguePatternReport:
    analysed: int = 0
    repeated_openings: list[str] = field(default_factory=list)
    trailing_question_ratio: float = 0.0
    repeated_phrases: list[str] = field(default_factory=list)
    name_overuse: list[str] = field(default_factory=list)
    assurance_repeats: int = 0
    emotion_mirror_repeats: int = 0
    syntax_repeats: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "analysed": self.analysed,
            "repeated_openings": list(self.repeated_openings),
            "trailing_question_ratio": round(self.trailing_question_ratio, 3),
            "repeated_phrases": list(self.repeated_phrases),
            "name_overuse": list(self.name_overuse),
            "assurance_repeats": self.assurance_repeats,
            "emotion_mirror_repeats": self.emotion_mirror_repeats,
            "syntax_repeats": list(self.syntax_repeats),
            "guidance": list(self.guidance),
        }


class DialoguePatternMonitor:
    def analyze(
        self,
        replies: Iterable[str],
        *,
        agent_name: Optional[str] = None,
        user_name: Optional[str] = None,
        max_replies: int = MAX_ANALYSED_REPLIES,
    ) -> DialoguePatternReport:
        texts = [
            str(reply).strip()
            for reply in list(replies)[-max_replies:]
            if str(reply).strip()
        ]
        if len(texts) < MIN_REPEATS:
            return DialoguePatternReport(analysed=len(texts))

        openings = [
            _first_opening(text, _COMFORT_OPENINGS) for text in texts
        ]
        repeated_openings = _repeated(
            [value for value in openings if value], MIN_REPEATS
        )

        trailing = sum(1 for text in texts if _QUESTION_RE.search(text))
        trailing_ratio = trailing / len(texts)

        phrases = [
            phrase
            for text in texts
            for phrase in _repeated_phrases_in(text)
            if phrase
        ]
        repeated_phrases = _repeated(phrases, MIN_REPEATS)

        name_overuse: list[str] = []
        for name in (agent_name, user_name):
            clean = (name or "").strip()
            if len(clean) < 2:
                continue
            if sum(1 for text in texts if clean in text) >= MIN_REPEATS:
                name_overuse.append(clean)

        assurances = sum(
            1
            for text in texts
            if any(marker in text for marker in _ASSURANCES)
        )
        emotion_mirrors = sum(
            1
            for text in texts
            if sum(1 for word in _EMOTION_WORDS if word in text) >= 2
        )

        syntax_tokens = [_syntax_token(text) for text in texts]
        syntax_repeats = _repeated(
            [token for token in syntax_tokens if token], MIN_REPEATS
        )

        guidance: list[str] = []
        if repeated_openings:
            guidance.append(
                "最近几轮反复用同一种安抚式开头（"
                + "、".join(repeated_openings[:3])
                + "）；这一轮换一种自然的进入方式，只有对方确实难受时才安抚。"
            )
        if trailing_ratio >= 0.6:
            guidance.append(
                "最近几轮大多以问题收尾；这一轮如果没必要就不追问，"
                "用陈述、判断或具体回应自然结束。"
            )
        if repeated_phrases:
            guidance.append(
                "避免重复这些近期用过的说法："
                + "、".join(repeated_phrases[:3])
                + "。意思可以保留，措辞不要复读。"
            )
        if name_overuse:
            guidance.append(
                "不要频繁重复名字（"
                + "、".join(name_overuse)
                + "）；自然对话里名字出现得很少。"
            )
        if assurances >= MIN_REPEATS:
            guidance.append(
                "不要再用“会一直陪着你”一类保证；用这一轮具体做了什么来体现在意。"
            )
        if emotion_mirrors >= MIN_REPEATS:
            guidance.append(
                "不要只是复述对方的情绪标签；回应具体内容，或直接给出你的判断。"
            )
        if syntax_repeats:
            guidance.append(
                "最近几轮的句首结构重复（"
                + "、".join(syntax_repeats[:2])
                + "）；这一轮换一种句式。"
            )

        return DialoguePatternReport(
            analysed=len(texts),
            repeated_openings=repeated_openings,
            trailing_question_ratio=trailing_ratio,
            repeated_phrases=repeated_phrases,
            name_overuse=name_overuse,
            assurance_repeats=assurances,
            emotion_mirror_repeats=emotion_mirrors,
            syntax_repeats=syntax_repeats,
            guidance=guidance,
        )

    def guidance(self, replies: Iterable[str], **kwargs) -> list[str]:
        return self.analyze(replies, **kwargs).guidance


def _first_opening(text: str, openings: tuple[str, ...]) -> Optional[str]:
    head = text[:24]
    for marker in openings:
        if head.startswith(marker):
            return marker
    return None


def _repeated(items: list[str], minimum: int) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    repeated = [item for item, count in counts.items() if count >= minimum]
    return sorted(repeated, key=lambda item: (-counts[item], item))


def _repeated_phrases_in(text: str) -> list[str]:
    """Find suspicious repeated clauses inside one reply (4–10 chars each)."""
    pieces = [
        piece.strip()
        for piece in re.split(r"[，,。.!！？?；;：:\n]", text)
        if piece.strip()
    ]
    return [piece for piece in pieces if 4 <= len(piece) <= 10]


def _syntax_token(text: str) -> Optional[str]:
    head = re.split(r"[，,。.!！？?；;：:\n]", text, maxsplit=1)[0].strip()
    if len(head) < 2:
        return None
    return head[:3]
