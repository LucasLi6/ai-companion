"""Deterministic, local detection of explicit silence and remember requests.

These signals are read from the *current utterance text* — they are trusted
application input, not model output — so they can outrank a provider's
suggestion. Detection is deliberately narrow:

- a quoted or reported "don't reply" ("我的朋友刚才说：不要回答我") is a
  *discussion* of silence, not a request, and must never suppress a reply;
- a question about why an AI stays silent ("为什么 AI 有时候不回答？") is not a
  request either;
- an explicit "记住…" is a request to persist now, so the durable write can
  finish before the acknowledgement is generated.

No probabilistic rules, no random non-response percentages.
"""

from __future__ import annotations

import re

# Quoted spans are removed before matching. ASCII single quotes are omitted on
# purpose so contractions ("don't") do not look like a quotation.
_QUOTED_RE = re.compile(
    r"「[^」]*」|『[^』]*』|“[^”]*”|‘[^’]*’|\"[^\"]*\"|`[^`]*`"
)

# "…说：X" / "…said X" immediately before a match marks reported speech.
_REPORT_PREFIX_RE = re.compile(
    r"(?:说|说道|写道|告诉|问|回答|said|says|wrote|told|asked)\s*[：:,，.。]?\s*$",
    re.IGNORECASE,
)

# A match inside an explicit hypothetical ("如果我说…", "假如你不想回答…")
# describes a possibility, not a current request. The window is deliberately
# short so an unrelated "如果" earlier in a long sentence cannot suppress a
# real request.
_HYPOTHETICAL_PREFIX_RE = re.compile(
    r"(?:如果|假如|假设|要是|万一|若|when|if)\s*[^。！？!?，,；;]{0,16}$",
    re.IGNORECASE,
)

_SILENCE_PATTERNS = [
    # "这句话不要回复。" / "这条消息不用回。" / "别理我。"
    re.compile(
        r"(?:这|那|此|该|本)?\s*(?:一)?\s*(?:句|条|个)?\s*(?:话|消息|信息|内容|输入)?\s*"
        r"(?:就|都|也)?\s*(?:不用|不要|无需|不需要|别)\s*(?:再)?\s*"
        r"(?:回复|回答|回应|理(?:我|会)?)"
    ),
    # "不要回复我这条消息。" / "不用回答这个问题。"
    re.compile(
        r"(?:请|麻煩|麻烦)?\s*(?:不用|不要|无需|不需要|别)\s*(?:再)?\s*"
        r"(?:回复|回答|回应)\s*(?:我)?\s*"
        r"(?:这|那|此|该)?\s*(?:一)?\s*(?:句|条|个)?\s*(?:话|消息|信息|问题|内容)?"
    ),
    # "接下来这句不用回复。"
    re.compile(
        r"接下来\s*(?:这|那)?\s*(?:一)?\s*(?:句|条|个)?\s*(?:话|消息|内容)?\s*"
        r"(?:不用|不要|无需|不需要|别)\s*(?:再)?\s*(?:回复|回答|回应)"
    ),
    # "别回我。" / "别理我。" / "不用管我。"
    re.compile(
        r"(?:请|麻煩|麻烦)?\s*(?:别|不要|不用|无需|不需要)\s*"
        r"(?:回|回复|回答|回应|理|搭理|管)\s*(?:我|这条|这句|这个消息)?"
    ),
    # "免回。" / "勿回。" / "不必回复。" / "请勿回复。"
    re.compile(
        r"(?:请)?\s*(?:免回|勿回|勿回复|不必回复|不必回答|无须回复|無須回覆)"
    ),
    # "先别说话。" / "安静。" / "保持安静。" / "别出声。"
    re.compile(
        r"(?:请|麻烦)?\s*(?:先)?\s*(?:别说话|不要说话|别出声|保持安静|安静(?:一下)?)"
    ),
    # English: "Don't respond to this message." / "No need to reply."
    re.compile(
        r"\b(?:please\s+)?(?:do not|don't|dont|no need to|needn'?t)\s+"
        r"(?:reply|respond|answer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:please\s+)?(?:leave me alone|stay quiet|keep quiet|hold (?:your )?(?:reply|response))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reply|respond|answer)\s+(?:to\s+)?(?:this|that)\s+"
        r"(?:message|one|line)?\s*(?:is\s+)?(?:not\s+needed|unnecessary)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|skip)\s+(?:reply|response)\s+(?:to\s+(?:this|that)\b|needed\b)",
        re.IGNORECASE,
    ),
]

_REMEMBER_PATTERNS = [
    re.compile(
        r"(?:请|麻煩|麻烦)?\s*(?:你)?\s*(?:帮我|帮忙)?\s*"
        r"(?:记住|记一下|记下|牢记|别忘了|不要忘记)"
    ),
    re.compile(
        r"\b(?:please\s+)?(?:remember|memorize|keep in mind|note that|"
        r"don't forget|do not forget)\b",
        re.IGNORECASE,
    ),
]


def _outside_quotes(text: str) -> str:
    return _QUOTED_RE.sub(" ", text)


_NEGATION_PREFIX_RE = re.compile(
    r"(?:不要|不用|别|无需|无须|不必|不需要|do not|don't|dont)\s*(?:再)?\s*$",
    re.IGNORECASE,
)


def _is_reported(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 16) : match.start()]
    return bool(_REPORT_PREFIX_RE.search(window))


def _is_hypothetical(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 24) : match.start()]
    return bool(_HYPOTHETICAL_PREFIX_RE.search(window))


def _is_negated(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 8) : match.start()]
    return bool(_NEGATION_PREFIX_RE.search(window))


def _has_direct_match(
    text: str, patterns: list[re.Pattern[str]], *, skip_negated: bool = False
) -> bool:
    stripped = _outside_quotes(text)
    for pattern in patterns:
        for match in pattern.finditer(stripped):
            if _is_reported(stripped, match) or _is_hypothetical(stripped, match):
                continue
            if skip_negated and _is_negated(stripped, match):
                continue
            return True
    return False


def detect_explicit_silence(text: str | None) -> bool:
    """True when the speaker explicitly asks the agent not to reply now."""
    if not text:
        return False
    return _has_direct_match(text, _SILENCE_PATTERNS)


def detect_explicit_remember(text: str | None) -> bool:
    """True when the speaker explicitly asks the agent to remember something."""
    if not text:
        return False
    return _has_direct_match(text, _REMEMBER_PATTERNS, skip_negated=True)
