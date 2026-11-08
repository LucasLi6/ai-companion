"""Real-DeepSeek evaluation for the Ava speech-expression protocol.

This is an offline-style development harness: it sends **synthetic** Chinese
turns from ``cases.jsonl`` to the real DeepSeek Speaker path and records what
the model actually does with the voice labels. It never writes to Ava's
database, memory, profile or emotion state, and it never uses private user
dialogue.

Usage::

    DEEPSEEK_API_KEY=... .venv/bin/python evals/speech_expression/run_eval.py
    .venv/bin/python evals/speech_expression/run_eval.py --prompt-version v1
    .venv/bin/python evals/speech_expression/run_eval.py --prompt-version both

Report metrics are descriptive: the coverage target (70%-85% of suitable
everyday replies carrying a fitting control tag) is an evaluation goal, not a
runtime probability. The harness compares the frozen V1 protocol with V1.1 on
the same model, parameters and inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from companion.cognition.models import (  # noqa: E402
    ROLE_ASSISTANT,
    ROLE_HUMAN,
    CognitiveContext,
    CognitiveTurn,
)
from companion.domain.enums import EventSource  # noqa: E402
from companion.domain.models import (  # noqa: E402
    CognitiveAction,
    CognitiveResult,
    ReasonCode,
    ReplyNecessity,
    make_utterance_event,
)
from companion.prompts import response as response_prompts  # noqa: E402
from companion.prompts.layout import (  # noqa: E402
    RESPONSE_LAYOUT_CHAT,
    RESPONSE_LAYOUT_STRUCTURED,
    response_layout_for,
)
from companion.prompts.speech_expression import (  # noqa: E402
    build_speech_expression_prompt,
)
from companion.providers.deepseek_client import (  # noqa: E402
    build_deepseek_client,
    load_deepseek_config,
)
from companion.providers.deepseek_response import (  # noqa: E402
    DeepSeekResponseProvider,
)
from companion.speech_expression.catalog import (  # noqa: E402
    KIND_NONVERBAL,
    render_tag_catalog,
)
from companion.speech_expression.parser import (  # noqa: E402
    VoiceResponseParser,
    parse_voice_response,
)

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "cases.jsonl"
DEFAULT_REPORT = HERE / "report.json"

# Frozen V1 protocol, kept here only so the harness can A/B the old wording.
# Production code no longer imports this text.
_V1_PROMPT = f"""【语音表达】

这段协议只规定语音表达方式，不改变你的人格、判断或说话内容。下面的标签会从最终保存的文字中移除，只交给语音合成；它们不是你要说出的话，也不是情绪记录。

{render_tag_catalog()}

使用原则：

- 语气控制标签改变它后面文字的读法，并持续到下一个语气控制标签或实际语音分段结束。声音动作标签只在当前位置发出声音，不改变周围文字的持续语气。
- 只在当前这一轮确实需要时使用少量标签。不要机械地给每句话加标签，也不要用标签代替文字表达。
- 不要建立固定映射：疑问句不一定用 [curious]，技术或严肃话题不一定用 [serious]，开心不一定用 [excited]，也不要按固定间隔插入笑声或叹气。
- 标签直接写在正文里，使用普通文本，不要使用 JSON、代码块或额外的标签清单，也不要解释、朗读或向对方提及标签。
- 如果当前内容是代码、表格、Markdown、精确语法、公式或需要逐字准确的输出，不要使用任何标签。
- 只能使用上面列出的标签和完全一致的写法。不要创造新标签，也不要改写拼写。没有必要时不使用标签，保持原本的自然表达。
"""


def _load_rows() -> list[dict]:
    rows = []
    with CASES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    return rows


# Frozen V1.1 prompt, kept only for the real-model A/B. Production code uses
# the V1.2 VOICE/TEXT protocol.
_V11_PROMPT = f"""【语音表达】

你正在通过声音与对方交流。除了文字内容，说话时的语气、节奏、态度和偶尔出现的声音动作，也是你自然表达的一部分。

这段协议只规定语音表达方式，不改变你的人格、独立判断或说话内容。下面的标签会从最终保存的文字中移除，只交给语音合成；它们不是你要说出的话，也不是情绪记录。

{render_tag_catalog()}

语气表达不是情绪状态分类。它体现你此刻想怎样说这句话，不代表你的内部情绪一定发生变化，也不需要始终与当前情绪状态一致。

日常交流中，多数适合语音表达的回复可以主动考虑一个最符合当前表达意图的语气标签，让听者能够从声音中感受到你的态度，而不只是听到平淡的文字朗读。不需要每一轮都使用标签，有些普通轮次保持自然说话即可。

根据真实对话、自己的性格、与对方的关系、当前情境以及你实际想表达的态度，自然选择语气。当你在好奇地追问、调皮地回应、有点不情愿、表示讽刺、感到惊讶、明显兴奋或认真提醒对方时，可以使用合适的语气控制标签，不必等到情绪特别强烈时才使用。

正常说话不一定需要特殊语气。普通事实陈述、单纯确认信息、记录时间地点等没有明显态度的内容，通常直接输出普通文字，不要为了显得生动而加标签。如果只是在确认、记录或复述信息，通常不需要标签。只有确实存在想表达的态度时，才选择标签。明显的情绪变化或态度转折处，可以使用多个控制标签。

语气控制标签改变它后面文字的读法，并持续到下一个语气控制标签或实际语音分段结束。声音动作标签只在当前位置发出声音，不改变周围文字的持续语气。只有你确实想在某个位置笑一声、叹气、清嗓或发出其他声音时，才使用动作标签；不要用声音动作代替普通文字，也不要把它们当成装饰或标点。

通常一个合适的控制标签已经足够。只有表达态度确实发生变化时，才考虑切换语气。不要为了提高标签数量而不断改变语调。不要把语气变成固定映射：疑问句不一定用 [curious]，技术或严肃话题不一定用 [serious]，开心不一定用 [excited]，也不要机械地重复笑声或叹气，同一个声音动作不要连续多轮反复使用。

标签直接写在正文里，使用普通文本，不要使用 JSON、代码块或额外的标签清单，也不要解释、朗读或向对方提及标签。

如果当前内容是代码、表格、Markdown、精确语法、公式或需要逐字准确的输出，不要使用任何标签。

只能使用上面列出的标签和完全一致的写法。不要创造新标签，不要翻译、简写或改写拼写，也不要把其他语音系统的标签当作有效指令。像 [happy]、[playful]、[neutral]、[confused]、[thoughtful] 这类不在列表里的写法都不是有效标签，不要使用。

任何正常回复都必须包含实际的文字内容。不能只输出控制标签或声音动作标签。
"""


def load_cases() -> list[dict]:
    return [row for row in _load_rows() if not row.get("dialogue")]


def load_dialogues() -> list[dict]:
    return [row for row in _load_rows() if row.get("dialogue")]


def _context(case: dict, history: list[dict] | None = None) -> CognitiveContext:
    turns = []
    history = case.get("history") or [] if history is None else history
    for index, turn in enumerate(history):
        role = ROLE_ASSISTANT if turn.get("role") == "assistant" else ROLE_HUMAN
        turns.append(
            CognitiveTurn(
                event_id=f"eval_hist_{case['id']}_{index}",
                role=role,
                speaker_person_id=None if role == ROLE_ASSISTANT else "eval_user",
                speaker_label="Ava" if role == ROLE_ASSISTANT else "用户",
                addressee_hint="human" if role == ROLE_ASSISTANT else "agent",
                text=str(turn.get("text") or ""),
            )
        )
    return CognitiveContext(
        conversation_id=f"eval_{case['id']}",
        current_event_id=f"eval_evt_{case['id']}",
        current_speaker_id="eval_user",
        current_speaker_label="用户",
        current_text=case["input"],
        current_addressee_hint="agent",
        turns=turns,
    )


def _event(case: dict):
    return make_utterance_event(
        conversation_id=f"eval_{case['id']}",
        text=case["input"],
        addressee_hint="agent",
        speaker_person_id="eval_user",
        source=EventSource.TEST,
        event_id=f"eval_evt_{case['id']}",
    )


def _analysis(case: dict) -> CognitiveResult:
    return CognitiveResult(
        event_id=f"eval_evt_{case['id']}",
        meaning="synthetic speech-expression evaluation",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        response_style="chat",
        provider="eval",
        intent="eval",
        addressed_to="agent",
        reply_necessity=ReplyNecessity.USEFUL,
        conversation_state="eval",
    )


async def _run_case(
    provider: DeepSeekResponseProvider,
    case: dict,
    semaphore: asyncio.Semaphore,
    *,
    history: list[dict] | None = None,
) -> dict:
    async with semaphore:
        event = _event(case)
        context = _context(case, history=history)
        analysis = _analysis(case)
        parser = VoiceResponseParser()
        raw_model_parts: list[str] = []
        t0 = time.monotonic()
        first_token_ms = None
        first_display_ms = None
        finish_reason = None
        usage = None
        error = None
        try:
            async for delta in provider.generate_stream(event, analysis, context):
                if delta.content:
                    raw_model_parts.append(delta.content)
                    if first_token_ms is None:
                        first_token_ms = (time.monotonic() - t0) * 1000
                    parser.feed(delta.content)
                    if first_display_ms is None and parser.text.strip():
                        first_display_ms = (time.monotonic() - t0) * 1000
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                if delta.usage_reported and delta.total_tokens is not None:
                    usage = {
                        "prompt_tokens": delta.prompt_tokens,
                        "completion_tokens": delta.completion_tokens,
                        "total_tokens": delta.total_tokens,
                    }
        except Exception as exc:  # keep the eval running; record the failure
            error = f"{type(exc).__name__}: {exc}"
        parser.finish()
        total_ms = (time.monotonic() - t0) * 1000
        result = parser.result()
        raw_model_output = "".join(raw_model_parts)

        valid_tags = [
            segment.tag
            for segment in result.segments
            if segment.tag is not None
        ]
        nonverbal_tags = [
            segment.tag
            for segment in result.segments
            if segment.kind == KIND_NONVERBAL and segment.tag is not None
        ]
        control_tags = [
            segment.tag
            for segment in result.segments
            if segment.kind == "control" and segment.tag is not None
        ]
        effective_controls = ([result.voice] if result.voice else []) + control_tags
        invalid_tags = list(result.invalid_tags)
        canonical = result.text.strip()
        empty = not canonical

        recovery_attempted = False
        recovery_succeeded = False
        recovery_ms = None
        if empty and finish_reason != "length" and error is None:
            recovery_attempted = True
            t_recovery = time.monotonic()
            try:
                generated = await provider.generate_plain_recovery(
                    event, analysis, context
                )
                recovered = parse_voice_response(str(generated or ""))
                recovery_succeeded = bool(recovered.text.strip())
            except Exception as exc:
                recovery_succeeded = False
                if error is None:
                    error = f"recovery:{type(exc).__name__}: {exc}"
            recovery_ms = (time.monotonic() - t_recovery) * 1000

        return {
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "speech_tag_desirable": bool(case.get("speech_tag_desirable")),
            "nonverbal_desirable": bool(case.get("nonverbal_desirable")),
            "layout": response_layout_for(case["input"]),
            "raw_model_output": raw_model_output,
            "raw": raw_model_output,
            "display_text": result.display_text,
            "canonical_text": canonical,
            "voice": result.voice,
            "invalid_voice": result.invalid_voice,
            "protocol_valid": result.protocol_valid,
            "protocol_error": result.protocol_error,
            "empty_markup_count": parser.empty_markup_count,
            "valid_tags": valid_tags,
            "control_tags": control_tags,
            "effective_control_tags": effective_controls,
            "nonverbal_tags": nonverbal_tags,
            "invalid_tags": invalid_tags,
            "empty": empty,
            "finish_reason": finish_reason,
            "error": error,
            "first_token_ms": round(first_token_ms, 1)
            if first_token_ms is not None
            else None,
            "first_display_ms": round(first_display_ms, 1)
            if first_display_ms is not None
            else None,
            "total_ms": round(total_ms, 1),
            "usage": usage,
            "recovery_attempted": recovery_attempted,
            "recovery_succeeded": recovery_succeeded,
            "recovery_ms": round(recovery_ms, 1)
            if recovery_ms is not None
            else None,
        }


def _mean(values):
    return round(statistics.mean(values), 1) if values else None


def _median(values):
    return round(statistics.median(values), 1) if values else None


def summarize(records: list[dict]) -> dict:
    total = len(records)
    valid = sum(len(record["valid_tags"]) for record in records)
    invalid = sum(len(record["invalid_tags"]) for record in records)
    emitted = valid + invalid
    desirable = [r for r in records if r["speech_tag_desirable"]]
    plain_chat = [
        r
        for r in records
        if not r["speech_tag_desirable"]
        and r["layout"] == RESPONSE_LAYOUT_CHAT
        and r["category"] != "technical_qa"
    ]
    structured = [
        r for r in records if r["layout"] == RESPONSE_LAYOUT_STRUCTURED
    ]
    chat = [r for r in records if r["layout"] == RESPONSE_LAYOUT_CHAT]
    valid_headers = [r for r in chat if r["protocol_valid"]]
    header_attempts = [
        r for r in chat if r["protocol_valid"] or r["invalid_voice"]
    ]
    chat_usable = [
        r for r in chat if not r["error"] and not r["empty"]
    ]
    nonverbal_forbidden = [
        r
        for r in records
        if r["category"] in ("plain_factual", "technical_qa")
    ]
    nonverbal_only = [
        r
        for r in records
        if r["nonverbal_tags"] and not r["control_tags"]
    ]
    nonverbal_counts: dict[str, int] = {}
    for record in records:
        for tag in record["nonverbal_tags"]:
            nonverbal_counts[tag] = nonverbal_counts.get(tag, 0) + 1
    with_recovery = [r for r in records if r["recovery_attempted"]]
    tokens = [
        r["usage"]["completion_tokens"]
        for r in records
        if r.get("usage") and r["usage"].get("completion_tokens") is not None
    ]

    def share(rows, predicate):
        return round(sum(1 for row in rows if predicate(row)) / len(rows), 4) if rows else None

    return {
        "cases": total,
        "errors": sum(1 for r in records if r["error"]),
        "accepted_tags": valid,
        "accepted_tags_all_whitelisted": True,
        "raw_invalid_tags": invalid,
        "raw_invalid_tag_rate": round(invalid / emitted, 4) if emitted else 0.0,
        "valid_tag_rate_after_parser": round(valid / emitted, 4) if emitted else 1.0,
        "empty_replies": sum(1 for r in records if r["empty"]),
        "empty_reply_rate": share(records, lambda r: r["empty"]),
        "coverage_desirable": share(
            desirable, lambda r: bool(r["effective_control_tags"])
        ),
        "control_coverage": share(
            desirable, lambda r: bool(r["effective_control_tags"])
        ),
        "natural_rate": share(
            chat_usable, lambda r: not r["effective_control_tags"]
        ),
        "invalid_voice_rate": share(
            header_attempts, lambda r: bool(r["invalid_voice"])
        ),
        "nonverbal_rate": share(
            chat, lambda r: bool(r["nonverbal_tags"])
        ),
        "protocol_failure_rate": share(
            chat, lambda r: not r["protocol_valid"]
        ),
        "voice_appropriateness": {
            "status": "manual_review_required",
            "scored_cases": 0,
            "note": "VOICE choices are judged in the report by a human reviewer.",
        },
        "desirable_cases": len(desirable),
        "unnecessary_tag_rate_plain_chat": share(
            plain_chat, lambda r: bool(r["valid_tags"])
        ),
        "plain_chat_cases": len(plain_chat),
        "tags_on_structured_replies": sum(
            1 for r in structured if r["valid_tags"]
        ),
        "structured_cases": len(structured),
        "nonverbal_abuse_rate": share(
            nonverbal_forbidden, lambda r: bool(r["nonverbal_tags"])
        ),
        "nonverbal_plain_or_structured_cases": len(nonverbal_forbidden),
        "nonverbal_only_cases": len(nonverbal_only),
        "nonverbal_tag_counts": dict(
            sorted(nonverbal_counts.items(), key=lambda item: -item[1])
        ),
        "control_tags_per_reply_mean": _mean(
            [len(r["effective_control_tags"]) for r in records]
        ),
        "recovery_attempts": len(with_recovery),
        "recovery_successes": sum(
            1 for r in with_recovery if r["recovery_succeeded"]
        ),
        "first_token_ms_mean": _mean(
            [r["first_token_ms"] for r in records if r["first_token_ms"] is not None]
        ),
        "first_token_ms_median": _median(
            [r["first_token_ms"] for r in records if r["first_token_ms"] is not None]
        ),
        "first_display_ms_mean": _mean(
            [r["first_display_ms"] for r in records if r["first_display_ms"] is not None]
        ),
        "first_display_ms_median": _median(
            [r["first_display_ms"] for r in records if r["first_display_ms"] is not None]
        ),
        "total_ms_mean": _mean([r["total_ms"] for r in records]),
        "total_ms_median": _median([r["total_ms"] for r in records]),
        "completion_tokens_mean": _mean(tokens),
        "completion_tokens_median": _median(tokens),
        "recovery_ms_mean": _mean(
            [r["recovery_ms"] for r in with_recovery if r["recovery_ms"] is not None]
        ),
        "raw_errors": [
            {"id": r["id"], "error": r["error"]}
            for r in records
            if r["error"]
        ][:20],
    }


def _set_prompt(version: str) -> None:
    if version == "v1.1":
        response_prompts.SPEECH_EXPRESSION_PROMPT = _V11_PROMPT
        # V1.1 has no header protocol; the per-turn V1.2 reminder would
        # otherwise contaminate the A/B by telling the model to emit VOICE=.
        response_prompts.SPEECH_EXPRESSION_REMINDER = ""
    elif version == "v1.2":
        response_prompts.SPEECH_EXPRESSION_PROMPT = (
            build_speech_expression_prompt()
        )
        from companion.prompts.speech_expression import (
            SPEECH_EXPRESSION_REMINDER,
        )

        response_prompts.SPEECH_EXPRESSION_REMINDER = SPEECH_EXPRESSION_REMINDER
    else:
        raise ValueError(f"unknown prompt version: {version}")


async def _run_dialogue(
    provider: DeepSeekResponseProvider,
    group: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    history: list[dict] = []
    records: list[dict] = []
    for index, text in enumerate(group["turns"]):
        case = {
            "id": f"{group['id']}_{index + 1:02d}",
            "category": group.get("category", "dialogue"),
            "input": text,
            "speech_tag_desirable": bool(group.get("speech_tag_desirable")),
            "nonverbal_desirable": bool(group.get("nonverbal_desirable")),
        }
        record = await _run_case(
            provider, case, semaphore, history=list(history)
        )
        records.append(record)
        history.append({"role": "human", "text": text})
        history.append(
            {"role": "assistant", "text": record["canonical_text"]}
        )
    return {
        "id": group["id"],
        "category": group.get("category", "dialogue"),
        "turns": len(group["turns"]),
        "summary": summarize(records),
        "records": records,
    }


async def evaluate(
    version: str,
    cases: list[dict],
    dialogues: list[dict],
    concurrency: int,
) -> dict:
    _set_prompt(version)
    config = load_deepseek_config()
    client = build_deepseek_client(config)
    provider = DeepSeekResponseProvider(client)
    provider.speech_expression_enabled = True
    semaphore = asyncio.Semaphore(max(1, concurrency))
    started = time.monotonic()
    case_records = await asyncio.gather(
        *[_run_case(provider, case, semaphore) for case in cases]
    )
    dialogue_records = await asyncio.gather(
        *[_run_dialogue(provider, group, semaphore) for group in dialogues]
    )
    wall_ms = (time.monotonic() - started) * 1000
    dialogue_turns = [
        record
        for dialogue in dialogue_records
        for record in dialogue["records"]
    ]
    report = {
        "prompt_version": version,
        "model": config.model,
        "base_url": config.base_url,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_records),
        "dialogue_count": len(dialogue_records),
        "dialogue_turn_count": len(dialogue_turns),
        "wall_ms": round(wall_ms, 1),
        "summary": summarize(case_records),
        "dialogue_summary": summarize(dialogue_turns),
        "dialogues": dialogue_records,
        "cases": case_records,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-version",
        choices=("v1.1", "v1.2", "both"),
        default="both",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dialogue-limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]
    dialogues = load_dialogues()
    if args.dialogue_limit:
        dialogues = dialogues[: args.dialogue_limit]

    versions = (
        ["v1.1", "v1.2"]
        if args.prompt_version == "both"
        else [args.prompt_version]
    )
    reports = {}
    for version in versions:
        report = asyncio.run(
            evaluate(version, cases, dialogues, args.concurrency)
        )
        reports[version] = report
        summary = report["summary"]
        dialogue = report["dialogue_summary"]
        print(
            f"[{version}] cases={summary['cases']}"
            f" coverage={summary['control_coverage']}"
            f" natural={summary['natural_rate']}"
            f" invalid_voice={summary['invalid_voice_rate']}"
            f" nonverbal={summary['nonverbal_rate']}"
            f" protocol_fail={summary['protocol_failure_rate']}"
            f" empty={summary['empty_reply_rate']}"
            f" first_token_ms={summary['first_token_ms_median']}"
            f" total_ms={summary['total_ms_median']}"
        )
        print(
            f"[{version}] dialogue_groups={report['dialogue_count']}"
            f" turns={report['dialogue_turn_count']}"
            f" coverage={dialogue['control_coverage']}"
            f" natural={dialogue['natural_rate']}"
            f" invalid_voice={dialogue['invalid_voice_rate']}"
            f" nonverbal={dialogue['nonverbal_rate']}"
            f" protocol_fail={dialogue['protocol_failure_rate']}"
            f" empty={dialogue['empty_reply_rate']}"
        )

    payload = (
        reports[versions[0]]
        if len(reports) == 1
        else {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(cases),
            "dialogue_count": len(dialogues),
            "versions": reports,
        }
    )
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
