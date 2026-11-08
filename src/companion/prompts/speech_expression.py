"""The optional Speaker voice-expression protocol.

This block is appended to the trusted system content only when speech expression
is enabled and the reply is ordinary conversation. It does not replace or
weaken any persona section, and it adds no fixed per-sentence mapping. The tag
list itself is rendered from :mod:`companion.speech_expression.catalog`, so the
prompt, parser and future TTS layer share one whitelist.

V1.1 changes the guidance from "use tags only when truly necessary" to "in
ordinary speech, actively consider one fitting delivery tag", while keeping
the anti-template rules: no fixed mapping, no forced frequency, no tags for
structured output, and never a tag-only reply.
"""

from __future__ import annotations

from companion.speech_expression.catalog import render_tag_catalog

SPEECH_EXPRESSION_PROMPT_VERSION = "speech-expression-v1.1"


def build_speech_expression_prompt() -> str:
    return f"""【语音表达】

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


SPEECH_EXPRESSION_PROMPT = build_speech_expression_prompt()


__all__ = [
    "SPEECH_EXPRESSION_PROMPT",
    "SPEECH_EXPRESSION_PROMPT_VERSION",
    "build_speech_expression_prompt",
]
