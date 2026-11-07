"""The optional Speaker voice-expression protocol.

This block is appended to the trusted system content only when speech expression
is enabled and the reply is ordinary conversation. It does not replace or
weaken any persona section, and it adds no fixed per-sentence mapping. The tag
list itself is rendered from :mod:`companion.speech_expression.catalog`, so the
prompt, parser and future TTS layer share one whitelist.
"""

from __future__ import annotations

from companion.speech_expression.catalog import render_tag_catalog

SPEECH_EXPRESSION_PROMPT_VERSION = "speech-expression-v1"


def build_speech_expression_prompt() -> str:
    return f"""【语音表达】

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


SPEECH_EXPRESSION_PROMPT = build_speech_expression_prompt()


__all__ = [
    "SPEECH_EXPRESSION_PROMPT",
    "SPEECH_EXPRESSION_PROMPT_VERSION",
    "build_speech_expression_prompt",
]
