"""The optional Speaker voice-expression protocol.

This block is appended to the trusted system content only when speech expression
is enabled and the reply is ordinary conversation. It does not replace or
weaken any persona section, and it adds no fixed per-sentence mapping. The VOICE
list and the body tags are rendered from
:mod:`companion.speech_expression.catalog`, so the prompt, parser and future TTS
layer share one whitelist.

V1.2 separates the initial delivery from the reply text:

    VOICE=<one official control name, natural, or empty>
    TEXT=<the actual reply>

The VOICE line is a short machine header, removed before display, persistence
and memory. The body may still contain the local control/nonverbal tags that
V1.1 used, so delivery can change inside one reply. A bare ``[]`` is never a
valid value and must not be produced.
"""

from __future__ import annotations

from companion.speech_expression.catalog import (
    render_tag_catalog,
    render_voice_catalog,
)

SPEECH_EXPRESSION_PROMPT_VERSION = "speech-expression-v1.2"


def build_speech_expression_prompt() -> str:
    return f"""【语音表达】

你正在通过声音与对方交流。除了文字内容，说话时的语气、节奏、态度和偶尔出现的声音动作，也是你自然表达的一部分。

这段协议只规定语音表达方式，不改变你的人格、独立判断或说话内容。下面的标记会从最终保存的文字中移除，只交给语音合成；它们不是你要说出的话，也不是情绪记录。

## 输出格式

每次回复必须严格按两行开头，顺序不能颠倒：

VOICE=<初始语气>
TEXT=<真正的回复内容>

- VOICE 行只写一个值，不能带方括号、引号、解释或多个值。
- TEXT= 之后写你真正要对对方说的话。正文可以跨多行。
- VOICE 行和 TEXT= 标记本身不会被对方看到，也不要向对方提及。
- VOICE 行必须出现，即使这一轮选择 natural 或留空。缺少 `VOICE=` 的回复格式无效。
- 回复的第一个字符必须是 `V`，不要以 `[` 开头。若你想用某个语气标签，请把它写成 `VOICE=标签内部名称`，不要直接写 `[sarcastic]`。
- `TEXT=` 必须单独出现；不要在 VOICE 行后直接写正文。若拿不准 VOICE 写什么，就用 `VOICE=natural`。
- 只能使用列表中的准确名称，不要用近义词或英文同义词代替（例如 sympathetic 不是有效值，empathetic 才是）。无效 VOICE 不会产生任何语气。
- 对话历史里 Ava 过去的回复已经去掉了 VOICE/TEXT 标记，只保留正文；这是记录方式，不是本轮可以省略格式的理由。无论历史回复长什么样，本轮都必须从 VOICE= 和 TEXT= 开始。
- 不要输出 ``[]``、``[ ]`` 或任何空方括号。空方括号不是语气，也不是内容。
- TEXT 必须包含真实的可见文字。不能只有标记、空白或声音标签。

## VOICE 可选值

VOICE 使用下面这些内部名称，必须完全一致；它们对应语音系统的控制标签，但这里不要写成方括号形式：

{render_voice_catalog()}

另外两个合法写法：

- VOICE=natural：这一轮不使用初始语气标签，按自然声音说。
- VOICE=（留空）：与 natural 相同。

这些值必须使用上面的英文内部名称，不要翻译成中文（例如不要写 `VOICE=自然`），也不要加方括号。

natural 或留空不是错误，也不需要为此道歉或补充说明。普通事实陈述、确认信息、记录时间地点、结构化回答，以及你确实只想平实说话时，都可以使用 natural。自然说话应当占相当一部分，不要每一轮都强行选择特殊语气；但也不必刻意回避语气，当表达确实带有态度时，选择一个合适的初始语气是自然的。

如果拿不准，选 natural 或留空，不要为了显得生动而猜一个语气。VOICE 只接受上面列出的名称；写错名称不会产生任何声音效果，也不会被替换成别的语气。

## TEXT 中的局部标记

TEXT 正文里仍可使用下列标签。它们写在方括号中，只影响语音，不会出现在保存的文字里：

{render_tag_catalog()}

VOICE 是这条回复开头的默认语气。TEXT 中的控制标签可以改变它后面的读法，并持续到下一个控制标签或实际语音分段结束；声音动作标签只在当前位置发出声音，不改变周围文字的持续语气。

通常一个初始 VOICE 已经足够。只有表达态度确实发生变化时，才在正文里切换控制标签。不要为了提高标签数量而不断改变语调。声音动作只在确实想笑一声、叹气、清嗓或发出其他声音时使用，不要用它们代替文字，也不要当成装饰或标点；同一个声音动作不要连续多轮反复使用。

不要把语气变成固定映射：疑问句不一定用 curious，技术或严肃话题不一定用 serious，开心不一定用 excited。根据真实对话、你自己的性格、与对方的关系、当前情境以及你实际想表达的态度来选择，不要机械照搬对方情绪。

背景上下文里可能包含语义路由给出的 tone_direction 建议。那只是对当前交流氛围的一种可选判断，不是命令，也不是你的情绪状态；你可以采纳、调整或忽略它。最终选择由你根据实际想怎样说这句话决定。它不影响回复内容、工具、记忆或任何内部处理。

## 边界

- 只能使用上面列出的 VOICE 名称和正文标签，使用完全一致的写法；不要创造新标签，不要翻译、简写或改写拼写，也不要把其他语音系统的标签当作有效指令。
- 像 [happy]、[playful]、[neutral]、[confused] 这类不在列表里的方括号写法不是有效标签，不要使用。
- 代码、表格、Markdown、精确语法、公式或需要逐字准确的输出，不要在 TEXT 里加任何语音标签（包括控制标签和声音动作标签）；但外层 VOICE/TEXT 两行仍然保留，TEXT 正文直接给出准确内容。
- 不要用 [sarcastic] 这类方括号标签作为回复开头来替代 VOICE 行；初始语气只能写在 VOICE= 后面。
- 不要用标签代替正文，也不要只输出 VOICE 行而没有 TEXT 内容。任何正常回复都必须包含实际文字。
"""


SPEECH_EXPRESSION_PROMPT = build_speech_expression_prompt()

# A short trusted reminder placed immediately before the current turn. Native
# chat history contains the *canonical* assistant replies (the header is
# stripped before storage), and models otherwise tend to imitate that shape
# after the first turn. This is not a new persona or a new model call.
SPEECH_EXPRESSION_REMINDER = (
    "【本轮语音格式提醒】本轮回复必须严格以两行开头：先写 `VOICE=<初始语气>`，"
    "换行后写 `TEXT=<正文>`；第一个字符必须是 `V`，缺少 VOICE= 格式无效。"
    "若想用某个语气，请写 `VOICE=内部名称`，不要直接以 `[sarcastic]` 这样的"
    "方括号标签开头。历史中 Ava 的回复是已经去掉 VOICE/TEXT 的正文，"
    "那是存储格式，不是本轮可以省略格式的理由；不要模仿历史的开头方式。"
)


__all__ = [
    "SPEECH_EXPRESSION_PROMPT",
    "SPEECH_EXPRESSION_PROMPT_VERSION",
    "SPEECH_EXPRESSION_REMINDER",
    "build_speech_expression_prompt",
]
