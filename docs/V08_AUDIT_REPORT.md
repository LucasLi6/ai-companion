# Ava V0.7 — Bug Fix & Reliability Audit 报告

日期：2026-11-04  
分支：`v0.1-deepseek-integration`  
基线提交：`a98e861`（与 `origin/v0.1-deepseek-integration` 同步）

---

## A. 环境与基线

- 分支：`v0.1-deepseek-integration`；未修改 `main`。
- 提交：`a98e861 V0.7: selective response and person understanding profiles`；本地与远端一致。
- Python：3.12.15（项目 `.venv`）；pytest / pytest-asyncio / pydantic v2 / aiosqlite / NiceGUI。
- 审计前基线：`1069 passed`，0 failed（V0.7 修复完成后的状态）。
- 审计后：`1103 passed`，0 failed，约 32.85 秒。
- 所有测试离线运行；`tests/conftest.py` 的 autouse fixture 阻断真实网络连接。
- 未调用真实 DeepSeek API，未读取或输出 API Key。

---

## B. 确认的 Bug

### B1. `不回答我说的话` / `跳过这个话` 被回复确认语

- 复现输入：`不回答我说的话`、`跳过这个话`、`这段先略过`。
- 症状：助手回复“没问题，翻篇了。”/“嗯，那就不回。”，而不是沉默。
- 根因：`detect_explicit_silence()` 只识别“不要/别/不用 + 回复”等助动词形式；裸否定
  `不 + 回答 + 我 + 说的话` 不匹配；“跳过这个话/这段先略过”也没有对应模式。
- 涉及文件/函数：`src/companion/cognition/silence.py` 的 `_SILENCE_PATTERNS`、
  `_SKIP_PATTERNS`、`detect_explicit_silence()`。
- 修复：
  - 新增裸否定模式 `(?<!得)不(回复|回答|回应)我(说|讲|打|发|写)?的?(话|消息|信息|内容|字)`；
  - 新增当前消息跳过模式（“跳过/略过 + 这个话/这段”及“这段先略过”），并对
    `跳过` 模式做否定守卫（“不要略过这段”不算沉默）。
- 验证：端到端断言 `IGNORE`、无 `ResponseProvider` 调用、无流事件、数据库
  action 为 `IGNORE`、`response_text=None`。反例对照
  `为什么 AI 有时候不回答？`、`我朋友刚才说“不要回答我”。`、`我朋友说“别回复我”`、
  `如果我让你不回答，你会怎么处理？`、`跳过刚才的话题，现在帮我解释 Python。`
  均正常回复。
- 一次性沉默不会持久化：同一会话随后说 `你好` 仍正常 `RESPOND`。

### B2. 普通聊天会打印 WAIT/IGNORE 内部标签

- 症状：CLI 在 WAIT/IGNORE 时输出 `[WAIT (...)]`、`[IGNORE (...)]`。
- 根因：`_run_streaming_turn()` 把“没有响应事件”当作需要报告的状态。
- 涉及文件/函数：`src/companion/cli.py` 的 `_run_streaming_turn()`、`cmd_repl()`。
- 修复：普通聊天面完全静默；新增 `--debug` 时才打印持久化的 action/reason。
  未删除 `NonresponseTrace`，诊断信息仍可从数据库和 `--debug` 获取。
- 验证：CLI 测试断言普通模式无输出、`--debug` 输出
  `[IGNORE (direct_human_address)]`；同时更新了原有 CLI 测试到新契约。
- UI：`src/companion/ui/app.py` 在非完成流（WAIT/IGNORE）时删除回复气泡，
  与 CLI 语义一致（代码审查确认，未启动真实 UI 交互测试）。

### B3. Fast Path 对纯语气词过度回复

- 复现：`嗯`、`...`、`……` 原本得到 `RESPOND`。
- 根因：Fast Path 对可信 `agent` 地址的普通发言默认 `RESPOND`，没有“短暂停顿”判断。
- 涉及文件/函数：`src/companion/core/fast_path.py` 的 `looks_like_backchannel()`、
  `assess_turn()`；`_WAIT_RE` 增加“整句思考停顿”（`我再想想` / `让我想想`）。
- 修复：整句 backchannel 与思考停顿 → `WAIT`，不调用模型；`嗯，那你觉得呢？`
  仍正常回答；`今天真累` 仍保留 `RESPOND/USEFUL`（不强制沉默）。
- 验证（离线端到端实测）：

| 输入 | 路由 | reply_necessity | 最终动作 | Response 调用 |
|---|---|---|---|---|
| 你好 | local_direct_address | USEFUL | RESPOND | 1 |
| 今天天气怎么样？ | local_direct_address | REQUIRED | RESPOND | 1 |
| 嗯 | local_backchannel | UNNECESSARY | WAIT | 0 |
| ... | local_backchannel | UNNECESSARY | WAIT | 0 |
| 我再想想 | local_wait_request | UNNECESSARY | WAIT | 0 |
| 等我一下 | local_wait_request | UNNECESSARY | WAIT | 0 |
| 今天真累 | local_direct_address | USEFUL | RESPOND | 1 |
| 我还没说完 | local_wait_request | UNNECESSARY | WAIT | 0 |
| 这个我觉得可以。（unknown） | deliberation | （模型 USEFUL） | RESPOND（balanced 策略） | 1 |

### B4. 长中文输入的重要关键词在句尾时召回为 0

- 复现：记忆 `Alice 的机器学习项目这个月进入部署阶段`；查询
  `我想问一下关于机器学习的项目进度`。
- 症状：`SqliteMemoryStore.retrieve()` 返回空集合。
- 根因：中文整句被 `re.split(r"[^\w]+")` 当成一个 token；`extract_keywords()`
  先加入整句，再用从左到右的 bigram 填满 6 个关键词预算，结果只得到
  `我想/想问/问一/一下/下关`，`机器学习/项目/进度` 永远轮不到。
- 涉及文件/函数：`src/companion/memory/store.py` 的 `extract_keywords()`、
  新增 `_cjk_keyword_candidates()`、`SqliteMemoryStore.retrieve()`。
- 修复：
  - 超过 8 字且无标点的中文整串不再作为整句关键词（它不可能是某条记忆的子串）；
  - 长中文串内嵌的 ASCII 词（如 `Python`）单独保留；
  - 两种奇偶位置的 2 字窗口覆盖全部相邻 bigram，先按“纯中文内容词”排序，
    再放混合/功能词窗口，最多 6 个；
  - 多个 token 的候选轮转分配，避免一个长句独占预算；
  - `retrieve()` 记录每条记忆命中的关键词数，按
    `identity → 命中数 → 更新时间` 排序，使前几条（也是进入 prompt 的条目）
    更相关。
- 验证：修复前 `retrieve()` 返回 `set()`；修复后命中项目记忆且排除无关记忆；
  多记忆同时命中；`今天` 这类单词重合但相关度低的记忆不会排在多关键词命中的
  项目记忆前面。

### B5. “我们有多少条对话”没有可信数字，模型只能猜

- 复现输入：`到现在你和我有多少条对话了？`
- 症状：回答不准确，并转向时间信息。
- 根因：代码中没有任何会话/消息计数查询，prompt 也没有可信数字；模型只能从
  最近历史窗口或聊天时间猜测。
- 涉及文件/函数：
  - `src/companion/storage/repositories.py` 新增
    `UtteranceRepository.conversation_stats(person_id)`（只读、参数化 SQL，不读正文）；
  - `src/companion/memory/requests.py` 新增 `is_conversation_count_request()`；
  - `src/companion/core/orchestrator.py` 在显式计数问题时读取统计；
  - `src/companion/cognition/models.py` 新增 `conversation_stats` 字段；
  - `src/companion/prompts/response.py` 渲染
    `【对话统计】`（对方消息数 / Ava 可见回复数 / 会话数）并明确“不得估算”。
- 统计范围：仅 `utterances.speaker_person_id = 当前说话人`；多人会话中别人的
  消息和回复不计入；回复只统计 `RESPOND`、`COMPLETED/TRUNCATED` 的真实可见回复。
- 验证：测试直接查询 SQLite 并与 prompt 中的数字比对；另一个人的消息不改变
  当前说话人的计数；`这段对话有多少条建议？` 这类话题性问句不触发统计。

### B6. 时间语义文档与实现不一致（非行为 bug）

- `runtime_context/temporal.py` 的模块文档说 `last_interaction` 包含 WAIT/IGNORE；
  实际 `RuntimeSnapshotBuilder._temporal()` 只统计 `RESPOND`。已把文档改成与实现
  一致的语义。未改动时间系统本身。
- WAIT/IGNORE 不计入 last interaction 已有测试
  `test_wait_and_ignore_do_not_count_as_last_interaction`。

### B7. 情绪系统

- 未发现新的可复现代码 bug；未重写情绪架构。
- 既有测试覆盖：subject/quoted/negated/hypothetical 拒绝、第三方目标拒绝、
  重试不重复应用、失败不改变情绪、隐私过滤后的记忆。
- 新增一条连续对话集成测试：直接批评 Ava → 第三方批评 → 引用式“你完全错了”
  → 和解 → 普通技术问题。验证只有两次规则生效（批评、和解）、第三方与引用
  被本地校验拒绝、和解降低 tension、切换话题后 tension 不再上升。

---

## C. 修改文件

| 文件 | 用途 |
|---|---|
| `src/companion/cognition/silence.py` | 裸否定与“跳过当前消息”的显式沉默检测 |
| `src/companion/core/fast_path.py` | backchannel / 思考停顿 → WAIT；保留直接问答快速路径 |
| `src/companion/cli.py` | WAIT/IGNORE 普通聊天静默；`--debug` 保留诊断 |
| `src/companion/memory/store.py` | 中文关键词拆分/排序；命中数排序 |
| `src/companion/memory/requests.py` | 会话计数请求的窄匹配 |
| `src/companion/core/orchestrator.py` | 计数请求时读取可信统计 |
| `src/companion/cognition/models.py` | `conversation_stats` 可信上下文字段 |
| `src/companion/prompts/response.py` | 渲染【对话统计】并禁止估算 |
| `src/companion/storage/repositories.py` | 只读、按说话人限定的会话统计查询 |
| `src/companion/runtime_context/temporal.py` | 修正 last interaction 语义文档 |
| `tests/test_reported_bug_regressions.py` | 新增 27 条端到端回归测试 |
| `tests/test_memory_retrieval.py` | 新增 5 条中文召回/排序测试 |
| `tests/test_emotion_semantic.py` | 新增 1 条连续对话情绪测试 |
| `tests/test_deepseek_repl.py` | 更新静默契约；新增 `--debug` 测试 |
| `tests/test_v061_repairs.py` | 更新 WAIT 输出契约测试 |
| `docs/V07_REPORT.md` | 修正过时的推送状态说明 |

未新增 Planner、第二次 LLM、第二套长期记忆、CMF 重写、情绪框架重写或
Persona Core 重写；未删除现有功能；未清理用户数据库或聊天历史。

---

## D. 测试结果

- 新增回归文件：`27 passed`（`tests/test_reported_bug_regressions.py`）。
- 记忆检索：`16 passed`（含 5 条新增）。
- 受影响模块（情绪、持久化、重试、流式、CLI、记忆）：`164 passed`。
- 全套：`1103 passed`，`0 failed`，`32.85s`。
- 覆盖方式：`build_runtime(...)` + 临时 SQLite + Fake DeepSeek transport，
  真实调用 `process_utterance` / `process_utterance_stream`，断言最终
  `AgentAction`、prompt 内容、provider 调用次数、流事件、CLI 输出与数据库记录。
- 真实 DeepSeek：**未运行**。按任务约束禁止在开发/验收中调用真实 API；
  因此自然语言层面的效果仍属于未在线验证项。
- CLI：已用测试验证。
- 流式：已用测试验证 WAIT/IGNORE 无任何事件、失败/取消不标记 COMPLETED。
- UI：仅代码审查确认气泡删除逻辑；未启动真实 UI 做人工验收。

---

## E. 未解决问题与设计边界

1. **balanced 策略仍允许 USEFUL 的未明确对象发言**：`addressee=unknown` 时，
   本地会走 Deliberation；若模型给出 `USEFUL`，默认 `balanced` 策略允许插话。
   这是既有可配置行为（`test_unaddressed_responses_follow_the_configured_conservatism`
   已固定），本次未改动。需要更严格时可调 `response_conservatism=conservative`。
2. **可信直呼场景的 USEFUL 仍会 RESPOND**：这是 Fast Path 的低延迟契约；
   `你好`、`今天真累` 均保留回复。本次只修掉了纯语气词和明确停顿的过度回复。
3. **同义词 / “项目名 vs 日常描述”的词汇差异**：纯关键词回退无法解决，
   需要配置显式 embedding provider 才能启用 Hybrid/向量召回；未把“需要显式
   provider”当作 bug，也未自动加载重型模型。
4. **代词指代上一轮的长期记忆召回**：`SqliteMemoryStore.retrieve()` 是无状态的，
   只看到当前句；同一会话内的指代由对话历史块解决（新增测试验证
   `那个现在怎么样了？` 能看到上一轮内容）。跨会话、历史被截断后的纯代词
   无法仅靠关键词召回解析，属于设计边界。
5. **向量索引使用 Python 排序而非 sqlite-vec KNN**：已知实现限制，按要求未重建。
6. **真实模型自然语言质量**：问候是否仍会展开成长时间间隔解释、计数的措辞是否
   自然、Fast Path 在实际模型下的语气，均需要真实 DeepSeek 验证；当前 Fake
   Provider 结果不能作为线上验收。
7. **“我再想想”改为 WAIT 是产品判断**：它属于明确停顿，不再强制回复；若希望
   给一句简短回应，可调整该规则。
8. **跨设备/跨账号数据**：统计只覆盖当前本地数据库中当前说话人的记录，prompt
   已明确说明范围；不承诺不存在的跨设备历史。
9. **未提交/未推送**：本次审计修改全部保留在工作区，未 commit、未 push，
   等待用户明确授权。
