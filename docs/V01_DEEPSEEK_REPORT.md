# V0.1 真实 DeepSeek 集成 — 交付报告

日期：2026-11-03  
分支：`main`（**未 commit、未 push、未 merge、未发布**）  
自动化测试：**250 passed**（基线 153，新增 97）

> 本报告中的「Real」示例**没有在本机执行**：开发环境没有
> `DEEPSEEK_API_KEY`，且未安装 `openai` SDK（按设计，核心运行时与测试都不
> 需要它）。真实 API 验收步骤见
> [DEEPSEEK_ACCEPTANCE.md](DEEPSEEK_ACCEPTANCE.md)，必须由持有 Key 的用户在
> 本机手动执行。本文所有自动化测试均使用 fake client，不产生任何网络请求或
> 费用。

---

## 1. 目标逐条核对

| # | 目标 | 实现与验证 |
|---|---|---|
| 1 | 每个已授权有效输入都由真实 DeepSeek 分析 | `build_runtime(provider="deepseek")` 把 `DeepSeekCognitiveProvider` 接入 Orchestrator；除服务端拒绝外，所有通过本地同意检查的输入都会调用分析接口 |
| 2 | 理解当前输入 + 最近多轮对话 | `ContextBuilder` 按时间顺序交错人类轮次与 assistant 回复，带说话人标签；`tests/test_context_builder.py`、`test_deepseek_runtime.py::test_deepseek_second_turn_sees_prior_history` |
| 3 | 读取相关且已授权的长期记忆 | `SqliteMemoryStore.retrieve`（关键词 + CJK bigram），再经隐私网关按记忆主体过滤；`tests/test_memory_retrieval.py` |
| 4 | 区分人物与对话对象 | 历史轮次标签 + `addressee_hint` 写入 prompt；当前说话人单独标注 |
| 5 | 自主判断 RESPOND / WAIT / IGNORE | 模型返回严格 JSON 建议；`ReasonCode` 扩展为 4 个新值；本地 `Policy` 拥有最终决定权 |
| 6 | 仅在 RESPOND 时调用 DeepSeek 生成回复 | Orchestrator 第 7 步只在 `final_action == RESPOND` 时调用 `DeepSeekResponseProvider`；`test_deepseek_runtime.py::test_deepseek_quiet_mode_suppresses_only_the_response` |
| 7 | 即使不回复也保留认知结果与候选记忆 | 分析、候选、任务在同一事务中持久化，与最终动作无关（`persist=True` 时） |
| 8 | 保留完整事件处理、错误恢复、幂等 | `tests/test_deepseek_recovery.py`、既有 recovery/idempotency 测试 |
| 9 | REPL 支持真实 DeepSeek 连续文本对话 | `repl --provider deepseek`，新增 `/status`；手动验收见第 6 节 |
| 10 | 继续支持 Mock 与全部既有 CLI | 默认 `mock`，250 个测试全绿 |

明确**未**开发：ASR/TTS/声纹/麦克风、真实 DeepSeek 空闲反思、自主联网研究、
向量数据库、完整自动长期事实合并、React/复杂 UI、模型工具调用/自动执行代码、
未经确认的数据库重建。

---

## 2. 新增与修改文件

### 新增

| 文件 | 作用 |
|---|---|
| `src/companion/providers/deepseek_client.py` | 唯一直接访问 DeepSeek API 的模块：配置、重试/退避、错误分类、审计日志、懒加载 `openai` |
| `src/companion/cognition/privacy.py` | `ForegroundPrivacyGateway`：逐说话人、逐记忆主体校验 `allow_external_ai` |
| `src/companion/cognition/context_builder.py` | 多轮上下文构建、说话人标签、截断、隐私过滤 |
| `src/companion/cognition/models.py` | `CognitiveTurn` / `CognitiveContext` |
| `src/companion/prompts/cognitive.py` | 版本化认知 prompt（`cognitive-v1`）+ 一次性修复 prompt |
| `src/companion/prompts/response.py` | 版本化回复 prompt（`response-v1`） |
| `src/companion/providers/deepseek_cognitive.py` | 严格 JSON 解析、信任边界、修复重试 |
| `src/companion/providers/deepseek_response.py` | 回复生成 + 二次隐私校验 |
| `src/companion/memory/candidates.py` | 候选记忆归一化去重（可信侧） |
| `src/companion/cognition/__init__.py`、`src/companion/prompts/__init__.py` | 包声明 |
| `tests/test_deepseek_client.py` (17) | 请求映射、401/402/400/429/5xx/timeout/空响应、日志卫生 |
| `tests/test_foreground_privacy.py` (12) | 逐人授权、多主体记忆、非持久化模式 |
| `tests/test_context_builder.py` (9) | 多轮交错、排除当前事件、截断、隐私过滤 |
| `tests/test_memory_retrieval.py` (11) | 关键词/CJK、状态过滤、伪造引用 id |
| `tests/test_deepseek_cognitive.py` (11) | JSON 校验、修复重试、显式人类对象保护、信任 id |
| `tests/test_deepseek_response.py` (6) | 回复生成、权限拒绝、prompt 内容 |
| `tests/test_deepseek_runtime.py` (12) | 运行时装配、禁止混配、端到端、非持久化 |
| `tests/test_deepseek_repl.py` (6) | REPL/`say` 无 Key 明确失败、`deepseek check` |
| `tests/test_memory_candidate_safety.py` (8) | 去重、幂等、永不自动 `ACTIVE` |
| `tests/test_deepseek_recovery.py` (5) | 分析/回复失败恢复、重复投递、Key/prompt 不入库 |
| `docs/DEEPSEEK_ACCEPTANCE.md` | 真实 API 手动验收清单（未执行） |

### 修改

- `pyproject.toml`：新增 `deepseek = ["openai>=1.40"]` 可选依赖。
- `src/companion/domain/enums.py`：新增
  `FOLLOW_UP_TO_AGENT`、`CONVERSATION_BETWEEN_HUMANS`、
  `AMBIGUOUS_ADDRESSEE`、`SOCIAL_RESPONSE_NOT_NEEDED`。
- `src/companion/memory/store.py`：关键词/CJK 检索、按状态/类型过滤、
  `validate_referenced_memory_ids`。
- `src/companion/storage/repositories.py`：批量读取历史动作；候选写入改为
  `INSERT OR IGNORE`。
- `src/companion/storage/schema.py`：迁移 v7，去重现有候选并建立
  `(source_event_id, content)` 唯一索引；`SCHEMA_VERSION = 7`。
- `src/companion/core/orchestrator.py`：可选 `context_builder`，构建隐私过滤
  后的 `CognitiveContext`，同一上下文同时交给分析与回复；候选去重。
- `src/companion/runtime.py`：`provider="mock"|"deepseek"` 装配；缺失 Key
  抛 `DeepSeekConfigError`；禁止真实/模拟混配。
- `src/companion/jobs/handlers.py`：候选进入终态后不再重复处理。
- `src/companion/cli.py`：`--provider`、`person add --allow-external-ai`、
  `deepseek check [--live]`、REPL `/status` 与聊天式输出。
- `README.md`：新增「Real DeepSeek mode」章节。

---

## 3. DeepSeek 配置与密钥安全

仅从环境变量读取：

| 变量 | 默认值 |
|---|---|
| `DEEPSEEK_API_KEY` | 无（缺失即报错） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | `deepseek-flash` |
| `DEEPSEEK_TIMEOUT` | `60` |
| `DEEPSEEK_MAX_RETRIES` | `2` |
| `DEEPSEEK_ANALYSIS_MAX_TOKENS` | `1024` |
| `DEEPSEEK_RESPONSE_MAX_TOKENS` | `2048` |
| `DEEPSEEK_ANALYSIS_REASONING_EFFORT` | 未设置 |
| `DEEPSEEK_RESPONSE_REASONING_EFFORT` | 未设置 |

已强制的不变量：

- Key 不接受 CLI 明文参数，不写入 SQLite，不打印到日志/异常，
  不出现在测试快照中（`test_deepseek_recovery.py::test_key_and_prompt_are_never_written_to_sqlite`）。
- Real 模式无 Key 时 `build_deepseek_client` 立即抛错；
  `say/repl/demo/recovery retry --provider deepseek` 打印
  `deepseek error: ... DEEPSEEK_API_KEY ...` 并返回 1，**绝不悄悄改用 Mock**。
- Mock 仍是所有命令的默认值。
- `provider="deepseek"` 时若只注入真实 cognitive 或只注入真实 response，
  `build_runtime` 抛 `ValueError`；只有显式
  `allow_mixed_providers=True` 才允许混合。

---

## 4. 运行时链路

```
CLI / REPL
  └─ build_runtime(db, provider="deepseek")
       ├─ ForegroundPrivacyGateway(registry)
       ├─ ContextBuilder(repos, registry, gateway)
       ├─ DeepSeekCognitiveProvider(client, gateway)
       ├─ DeepSeekResponseProvider(client, gateway)
       └─ Orchestrator(..., context_builder=...)

Orchestrator._process
  1 校验 person / ACTIVE / allow_processing
  2 读取或创建会话（非持久化时只读）
  3 去重 / 冲突 / 恢复
  4 关键词检索记忆
  4b ContextBuilder：过滤历史说话人 + 过滤记忆主体 + 交错多轮 + 截断
  5 CognitiveProvider.analyze（或复用已存分析）
  6 Policy.decide（quiet mode 等本地规则）
  7 仅 RESPOND 时 ResponseProvider.generate
  8 原子持久化分析 + 去重后的候选 + 记忆任务；保存动作
```

信任边界：

- `event_id`、`source_event_id`、`subject_person_id`、候选 id 全部由程序填写；
  模型输出里的同名字段被忽略。
- `referenced_memory_ids` 只保留检索结果中真实存在的 id，伪造项丢弃。
- `addressee_hint == human` 时，即使模型返回 RESPOND，也强制降级为 IGNORE。
- 模型不参与权限、静音、是否保存的决定。

---

## 5. 三种模式示例

### 5.1 Mock（本机实际执行，无网络）

```text
$ python -m companion --db /tmp/c.db repl --person person_4f4c... --provider mock
AI Companion REPL. Type /help for commands.
speaking as person_4f4c... (Owner)
provider: mock
you> 你好，今天怎么样？
assistant> 模拟回复：我收到你的话了。
you> /status
provider: mock
person: person_4f4c... (Owner)
conversation: default
addressee: agent
quiet_mode: off
```

### 5.2 Fake client（自动化测试，无网络）

`tests/test_deepseek_runtime.py` 注入 `FakeDeepSeekClient`：

- 分析请求返回严格 JSON；
- 回复请求返回 `"你好！很高兴见到你。"`；
- 断言 `[c["request_type"] for c in client.calls] == ["analysis", "response"]`；
- 第二轮分析的 prompt 同时包含上一轮用户文本与 assistant 回复；
- 无 `allow_external_ai` 时 `client.calls == []`。

### 5.3 Real（**未执行**，需用户在本机提供 Key）

```text
# 前置：export DEEPSEEK_API_KEY=...（只在当前 shell）
$ python -m companion --db "$DB" deepseek check --live
live check passed: model=deepseek-flash tokens=... reply='OK'

$ python -m companion --db "$DB" repl --person "$ALICE" --provider deepseek
AI Companion REPL. Type /help for commands.
speaking as person_... (Alice)
provider: deepseek
you> 你好，我叫 Alice，最近在学做饭。
assistant> <DeepSeek 根据真实上下文生成的回复>
you> 我特别喜欢川菜。
assistant> <DeepSeek 结合上一轮生成的回复>
you> 你觉得我今晚应该做什么菜？
assistant> <DeepSeek 结合「学做饭」「喜欢川菜」生成的建议>
```

完整验收步骤（含隐私、记忆、候选、错误、崩溃恢复）见
[docs/DEEPSEEK_ACCEPTANCE.md](DEEPSEEK_ACCEPTANCE.md)。

---

## 6. 测试覆盖（A–O 对照）

| 场景 | 覆盖位置 |
|---|---|
| A 请求/响应映射 | `test_deepseek_client.py` |
| B 无 Key 明确报错、不回退 Mock | `test_deepseek_client.py`、`test_deepseek_repl.py`、`test_deepseek_runtime.py` |
| C 401 快速失败 | `test_deepseek_client.py::test_fails_fast_on_auth` |
| D 402 余额不足快速失败 | `test_deepseek_client.py::test_fails_fast_on_insufficient_balance` |
| E 400 快速失败 | `test_deepseek_client.py::test_fails_fast_on_400` |
| F 429 重试后成功 | `test_deepseek_client.py::test_retries_then_succeeds` |
| G 5xx 重试后放弃 | `test_deepseek_client.py::test_gives_up_after_max_retries` |
| H 超时分类可重试 | `test_deepseek_client.py::test_timeout_classified_retryable` |
| I 空/畸形响应可重试 | `test_deepseek_client.py::test_empty_content_raises_response_error` |
| J 前台隐私网关 | `test_foreground_privacy.py`、`test_deepseek_runtime.py` |
| K 多轮上下文 | `test_context_builder.py`、`test_deepseek_runtime.py` |
| L 记忆检索与伪造引用 | `test_memory_retrieval.py` |
| M RESPOND/WAIT/IGNORE 决策 | `test_deepseek_runtime.py`、`test_deepseek_cognitive.py` |
| N 恢复与幂等 | `test_deepseek_recovery.py` |
| O 敏感内容不落库 | `test_deepseek_recovery.py`、`test_deepseek_client.py` |

最终结果：

```text
$ python -m pytest -q
250 passed
```

---

## 7. 使用方式

```bash
# 默认 Mock，零网络零费用
python -m companion say --person "$P" --to agent --text "你好"
python -m companion repl --person "$P"

# 安装真实 SDK
uv pip install --python .venv/bin/python -e ".[deepseek]"

# 只从环境变量读取 Key
export DEEPSEEK_API_KEY="sk-..."

# 先离线检查，再显式做一次最小真实调用
python -m companion deepseek check
python -m companion deepseek check --live

# 授权外部 AI（与 allow-memory-learning 独立的开关）
python -m companion person add --name Alice --role owner \
  --allow-processing --allow-persistence --allow-memory-learning \
  --allow-external-ai

# 真实模式
python -m companion say --provider deepseek --person "$P" --to agent --text "你好"
python -m companion repl --provider deepseek --person "$P"
python -m companion recovery retry <EVENT_ID> --provider deepseek
```

---

## 8. 剩余限制

- **未做真实 API 联调**：本环境无 Key、无 `openai` 包；真实链路需按验收文档
  手动跑一次。
- **记忆检索是关键词/CJK bigram**，不是语义向量检索。
- **候选记忆不会自动成为长期事实**：只进入 `CONSOLIDATED`/`SKIPPED`，等待
  后续人工确认里程碑。
- **未实现真实空闲反思、联网研究、语音输入输出、声纹识别**。
- **单进程 SQLite**：并发控制仍限于每会话串行 + 作业租约。
- 历史过滤按「说话人是否授权外部 AI」进行；若某条 assistant 回复本身转述了
  第三方未授权内容，当前不做传递式再脱敏（超出本阶段范围）。

---

## 9. 未执行的操作

按任务约束：

- 未 `git commit`
- 未 `git push`
- 未 merge
- 未发布
- 未调用真实 DeepSeek API
- 未把任何 Key 写入代码、数据库、日志或测试
