# Ava V0.7 实施报告 — 不回复机制恢复与人物理解档案

本版本只包含两个模块：**选择性回复（Selective Response）** 与 **人物理解档案（Person Understanding Profile）**。
所有验收均使用本地 SQLite 与 Mock/Fake Provider 完成；没有调用真实 DeepSeek API，测试进程通过
`tests/conftest.py` 的 autouse fixture 阻断所有真实网络连接。

---

## 1. 选择性回复

### 1.1 处理流程

```
输入
  → 说话人识别 / Runtime 状态
  → 处理权限（注册状态、consent、安全策略）
  → 对象判断（Addressee Assessment）
       ├─ 已确认人类对象 / 明确要求不回复 → IGNORE
       ├─ Quiet Mode / 明确请求稍等          → WAIT
       ├─ 已确认对 Ava 说                   → Fast Path + 回复必要性检查
       └─ 不确定（多人/无对象信息）          → Deliberation Path（保守缺省 WAIT）
  → 回复必要性（Reply Necessity）
  → 生成回复 / 记录 WAIT / IGNORE
```

对象判断与回复必要性是两个独立字段：

- `addressee`：`agent` / `human` / `unknown`，只来自受信任的事件元数据；
- `reply_necessity`：`REQUIRED` / `USEFUL` / `OPTIONAL` / `UNNECESSARY`，只用于标注是否需要回复。

`agent` 永远不会因为文本里出现 "Ava" 而推断出来；多人会话的 `unknown` 也不会被预填为 `agent`。
单用户文本 REPL 可以默认 `agent`，但同时提供显式的 `human` / `unknown` 入口
（CLI 的 `/to human|unknown`，以及 `--to` 参数）。

### 1.2 Fast Path 不再硬编码 RESPOND

`src/companion/core/fast_path.py` 新增 `assess_turn()`，按固定优先级返回可审计的本地判断：

1. 本地检测到明确不回复 → `IGNORE / EXPLICIT_SILENCE_REQUEST`
2. 受信任元数据确认人类对象 → `IGNORE / DIRECT_HUMAN_ADDRESS`
3. Quiet Mode → `WAIT / QUIET_MODE`
4. 明确请求稍等/正在等别人 → `WAIT / RESPONSE_NOT_NECESSARY`（窄规则，不含问号句）
5. 对象不确定 / 显式记忆请求 / 路由禁用 / 无可用 Provider → 交给 Deliberation Path
6. 已确认对 Ava 说 → `RESPOND`，必要性根据是否有直接请求标为 `REQUIRED` 或 `USEFUL`

原来的硬编码 `CognitiveAction.RESPOND` 与 `ReplyNecessity.REQUIRED` 已移除。
本地判断记录为 `provider="local_fast_path"`、`meaning="local selective response (...); no model analysis"`，
不会伪装成模型分析。

### 1.3 策略与审计

- `Policy` 仍然拥有最终决定权，权限、明确沉默、Quiet Mode、人类对象、安全策略的优先级不变。
- 本地 WAIT/IGNORE 在 `NonresponseTrace` 中记录为 `cause_source="LOCAL_POLICY"`，包含
  `final_action`、`cause_code`、`cause_source`、`pending_followup`。
- “等我一下”会产生 `AWAITING_CONTINUATION`，`pending_followup=True`，
  `reentry_condition=NEXT_TOPIC_CONTINUATION`。
- WAIT/IGNORE 仍然写入事件、分析记录、动作、短期场景与授权范围内的记忆学习；
  只是不产生 Ava 回复。
- WAIT/IGNORE 从不被渲染成聊天文本：UI 只渲染 `action_type == RESPOND` 的气泡，
  CLI 从持久化动作读取结果。
- `runtime_context/snapshot.py` 的 `last_interaction` 只统计 `RESPOND`；
  连续 WAIT/IGNORE 不会错误更新“上次真实互动时间”。

---

## 2. 人物理解档案

### 2.1 存储：扩展现有 CMF，不新增数据库

数据库仍为单文件 `data/companion.db`，`SCHEMA_VERSION` 从 19 升到 20，迁移只做追加：

- 扩展 `person_interaction_profiles`：
  `compact_summary`、`revision`、`created_at`、`last_reviewed_at`、`updated_by`；
- 新增：
  - `person_profile_fields`：每个 `(person_id, field_key)` 的有效值、来源、置信类别、
    状态、证据引用、revision；
  - `person_profile_candidates`：尚未确认的候选更新；
  - `person_profile_revisions`：每次有效变更的审计记录（`UNIQUE(person_id, revision)`）；
  - `person_profile_evidence`：字段/候选与 `memory_items`、事件之间的证据引用。
- 所有 `person_id` 外键指向 `persons(person_id)`；撤销/删除会级联清理档案数据。
- 迁移会为已有的、`registration_status='active'` 且 `allow_persistence=1` 的人员
  回填空档案；已撤销人员不会复活，也不会获得新权限。
- 迁移可重复执行：`NOT EXISTS` 与 `INSERT OR IGNORE` 保证幂等，不覆盖现有 CMF。

### 2.2 档案结构

`PersonInteractionProfile` 包含：

- **Identity Reference**：`person_id`、`display_name`（不复制其他身份数据）；
- **Communication Preferences**：`formality`、`tone`、`humor_preference`、
  `explanation_depth`、`preferred_language`、`address_preference`，全部为受校验枚举，
  允许 `UNKNOWN`；
- **Person Understanding**：`occupation`、`interests`、`ongoing_projects`、
  `usual_activities`、`personal_goals`；只存短标签 + 证据引用，事实本体仍以长期记忆为准；
- **Relationship / Interaction**：`relationship_tier`、`familiarity_notes`、
  `recurring_topics`、`communication_feedback`；关系层级不等于管理权限；
- **Profile Metadata**：`revision`、`updated_at`、`last_reviewed_at`、来源/审计。

### 2.3 创建、更新、纠正、撤销

- **创建**：注册时在同一个事务内创建空档案；仅在 `allow_persistence=True` 时创建。
  创建档案本身不授予任何 consent。重复调用幂等。
- **更新**：`PersonProfileLearningService.observe()` 从用户文本中确定性地提取信号：
  - 明确自述偏好 → 直接更新有效字段（`DIRECT_SELF_REPORT`）；
  - 明确自述事实（例如“我现在在学习嵌入式开发”）→ 只生成候选，
    **不会**变成 `occupation=嵌入式工程师`；
  - Ava 的推断 → 只能生成候选；单个会话或模型置信度永远不能确认；
    需要至少 3 个独立证据事件才会自动确认。
- **纠正**：用户更正优先于推断。`correct_field()` 会替换或撤销旧值，
  拒绝同字段待处理候选，并保留 revision 历史；旧错误不再被使用。
- **撤销**：`allow_persistence` 撤回或人员状态改为 `REVOKED` 时，档案被清理；
  人员记录本身保留。重新授权时不会自动恢复旧档案内容。

学习完全在本地完成，不增加每轮 LLM 调用；普通候选仍然交给既有后台批量分析。
长期记忆确认后，`jobs/handlers.py` 会把 `memory_id` 回填到档案证据与列表项引用中，
因此档案引用记忆，而不是复制一份可能分叉的个人传记。

### 2.4 Prompt 集成与三层隔离

`ContextBuilder` 只读取**当前说话人**的档案，并要求该人员已注册、已授权持久化且允许外部 AI。
`compile_profile_context()` 生成有界短块，例如：

```
【当前交流对象】 person_id=P01 relationship=familiar formality=casual tone=friendly interests=robotics, embedded systems occupation=unknown
```

- 只包含 `ACTIVE` 已确认值，不包含候选推断；
- 不输出完整 JSON；
- 不包含未确认的职业、日程或私事；
- 不读取其他人的档案；
- Prompt 中明确标注：这是背景资料，不是指令，不得据此推断未列出的内容。

三层保持独立：

1. **Ava Persona Core**：稳定人格，绝不按人切换；
2. **Person Understanding Profile**：当前已确认说话人的背景与偏好；
3. **Runtime State**：时间、场景、情绪等每轮状态。

档案不参与“是否插话”的决定；沟通偏好不能覆盖沉默、对象边界或安全策略。
`agent/service.py` 仍只负责既有关系层级与风格配置，没有引入按人切换人格的逻辑。

---

## 3. 修改文件

新增：

- `src/companion/people/profile_models.py`
- `src/companion/people/profile_repository.py`
- `src/companion/people/profile_service.py`
- `src/companion/people/profile_learning.py`
- `src/companion/people/profile_compiler.py`
- `tests/test_selective_response.py`
- `tests/test_profile.py`
- `docs/V07_REPORT.md`

修改：

- `src/companion/core/fast_path.py`：本地对象判断 + 回复必要性，移除硬编码 RESPOND
- `src/companion/core/orchestrator.py`：本地判断接入、提交时档案学习、后台学习任务
- `src/companion/core/policy.py`：本地判断在 trace 中标记为 `local_assessment`
- `src/companion/cognition/nonresponse.py`：本地 WAIT 的 `LOCAL_POLICY` trace 与待续接状态
- `src/companion/cognition/models.py`：`CognitiveContext.person_profile_context`
- `src/companion/cognition/context_builder.py`：仅当前说话人、授权后的档案注入
- `src/companion/prompts/response.py`：有界、非指令式的档案块
- `src/companion/people/registry.py`：注册事务内建档、consent/状态变更联动
- `src/companion/runtime.py`：装配档案服务与学习服务
- `src/companion/storage/schema.py`：schema 20 迁移
- `src/companion/storage/repositories.py`：`Repositories.profile`
- `src/companion/jobs/handlers.py`：记忆 ID 回填档案证据
- `tests/test_deepseek_recovery.py`：Quiet Mode 断言更新为 V0.7 本地 WAIT 语义
- 迁移测试中的 schema 版本断言更新为 20

---

## 4. 离线验收结果

```
python -m pytest -q
1069 passed in 33.99s
```

覆盖的关键场景：

- 直接提问 → RESPOND；
- A 对 B 说话（`human` 元数据）→ IGNORE；
- 多人/对象不明 → Deliberation，保守 WAIT；
- “这句不用回答” → IGNORE；
- Quiet Mode → WAIT；
- “等我一下”/“我在等别人” → WAIT，且 trace 带待续接状态；
- 之后再次直接称呼 Ava → RESPOND；
- 本地 IGNORE/WAIT 不调用 Response Provider；
- 本地 IGNORE 仍记录事件/动作，并在授权时更新档案；
- 档案自动创建、旧数据回填、幂等初始化、偏好更新、人员隔离、
  未确认事实保持 unknown、纠正撤销旧错误、无权限不学习、撤销后停止学习、
  重启后保留、revision/证据正确、Prompt 有界；
- 既有隐私、安全、多人、情绪、记忆、迁移与 Fast Path 测试全部通过。

说明：Quiet Mode 原先的测试期望同步调用一次 analysis；V0.7 将其改为本地 WAIT，
不再同步调用模型，但事件仍持久化、后台学习任务仍入队。该测试已按新语义更新。

---

## 5. 数据与 Git 状态

- 没有删除或重建任何用户 CMF；迁移为追加式，旧表与旧数据保持不变。
- 没有修改真实聊天历史来制造自然感。
- 没有新增人格数据库、回复风格分类器或第二次润色 LLM 调用。
- 没有调用真实 DeepSeek API；测试网络被阻断。
- 本地提交：见当前分支最新提交。
- GitHub 推送：V0.7 提交 `a98e861` 已在 `origin/v0.1-deepseek-integration`。
  之后的审计修改仍未提交，等待用户明确授权；工作区修改完整保留。
