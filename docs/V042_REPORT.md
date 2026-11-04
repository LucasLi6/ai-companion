# V0.4.2 实施报告 — 角色优先认知与对话驱动情绪

**分支：** `v0.1-deepseek-integration`（未 commit、未 push）
**基线：** `812097a` (V0.4.1，schema 15)
**迁移：** schema 15 → 16（只追加，未修改历史迁移）
**测试：** `.venv/bin/python -m pytest -q` → **779 passed in 40.28s**
**真实模型：** 本机没有 `DEEPSEEK_API_KEY`，因此**没有执行在线 A/B**，也没有任何关于线上模型行为的结论。

本报告只描述本地实际完成的代码、测试和已知失败/限制。凡未验证的内容均明确标注。

---

## 1. 分层结果

| 层 | 决定什么 | V0.4.2 的实现 |
|----|----------|----------------|
| 人格 / Persona Core | Ava 是谁：价值观、偏好、边界、长期自述 | 可信 `character_description` 全量保留；persona-first 编译；立场（stance）来自实际说出口的回复或管理员配置 |
| 对话事件 / Pragmatic | 这一轮对方在做什么 | 模型给出语用提示，本地校验后才生效；交互框架进入 SceneManager |
| 情绪 / Emotion | 这些事件如何影响她 | 事件优先；场景相关性抑制旧情绪；新事件可接管表达；时间衰减降为辅助 |
| 回复 / Response | 她如何表达反应 | `response-v7` 精简为角色优先；删除重复安慰/追问指令；CONCERN 不再默认变温柔或追问 |

---

## 2. 按步骤的实现

### Step 1 — 修复确定的 Bug

- `agent_persona_core` 的更新/读取不再丢失字段；`character_description` 纳入
  `as_prompt_dict()`、`_PROMPT_CORE_FIELDS`、repository 的 SELECT/INSERT/UPDATE。
- 情绪主体/目标校验：新增 `emotion/interaction.py`，按事件类型校验 target /
  social_intent / factuality，actor 只来自可信说话者 id。
- 回归测试：`tests/test_v042_step1.py`（11）。

### Step 2 — 语用理解

- `PragmaticAnalysis`、`validate_pragmatic_analysis()`：主行为非法或证据不是当前输入的
  逐字子串时，整个提示失效；次要行为/框架/角色/话题转换逐字段降级；指代 id 只接受
  本轮可见事件。
- `InteractionFrame` 写入 SceneManager；`cognitive_analyses.pragmatic_json` 持久化，
  重试/恢复时不再重新分析。
- 测试：`tests/test_v042_pragmatic.py`（8）。

### Step 3 — 对话驱动情绪

新增 `src/companion/emotion/transition.py`：

- `time_decay_mode`: `OFF | AUXILIARY | LEGACY`，默认 `AUXILIARY`。
- 场景相关性：`CONTINUE 1.0`、`RETURN 0.85`、`SHIFT 0.25`、`UNCERTAIN 0.60`；
  指向上一轮 +0.15；SETUP/PAYOFF +0.10；框架变化再 ×0.85。
- 框架刺激：CONFLICT→FRUSTRATION + tension；REPAIR→RELIEF − tension；
  SUPPORT→CONCERN；PLAYFUL→JOY；SERIOUS→SERIOUS。
- 新事件影响足够大时可覆盖旧情绪的排序（`ranking_override_*`）。
- 事件规则优先于框架提示；同一轮两者不会互相矛盾。
- 测试：`tests/test_v042_transition.py`（10）。

### Step 4 — 强化人格与自有立场

- `PersonalityCompiler.compile(..., stances=...)`：角色核心 → 既往立场 → 表达优先级
  → 人格准则；确定性 `character_guidance`。
- `agent_stances` 表：来源只允许 `ADMIN` / `RESPONSE`；**没有** `THOUGHT` 来源，
  模型私有分析不能变成“她说过的话”。
- `POST` 接口/服务：`import_persona_description`、`add_stance`、`retire_stance`、
  `list_stances`；回复生成后仅从实际发出的文本中保守抽取立场。
- 测试：`tests/test_v042_persona.py`（7）。

### Step 5 — 重构回复

- `RESPONSE_PROMPT_VERSION = "response-v7"`；system prompt 压缩为 10 条。
- 用户消息顺序：身份 → 角色核心/立场/优先级/人格准则 → 当前场景 → 语用理解 →
  对话风格/衔接 → 情绪影响 → 表达习惯 → 风格参考 → 最近对话 → 行为历史 → 记忆 →
  当前输入 → 认知分析。
- 不再渲染 `想要的回应风格` / `可考虑的方向`；`response_style` 不再是模板指令。
- `CONCERN` 表达映射改为 `(-0.05, 0.15, -0.35, -0.10, 0.35)`：降低轻松度、提高认真度，
  但**不**提高温柔度或追问倾向。
- 测试：`tests/test_natural_response.py`（更新）、`tests/test_v042_response.py`（2）。

### Step 6 — 可靠性

- Schema 16：`agent_persona_core.character_description`、`agent_stances`、
  `cognitive_analyses.pragmatic_json`、情绪事件的 `transition_cause/frame_from/frame_to/scene_relevance`。
- `AnalysisRepository.save_analysis/get_analysis` 现在同时持久化并恢复
  `dialogue_impulse_json` 与 `pragmatic_json`，崩溃恢复不再丢理解结果。
- **独立安全路径** `src/companion/core/safety.py`：
  - 本地正则识别明确的自伤/即时伤害他人表达（中英）；
  - 固定本地回复，不调用 response provider，不受人格、情绪、场景影响；
  - 可覆盖模型 WAIT、quiet mode、当轮“别回复我”；
  - 仍尊重 explicit human addressee 与 permission 拒绝；
  - 危机轮不生成记忆候选、不记录立场；
  - 审计原因 `ReasonCode.SAFETY_RESPONSE`。
- **记忆 vs 短时情绪**：新增 `memory/transience.py`；“第一人称 + 明确时间锚点 + 情绪词”
  且无持续/偏好措辞的候选写为 `NO_MEMORY / temporary_emotion_not_durable`。
  情绪本身仍进入 scene/emotion 层，只是不进入长期记忆。
- 测试：`tests/test_v042_reliability.py`（9）。

### Step 7 — 测试与捕获

- `tests/test_v042_scenarios.py`（32）：
  - 24 个语用类别：提问、信息、任务、边界、批评、道歉/修复、玩笑、自我披露、感谢、
    邀请、告别、问候、抱怨、闲聊、确认、指代前文、SETUP、PAYOFF、话题回归、话题切换、
    “你是不是机器人”、越权请求、立场矛盾、情绪宣泄；
  - 跨轮场景：冲突→修复→中性、话题切换抑制旧情绪、玩笑 SETUP→PAYOFF、
    实际发出的立场进入下一轮请求、完整请求链捕获、安全轮少一次模型调用、
    安全覆盖明确沉默。
- `scripts/v042_capture_requests.py` 生成
  `docs/V042_REQUEST_CAPTURE.json`：3 轮 × (analysis + response) = 6 个完整请求，
  使用真实 DeepSeek provider 类 + 脚本化 transport，无网络。
- `scripts/v042_deepseek_ab.py`：可选在线 A/B（AUXILIARY vs LEGACY），需要真实
  `DEEPSEEK_API_KEY`；无 key 时以退出码 2 明确说明“未执行”。

---

## 3. 与 V0.4.1 的实际差异

1. **人格不再被场景/情绪重写**：`character_description` 与 `character_guidance` 在
   场景、语用、情绪之前；风格参考与临时标签只能影响措辞。
2. **旧情绪不再跨话题主导表达**：话题切换时场景相关性会压缩旧激活；新事件强度足够
   时可以接管排序。
3. **时间衰减从主因降为辅助**：`AUXILIARY` 下仍有衰减，但不再靠“放久了”解释一切。
4. **CONCERN 不再是安慰脚本**：它提高认真度、降低俏皮和追问，不增加温柔和问题。
5. **回复提示噪音显著减少**：去掉 `response_style` 指令和 `response_direction` 方向，
   删除重复的“安慰/追问/分类模板”段落，保留一条明确规则。
6. **立场连续可审计**：只记录实际发出的回复或管理员配置；模型私有分析不进入立场。
7. **危机表达有独立下限**：固定本地回复 + 一次 analysis 调用，不再依赖模型是否愿意接住。
8. **短时情绪不进入长期记忆**：`我现在很焦虑` / `我最近很累` 不写事实；
   `我喜欢安静` / `我一直很容易焦虑` 仍可成为长期记忆。
9. **恢复不丢理解**：pragmatic 与 dialogue impulse 现在随 analysis 一起持久化。

---

## 4. 已知限制与失败

- **没有在线 A/B。** 本机无 `DEEPSEEK_API_KEY`。`scripts/v042_deepseek_ab.py` 只验证了
  “无 key 时明确失败”，未产生任何真实模型对比结果。离线 scripted transport 不能替代
  人工盲评，也不能证明线上模型会遵守 prompt。
- **安全检测是窄规则，不是临床分类器。** 它只覆盖明确措辞，可能漏掉委婉/间接表达；
  回复是固定文本，不掌握用户所在地的完整急救资源，只举例常见号码。它不是心理危机干预服务。
- **安全路径有意覆盖 quiet mode 和当轮明确沉默。** 这是安全下限的产品决定；若需改成
  尊重静音，必须改代码并重新评估，当前没有开关。
- **立场抽取保守。** 需要明确的“我觉得/我认为/我更喜欢……”句式；可能漏掉含蓄立场，
  但不会把模型想法或普通附和记录成立场。
- **短时情绪过滤保守。** 需要第一人称 + 明确时间锚点 + 情绪词。没有这些标记的陈述仍可能
  被当作事实，交由 consolidation 的其余规则处理。
- **冲突后的情绪不是归零。** 饱和累积与半衰期会留下低于主导阈值的残余激活；测试断言的是
  主导情绪与张力下降，而不是绝对值归零。
- **恢复时的 pragmatic_json 不做证据复核。** 它信任本地已校验并写入的数据库行；Pydantic
  仍校验类型/范围，但不会拿原文重新核对 `evidence_span`。数据库本身被视为可信边界。
- **未 push。** 按既有约束，未经明确授权没有对远程仓库做任何操作。

---

## 5. 文件

### 新增生产代码

| 文件 | 作用 |
|------|------|
| `src/companion/emotion/transition.py` | 对话事件/框架驱动的情绪转移与场景相关性抑制 |
| `src/companion/emotion/interaction.py` | 按事件类型校验 target/intent/factuality |
| `src/companion/agent/stance.py` | 立场模型、保守抽取、持久化 |
| `src/companion/core/safety.py` | 独立本地安全检测与固定回复 |
| `src/companion/memory/transience.py` | 短时情绪陈述过滤 |

### 新增脚本 / 文档

| 文件 | 作用 |
|------|------|
| `scripts/v042_capture_requests.py` | 离线生成完整请求链捕获 |
| `scripts/v042_deepseek_ab.py` | 有 key 时的可选在线 A/B |
| `docs/V042_REQUEST_CAPTURE.json` | 6 个完整请求（3 轮 analysis+response） |

### 修改（主要）

`core/orchestrator.py`、`domain/models.py`、`domain/enums.py`、`emotion/{appraisal,dynamics,expression_policy,models,parameters,service}.py`、
`agent/{persona_core,personality_compiler,service}.py`、`cognition/{scene_manager,scene_models}.py`、
`prompts/{cognitive,response}.py`、`providers/deepseek_cognitive.py`、
`storage/{schema,repositories}.py`、`memory/consolidation.py`。

### 新增测试

| 文件 | 数量 |
|------|------|
| `tests/test_v042_step1.py` | 11 |
| `tests/test_v042_pragmatic.py` | 8 |
| `tests/test_v042_transition.py` | 10 |
| `tests/test_v042_persona.py` | 7 |
| `tests/test_v042_response.py` | 2 |
| `tests/test_v042_reliability.py` | 9 |
| `tests/test_v042_scenarios.py` | 32 |
| **合计** | **79** |

另有既有测试更新：`test_natural_response.py`、`test_v041_integration.py`、
`test_v041_migration.py`、`test_scene_repository.py`（版本与 v7 断言）。
