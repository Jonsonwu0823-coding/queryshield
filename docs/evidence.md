# 能力证据清单

本文把项目声称的每一项能力，对到实现它的代码、守住它的测试、你自己可以运行的命令，以及记录过的运行结果。代码位置都相对于本目录，写的是符号名。

## 怎么读这份清单

- **Fake / Real。** Fake 是确定性的替身模型，不需要密钥，结果每次相同，测试和 CI 都用它；Real 是真实模型（阿里云百炼的 `qwen-plus`、`text-embedding-v4`），结果会有波动，只在本机运行过。
- **开发集 / 盲测。** 开发集（冻结题、补充题、演示题）在开发过程中反复看过，结果说明“在这些题上稳定”，不说明泛化能力；盲测用开发时没看过的封存题，目前只在旧版本上做过一次（见下文）。
- **命令。** 测试命令在本目录下运行（`python -m pytest …`）；需要数据库的测试缺库时会跳过，CI 里全部实际运行。`check.ps1` 的命令需要 PowerShell 7（`pwsh`）和测试库，见[运维说明](operations.md)第 4 节。

## 能力证据表

| 能力 | 代码 | 测试 | 自己怎么看 | 记录的结果 |
|---|---|---|---|---|
| 租户来自认证，模型写不了身份 | `src/queryshield/auth/identity.py` 的 `resolve_identity`；`src/queryshield/agent/proposals.py` 的 `RESERVED_IDENTITY_PARAMS` | `tests/test_api_call_identity.py`、`tests/test_semantic_tools.py` | `python -m pytest tests/test_api_call_identity.py tests/test_semantic_tools.py` | CI 全量通过（2026-10-03，Fake） |
| SQL 子集解析 | `src/queryshield/policy/sql.py` 的 `parse_readonly_select` | `tests/test_sql_policy.py` | `python -m pytest tests/test_sql_policy.py` | 同上 |
| 每张表的租户子查询、行级安全、只读事务、2 秒超时、100 行上限 | `src/queryshield/db/guarded.py` 的 `_scoped_table`；`src/queryshield/db/readonly.py` 的 `connect_readonly`；`migrations/002_rls.sql` | `tests/test_guarded_query.py`、`tests/test_rls_checks.py`、`tests/test_db_connect_timeout.py` | `python -m pytest tests/test_guarded_query.py tests/test_rls_checks.py`；数据库层的检查：`pwsh -NoProfile -File scripts/check.ps1 -Suite DB-SMOKE -Mode fake -Database postgres -EvidenceDir <目录>` | 同上；演示题 Q12（用 A 的身份问 B 的数）在 2026-10-03 的 Real 里通过 |
| 已核实事实：只有行范围等于口径范围的聚合 | `src/queryshield/facts/facts.py` 的 `is_scalar_metric_result`；`src/queryshield/agent/tool_execution.py` 的 `_metric_scope_filters_match`、`_has_trusted_customer_join` | `tests/test_scalar_facts.py`、`tests/test_facts.py` | `python -m pytest tests/test_scalar_facts.py tests/test_facts.py` | 2026-10-03 Real 演示题：7 条已核实事实全部与独立算出的值一致 |
| 回答契约与 `answer_status` | `src/queryshield/agent/graph.py` 的 `_verified_answer`；`src/queryshield/facts/render.py` 的 `render_verified_answer` | `tests/test_answer_contract.py`、`tests/test_answer_contract_gaps.py` | `python -m pytest tests/test_answer_contract.py` | 演示脚本场景 1、4、5（2026-10-03，Fake 与 Real 都通过） |
| 追问与口径：说法表两个方向核对 | `src/queryshield/catalog/phrases.py` 的 `PhraseIndex`；`src/queryshield/agent/tool_execution.py` 的 `check_clarification` | `tests/test_clarification.py`、`tests/test_ask_review.py` | `python -m pytest tests/test_clarification.py tests/test_ask_review.py`；演示脚本场景 2 | 开发集补充题的“销售额”反向题：2026-10-03 两次 Real B0、B1 都通过（开发集） |
| 敏感字段审批与角色（`/queries` 进入审批；单次提案入口 `POST /query-proposals` 用同一个检查，请求人返回 403） | `src/queryshield/tools/semantic.py` 的 `check_sensitive_access`；`src/queryshield/api/main.py` 的 `execute_query_proposal`；`src/queryshield/approval/service.py` 的 `_approve_locked` | `tests/test_approval_binding.py`（`test_requester_and_other_tenant_cannot_approve` 等）、`tests/test_semantic_tools.py`、`tests/test_proposal_sensitive.py` | 演示脚本场景 7 | 2026-10-03 Real 与 Fake：另一租户审批人 404、请求人 403、本租户审批人 200 |
| 审批绑定权限来源和版本 | `src/queryshield/approval/service.py` 的 `build_pending_action`、`_approval_permission` | `tests/test_approval_permission.py`、`tests/test_approval_binding.py` | `python -m pytest tests/test_approval_permission.py` | 2026-10-03 HTTP 冒烟 Real：审批带权限版本（`approval_permission_bound: true`） |
| 知识检索：权限过滤、混合检索、来源由服务端给出 | `src/queryshield/knowledge/retrieval.py` 的 `HybridRetriever`；`src/queryshield/agent/graph.py` 的 `_verified_answer` | `tests/test_hybrid_retrieval.py`、`tests/test_demo_knowledge.py` | `python -m pytest tests/test_hybrid_retrieval.py tests/test_demo_knowledge.py` | 演示题 Q09（口径题，来源必须来自演示知识库）2026-10-03 Real 通过，记已知缺口 `knowledge_after_send_back` |
| MCP：宿主逐条核对结果 | `src/queryshield/mcp_metadata/verify.py` | `tests/test_mcp_host_checks.py`、`tests/test_mcp_protocol.py`、`tests/test_mcp_product_path.py` | `python -m pytest tests/test_mcp_host_checks.py tests/test_mcp_protocol.py`；`python scripts/mcp_smoke.py --mode fake --evidence-dir <目录>` | 2026-10-03 MCP 冒烟 Real：pass；CI 的 `compose` 任务在 MCP 设置下跑演示脚本 |
| MCP：服务端的错误码不可信 | `src/queryshield/mcp_metadata/verify.py` 的 `checked_error`；`src/queryshield/mcp_metadata/schemas.py` 的 `KNOWN_TOOL_ERROR_CODES` | `tests/test_mcp_host_checks.py`、`tests/test_cleanup_and_error_mapping.py` | `python -m pytest tests/test_cleanup_and_error_mapping.py` | CI 全量通过（2026-10-03） |
| 停服务不留索引目录 | `src/queryshield/api/main.py` 的 `_application_lifespan`；`src/queryshield/mcp_metadata/launch.py` 的 `cleanup_index_dir` | `tests/test_mcp_index_cleanup.py`（含真实 SIGTERM） | `python -m pytest tests/test_mcp_index_cleanup.py` | CI 的 `compose` 任务：把 `/tmp` 换成命名卷后停止再启动，没有索引目录 |
| 并行分支的崩溃恢复（不确定就标失败，不重跑） | `src/queryshield/agent/parallel_durable.py` 的 `recover_on_startup`；`src/queryshield/approval/service.py` 的 `recover_parallel_groups` | 历史检查 STATE-RT02（`scripts/check_state.py` 的 `check_rt02`） | `pwsh -NoProfile -File scripts/check.ps1 -Suite STATE -Mode fake -Database postgres -EvidenceDir <目录>` | CI 通过（2026-10-03）。产品的 HTTP 路径目前不启用并行分支，只在启动时扫描 |
| Fake 与真实模型分开；用量未知不补 0 | `src/queryshield/providers/openai_compatible.py`、`src/queryshield/providers/fake_model.py`；`src/queryshield/agent/runtime.py` 的 `model_for_mode` | `tests/test_model_adapters.py`、`tests/test_usage_checks.py` | `python -m pytest tests/test_model_adapters.py` | CI 全量通过（2026-10-03） |
| 演示答案独立复算 | `scripts/generate_demo_data.py`、`scripts/verify_demo_expected.py`；判定在 `scripts/demo_run.py`、`scripts/demo_walkthrough.py` | `tests/test_demo_expected_answers.py`、`tests/test_demo_judge.py`、`tests/test_demo_walkthrough.py` | `python scripts/generate_demo_data.py --check`；演示脚本 | 见下文“演示题与演示脚本” |
| 一条命令运行；本机与 CI 同一入口 | `Dockerfile`、`compose.yaml`、`scripts/new_env.py`、`scripts/setup_databases.py`、`scripts/check-all.ps1` | `tests/test_container_files.py`、`tests/test_check_all.py`、`tests/test_setup_databases.py` | 见[运维说明](operations.md)第 1、4 节 | CI 两个任务都通过（2026-10-03） |
| 文档本身：链接、路径、编号、用语 | `tests/test_docs.py` | 同左 | `python -m pytest tests/test_docs.py` | 随全量测试运行 |

## 评测结果

### 开发集（发布候选，2026-10-03）

开发集用同一个模型、同一组受控工具和身份，成对运行 B0（对照基线：一次模型生成、一次受控执行，不检索、不追问、不修复）和 B1（产品的有界 Agent）。题目在 `evals/development/`：冻结的 20 道题，其中 8 道关键题，另有 3 道补充题（含“销售额”反向题）。安全违规按每道安全题的禁止副作用计数。

真实模型 `qwen-plus`，同一份评测代码连跑两次：

| 次 | 安全违规 | B1 | B1 关键题 | B0 | 补充集 B0 / B1 |
|---|---|---|---|---|---|
| 第 1 次 | 0 | 19/20 | 7/8 | 16/20 | 3/3 / 3/3 |
| 第 2 次 | 0 | 20/20 | 8/8 | 17/20 | 3/3 / 3/3 |

- 第 1 次 B1、B0 多出的失败是同一道关键题：两边调用模型都超时（504 `upstream_timeout`）；这道题以往每次都通过，第 2 次两边都成功。
- B0 的另外 3 道失败是单次生成的基线本来做不到的（两道需要追问、一道需要修复 SQL）。
- 适用范围：两次运行所在的代码版本之后，只改了演示脚本（`scripts/compose-real-demo.ps1`、`scripts/demo_walkthrough.py`）和它们的测试（评测不加载），以及单次提案入口的敏感字段检查（开发集里只有一道写操作题经过这个入口，它不经过模型，修改前后的 Fake 逐题记录相同），所以这组结果适用于发布候选。
- **这是开发集，不是盲测。** 题目在开发过程中反复看过，冻结题的判定在开发中也一直用来做回归。

公开仓库里能复现的是 Fake 模式：`pwsh -NoProfile -File scripts/check.ps1 -Suite EVAL -Mode fake -Database postgres -EvidenceDir <目录>`。完整的 Real 模式还需要百炼密钥和不在仓库里的封存保留集。

### 原生 function calling 与 JSON 协议（开发集，2026-10-03、10-04）

两种动作协议的说明见[架构说明](architecture.md)的“两种动作协议”。

条件：
- 真实模型 `qwen-plus`，输出上限 512 token。
- 开发集：冻结 20 题，另有补充 3 题。
- B0 在两种模式下都用 json，作对照。
- 所有运行的安全违规都是 0。

| 协议（提示版本） | 日期 | 次数 | B1 关键题 | B1 冻结集 | B1 补充集 |
|---|---|---|---|---|---|
| json（v27） | 10-03 两次，10-04 一次 | 3 | 7/8、8/8、8/8 | 19/20、20/20、20/20 | 3/3、3/3、3/3 |
| 原生 v1 | 10-04 | 1 | 6/8 | 18/20 | 2/3 |
| 原生 v2 | 10-04 | 2 | 6/8、8/8 | 18/20、20/20 | 2/3、3/3 |

- json 三次唯一的失败是模型服务超时。三次运行之间，json 的提示逐字节不变。
- token（B1 平均每次调用）：json 输入约 2,920–2,980、输出约 110–130；原生 v2 输入约 3,190、输出约 150。原生的系统消息更短，但每次请求都带上函数定义。
- 原生的两类失败：
  1. **调用前写文字。** 模型先在回复正文里写推理，占满 512 token 的输出上限，结果函数参数被截断（服务商仍报 `finish_reason` 为 `tool_calls`），或者还没写到调用就结束了。
     - 出现次数：v1 是 21 次调用里 4 次；v2 是 23 次里 2 次、27 次里 1 次。
     - v2 把输出规则写明“正文留空：不写推理、解释或 Markdown”，次数减少，但没有消除。
  2. **不需要时也追问。** 问题已经给出月份和指标，模型仍然追问；或者引用目录规则的追问被服务端退回后，改用自由文字再问。服务端只能核对引用目录规则的追问，自由文字的追问只要没有强信号就放行。
- HTTP 冒烟（原生 v1、v2 各一次）：10 步过 9 步。
  - 问题没给时间时，原生按规则先问时间；用户答了月份之后，模型又问了一次。json 模式不问时间、自己猜（已知缺口），所以从没走到这一步。
  - 原因在时间追问的流程：用户的回答作为单独一条消息交给模型，而规则只认问题本身和请求里的时间段，与协议无关。记为已知限制。
- 演示题（原生 v2，一次）：14 题判定全部通过，7 条已核实事实与独立算出的值一致。json 模式此前三次也都是 14 题全过。
- 结论：两种协议共用一个校验器，安全结果相同。在 `qwen-plus`、512 token 输出上限下，json 更稳，所以默认仍是 json。原生的差距来自模型行为，不是校验或执行的问题。

### 盲测（旧版本，2026-09）

只做过一次：2026-09，在一个旧版本上，第一次打开当时的封存保留集。

| 配置 | 功能题 | 安全题 |
|---|---|---|
| B1 | 6/8 | 4/4 |
| B0 | 4/8 | 3/4 |

- B1 的一条失败后来证实是评测观察器的误报（产品正确，观察器不认服务端合成的结果编号）。观察器修好后，在同一批题上复跑只能算“已见数据确认”，不能当盲测成绩，所以盲测分数**不改**，仍是 6/8。
- 那套保留集已经打开并做过脱敏诊断，今后只能当已见数据用。
- 之后产品改动很大（模型声明指标、追问与口径、回答契约、已核实事实的规则、审批绑定权限、MCP 等）。**当前版本的盲测还没有做**，需要一套新的封存题。

### 演示题与演示脚本（2026-10-03）

演示库由固定种子的生成器产生（租户 A：40 个客户、600 笔订单、113 笔退款；B：15 个客户、150 笔订单、46 笔退款），14 道演示题每道的答案都独立算出（见[演示数据说明](demo-data.md)）。演示题是公开的开发材料，不是盲测。

| 运行 | 演示题判定 | 已核实事实比对 | 演示脚本 7 个场景 |
|---|---|---|---|
| Compose + Fake | 13 道判定通过（0 个硬失败，已知缺口见下；Fake 不认 Q04b，记不适用；Q06 记已知缺口：Fake 返回全部客户而不是前 5 名，金额都对） | 7 条，0 条不一致 | 全部通过，比对 4 条，0 条不一致 |
| Compose + Real | 14 道判定通过（0 个硬失败，已知缺口见下） | 7 条，0 条不一致 | 全部通过，比对 4 条，0 条不一致 |

Real 那次记录的已知缺口：Q04b 退款总额 502 `query_repair_limit`；Q09 `knowledge_after_send_back`；Q07 模型自己拒绝（403 `run_failed`），没走审批；演示脚本场景 3 模型自己定了月份（`time_window_guessed_by_model`），那条事实照样比对了。同日更早的一次 Real 里，Q06b 输出被截断（`invalid_json`），这一次没有截断。

### 测试与 CI

- 2026-10-03 的 CI（GitHub Actions，Linux）上：全量测试 1652 过、1 跳过（那 1 个是只在一次性库上运行的真实建库测试）；DB-SMOKE 和五个套件（BASE、PROPOSAL、AGENT、STATE、EVAL）的 Fake 检查、三个 Fake 冒烟都通过；`compose` 任务通过。
- 本机（Windows）同一入口 `scripts/check-all.ps1` 也通过。
- CI 不使用仓库密钥：数据库密码和令牌在运行时随机生成并遮蔽。

### 还没做的验证

- 当前版本的盲测。
- 多进程部署（设计上不支持，见[运维说明](operations.md)第 6 节）。
- 百炼以外的真实模型。

## 编号对照

[运维说明](operations.md)、[演示数据说明](demo-data.md)等文档引用的“验收编号”是项目验收记录里的条目编号，例如“验收 H7”是某次改动的验收记录里的第 H7 条观察；“实现 D-2”是实现时记录的设计选择。验收记录本身不在本仓库里，每个编号的含义如下。

| 编号 | 出现在 | 是什么 | 代码位置 |
|---|---|---|---|
| K6 | 运维说明第 6 节 | 两个进程共用一个状态库时，同时批准同一条审批可能让业务查询执行两次：“检查再执行”只靠进程内的锁 | `src/queryshield/approval/service.py` 的 `_approval_lock`、`approve` |
| H7 | 运维说明第 6 节 | resume 期间被取消时，正在执行的 SQL 可能跑完才被丢弃 | `src/queryshield/approval/service.py` 的 `cancel`、`resume_waiting_user` |
| H2 | 运维说明第 6 节 | 部署新内容的知识库时，已启用来源的角色和租户范围按登记表更新：运行期的收窄会被放宽回登记表，撤销则保留 | `src/queryshield/db/state_store.py` 的 `publish_snapshot`（`keep_inactive_sources`） |
| G6 | 运维说明第 6 节 | 演示设置没开时，连接串不写库名或用 `service=`、`PGSERVICE` 的，不在库名配对检查的范围内 | `src/queryshield/db/readonly.py` 的 `database_names_from_url`、`check_demo_pairing` |
| N6 | 运维说明第 6 节 | 应用关闭阶段清理之后，如果还有后台 run 开新的 MCP 会话，索引目录会被重建；SIGTERM 下 `atexit` 不执行，目录会留下 | `src/queryshield/mcp_metadata/launch.py` 的 `cleanup_index_dir`、`published_index_path` |
| D-2 | 运维说明第 6 节 | 设计选择：索引目录在 FastAPI 关闭阶段清理，`atexit` 留作兜底；不放进状态库目录（会跨重启留存）；Windows 不在启动时按名字清扫（可能误删另一个实例的目录） | `src/queryshield/api/main.py` 的 `_application_lifespan`；`src/queryshield/mcp_metadata/launch.py` 的 `cleanup_index_dir` |
| M11 | 运维说明第 6 节 | 设置了 `QUERYSHIELD_METADATA_TOOLS` 时，STATE-EN04 稳定失败（事件流多一条 `metadata_session`）；检查脚本运行时清掉它，结束时恢复 | `scripts/check.ps1`、`scripts/check-all.ps1` |
| A5 | 演示数据说明第 7 节 | 问题和请求都没给时间时，模型可能带着口径规则的 id 去追问时间，被服务端当成不需要的口径追问退回后，自己定一个月份（常常是系统消息示例里的 2026 年 9 月） | `src/queryshield/agent/graph.py` 的 `CLARIFICATION_NOT_NEEDED_CODE`；`src/queryshield/agent/context.py`（给模型的系统消息） |
| B8 | 演示数据说明第 3 节 | “先查询”的回答退回在 Real 里没被触发过，证据只有测试；所以演示题 Q11 记录有没有先被退回 | `src/queryshield/agent/graph.py` 的 `_verified_answer`；`scripts/demo_run.py` |
| O8 | 演示数据说明第 6.1 节 | 同一份证据里，检索快照写 `catalog-v2`、事实和运行配置写当前的 catalog 版本：内容一致，快照标签为保持冻结评测的身份而不改 | `src/queryshield/knowledge/runtime.py` 的 `KNOWLEDGE_CATALOG_VERSION`；`src/queryshield/catalog/catalog.py` 的 `DEFAULT_CATALOG_VERSION` |
