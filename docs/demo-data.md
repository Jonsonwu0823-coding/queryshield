# 演示数据、中文演示知识库和演示题

本文说明：怎么建演示库、怎么跑演示题、演示数据和演示知识库是什么、两个 catalog 版本号的来由，以及已知限制。命令里的凭据一律是占位符或提示输入，不要写进命令、日志或证据。

## 1. 它是什么

- **演示库 `queryshield_demo`**：表结构与 commerce-v1 相同（客户、订单、退款三张表，同样的行级权限），租户 A（小网店）和 B（批发商），数据比 commerce-v1 多、更像真实业务，由固定种子的生成器写出。数据版本 `commerce-demo-v1`。
- **演示知识库**：`fixtures/demo/knowledge/`，13 篇中文文档（含一篇已停用的旧版退款规则），有自己的 `source_registry.json`，知识版本 `knowledge-demo-v1`，catalog 标 `catalog-v4`。只供产品用；评测和历史检查读不到。
- **演示题清单**：`fixtures/demo/demo-questions-v1.json`，14 道题（Q01–Q12 加观察题 Q04b、Q06b），每道有独立算出的答案。它是公开的开发/演示材料，**不是盲测题**。
- **演示设置**：服务端环境变量 `QUERYSHIELD_DEMO_DATASET=commerce-demo-v1`。打开后，产品检索器用演示知识库；客户端没有任何办法选择。没打开时，产品的行为、知识快照、版本标签和缓存键都与本单之前完全相同。

> 证据和事实里出现的 `commerce-v1`，指的是**表结构和口径**，不是数据行。演示库的数据是另一份，表结构和口径与它相同，所以 catalog 不变。

文件一览：

| 文件 | 作用 |
|---|---|
| `scripts/generate_demo_data.py` | 生成器：写出 `fixtures/demo/commerce-demo-v1.{sql,md}` 和 `demo-questions-v1.json`；`--check` 比较已提交文件 |
| `fixtures/demo/commerce-demo-v1.md` | 由生成器写出的说明：口径、各种边界情况的数量、按月汇总表、演示题预期答案 |
| `scripts/bootstrap_demo_db.py`、`scripts/bootstrap-demo-db.ps1` | 建表、载入数据、授权（只接受 `_demo` 库名） |
| `scripts/verify_demo_expected.py` | 预期答案的第二遍计算：对建好的演示库跑手写 SQL |
| `scripts/demo_run.py`、`scripts/demo-local.ps1` | 起真实 uvicorn 进程，经 HTTP 逐题运行并比对 |

## 2. 建演示库

`CREATE DATABASE` 要数据库管理员权限，建表脚本自己不做。Linux / macOS 和 Windows 各有一种办法；之后都用同一个建表脚本。脚本只接受库名以 `_demo` 结尾的连接，`_test` 一律拒绝；它在一个事务里：执行迁移 001、临时关闭三表行级权限、`TRUNCATE` 三张表并载入演示数据、核对行数与生成器一致（不一致就回滚并失败）、执行迁移 002（开启并强制行级权限）、给**已存在**的 `queryshield_ro` 授权。不创建角色、不设密码、只打印库名和行数，重复执行结果相同。

### 2.1 Linux / macOS（Docker 里的 PostgreSQL）

沿用 README“不用 Compose 的本机安装”的容器名和端口（容器的超级用户叫 `queryshield`）。尖括号里换成你自己的密码，只放在当前 shell 里，不要写进文件。前提：只读角色 `queryshield_ro` 已存在（见 README）。

```bash
cd queryshield

# 1. 建库（容器内本地连接）
docker exec queryshield-postgres psql -U queryshield -d postgres -c "CREATE DATABASE queryshield_demo"

# 2. 建表、载入数据、授权（管理员连接，库名必须以 _demo 结尾）
export QUERYSHIELD_DEMO_BOOTSTRAP_DATABASE_URL='postgresql://queryshield:<管理员密码>@127.0.0.1:5433/queryshield_demo'
python scripts/bootstrap_demo_db.py
# 期望输出：bootstrap_demo_db_ok database=queryshield_demo fixture=commerce-demo-v1 A=customers:40/orders:600/refunds:113 B=customers:15/orders:150/refunds:46 customers=55 orders=750 refunds=159

# 3. 预期答案的第二遍计算（对真实库手写 SQL，用只读角色连接）
export QUERYSHIELD_DEMO_DATABASE_URL='postgresql://queryshield_ro:<只读角色密码>@127.0.0.1:5433/queryshield_demo'
python scripts/verify_demo_expected.py
# 期望输出末行：checked=23 mismatches=0
```

### 2.2 本机（Windows，Docker 里的 PostgreSQL）

沿用 README 的容器名和端口。管理员密码、只读角色密码都是提示输入（隐藏），只放进本进程的环境变量，结束时恢复。

```powershell
# 1. 建库（容器内本地连接，同 README 的做法）
docker exec queryshield-postgres psql -U queryshield -d postgres -c "CREATE DATABASE queryshield_demo"

# 2. 建表、载入数据、授权（提示输入管理员密码）
.\scripts\bootstrap-demo-db.ps1                  # 端口不是 5433 时加 -PostgresPort <端口>
```

前提：只读角色 `queryshield_ro` 已存在（README 里为 `queryshield_test` 建过，角色是整个集群共用的）；脚本不创建它。

## 3. 跑演示题

演示脚本每次都起一个真实的 uvicorn 进程，经 HTTP 逐题运行。服务进程的 `QUERYSHIELD_DEMO_DATASET` 由脚本给，不依赖你的会话环境；库名必须以 `_demo` 结尾，否则 blocked（不起服务、不调模型）。

**Fake（免费）：** Fake 模型只认脚本里的题，其余记 `not_applicable`（现在只有 Q04b）。Fake 不会只取前 5 名：Q06 在 Fake 下会返回全部客户，金额都对，记已知缺口 `rowset_customers_differ`。Fake 能答的题，已核实的数值**必须**与预期答案一致，等于用产品自己的核实路径再算一遍答案。另有固定检查：库能连上、各表各租户行数与生成器一致、演示知识库能建出快照、每个口径题的期望来源能在 Fake 嵌入下检索到、租户/角色/已停用文档的隔离。

```bash
# Linux / macOS（接上一节的 shell；QUERYSHIELD_DATABASE_URL 指向演示库）
QUERYSHIELD_DATABASE_URL="$QUERYSHIELD_DEMO_DATABASE_URL" \
  python scripts/demo_run.py --mode fake --evidence-dir <证据目录>
```

```powershell
# 本机
.\scripts\demo-local.ps1 -FakeDryRun
```

**Real（本机，由你运行，会调用付费模型）：** 提示输入百炼 API Key 和只读角色密码，只放进进程环境。你会话里已有的 `QUERYSHIELD_DATABASE_URL`（指向测试库）会被忽略并在结束时恢复。

```powershell
.\scripts\demo-local.ps1 -BailianBaseUrl '<复制自百炼 API Key 设置页的北京工作区 OpenAI 兼容地址>'
```

退出码：0 全部通过，1 有硬失败，2 blocked。证据默认写在项目目录下的 `evidence\demo-<fake|real>-<时间戳>\`（`-EvidenceRoot <目录>` 可换到别处；脚本结束时打印 `Evidence root:` 和完整路径），两个文件：

- `demo-summary.json`：只有固定字段和数值（题号、身份、HTTP 码、终态、answer_status、动作轨迹、预期值与实际值、判定、已知缺口、版本信息），没有问题原文、回答、客户姓名、URL 和凭据；每道题还记下每次模型调用的 completion_tokens（来自响应里的 usage，没有就是 null，不会补成 0）、本次运行的输出上限（读 `QUERYSHIELD_MODEL_MAX_TOKENS`，缺省 512，只读不改）和用满输出上限的调用次数，只有数字，没有文字。每条记录还有 `run_id`（`run-` 加随机 UUID，不含租户和用户信息）和 `usage_total`（这道题最后一个响应里的聊天用量合计：固定的状态和三个数），用来和模型网关按 `X-Run-Id` 记的调用、用量逐 run 对上；没有建 run 的题（例如在入口被拒绝的隔离题）两个都是 `null`。
- `demo-raw.json`：问题、回答文字、客户姓名。只在本机看，不要分享，也不要提交。

### 判定规则

**所有题共用的一条：** 回答里的每条已核实事实，都按题目身份的租户、事实自己的指标和时间窗，用生成器的 `expected_metrics`（第一种算法，纯标准库，不调用产品构建 SQL 的代码）独立算一遍，值不相等就是**硬失败** `verified_fact_mismatch`；算不出来（指标、租户或窗口不认识）也算不相等。摘要每题记 `verified_facts_checked`、`verified_facts_mismatched`，顶层记 `verified_fact_check: {checked, mismatched}`，只有数字。早先的 Q07 只比名字，漏掉了一条口径标错的已核实事实，这条规则就是为它加的。

| 题 | 判定 |
|---|---|
| Q01 Q02 Q03 Q04 Q05 Q11 | 已核实的值必须等于预期，窗口必须等于预期窗口；不一致是**硬失败**。模型没给出已核实的值（追问、没查询就回答）只记已知缺口。Q11 另外记录有没有先被退回（验收 B8） |
| Q04b | **观察题**：`refund_fen` 没有服务端核实器，只记录终态、HTTP 码、answer_status、动作轨迹，以及给出的数值是否等于预期。**HTTP ≥ 500 也只记已知缺口**（`observed_server_error:<错误码>`，产品实际会两次声明 `refund_fen`、两次被 `metric_not_verifiable` 拒绝，修复次数用完后 502 `query_repair_limit`）。唯一的硬失败是标了 verified 却没有事实 |
| Q06 | **有界行集**：某月（6 月）已支付金额最高的 5 个客户，只要客户编号和金额。按客户编号比对金额（列名是模型起的别名，按值比对）：任何一个返回的客户，金额不等于该客户的真实金额是**硬失败**；多出的客户、缺客户、行不可解析、没有行只记已知缺口。模型选了姓名触发审批时，脚本用同租户审批人批准再比对。生成器保证第 5 名和第 6 名不并列 |
| Q06b | **观察题**：同一个月按客户的**全量**汇总（33 行）。只记录行数、匹配数、终态、完成 token 数，不算硬失败（标了 verified 除外）：行集由模型逐行抄写，受输出上限限制，这道题用来记录这个缺口 |
| Q07 | 问法是“已支付金额最高的客户姓名是什么”（不用“消费最多”，那个词在支付金额和退款后净额之间有歧义，模型追问口径是合理的）。姓名要审批：发起 → 审批人批准 → 取结果。1 行时名字必须等于预期；多行时预期名字必须出现；0 行硬失败；预期名字是否排第一行只作观察。摘要只写是否匹配和行数，不写姓名。**当前规则下**：第一名客户的金额是分组值，不是全租户口径，所以预期 `answer_status` 为 unverified、没有标量事实（审批回答是固定文字“审批通过，已执行只读查询：返回 N 行，见 result.rows。”）。如果仍有事实，按上面的共用规则核对，不一致是硬失败；全部一致但标了 verified（例如前一步的全租户总额作为 supporting fact）记已知缺口 `verified_supporting_fact` |
| Q08 | “销售额”：必须停在 `WAITING_USER`，追问是 catalog 原文；resume 后 `gross_fen` 已核实、等于预期、窗口是 8 月、回答写明口径 |
| Q09 | 口径题：模型文字、unverified、0 次 SQL，`source_ids` 里的文档必须来自演示知识库（`commerce-v1` 是 catalog 自己的来源，允许），不得出现默认知识库的文档 |
| Q10 | 问候：服务端固定回复，no_data，没有事实、没有 SQL |
| Q12 | 用 A 的身份问 B：B 的数值（`net_fen`、`gross_fen`）不得出现在事实、行或回答里；终态不限定，如实记录 |

模型行为上的偏差（被退回一次才答对、自己定月份、没有给出已核实的值）一律记为已知缺口，口径与 HTTP 冒烟一致。

### 演示里看到的已知缺口（2026-10-01 的本机 Real）

这三条都是产品或模型的行为，不是演示脚本的问题；产品层面的修复不在本说明范围，留作后续修复：

1. **“退款总额”会 502**：`refund_fen` 没有核实器，模型两次声明它都被 `metric_not_verifiable` 拒绝，修复次数用完，502 `query_repair_limit`（Q04b 只记录）。
2. **行集回答由模型逐行抄写，受输出上限限制**：按客户的全量汇总（33 行）查询成功后，模型的下一步输出不是合法 JSON，502 `invalid_json`。推断是模型把 33 行逐行写进回答，超过输出上限（`QUERYSHIELD_MODEL_MAX_TOKENS` 缺省 512），JSON 被截断；适配器不读 `finish_reason`，所以只报 `invalid_json`。现有证据确认不了，所以摘要现在记每次模型调用的 completion_tokens 和输出上限。演示里的 Q06 因此只要前 5 名，全量汇总是观察题 Q06b。
3. **“消费”会触发口径追问**：“消费最多的客户”在支付金额和退款后净额之间确实有歧义，模型查询一次后追问口径，停在 `WAITING_USER`，没走到审批。Q07 改成“已支付金额最高”。


## 4. 演示数据

由 `scripts/generate_demo_data.py` 生成：只用标准库，自带 SplitMix64 随机数（不用 `random`，因为 `shuffle`/`sample` 不保证跨 Python 版本稳定），金额用整数分，时间用整数 UTC 秒；固定种子写在文件头；重跑输出逐字节相同（`python scripts/generate_demo_data.py --check`）。

规模（订单 2026-06 至 2026-09，退款最晚 2026-10-31T23:59:59Z，2026 年 5 月没有任何订单和退款）：

| 项 | A（小网店） | B（批发商） |
|---|---:|---:|
| 客户 / 从未下单 | 40 / 3 | 15 / 1 |
| 订单 / 已支付 / 已取消 | 600 / 552 / 48 | 150 / 141 / 9 |
| 单笔金额 | 30–600 元 | 3,000–80,000 元 |
| 有退款的已支付订单（全额 / 部分 / 多次） | 80（20 / 40 / 20） | 30（8 / 14 / 8） |
| 跨月退款订单 | 42 | 18 |
| 取消订单上的退款 | 3 | 1 |

边界情况（完整数字和按月汇总表见 `fixtures/demo/commerce-demo-v1.md`）：

- **半开区间边界**：每个租户有 7 笔订单落在月初第一秒或月末最后一秒，用来检验 `[start, end)`。
- **跨月退款**：订单在 8 月、退款在 9 月，按单月口径哪个月都不计入；按 7–9 月的窗口才计入。10 月的退款在任何 6–9 月窗口里都不计入。
- **取消订单上的退款**：订单付款后取消并退款，这笔退款**不计入** `refund_fen`（catalog 口径明确排除）。这类数据在商业上少见，作为数据质量异常的边界样本有意放进来。
- **一单多次退款**：同一订单 2–3 笔退款，合计不超过订单金额。
- **客户姓名**：虚构的中文姓名，敏感字段，查询需要审批。
- 生成时断言：每月“消费最多的客户”唯一，并且至少领先第二名 5%；Q07 对应的月份里，按总额和按退款后净额排名的第一名是同一人。

## 5. 预期答案怎么来的

预期答案算两遍，两遍一致才写进清单，**互不共享代码**：

1. **第一遍（生成器内）**：`expected_metrics`，直接遍历内存里的订单和退款元组，按 catalog 口径算：已支付订单、`created_at ∈ [start, end)`；退款自己的时间和它所属的已支付订单的时间都在窗内、同一租户。结果写进清单。
2. **第二遍（对库手写 SQL）**：`scripts/verify_demo_expected.py`。不 import `queryshield`，也不 import 生成器；SQL 是手写的 CTE + `EXISTS`，不照搬产品的 `NET_FEN_*` JOIN 写法；读清单里的（租户、窗口、预期值），通过只读角色在每个租户的事务里绑定租户（强制行级权限生效）逐项比对。窗口以字面 UTC 字符串写在清单里，测试再断言窗口与问题里的月份一致。

测试里还有第三种读法：`tests/test_demo_generator.py` 用正则直接解析已提交的 SQL 文件，按另一种写法重算所有数值题。独立验收可以另用自己的算法。

## 6. 演示知识库

13 篇中文文档，`source_id` 都以 `demo-` 开头（与默认库无交集）：四个指标的口径（已支付订单数、支付订单总额、退款金额、退款后净额）、“销售额”歧义要先问清楚、时间口径、数据字典、客户姓名审批人的核对要点、现行和已停用的退款规则、租户 A 和租户 B 的业务概况、常见问题。

- 文档内容与 catalog-v4 一致，不引入 catalog 里没有的指标。`demo-metric-refund` 如实写明：退款金额目前是口径说明，服务端不单独核实，它的数值通过退款后净额间接核实。
- **文档里不写任何具体的业务数值**（测试扫描：除 `catalog-v4`、`knowledge-demo-v1`、`commerce-v1` 外没有任何数字，也没有“数字 + 元/笔/分…”的写法），免得模型把文档里的数字当成答案。文档只用来解释口径。
- `demo-tenant-a-overview` 的范围是 A，`demo-tenant-b-overview` 是 B：A 的身份检索不到 B 的文档，B 也检索不到 A 的。`demo-sensitive-customer-name` 只给审批人（与原知识库一致），内容是审批人核对 SQL 的要点；requester 检索不到。`demo-refund-policy-v1` 登记为 deleted，不进索引，任何身份都检索不到。
- 导入限制内：13 个文件、26 个分块（上限 100 个文件、200 个分块），每块不超过 800 字符。

### 6.1 两个 catalog 版本号的来由（验收 O8）

同一份证据里，知识快照标 `catalog-v2`，事实和运行配置标 `catalog-v4`：

- 现有知识快照建立时对应的是 catalog-v2，这个标签已经记在早期冻结评测的证据里，**不改**；改标签就改变了冻结评测的身份。
- catalog v3、v4 只加了名称、说法和识别词，四个指标的定义文本与 v1、v2 相同（测试 `test_catalog_v3_keeps_the_v1_entries_and_adds_only_names_and_phrases`、`test_catalog_v4_is_v3_without_the_generic_markers_and_with_one_business_phrase`）。所以 v2 的知识快照对 v4 的事实仍然有效。
- 演示知识库是新的，直接标 `catalog-v4`（取产品当前的 catalog 版本常量）。
- 注释写在 `src/queryshield/knowledge/runtime.py` 的 `KNOWLEDGE_CATALOG_VERSION` 旁。冻结证据里**没有**加字段；只有新的演示证据（`demo-summary.json` 的 `versions`）把知识快照的 catalog 版本和事实的 catalog 版本分列。

## 7. 演示设置与评测隔离

- 库名配对检查放在产品取数据库连接的地方（`db/readonly.py` 的 `get_database_url()`）：设置打开 → 库名必须以 `_demo` 结尾，URL 缺失也拒绝；设置没开、库名却以 `_demo` 结尾 → 拒绝；设置没开且库名不以 `_demo` 结尾 → 与以前完全相同。错误码固定：`demo_dataset_database_mismatch`、`demo_database_without_demo_dataset`、`invalid_demo_dataset`；错误信息和日志里没有 URL、主机名或凭据。只解析 URL 的路径（以及 `?dbname=` 和关键字形式），百分号编码会解码。
- `product_retriever`（HTTP 入口和审批服务的 `default_dependencies` 都经过它）在选检索器之前做同样的检查，不通过时 HTTP 返回 503 和固定错误码。
- 设置打开时评测和历史检查的表现：
  - `check_state.py`、`check_eval.py` 里不带依赖的 `start_async`：连的是 `_test` 库或没有 URL → **拒绝**（blocked，上面的错误码）；
  - 开发集的有状态评测把检索器换成 `CATALOG_SEARCH_ONLY`、`check_eval.py` 直接构建默认检索，二者都不读这个设置 → **不受影响**，知识快照和默认完全相同；
  - 所有读库都经过 `get_database_url()`，所以设置打开 + `_test` 库一律拒绝。

### 知识快照与审批权限

- **产品发布自己用的知识库。** HTTP 产品服务（`shared_run_service()` 建的那个）在每次建 run 之前确认一次：它用的知识库（默认，或演示设置打开时的演示知识库）的快照已经在状态库里；不在就发布。同一个服务实例对同一个快照只做一次。发布在第一次运行之前，不放在应用启动时，所以不进 lifespan 的 TestClient 也成立。catalog-only、关闭检索和 B0 也发布：权限是访问控制，不是检索。
- **快照 id 是嵌入前的内容 id。** `run_config.knowledge_snapshot_id`（顶层）就是这个 id，不再回落到写死的值；发布时不需要嵌入，也与 Python 版本无关。默认知识库是 `knowledge-v1-9f580dd7f887ed0a`（与以前写死的值相同），演示知识库是 `knowledge-demo-v1-142c2433a63cc6a8`。混合检索时，`run_config.agent_run_config.knowledge_snapshot_id` 是检索器把同一份内容嵌入之后的 id，它的 `base_snapshot_id` 等于顶层的 id（测试 `test_hybrid_retrieval_embeds_the_snapshot_named_at_the_top_level`、`test_the_demo_setting_binds_the_demo_permission_source`）。
- **审批绑定权限。** 客户姓名的权限来源由服务端按知识库给出：默认知识库 `semantic-sensitive-customer-name`，演示知识库 `demo-sensitive-customer-name`（`knowledge/runtime.py`，模型和请求都改不了）。产品发起审批时，待批动作带上 `permission_source_id` 和当时的 `permission_version`；批准时来源已撤销、角色或租户范围变了、版本对不上，一律 409 `authorization_revoked`，不执行查询，审批仍待批。状态库里找不到这个来源、或它不是 active 时，**不建审批**，run 以 503 `approval_permission_unavailable` 失败。
- **发布不会撤回运行期的改动。** 重复发布相同内容，`acl_version` 不变；只有 tenant_scope、allowed_roles、status 变了才加 1。产品不会再次发布状态库里已有的快照 id，所以重启或重复运行都不碰已有的 ACL 行（撤销、改角色都保留）。产品第一次发布一个新内容的快照时，状态库里已经不是 active 的来源整行保持不变（撤销仍是撤销）；active 的来源按新内容更新。运维人员显式重新发布（`KnowledgeSnapshotRepository.publish`）仍按快照内容恢复 ACL，与以前相同。
- **评测和检查不受影响。** 自己构造 `RunService(store=…)` 的路径（开发集的有状态评测、直接构造服务的状态检查、测试）不发布，审批照旧：状态库有 ACL 才带权限字段，没有也照建。STATE-R03、EN05、FS02 跑的就是产品 app（自带状态库路径），产品会把自己的快照发布进去；ACL 内容相同，版本不变。

### 已知限制

- **没封死的组合**：有人打开设置、又让评测的 `QUERYSHIELD_DATABASE_URL` 指向 `_demo` 库。这时评测读到演示数据；EVAL-DB01 和 STATE 套件的库检查对 commerce-v1 的固定数值（例如 15000、3000）会失败，**不会悄悄通过**。评测的本机包装脚本使用的一直是 `queryshield_test`；演示包装脚本会忽略并恢复你会话里的测试库 URL。要彻底封死只能改评测脚本，本单不改。
- **resume 的时间窗**：追问后 resume 时，服务端从问题、追问和回答的文字里取**第一个**“YYYY年M月”，取不到就回退到 2026 年 9 月（`approval/service.py`，本单不改）。所以追问的演示题（Q08）在问题里写了一个不是 9 月的单月，范围窗不能和追问同题。已转后续评估。
- **`refund_fen` 没有核实器**：产品只核实 `paid_count`、`gross_fen`、`net_fen`。“退款总额”问不出已核实的值（Q04b 只观察）。已转后续评估。
- **不检索知识时 `agent_run_config` 的快照 id 是占位值（转后续评估）**：catalog-only、关闭检索和 B0 下，agent 不使用知识快照，`run_config.agent_run_config.knowledge_snapshot_id` 是固定的占位值（默认知识库的内容 id）。默认知识库下它恰好与顶层相同；演示设置打开再配这几种模式时，它与顶层的演示 id 不同。演示运行本身走混合检索，不受影响。要让两者一致得改 `product_run_config` 及其调用，超出这次改动的范围。（早先记的“审批路径顶层 id 写死”已经修复。）
- **旧记录里的分组单行事实认不出来**：现在，带 GROUP BY 的查询结果在 binding 上记 `grouped: true`，只作行集，不再生成标量事实，`/result` 复核也拒绝这种事实。这条规则之前写进状态库的记录没有这个标记，结果证据只存 SQL 的哈希，读取时无法判断当时是不是分组查询，所以那时由“分组取第一名”得到的事实读取时照旧通过。本机状态库里 演示数据首次运行时 Q07 的 run 就是这种情况；演示脚本每次用新的临时状态库，新的运行不受影响。
- **系统消息的示例窗是 2026 年 9 月**：问题和请求都不给时间时，模型可能照着示例查 9 月（验收 A5）。所以除追问时间的演示外，每道题都在问题里写明月份，也不用 9 月（Q05 的范围窗只在末端含 9 月）。
- **快照 id 与 Python 版本有关**：Fake 嵌入向量里的浮点运算在不同 Python 版本上最后几位可能不同，所以“嵌入后”的快照 id 在不同解释器上不同（这是本单之前就有的；项目虚拟环境 Python 3.14.3 上默认知识库是 `knowledge-v1-97d2061e0b7c34b6`，演示知识库每次重算）。嵌入之前的快照 id（`knowledge-v1-9f580dd7f887ed0a`）和 index hash 与版本无关。
