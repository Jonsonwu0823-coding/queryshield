# 运维说明

本文说明怎么用 Docker Compose 把 QueryShield 跑起来、怎么接真实模型、怎么跑检查，以及已知的部署限制。命令里的凭据一律由脚本生成或在你自己的 shell 里设置，不要写进文件、命令历史、日志或截图。

引用的“验收编号”（例如“验收 H7”）是项目验收记录里的条目编号，每个编号的含义和对应的代码位置见 [能力证据清单的编号对照](evidence.md#编号对照)；代码位置都在本目录（`queryshield/`）内。

## 1. 用 Compose 跑起来

前提：Docker（Docker Desktop 或 Docker Engine + Compose v2），以及用来生成 `.env` 的 Python 3（只用标准库）。

```bash
cd queryshield
python scripts/new_env.py          # 一次：写出 .env（随机密码和令牌，不打印任何值）
docker compose up -d --build       # 建镜像、建演示库并载入数据、启动服务
curl http://127.0.0.1:8000/health  # {"status":"ok"}
```

默认用确定性的 Fake 模型，不需要任何密钥。三个服务：

| 服务 | 作用 |
|---|---|
| `db` | PostgreSQL 16，数据放命名卷 `qs_pgdata`，**不向本机发布端口** |
| `setup` | 一次性任务：建两个角色和演示库 `queryshield_demo`，载入演示数据，成功后退出 |
| `app` | 产品服务，等 `setup` 成功后才启动；只发布到 `127.0.0.1:${QUERYSHIELD_HTTP_PORT:-8000}`；状态库和调用记录放命名卷 `qs_state` |

没有本机 Python 时，可以用容器生成 `.env`（Windows PowerShell，在 `queryshield` 目录里）：

```powershell
docker run --rm -v "${PWD}:/w" -w /w python:3.14.3-slim-bookworm python scripts/new_env.py
```

### 常用操作

| 做什么 | 命令 |
|---|---|
| 看状态 | `docker compose ps -a` |
| 看日志 | `docker compose logs -f app`（`setup` 的结果：`docker compose logs setup`） |
| 停止（保留数据） | `docker compose stop`；再启动 `docker compose start`（`start` 也会重跑 `setup`：重载演示数据、按 `.env` 重设两个角色的密码，见下一节） |
| 停止并删除容器（保留数据卷） | `docker compose down` |
| **重置为空环境** | `docker compose down -v`（删除两个命名卷：数据库和状态库） |
| 换端口 | 在 `.env` 里加 `QUERYSHIELD_HTTP_PORT=8080`，再 `docker compose up -d` |
| 只重启或重建 app（不重跑 `setup`） | `docker compose up -d --no-deps app` |
| 打开 MCP 设置 | `QUERYSHIELD_METADATA_TOOLS=mcp docker compose up -d --no-deps app`（PowerShell：`$env:QUERYSHIELD_METADATA_TOOLS='mcp'` 之后同一命令） |

### 看核心能力：演示脚本

脚本连正在运行的服务，按顺序演示：已核实的数据回答、含糊说法追问并恢复、没给时间的追问、不需要数据的回答、带来源的知识回答、异步、敏感查询的审批（含被拒的尝试）。每条已核实事实都与演示数据生成器独立算出的值比对，不一致就是硬失败。

```bash
docker compose exec app python scripts/demo_walkthrough.py            # Fake 模型
docker compose exec app python scripts/demo_walkthrough.py --metadata-tools mcp   # MCP 设置打开之后
docker compose exec app python scripts/demo_run.py --mode fake --base-url http://127.0.0.1:8000 --evidence-dir /tmp/demo-run   # 整套演示题
```

要在容器里运行，是因为脚本要读服务的状态库（`QUERYSHIELD_STATE_STORE_PATH`）来判定动作轨迹；读不到时脚本以退出码 2 停下并说明原因，不会跳过判定。审批场景里你会看到：另一租户的审批人批准返回 **404 `not_found`**（对方租户看不到这条任务），本租户的请求人批准返回 **403 `forbidden`**，本租户的审批人批准返回 200。返回的行集标注为“原始查询结果，未核实；列名来自 SQL 里的别名”，不要把它和已核实的回答混在一起。

### 重复执行 `docker compose up` 的影响

`setup` 每次都会重载演示数据：在一个事务里清空三张表、重新载入、核对行数。重载后的数据与之前完全相同（固定种子），所以已核实的数值不变，审批和 run 记录在状态库里，也不受影响。但清空表要拿表锁：正在运行的 `app` 的查询会等锁，而每次查询的 `statement_timeout` 是 2 秒，等不到的查询会失败。实测一次完整的 `setup` 约 0.3 秒（本机一次性数据库上），窗口很短，但演示进行中不要再执行完整的 `docker compose up`，也不要 `docker compose start`（它同样会重跑 `setup`）；只想重启 app 用 `docker compose up -d --no-deps app`。

`setup` 每次还会按 `.env` 重设两个应用角色的密码（见 §4）。

## 2. 环境变量

`.env` 由 `scripts/new_env.py` 生成，已被 `.gitignore` 忽略。模型相关的变量在**当前 shell** 里设置，不写进文件；没设置的变量不会传给容器（不是空串）。

| 变量 | 作用 | 默认 | 敏感 |
|---|---|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL 超级用户密码，只给 `db` 和 `setup` | 无，必须设置 | 是 |
| `QUERYSHIELD_ADMIN_PASSWORD` | 角色 `queryshield`（库属主）的密码 | 无，必须设置 | 是 |
| `QUERYSHIELD_RO_PASSWORD` | 角色 `queryshield_ro`（服务连库用的只读角色）的密码 | 无，必须设置 | 是 |
| `QUERYSHIELD_TOKEN_A_REQUESTER`、`QUERYSHIELD_TOKEN_A_APPROVER`、`QUERYSHIELD_TOKEN_B_REQUESTER`、`QUERYSHIELD_TOKEN_B_APPROVER` | 四个身份令牌（租户 A、B；请求人、审批人），**四个值必须互不相同** | 无，必须设置 | 是 |
| `QUERYSHIELD_HTTP_PORT` | 发布到本机 `127.0.0.1` 的端口 | `8000` | 否 |
| `QUERYSHIELD_PROVIDER_MODE` | `fake` 或 `real`；两者从不混用 | `fake` | 否 |
| `QUERYSHIELD_MODEL_PROTOCOL` | 模型怎样给出动作：`json`（回复正文里的 JSON）或 `native`（原生 function calling）；其它值以 503 结束，不退回 `json` | `json` | 否 |
| `QUERYSHIELD_MODEL_BASE_URL`、`QUERYSHIELD_MODEL_NAME` | 真实模型（OpenAI 兼容）的地址和名字 | 未设置 | 地址不算敏感，见下 |
| `QUERYSHIELD_MODEL_API_KEY` | 真实模型的密钥 | 未设置 | **是** |
| `QUERYSHIELD_MODEL_MAX_TOKENS` | 模型单次输出上限 | 代码内 512 | 否 |
| `QUERYSHIELD_MODEL_TIMEOUT_SECONDS` | 模型调用超时（秒） | 代码内 15 | 否 |
| `QUERYSHIELD_EMBEDDING_BASE_URL`、`QUERYSHIELD_EMBEDDING_API_KEY`、`QUERYSHIELD_EMBEDDING_MODEL_NAME`、`QUERYSHIELD_EMBEDDING_MODEL_REVISION`、`QUERYSHIELD_EMBEDDING_DIMENSIONS` | 真实嵌入服务 | 未设置 | `QUERYSHIELD_EMBEDDING_API_KEY` 是 |
| `QUERYSHIELD_METADATA_TOOLS` | `mcp` 时，两个只读元数据工具走真实 MCP stdio 会话；不设或 `local` 为本地 | 未设置（本地） | 否 |

Compose 里由文件固定、不要覆盖的：`QUERYSHIELD_DATABASE_URL`（只读角色，库名写明为 `queryshield_demo`）、`QUERYSHIELD_DEMO_DATASET=commerce-demo-v1`、`QUERYSHIELD_STATE_STORE_PATH`、`QUERYSHIELD_CALL_STORE_PATH`。

密码只用 URL 安全的字符（字母、数字、`-`、`_`），因为它们会被拼进连接串；`new_env.py` 生成的值满足这一点。

## 3. 接真实模型

任何 OpenAI 兼容的接口都可以。实测用的是百炼的 `qwen-plus` 和 `text-embedding-v4`（1024 维），地址模板：

```
https://ws-<你的工作空间编号>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

密钥在**当前 shell** 里设置，不写进文件。Bash：

```bash
export QUERYSHIELD_PROVIDER_MODE=real
export QUERYSHIELD_MODEL_BASE_URL='https://ws-…/compatible-mode/v1' QUERYSHIELD_MODEL_NAME=qwen-plus
export QUERYSHIELD_EMBEDDING_BASE_URL="$QUERYSHIELD_MODEL_BASE_URL" QUERYSHIELD_EMBEDDING_MODEL_NAME=text-embedding-v4 \
       QUERYSHIELD_EMBEDDING_MODEL_REVISION=text-embedding-v4 QUERYSHIELD_EMBEDDING_DIMENSIONS=1024
read -rs QUERYSHIELD_MODEL_API_KEY; export QUERYSHIELD_MODEL_API_KEY QUERYSHIELD_EMBEDDING_API_KEY="$QUERYSHIELD_MODEL_API_KEY"
docker compose up -d --build
```

要让模型用原生 function calling，再设置 `QUERYSHIELD_MODEL_PROTOCOL=native`（Windows 脚本用 `-ModelProtocol native`）。

缺配置时服务以 503 结束（检查脚本记为 blocked），不会退回 Fake，也不会把未知的用量补成 0。Windows 上可以用 `scripts/compose-real-demo.ps1 -BailianBaseUrl '<地址>'`：隐藏输入密钥、只放进本进程、起 Compose、跑整套演示题和演示脚本、把两份摘要拷到证据目录、最后 `docker compose down`。

真实模型的行为问题（例如自己定月份、被退回一次才答对）按已知缺口记录，不当作服务端失败；判定口径与 HTTP 冒烟一致。

## 4. 检查与 CI

本机和 CI 用同一个入口 `scripts/check-all.ps1`（PowerShell 5.1 或 7；Linux 上用 `pwsh`）：

```powershell
# Windows：缺数据库连接串时隐藏输入两个密码，连 127.0.0.1:5433 的 queryshield_test / queryshield_demo
.\scripts\check-all.ps1 -EvidenceDir 'C:\path\to\evidence\check-all-<日期>'
```

```bash
# Linux：先设置 QUERYSHIELD_DATABASE_URL（只读角色）和 QUERYSHIELD_BOOTSTRAP_DATABASE_URL（属主），都指向 queryshield_test
pwsh -NoProfile -File scripts/check-all.ps1 -EvidenceDir "$PWD/evidence/check-all"
```

它依次运行：全量测试；经 `check.ps1` 的 DB-SMOKE、BASE、PROPOSAL（fake）、AGENT、STATE、EVAL（fake）；Fake 冒烟（HTTP、MCP、演示题）。结果汇总在 `check-all-summary.json`，全部通过才退出 0。需要演示库的测试必须真的运行，被跳过就失败。

没有在这里运行的检查，都写在 `check-all.ps1` 的一个清单里，每项带原因，并且“清单 + 实际运行的检查”必须正好等于 `check.ps1` 登记的全部检查，多一项少一项都失败：

| 检查 | 原因 |
|---|---|
| `EVAL-R01`、`EVAL-R06` | 需要封存的保留集，只在本机、不在仓库里；本机用 `scripts/eval-local-real.ps1` 的完整模式 |
| `STATE-X01`、`EVAL-X01` | 读上游资产登记表和两个验收标签；登记表或标签不在时记 `not_applicable` 并写明原因（例如导出的公开仓库）。CI 取完整历史和标签，所以在 CI 里跑 |

CI（`.github/workflows/queryshield-ci.yml`）有两个任务：`checks`（起 PostgreSQL，`setup_databases.py --test --demo`，再 `check-all.ps1`）和 `compose`（从空环境起 Compose、跑两遍演示脚本——先默认设置、再 MCP 设置——并做停止与重启的检查）。不使用仓库密钥：数据库密码和令牌在运行时随机生成并遮蔽；上传的产物只有摘要和检查的文字输出，不含 `*-raw.json`。

### 改了 `.env` 里的密码之后

- `QUERYSHIELD_ADMIN_PASSWORD`、`QUERYSHIELD_RO_PASSWORD`：直接 `docker compose up -d`。`setup` 每次都按环境变量重设这两个角色的密码，让 `.env` 成为唯一来源；`app` 因环境变化被重建。代价：别的程序如果拿着旧密码连这个库，下次 `up` 之后就连不上。
- `POSTGRES_PASSWORD`：**不会**改已有数据卷里的超级用户密码（Postgres 镜像只在数据卷第一次初始化时读它）。改了而不删卷，`setup` 会因认证失败退出，`app` 起不来。办法：`docker compose down -v` 重来；或先用旧密码进 `db` 容器执行 `ALTER ROLE postgres PASSWORD '…'` 再改 `.env`。
- 令牌：改 `.env` 后 `docker compose up -d` 重建 app 即可；记得四个值互不相同。

## 5. 常见问题

| 现象 | 原因与办法 |
|---|---|
| `error during connect` / `Cannot connect to the Docker daemon` | Docker Desktop 没开。启动它，等托盘图标显示运行中，再开一个新的终端 |
| `port is already allocated` / 8000 被占 | 换端口：`.env` 里设 `QUERYSHIELD_HTTP_PORT`。数据库不发布端口，所以与本机已有的 PostgreSQL 容器（例如 5433）不冲突 |
| `required variable … is not set` | `.env` 缺项，错误信息里有变量名。运行 `python scripts/new_env.py`（已有 `.env` 要加 `--force`，会换掉所有值） |
| `setup` 以非 0 退出，`app` 不启动 | `docker compose logs setup`。输出只有库名、行数和异常类型，没有连接串。常见原因：改了 `POSTGRES_PASSWORD` 却没删卷（见上） |
| 请求返回 503 `database_unavailable` | app 连不上库：`docker compose ps` 看 `db` 是否健康；本机直接起服务时，检查 `QUERYSHIELD_DATABASE_URL` 的主机、端口和角色密码 |
| 503 `demo_dataset_database_mismatch` / `demo_database_without_demo_dataset` | 演示设置与库名不配对：库名以 `_demo` 结尾必须同时设置 `QUERYSHIELD_DEMO_DATASET=commerce-demo-v1`，反之亦然 |
| 演示脚本退出码 2 | 读不到状态库、令牌没设或服务没响应。在 `app` 容器里运行（`docker compose exec app …`） |
| 自建的 PostgreSQL 上建演示库，只报 `setup_databases_failed: UnicodeEncodeError` | 服务器的编码不是 UTF8（例如用 POSIX 区域设置 `initdb` 得到的 `SQL_ASCII`）。`setup_databases.py` 建库时跟随服务器 `template1` 的编码，而演示数据是中文。请用 UTF8 初始化服务器（例如 `initdb -E UTF8`）。Compose 和 CI 用的官方 `postgres` 镜像默认就是 UTF8 |
| `docker compose config` 输出里有密码 | 它会展开 `.env`。不要把它的输出贴到任何地方 |

## 6. 部署限制

每条写明原因、建议和依据。

1. **只能单进程部署。** 审批的“检查再执行”只靠进程内的锁（`approval/service.py` 的 `_approval_lock`）串行化。两个进程共用同一个状态库时，同时批准同一条审批，可能让业务查询执行两次。另外，resume 期间被取消时，正在执行的 SQL 可能跑完才被丢弃。建议：Compose 里 `app` 只起一个容器，镜像里 uvicorn 不加 `--workers`。依据：验收 K6，验收 H7。上多进程之前，`decide_approval` 需要先返回“这次调用是否赢得了状态转换”。
2. **要长期收窄某个角色，改登记表或用撤销。** 部署新内容的知识库时，已启用来源的角色和租户范围按登记表更新：运行期把角色收窄（例如去掉 approver）之后，部署新知识库会把它放宽回登记表；撤销（状态不是 active）则保留。依据：验收 H2。
3. **连接串必须写明库名。** 演示设置没开时，不写库名（libpq 缺省取用户名）、用 `service=` 或 `PGSERVICE` 的连接，不在库名配对检查的范围内，指向 `_demo` 库也不会被拒。Compose 里连接串写明了 `queryshield_demo`。依据：`db/readonly.py` 的 `database_names_from_url`、`check_demo_pairing`；验收 G6。
4. **停服务前，先等异步 run 结束。** MCP 索引目录在 FastAPI 关闭阶段清理；关闭之后如果还有后台 run 开新的 MCP 会话，目录会被重建，SIGTERM 下 `atexit` 不执行，目录会留下（里面有各租户的分块文字，权限 0700）。**不在容器里、又在 Windows 上被强制结束时**，索引目录会留在系统临时目录。Compose 里 `/tmp` 是 tmpfs，容器停止时整个消失。依据：`api/main.py` 的 lifespan 关闭阶段、`mcp_metadata/launch.py` 的 `cleanup_index_dir`；验收 N6，实现 D-2。
5. **跑检查时不要设置 `QUERYSHIELD_METADATA_TOOLS`。** 设了以后 STATE-EN04 稳定失败（SSE 初始事件多一条 `metadata_session`），评测和检查本来也只走本地元数据工具。`check.ps1` 和 `check-all.ps1` 会自动清掉它并在结束时恢复你原来的值；`http_smoke.py`、`demo_run.py` 也各自清掉。依据：MCP 验收 M11。
6. **模型输出上限默认 512。** 按客户的全量列表（几十行）会被截断，所以演示用“前 N 名”的题。上限可以用 `QUERYSHIELD_MODEL_MAX_TOKENS` 调高，但评测数据都是在 512 下得到的。依据：`providers/openai_compatible.py` 里的 `QUERYSHIELD_MODEL_MAX_TOKENS` 默认值；演示题 Q06b。
7. **不用 Compose 时，状态库和调用记录默认在内存里。** `QUERYSHIELD_STATE_STORE_PATH`、`QUERYSHIELD_CALL_STORE_PATH` 不设就是 `:memory:`，服务一重启，run 和审批都没了。Compose 把它们放在命名卷里。依据：`approval/service.py` 的 `state_path_from_env`、`api/main.py` 的 `get_call_store`。
