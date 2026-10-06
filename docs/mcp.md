# MCP 只读元数据工具

服务端设置 `QUERYSHIELD_METADATA_TOOLS=mcp` 时，Agent 的两个元数据工具 `search_catalog`、`describe_tables` 改走真实的 MCP stdio 会话（官方 Python SDK `mcp` 2.x，协议版本 2025-11-25）。默认值 `local` 的行为和记录与以前完全相同。

## 结构

```
HTTP 请求 ──> QueryShield 服务（宿主）
               ├─ Agent 图 ──> 工具门面 McpMetadataTools
               │                ├─ search_catalog / describe_tables ──stdio──> MCP 服务进程（每个 run 一个）
               │                └─ query_readonly（本地，带审批和证据登记）
               └─ 状态库：每次调用的 transport，每个会话一条 metadata_session 记录
```

- 服务进程：`python -m queryshield.mcp_metadata.server`。由宿主在 run 第一次调用元数据工具时启动，run 的这次执行结束时关闭（成功、失败、拒绝、等待用户、等待审批、取消、异常都会关）。resume 是新的一次执行，需要时另开一个会话。
- 服务进程里跑的就是 `ControlledTools.search_catalog`、`describe_tables`。它不连数据库，也不重新给知识库做嵌入：混合检索时，读宿主写好的索引文件，并核对嵌入后的快照 id。

## 身份怎么绑定

- 启动参数（tenant、principal、role、run_id，检索设置、知识库、索引文件、预期快照 id）全部由宿主根据服务端创建的 `ExecutionContext` 生成。模型的输出影响不了进程命令、身份和环境变量。
- 工具参数里没有身份字段。带了 `tenant_id`、`role` 等字段，宿主和服务进程都会拒绝（`unknown_argument`）。
- 子进程的环境只包含 SDK 默认继承的那组变量，加上 Python 路径设置。混合检索的 Real 模式额外传入 `QUERYSHIELD_EMBEDDING_*`。数据库连接串、模型密钥、令牌、状态库路径一律不传。

## 宿主核对什么

MCP 返回的内容都不可信。宿主在下面几个环节核对：

1. **发送前**：用本地同一组函数校验参数，不通过就不发送请求。
2. **tools/list**：必须恰好两个工具，名字和输入、输出 Schema 都要与 `mcp_metadata/schemas.py` 一致。
3. **每次结果**：
   - `structuredContent` 要符合 outputSchema，并且与 JSON text 相同；
   - 检索条目必须是本角色可见的目录条目，或者宿主索引里来源有效、角色允许、租户匹配的分块，四个字段逐字相等；
   - 表描述必须与本地计算的结果完全相同。

任何一条不通过，整次调用失败，返回的条目一条也不会进入 run。交给图的结果由宿主用自己的记录重建。

### 为什么 `query_readonly` 不经 MCP

它是唯一读业务数据的工具，指标绑定、审批、结果证据都要在宿主里完成。

## 失败关闭

| 错误码 | HTTP | 情形 |
|---|---|---|
| `mcp_unavailable` | 503 | 起不来、拒绝启动、initialize 失败、调用中进程退出；服务进程里的上游服务（嵌入服务）出错（会话记录 `failure_source` 为 `upstream`） |
| `mcp_timeout` | 504 | 单次调用超过上限 |
| `mcp_protocol_error` | 502 | tools/list 不符、协议错误 |
| `mcp_result_invalid` | 502 | 宿主核对不通过；服务端返回了宿主发送前已经检查过的错误（参数类、身份类错误码）、`internal_error` 或未知错误码 |

这四种错误都不可修复，不会退回本地工具，也不会重试。

服务端的错误结果这样处理：

- 上游服务出错（目前只有嵌入服务的 `EmbeddingProviderError`）：服务端只返回固定的 `upstream_unavailable` 和固定文字，异常文字、上游返回的内容都不传。宿主以 `mcp_unavailable` 结束这次调用，会话记录写 `failure_source: "upstream"`，让人能分清“服务进程坏了”和“服务进程的上游坏了”。本地模式下同样的错误仍以它自己的错误码结束（例如 `upstream_timeout`）。
- 其它异常：服务端返回固定的 `internal_error`，宿主记 `mcp_result_invalid`。
- 参数类和身份类错误码（`invalid_arguments`、`missing_argument`、`unknown_argument`、`invalid_argument`、`table_not_allowed`、`unauthorized`、`forbidden`）：宿主发送前已经用同一组本地函数检查过参数和身份，诚实的服务端不会再返回它们。收到时一律记 `mcp_result_invalid`（run 以 FAILED 结束），不会变成 DENIED，所以服务端不能把一个 run 伪装成安全拒绝。发送前由宿主自己发现的这类错误，行为与本地模式相同。
- `retrieval_unavailable` 是服务端自己检索器的状态，宿主发送前无法知道，照原码返回（run 以 FAILED 结束），文字是宿主固定的那一句。

单次调用默认最多等 2 秒，可以用 `QUERYSHIELD_MCP_CALL_TIMEOUT_SECONDS` 调整，范围 1–10 秒。会话启动的上限是 20 秒。设置值不合法时，返回 503 `invalid_metadata_tools_configuration`。

## 打开设置与冒烟

```powershell
$env:QUERYSHIELD_METADATA_TOOLS = 'mcp'      # 只影响 HTTP 产品服务（shared_run_service）
.\scripts\mcp-local-smoke.ps1 -FakeDryRun                     # 协议部分 + 产品部分，Fake 模型
.\scripts\mcp-local-smoke.ps1 -BailianBaseUrl '<百炼地址>'      # Real，凭据提示输入
```

Linux / macOS 上也可以直接运行：`python scripts/mcp_smoke.py --mode fake --evidence-dir <目录>`（产品部分需要 `QUERYSHIELD_DATABASE_URL`）。

## 边界与限制

- 只有 `shared_run_service()` 读这个设置。直接构造的 `RunService`（开发集的有状态评测、各项检查、测试）一律走本地。
- 经 HTTP 应用运行的 STATE 套件和历史检查（`check_state.py` 的 TestClient、`check_api`、`check_faults`、`check_runtime`、`check_api_facts`）会跟随这个设置，运行这些检查时不要设它。
- 只支持 stdio 和本机进程：不做远程 MCP、OAuth、多用户共用进程或进程池。
- 状态库里的会话记录（server_pid、SDK 版本、协议版本、清理结果、`failure_code`、`failure_source`）对 run 的所有者可见，都是固定字段。
- 索引文件写在宿主进程自己的临时目录（`queryshield-mcp-index-*`），里面有 approver 专属和各租户的分块文字，权限 0700。宿主在应用关闭阶段删除它，并在解释器退出时再删一次兜底。
  - Linux（包括容器里的 `docker stop`）：uvicorn 收到 SIGTERM 后先跑完应用的关闭阶段再重新发出信号结束进程，`atexit` 不会执行，所以靠关闭阶段的清理；正常停服务后目录不存在。
  - Windows：`terminate()` 是 `TerminateProcess` 强制结束，关闭阶段和 `atexit` 都不会执行，目录会留在系统临时目录，需要手动删除。下次启动用的是新目录，不会碰旧目录；也不会在启动时按名字清扫，因为那可能误删同一台机器上另一个实例正在用的目录。
- 宿主不核对“上游故障”的真假：恶意的服务端可以把任何一次调用标成上游故障，宿主照样以 `mcp_unavailable` 结束（会话记录 `failure_source: "upstream"`）。后果只是失败原因的标注不准：run 仍以 FAILED 结束，服务端的文字一个字也带不进来，也变不成 DENIED。
- 宿主逐条核对服务端返回的条目，但不重跑检索：恶意的服务端能换成另一组本租户、本角色可见且完全合法的条目（文字逐字相同），换不进任何文字，也越不了权限。
- 清理确认用的是服务进程自己写到 stderr 的 pid：宿主确认“这个 pid 已经不存在”，不是独立找出这个进程。进程树由 SDK 在关闭时一起结束。
