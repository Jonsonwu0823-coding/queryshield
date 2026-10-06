import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import logging
from typing import Any, Callable, Iterator, Mapping
from fastapi import Depends, FastAPI, Request, Header
from uuid import uuid4
from fastapi.responses import JSONResponse, Response, StreamingResponse
from queryshield.models.queries import QueryProposalRequest, QueryRequest
from fastapi.exceptions import RequestValidationError
from queryshield.auth.identity import resolve_identity
from queryshield.agent.tenant_scope import has_explicit_foreign_tenant
from queryshield.providers.contracts import (
    ModelAdapter,
    ModelProviderError,
)
from queryshield.agent.runtime import (
    B1_PROFILE,
    FAILED_ERROR_HTTP,
    RuntimeConfigurationError,
    RuntimeDependencies,
    configured_profile,
    http_status_for_run,
    model_for_mode,
    product_retriever,
    provider_mode,
)
from queryshield.agent import (
    DurableModelCallStore,
    ExecutionContext,
    ProposalParseError,
    ToolCallAction,
    parse_query_proposal,
)
from queryshield.db.guarded import GuardedQueryError, GuardedQueryExecutor
from queryshield.policy.params import ordered_param_values
from queryshield.policy.sql import SQLPolicyError, parse_readonly_select
from queryshield.tools.semantic import ToolError, check_sensitive_access
import psycopg
from psycopg.errors import GroupingError, QueryCanceled, UndefinedColumn
from queryshield.approval.models import ApprovalRequest, EmptyRequest, PreferenceUpdate, ResumeRequest
from queryshield.approval.service import (
    TERMINAL_STATUSES,
    ApprovalConflict,
    ObjectNotFound,
    RunAuthorizationError,
    RunService,
    call_store_path_from_env,
    normalized_answer,
    shared_run_service,
)
from queryshield.memory.preferences import PreferenceError, PreferenceStore
from queryshield.facts import FactResolutionError
from queryshield.facts.persisted import validate_persisted_run_result


@asynccontextmanager
async def _application_lifespan(application: FastAPI):
    application.state.parallel_recovery = shared_run_service().recover_parallel_groups()
    yield
    # The MCP index directory holds chunk text for every identity: remove it on a normal stop.
    from queryshield.mcp_metadata.launch import cleanup_index_dir

    cleanup_index_dir()


app = FastAPI(title="QueryShield", version="0.1.0", lifespan=_application_lifespan)
# Uvicorn configures this channel at INFO, so normal structured events remain visible.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

def request_id_for(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id is None:
        request_id = str(uuid4())
        request.state.request_id = request_id
    return request_id


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any):
    request_id = str(uuid4())
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    run_id = getattr(request.state, "run_id", "-")
    error_code = getattr(request.state, "error_code", "-")
    log = logger.warning if error_code != "-" else logger.info
    log(
        "request_completed request_id=%s run_id=%s method=%s path=%s status_code=%s code=%s",
        request_id,
        run_id,
        request.method,
        request.url.path,
        response.status_code,
        error_code,
    )
    return response


def get_model_provider() -> ModelAdapter:
    """The server-configured model; Fake and real providers are never mixed."""

    return model_for_mode(provider_mode())


def get_call_store() -> Iterator[DurableModelCallStore]:
    with DurableModelCallStore(call_store_path_from_env()) as store:
        yield store


def get_run_service() -> RunService:
    return shared_run_service()


def get_guarded_executor(service: RunService = Depends(get_run_service)) -> object:
    """The product database boundary, owned by the run service.

    The fixture DB only serves the Fake provider (checked by the service).
    """

    return service.new_executor()


def get_agent_profile() -> str:
    """The server-configured profile (default B1); clients cannot choose it."""

    return configured_profile()


def get_retriever_source() -> Callable[[], object | None]:
    """Resolve the product retriever lazily, after the request is authenticated.

    Building the Real index calls the embedding service, so an unauthenticated
    request must never trigger it.  Tests and the evaluation override this.
    """

    mode = provider_mode()
    return lambda: product_retriever(mode)


def _usage_entries(service: RunService, run: Mapping[str, object]) -> list[dict[str, object]]:
    """One usage entry per actual model call, from the run's persisted Agent events."""

    entries: list[dict[str, object]] = []
    for event in service.store.events(str(run["run_id"]), after_event_id=0, limit=1000):
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        if event.get("type") != "agent_step" or payload.get("kind") != "model_call":
            continue
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        entries.append({
            "model_call_id": payload.get("model_call_id"),
            "provider_call_id": payload.get("provider_call_id"),
            "provider_request_id": payload.get("provider_request_id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "usage_status": payload.get("usage_status", "unknown"),
        })
    if not entries:
        checkpoint = run.get("checkpoint") if isinstance(run.get("checkpoint"), Mapping) else {}
        summary = run.get("usage") if isinstance(run.get("usage"), Mapping) else {}
        for model_call_id in checkpoint.get("model_call_ids", ()) or ():
            entries.append({
                "model_call_id": model_call_id,
                "provider_call_id": None,
                "provider_request_id": None,
                "prompt_tokens": summary.get("prompt_tokens"),
                "completion_tokens": summary.get("completion_tokens"),
                "total_tokens": summary.get("total_tokens"),
                "usage_status": summary.get("usage_status", "unknown"),
            })
    return entries


def _model_error_status(code: str) -> int:
    return FAILED_ERROR_HTTP.get(code, 502)


def _query_error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    run_id: str | None = None,
    usage: list[dict[str, object]] | None = None,
) -> JSONResponse:
    request_id = request_id_for(request)
    request.state.error_code = code
    content: dict[str, object] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if run_id is not None:
        content["run_id"] = run_id
    if usage is not None:
        content["usage"] = usage
    return JSONResponse(status_code=status_code, content=content)


def _unauthorized(request: Request) -> JSONResponse:
    return _query_error(request, status_code=401, code="unauthorized", message="身份认证失败")


def _proposal_params(action: ToolCallAction) -> tuple[object, ...]:
    """The ordered parameter values; the proposal parser already guarantees an object with string keys."""

    values = ordered_param_values(action.arguments["params"])  # type: ignore[arg-type]
    if values is None:
        raise ProposalParseError(
            "invalid_field",
            "query_readonly params keys must be consecutive indexes starting at zero without gaps",
        )
    return values


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _query_error(request, status_code=422, code="invalid_request", message="请求参数不符合要求")


@app.exception_handler(psycopg.OperationalError)
def database_error_handler(request: Request, exc: psycopg.OperationalError) -> JSONResponse:
    return _query_error(request, status_code=503, code="database_unavailable", message="数据库暂时不可用")


@app.exception_handler(QueryCanceled)
def query_timeout_handler(request: Request, exc: QueryCanceled) -> JSONResponse:
    return _query_error(request, status_code=504, code="query_timeout", message="查询超时")


@app.exception_handler(ModelProviderError)
def model_provider_error_handler(request: Request, exc: ModelProviderError) -> JSONResponse:
    return _query_error(request, status_code=_model_error_status(exc.code), code=exc.code, message="模型服务暂时不可用")


@app.exception_handler(RuntimeConfigurationError)
def runtime_configuration_error_handler(request: Request, exc: RuntimeConfigurationError) -> JSONResponse:
    # A refused configuration is blocked, never silently replaced.
    return _query_error(request, status_code=503, code=exc.code, message="服务配置不允许运行")


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


def _public_run(run: Mapping[str, object], *, include_result: bool = False, include_approval: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run["run_id"],
        "status": run["status"],
        "tenant_id": run["tenant_id"],
        "principal_id": run["principal_id"],
        "role": run["role"],
        "question": run["question"],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "mode": run["mode"],
        "model_call_count": run["model_call_count"],
        "tool_call_count": run["tool_call_count"],
        "sql_exec_count": run["sql_exec_count"],
    }
    if run.get("approval_id") is not None:
        payload["approval_id"] = run["approval_id"]
    if run.get("error_code") is not None:
        payload["error_code"] = run["error_code"]
    if include_result:
        payload.update({"answer": run.get("answer"), "result": run.get("result"), "facts": run.get("facts")})
        payload.update(_answer_fields(run))
    if include_approval and run.get("approval_id"):
        payload["approval"] = run.get("approval")
    return payload


def _answer_fields(run: Mapping[str, object]) -> dict[str, object]:
    """Machine-readable answer status and server source ids, read back from the stored run.

    verified: every number rendered by the server from this run's verified
    results; unverified: model text, or no verified fact; no_data: the
    server's fixed reply.  null without a SUCCEEDED answer.  A run stored
    without a status reads as unverified, never verified.
    """

    if run.get("status") != "SUCCEEDED" or run.get("answer") is None:
        return {"answer_status": None, "source_ids": None}
    envelope = run.get("checkpoint") if isinstance(run.get("checkpoint"), Mapping) else {}
    status, source_ids = normalized_answer(envelope.get("answer_status"), envelope.get("answer_source_ids"))
    return {"answer_status": status, "source_ids": source_ids}


@app.get("/runs/{run_id}")
def get_run_status(
    run_id: str,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
) -> JSONResponse:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        run = service.visible_run(run_id=run_id, identity=identity)
        include_approval = identity["role"] == "approver" and run.get("status") == "WAITING_APPROVAL"
        if include_approval and run.get("approval_id"):
            run = dict(run)
            run["approval"] = service.store.get_approval(str(run["approval_id"]))
        return JSONResponse(status_code=200, content=_public_run(run, include_approval=include_approval))
    except ObjectNotFound:
        return _query_error(http_request, status_code=404, code="not_found", message="任务不存在")


@app.get("/runs/{run_id}/result")
def get_run_result(
    run_id: str,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
) -> JSONResponse:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        run = service.visible_run(run_id=run_id, identity=identity, result=True)
        # A SUCCEEDED answer that needed no query (knowledge, no_data) has no
        # result and no facts to check; every stored result is still validated.
        if run.get("result") is not None or run.get("facts") is not None:
            validate_persisted_run_result(run)
        return JSONResponse(status_code=200, content=_public_run(run, include_result=True))
    except ObjectNotFound:
        return _query_error(http_request, status_code=404, code="not_found", message="结果不存在")
    except ApprovalConflict as exc:
        return _query_error(http_request, status_code=409, code=exc.code, message="结果尚未提交", run_id=run_id)
    except FactResolutionError:
        return _query_error(http_request, status_code=502, code="evidence_validation_failed", message="结果证据校验失败", run_id=run_id)


@app.post("/runs/{run_id}/resume")
def resume_run(
    run_id: str,
    body: ResumeRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
    model: ModelAdapter = Depends(get_model_provider),
    call_store: DurableModelCallStore = Depends(get_call_store),
    executor: object = Depends(get_guarded_executor),
    retriever_source: Callable[[], object | None] = Depends(get_retriever_source),
) -> JSONResponse:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        run = service.visible_run(run_id=run_id, identity=identity)
        # Asked first so a run that is not waiting gets this answer; the service checks the state again under its lock.
        if run.get("status") != "WAITING_USER":
            return _query_error(http_request, status_code=409, code="invalid_run_state", message="任务不在等待用户状态", run_id=run_id)
        run = service.resume_waiting_user(
            run_id=run_id,
            answer=body.answer,
            identity=identity,
            model=model,
            call_store=call_store,
            executor=executor,
            retriever=retriever_source(),
        )
        http_request.state.run_id = run_id
        return _resume_run_response(http_request, run)
    except ObjectNotFound:
        return _query_error(http_request, status_code=404, code="not_found", message="任务不存在")
    except ApprovalConflict as exc:
        status = 501 if exc.code == "not_supported" else 409
        message = "该运行配置不支持交互续跑" if status == 501 else "任务续跑状态或checkpoint无效"
        return _query_error(http_request, status_code=status, code=exc.code, message=message, run_id=run_id)


@app.post("/runs/{run_id}/approval")
def approve_run(
    run_id: str,
    body: ApprovalRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
) -> JSONResponse:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        run = service.approve(
            run_id=run_id,
            approval_id=body.approval_id,
            identity=identity,
            decision=body.decision,
        )
        if run.get("status") == "FAILED":
            code = str(run.get("error_code") or "execution_failed")
            return _query_error(http_request, status_code=http_status_for_run("FAILED", code) or 502, code=code, message="审批后的查询未完成", run_id=run_id)
        return JSONResponse(status_code=200, content=_public_run(run, include_result=True))
    except ObjectNotFound:
        return _query_error(http_request, status_code=404, code="not_found", message="任务不存在")
    except RunAuthorizationError as exc:
        status = 403 if exc.code == "forbidden" else 404
        return _query_error(http_request, status_code=status, code=exc.code, message="没有该审批权限", run_id=run_id)
    except ApprovalConflict as exc:
        return _query_error(http_request, status_code=409, code=exc.code, message="审批状态已失效", run_id=run_id)


@app.post("/runs/{run_id}/cancel")
def cancel_run(
    run_id: str,
    body: EmptyRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
) -> JSONResponse:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        run = service.cancel(run_id=run_id, identity=identity)
        status = 202 if run.get("status") == "CANCEL_REQUESTED" else 200
        return JSONResponse(status_code=status, content=_public_run(run))
    except ObjectNotFound:
        return _query_error(http_request, status_code=404, code="not_found", message="任务不存在")
    except ApprovalConflict as exc:
        return _query_error(http_request, status_code=409, code=exc.code, message="任务当前状态不能取消", run_id=run_id)


def _event_cursor(
    http_request: Request, service: RunService, run_id: str, last_event_id: str | None
) -> int | JSONResponse:
    """The id to resume after, or the error response for a malformed or expired Last-Event-ID."""

    text = (last_event_id or "").strip()
    if not text:
        cursor = 0
    elif not (text.isascii() and text.isdigit() and len(text) <= 18):
        return _query_error(http_request, status_code=400, code="invalid_event_cursor", message="Last-Event-ID格式错误", run_id=run_id)
    else:
        cursor = int(text)
    bounds = service.store.event_bounds(run_id)
    if bounds is not None and cursor < bounds[0] - 1:
        return _query_error(http_request, status_code=410, code="event_history_expired", message="事件游标已过期", run_id=run_id)
    return cursor


def _sse_frame(event: Mapping[str, object]) -> str:
    data = {
        "event_id": event["event_id"],
        "run_id": event["run_id"],
        "type": event["type"],
        "status": event["status"],
        "occurred_at": event["occurred_at"],
    }
    if event.get("result_id") is not None:
        data["result_id"] = event["result_id"]
    return (
        f"id: {event['event_id']}\nevent: {event['type']}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


async def _event_stream(http_request: Request, service: RunService, run_id: str, cursor: int):
    """Persisted events after ``cursor``, then heartbeats until the run is terminal.

    There is no detached producer or per-client task to leak; ASGI cancellation
    closes this generator without touching the run.
    """

    next_heartbeat = asyncio.get_running_loop().time() + 1.0
    last_sent = cursor
    yield ": heartbeat\n\n"
    while not await http_request.is_disconnected():
        events = service.store.events(run_id, after_event_id=last_sent, limit=32)
        if events:
            bounds = service.store.event_bounds(run_id)
            has_backlog = bounds is not None and bounds[1] > events[-1]["event_id"]
            for event in events:
                yield _sse_frame(event)
                last_sent = int(event["event_id"])
            # A slow client never accumulates an unbounded pending list.
            # It receives at most 32 persisted events and reconnects from
            # its last delivered ID to drain any remaining backlog.
            if has_backlog:
                break
            continue

        current = service.store.get_run(run_id)
        if current is None or current["status"] in TERMINAL_STATUSES:
            break
        now = asyncio.get_running_loop().time()
        if now >= next_heartbeat:
            yield ": heartbeat\n\n"
            next_heartbeat = now + 1.0
        await asyncio.sleep(0.1)


@app.get("/runs/{run_id}/events")
def run_events(
    run_id: str,
    http_request: Request,
    authorization: str | None = Header(default=None),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    service: RunService = Depends(get_run_service),
) -> Response:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        service.visible_run(run_id=run_id, identity=identity)
    except ObjectNotFound:
        return _query_error(http_request, status_code=404, code="not_found", message="任务不存在")
    cursor = _event_cursor(http_request, service, run_id, last_event_id)
    if isinstance(cursor, JSONResponse):
        return cursor
    return StreamingResponse(
        _event_stream(http_request, service, run_id, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/preferences/{key}")
def get_preference(
    key: str,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
) -> JSONResponse:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        record = PreferenceStore(service.store).get(tenant_id=identity["tenant_id"], principal_id=identity["principal_id"], key=key)
    except PreferenceError as exc:
        return _query_error(http_request, status_code=422, code=exc.code, message="偏好键不受支持")
    if record is None:
        return _query_error(http_request, status_code=404, code="preference_not_found", message="偏好不存在")
    return JSONResponse(status_code=200, content=record)


@app.put("/preferences/{key}")
def put_preference(
    key: str,
    body: PreferenceUpdate,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
) -> JSONResponse:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        record = PreferenceStore(service.store).put(
            tenant_id=identity["tenant_id"],
            principal_id=identity["principal_id"],
            key=key,
            value=body.value,
            confirmed=body.confirmed,
        )
    except PreferenceError as exc:
        return _query_error(http_request, status_code=422, code=exc.code, message="偏好请求未通过校验")
    return JSONResponse(status_code=200, content=record)


@app.delete("/preferences/{key}")
def delete_preference(
    key: str,
    http_request: Request,
    authorization: str | None = Header(default=None),
    service: RunService = Depends(get_run_service),
) -> Response:
    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)
    try:
        PreferenceStore(service.store).delete(tenant_id=identity["tenant_id"], principal_id=identity["principal_id"], key=key)
    except PreferenceError as exc:
        return _query_error(http_request, status_code=422, code=exc.code, message="偏好键不受支持")
    return Response(status_code=204)

@app.post("/queries")
def create_query(
    query: QueryRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    prefer: str | None = Header(default=None),
    call_store: DurableModelCallStore = Depends(get_call_store),
    executor: object = Depends(get_guarded_executor),
    run_service: RunService = Depends(get_run_service),
    model: ModelAdapter = Depends(get_model_provider),
    retriever_source: Callable[[], object | None] = Depends(get_retriever_source),
    profile: str = Depends(get_agent_profile),
) -> JSONResponse:
    """Run the server-configured Agent profile; sync and async share one runtime."""

    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)

    if has_explicit_foreign_tenant(query.question, identity["tenant_id"]):
        return _query_error(
            http_request,
            status_code=403,
            code="forbidden",
            message="请求涉及当前身份不可访问的租户",
        )
    if prefer is not None and prefer.strip() != "respond-async":
        return _query_error(
            http_request,
            status_code=400,
            code="unsupported_preference",
            message="只支持Prefer: respond-async",
        )

    time_window = query.time_window.as_window() if query.time_window is not None else None
    deps = RuntimeDependencies(
        model=model,
        executor=executor,
        # Resolved only after authentication; B0 never retrieves.
        retriever=retriever_source() if profile == B1_PROFILE else None,
        call_store=call_store,
        profile=profile,
    )

    try:
        if prefer is not None:
            # The request-scoped call store closes with this request; the
            # background worker opens its own durable store.
            run = run_service.start_async(
                identity=identity,
                question=query.question,
                time_window=time_window,
                deps=replace(deps, call_store=None),
            )
        else:
            run = run_service.run_sync(identity=identity, question=query.question, time_window=time_window, deps=deps)
    except ApprovalConflict as exc:
        return _query_error(http_request, status_code=503, code=exc.code, message="当前运行容量已满")
    http_request.state.run_id = str(run["run_id"])
    if prefer is not None:
        return JSONResponse(
            status_code=202,
            content={"run_id": run["run_id"], "status": run["status"]},
            headers={"Location": f"/runs/{run['run_id']}"},
        )
    return _sync_run_response(http_request, run_service, run)


# A resumed run that is still going (or done) answers 200: the frozen resume
# cases expect 200 for WAITING_USER and SUCCEEDED, unlike the 202 of /queries.
_RESUME_OK_STATUSES = frozenset({"WAITING_USER", "WAITING_APPROVAL", "SUCCEEDED"})


def _outcome_status(http_request: Request, run: Mapping[str, object], content: dict[str, object], message: str) -> int:
    """The HTTP status of a run's outcome, from the shared outcome table.

    A failure (status >= 400) also gets the ``error`` body; ``message`` is the route's wording.
    """

    status = str(run["status"])
    error_code = run.get("error_code")
    if status == "CANCELLED":
        http_status, error_code = 409, "run_cancelled"
    else:
        http_status = http_status_for_run(status, error_code) or 502
    if http_status >= 400:
        code = str(error_code or "run_failed")
        http_request.state.error_code = code
        content["error"] = {"code": code, "message": message, "request_id": request_id_for(http_request)}
    return http_status


def _resume_run_response(http_request: Request, run: Mapping[str, object]) -> JSONResponse:
    """The resumed run with /queries' error codes and body for every failure."""

    content = _public_run(run, include_result=True)
    if str(run["status"]) in _RESUME_OK_STATUSES:
        return JSONResponse(status_code=200, content=content)
    return JSONResponse(status_code=_outcome_status(http_request, run, content, "续跑未完成"), content=content)


def _sync_run_response(http_request: Request, service: RunService, run: Mapping[str, object]) -> JSONResponse:
    """The persisted run, rendered with the shared outcome table."""

    content = _public_run(run, include_result=True)
    content["profile"] = (run.get("run_config") or {}).get("profile") if isinstance(run.get("run_config"), Mapping) else None
    content["usage"] = _usage_entries(service, run)
    result = run.get("result")
    content["sources"] = (
        [{"source_id": result.get("result_id"), "version": result.get("catalog_version")}]
        if isinstance(result, Mapping)
        else []
    )
    checkpoint = run.get("checkpoint") if isinstance(run.get("checkpoint"), Mapping) else {}
    agent_checkpoint = checkpoint.get("agent_checkpoint") if isinstance(checkpoint.get("agent_checkpoint"), Mapping) else None
    if str(run["status"]) == "WAITING_USER" and agent_checkpoint is not None:
        content["pending_question"] = agent_checkpoint.get("waiting_question")
    http_status = _outcome_status(http_request, run, content, "查询未完成")
    headers = {"Location": f"/runs/{run['run_id']}"} if http_status == 202 else {}
    return JSONResponse(status_code=http_status, content=content, headers=headers)


# Each failure of the guarded path: exception type, HTTP status, error code (None: the
# exception's own) and message.  Matched in this order.
_PROPOSAL_FAILURES: tuple[tuple[type[Exception], int, str | None, str], ...] = (
    (ProposalParseError, 422, None, "模型提案参数未通过校验"),
    (ToolError, 403, None, "包含客户姓名的查询要通过 /queries 发起，由同租户审批人批准"),
    (SQLPolicyError, 403, None, "查询被SQL安全策略拒绝"),
    (UndefinedColumn, 403, "invalid_sql", "提案引用了不存在的数据库字段"),
    (GroupingError, 403, "invalid_sql", "提案的聚合字段分组不完整"),
    (GuardedQueryError, 422, None, "查询未通过受限执行器"),
)
_PROPOSAL_FAILURE_TYPES = tuple(row[0] for row in _PROPOSAL_FAILURES)


def _execute_proposal_sql(executor: GuardedQueryExecutor, context: ExecutionContext, action: ToolCallAction):
    params = _proposal_params(action)
    # Apply the same sensitive-field rule as ControlledTools.query_readonly
    # to every SQL that parses.  The executor's first step parses the same
    # SQL with the same function, so one that does not parse is always
    # rejected there: it is passed on unchanged, which keeps the policy
    # error's status and code, and the executor's attempt, as they were
    # (the frozen case security-mutating-sql-rejected relies on both).
    try:
        statement = parse_readonly_select(action.arguments["sql"])
    except SQLPolicyError:
        statement = None
    if statement is not None:
        check_sensitive_access(context, statement)
    return executor.execute(action.arguments["sql"], context=context, params=params)


def _proposal_failure(http_request: Request, exc: Exception, *, run_id: str, model_call_id: str) -> JSONResponse:
    if isinstance(exc, GuardedQueryError) and exc.code == "limit_reached":
        return JSONResponse(
            status_code=200,
            content={
                "run_id": run_id,
                "status": "LIMIT_REACHED",
                "error": {
                    "code": exc.code,
                    "message": "查询结果超过100行",
                    "request_id": request_id_for(http_request),
                },
                "usage": [{"model_call_id": model_call_id}],
            },
        )
    _, status_code, code, message = next(row for row in _PROPOSAL_FAILURES if isinstance(exc, row[0]))
    return _query_error(http_request, status_code=status_code, code=code or exc.code, message=message)  # type: ignore[attr-defined]


@app.post("/query-proposals")
def execute_query_proposal(
    request_body: QueryProposalRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    call_store: DurableModelCallStore = Depends(get_call_store),
    executor: GuardedQueryExecutor = Depends(get_guarded_executor),
) -> JSONResponse:
    """Execute one server-bound model proposal through the guarded path."""

    identity = resolve_identity(authorization)
    if identity is None:
        return _unauthorized(http_request)

    run_id = str(uuid4())
    http_request.state.run_id = run_id
    context = ExecutionContext(
        run_id=run_id,
        tenant_id=identity["tenant_id"],
        principal_id=identity["principal_id"],
        role=identity["role"],
    )
    call_identity = call_store.new_call(run_id, request_id=request_id_for(http_request))

    try:
        proposal = parse_query_proposal(
            request_body.proposal,
            context=context,
            model_call_id=call_identity.model_call_id,
        )
    except ProposalParseError as exc:
        return _query_error(http_request, status_code=422, code=exc.code, message="模型提案未通过结构校验")

    if not isinstance(proposal.action, ToolCallAction) or proposal.action.name != "query_readonly":
        return _query_error(
            http_request, status_code=422, code="unsupported_action", message="当前入口只执行query_readonly提案"
        )

    try:
        result = _execute_proposal_sql(executor, context, proposal.action)
    except _PROPOSAL_FAILURE_TYPES as exc:
        if isinstance(exc, ToolError) and exc.code != "approval_required":
            raise
        return _proposal_failure(http_request, exc, run_id=run_id, model_call_id=call_identity.model_call_id)

    evidence = result.evidence.as_dict()
    logger.info(
        "proposal_query_succeeded request_id=%s run_id=%s model_call_id=%s result_id=%s",
        request_id_for(http_request),
        run_id,
        call_identity.model_call_id,
        evidence["result_id"],
    )
    return JSONResponse(
        status_code=200,
        content={
            "run_id": run_id,
            "status": "SUCCEEDED",
            "rows": evidence["rows"],
            "result": evidence,
            "proposal_sha256": proposal.content_sha256,
            "usage": [
                {"model_call_id": call_identity.model_call_id}
            ],
            "mode": "proposal_ingress",
        },
    )
