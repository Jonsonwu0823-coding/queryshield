"""Demo walkthrough: drive a RUNNING QueryShield service through the core capabilities.

Seven scenarios, in order, each printed with the question, HTTP codes and terminal state,
the answer, ``answer_status``, the verified facts (metric, value, time window), source ids
and approval details:

  1. a verified data answer (synchronous);
  2. the ambiguous "销售额": the server asks, the user answers, the run resumes and states its basis;
  3. a question without a time range: the server asks for it, the run resumes;
  4. a question that needs no data: the server's fixed reply;
  5. a definition question: a knowledge answer with its sources;
  6. an asynchronous run: submit, poll, fetch the result;
  7. a sensitive query (customer names): approval, with the refused attempts shown first
     (the other tenant's approver: 404, the requester approving their own request: 403).

Every verified fact is compared with the value the demo-data generator computes on its own
for the same tenant, metric and time window (B3e rule, ``demo_run.verified_fact_counts``);
a difference is a hard failure.  Whatever path a scenario takes, every verified fact in the answer
it ends with must have been compared: a scenario that closes with more facts than it compared
fails with ``verified_fact_unchecked``.  The judgements are the ones of the HTTP smoke and of the
demo questions, imported, not copied.

The service is reached over HTTP; tokens come from the same environment variables the
service uses.  The action trace and the model usage are read from the service's state store
(QUERYSHIELD_STATE_STORE_PATH or --state-path): run this script where the store is readable,
for example ``docker compose exec app python scripts/demo_walkthrough.py``.  If it cannot be
read the script is blocked (exit 2): it never skips the trace-based judgements silently.

Fake mode: every scenario ends in a fixed state, and any deviation fails.  Real mode: model
behaviour that the smoke records as a known gap is recorded as a known gap here, hard failures
stay hard failures.

--evidence-dir writes b4b-walkthrough-summary.json: fixed fields and numbers only (scenario
ids, HTTP codes, terminal states, answer_status, check counts, which runs used MCP).  It holds
no question, answer text, customer name, URL or credential.

Exit codes: 0 pass, 1 a hard failure, 2 blocked (configuration, service, state store).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts import b2b_http_smoke as smoke  # noqa: E402
from scripts import demo_run  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
STATE_PATH_ENV = demo_run.STATE_PATH_ENV
TOKEN_ENVIRONMENT = demo_run.TOKEN_ENVIRONMENT
UNDATED_QUESTION = "支付金额是多少？"
UNDATED_ANSWER = "2026年9月"
APPROVAL_QUESTION = "查询本租户所有客户的姓名"
ASYNC_QUESTION = "2026年9月已支付订单有几笔？"
DEMO_QUESTION_IDS = {"sync": "Q02", "clarify": "Q08", "knowledge": "Q09", "no_data": "Q10"}
ROWSET_LABEL = "原始查询结果，未核实；列名来自 SQL 里的别名"
TERMINAL = frozenset({"SUCCEEDED", "DENIED", "FAILED", "LIMIT_REACHED", "CANCELLED", "USAGE_UNKNOWN"})
SSE_READ_SECONDS = 8.0
METADATA_CHOICES = ("any", "local", "mcp")


class Blocked(Exception):
    """The walkthrough cannot start; the message is safe to print."""


# --- pure judgements -------------------------------------------------------------


def judge_time_scenario(
    first: Mapping[str, object],
    reviews: list[dict],
    pending_is_catalog: bool,
    resumed: Mapping[str, object] | None,
    resumed_basis_stated: bool,
    resumed_window_ok: bool,
    resumed_has_gross: bool,
) -> dict:
    """Scenario 3: the undated question must wait for a time range, and the resume must verify it."""

    outcome, hard, gaps, run_resume = smoke.judge_time_request(
        int(first.get("http_status") or 0), first.get("status"), first.get("error_code"), reviews, pending_is_catalog
    )
    hard, gaps = list(hard), list(gaps)
    if run_resume:
        if resumed is None:
            hard.append("time_resume_missing")
        else:
            if resumed.get("http_status") != 200 or resumed.get("status") != "SUCCEEDED" or not resumed_has_gross:
                hard.append("time_resume")
            if not resumed_basis_stated:
                hard.append("time_resume_basis_not_stated")
            if not resumed_window_ok:
                hard.append("time_resume_window")
            if resumed.get("answer_status") != "verified":
                hard.append("time_resume_answer_status")
    return {"outcome": outcome, "hard_failures": hard, "known_gaps": gaps}


def judge_async_scenario(accepted_http: int, accepted_status: object, final: Mapping[str, object]) -> dict:
    """Scenario 6: 202 on submit, then a verified SUCCEEDED result with facts."""

    hard: list[str] = []
    if accepted_http != 202:
        hard.append("async_not_accepted")
    if final.get("status") != "SUCCEEDED":
        hard.append("async_not_succeeded")
    else:
        if final.get("answer_status") != "verified":
            hard.append("async_answer_status")
        if not final.get("facts"):
            hard.append("async_no_facts")
    return {"hard_failures": hard, "known_gaps": [], "accepted_status": accepted_status}


def judge_approval_scenario(step: Mapping[str, object]) -> dict:
    """Scenario 7, by the codes the service really returns.

    The other tenant's approver gets 404 not_found (the run is invisible to them), the
    requester (not an approver) gets 403 forbidden for approving their own request, and the
    same-tenant approver gets 200.  The rows stay unverified and are not copied into the answer.

    (The service also refuses an approver who approves an approval they requested themselves,
    but an approver's own sensitive query runs without an approval, so that branch cannot be
    reached through HTTP and is not part of the walkthrough.)
    """

    hard: list[str] = []
    if not (step.get("request_http") == 202 and step.get("request_status") == "WAITING_APPROVAL"):
        return {"hard_failures": ["approval_not_triggered"], "known_gaps": []}
    if not step.get("approval_permission_bound"):
        hard.append("approval_permission_unbound")
    if not (step.get("cross_tenant_http") == 404 and step.get("cross_tenant_error") == "not_found"):
        hard.append("cross_tenant_approval_not_404")
    if not (step.get("requester_http") == 403 and step.get("requester_error") == "forbidden"):
        hard.append("requester_approval_not_403")
    if step.get("approve_http") != 200 or step.get("approve_status") != "SUCCEEDED":
        hard.append("approval_not_executed")
    if int(step.get("row_count") or 0) < 1:
        hard.append("approval_no_rows")
    if step.get("answer_contains_row_values"):
        hard.append("answer_contains_row_values")
    if step.get("answer_status") != "unverified":
        hard.append("approval_answer_status")
    if int(step.get("fact_count") or 0) != 0:
        hard.append("approval_rows_became_facts")
    return {"hard_failures": hard, "known_gaps": []}


def metadata_expectation_failures(setting: str, mcp_run_count: int) -> list[str]:
    """--metadata-tools mcp needs at least one run with a metadata_session event, local none."""

    if setting == "mcp" and mcp_run_count == 0:
        return ["metadata_session_missing"]
    if setting == "local" and mcp_run_count > 0:
        return ["metadata_session_unexpected"]
    return []


def fake_gaps_are_failures(mode: str, hard: list[str], gaps: list[str]) -> tuple[list[str], list[str]]:
    """Fake mode is deterministic: a scenario that needed a known-gap excuse has failed."""

    if mode != "fake":
        return hard, gaps
    return [*hard, *[f"fake_gap:{item}" for item in gaps]], []


def unchecked_fact_failures(fact_count: int, checked: int) -> list[str]:
    """Every verified fact in a scenario's final answer must have been compared with the generator (R3).

    Whatever path a scenario took (a resumed run, a model that guessed the time window, an async result, ...),
    the number of verified facts in the answer it ends with must equal the number compared.  Fewer compared is
    a hard failure, so a scenario that forgets the comparison cannot pass quietly.
    """

    return [] if fact_count == checked else ["verified_fact_unchecked"]


def parse_sse_event_types(lines: Iterable[str]) -> list[str]:
    """The ``event:`` names of a run's SSE stream, in order."""

    return [line[len("event:"):].strip() for line in lines if line.startswith("event:")]


def runs_with_metadata_session(event_types_by_run: Mapping[str, list[str]]) -> list[str]:
    return sorted(run_id for run_id, types in event_types_by_run.items() if "metadata_session" in types)


def scenario_record(scenario_id: str, *, http_codes: list[int], terminal: object, answer_status: object, checked: int, mismatched: int, hard: list[str], gaps: list[str], runs: int) -> dict:
    """The evidence line: fixed fields and numbers only."""

    return {
        "scenario": scenario_id,
        "http_codes": http_codes,
        "terminal": terminal,
        "answer_status": answer_status,
        "verified_facts_checked": checked,
        "verified_facts_mismatched": mismatched,
        "run_count": runs,
        "hard_failures": sorted(set(hard)),
        "known_gaps": sorted(set(gaps)),
        "verdict": "fail" if hard else "pass",
    }


# --- output ----------------------------------------------------------------------


def _say(text: str = "") -> None:
    print(text, flush=True)


def show_scenario(number: int, title: str, question: str) -> None:
    _say()
    _say(f"=== 场景 {number}：{title} ===")
    _say(f"问题：{question}")


def show_response(label: str, code: int, body: Mapping[str, object]) -> None:
    error = body.get("error")
    error_code = error.get("code") if isinstance(error, Mapping) else body.get("error_code")
    suffix = f" 错误码={error_code}" if error_code else ""
    _say(f"{label}：HTTP {code} 终态={body.get('status')}{suffix}")
    if body.get("pending_question"):
        _say(f"  服务端追问：{body['pending_question']}")
    if body.get("answer"):
        _say(f"  回答：{body['answer']}")
    if body.get("answer_status"):
        _say(f"  answer_status={body['answer_status']}")
    facts = smoke._facts(dict(body))
    for fact in facts:
        window = fact.get("time_window") if isinstance(fact.get("time_window"), Mapping) else {}
        _say(
            f"  已核实事实：指标={fact.get('metric_id')} 数值={fact.get('value')} "
            f"时间窗={window.get('start')} 至 {window.get('end')}"
        )
    sources = body.get("source_ids")
    if isinstance(sources, list) and sources:
        _say(f"  来源：{', '.join(str(item) for item in sources)}")


def show_checks(checked: int, mismatched: int, hard: list[str], gaps: list[str]) -> None:
    _say(f"  与生成器独立算出的值比对：{checked} 条已核实事实，{mismatched} 条不一致")
    _say(f"  硬失败：{', '.join(sorted(set(hard))) or '无'}；已知缺口：{', '.join(sorted(set(gaps))) or '无'}")


# --- HTTP ------------------------------------------------------------------------


def event_types(base: str, run_id: str, token: str) -> list[str]:
    """Event names of a run from GET /runs/{id}/events.

    A finished run's stream ends by itself.  A waiting run's stream stays open and sends a
    heartbeat comment about once a second: stop at the first heartbeat after an event.
    """

    request = Request(f"{base}/runs/{run_id}/events")
    request.add_header("Authorization", f"Bearer {token}")
    lines: list[str] = []
    deadline = time.monotonic() + SSE_READ_SECONDS
    try:
        with urlopen(request, timeout=SSE_READ_SECONDS) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith(": heartbeat") and any(item.startswith("event:") for item in lines):
                    break
                lines.append(line)
                if time.monotonic() > deadline:
                    break
    except (URLError, OSError, TimeoutError):
        return parse_sse_event_types(lines)
    return parse_sse_event_types(lines)


class Walkthrough:
    def __init__(self, base: str, tokens: Mapping[str, str], state_path: Path, mode: str, metadata_tools: str) -> None:
        self.base = base
        self.tokens = dict(tokens)
        self.state_path = state_path
        self.mode = mode
        self.metadata_tools = metadata_tools
        self.records: list[dict] = []
        self.run_tokens: dict[str, str] = {}
        self.run_scenario: dict[str, str] = {}
        document = json.loads(demo_run.QUESTIONS_PATH.read_text(encoding="utf-8"))
        self.questions = {item["id"]: item for item in document["questions"]}
        from queryshield.agent.metric_intent import declarable_metric_ids
        from queryshield.catalog import load_default_catalog
        from queryshield.facts.render import render_no_data_answer

        catalog = load_default_catalog()
        self.no_data_reply = render_no_data_answer([catalog.metric_name(item) for item in declarable_metric_ids(catalog)])
        self.catalog_question = catalog.clarification("clarify.metric_basis").question
        self.demo_ids = demo_run.demo_source_ids()

    # -- helpers

    def _http(self, identity: str, path: str, *, method: str = "GET", body: dict | None = None, headers: dict | None = None, timeout: float = 300):
        return smoke._http(self.base, path, token=self.tokens[identity], method=method, body=body, headers=headers, timeout=timeout)

    def _remember(self, scenario_id: str, identity: str, body: Mapping[str, object]) -> None:
        run_id = body.get("run_id")
        if isinstance(run_id, str):
            self.run_tokens.setdefault(run_id, self.tokens[identity])
            self.run_scenario.setdefault(run_id, scenario_id)

    def _finish(self, scenario_id: str, codes: list[int], obs: Mapping[str, object], *, facts: list, checked: int, mismatched: int, hard: list[str], gaps: list[str], runs: int) -> None:
        """Close a scenario.  `facts` is the list of verified facts of the answer the scenario ended with (every
        scenario passes it, whatever the structure of its final response); `checked` is how many were compared."""

        hard = [*hard, *unchecked_fact_failures(len(facts), checked)]
        hard, gaps = fake_gaps_are_failures(self.mode, list(hard), list(gaps))
        show_checks(checked, mismatched, hard, gaps)
        self.records.append(
            scenario_record(
                scenario_id, http_codes=codes, terminal=obs.get("status"), answer_status=obs.get("answer_status"),
                checked=checked, mismatched=mismatched, hard=hard, gaps=gaps, runs=runs,
            )
        )

    # -- scenarios driven by the demo questions (1, 2, 4, 5)

    def demo_question(self, scenario_id: str, number: int, title: str, question_id: str) -> None:
        question = self.questions[question_id]
        show_scenario(number, title, str(question["question"]))
        obs, _ = demo_run._run_question(self.base, self.tokens, question, self.state_path, self.no_data_reply, self.catalog_question)
        codes = [int(obs["first"]["http_status"])] if "first" in obs else []
        codes.append(int(obs["http_status"]))
        if "first" in obs:
            _say(f"  第一次：HTTP {obs['first'].get('http_status')} 终态={obs['first'].get('status')} 追问是 catalog 的固定问题={obs['first'].get('pending_is_catalog_question')}")
            _say(f"  用户回答：{question.get('resume_answer')}")
        show_response("结果", int(obs["http_status"]), {**obs, "facts": {"facts": obs["facts"]}})
        if obs.get("answer_states_basis") is not None:
            _say(f"  回答写明口径：{obs['answer_states_basis']}")
        judgement = demo_run.judge_question(question, obs, self.demo_ids)
        self._remember(scenario_id, question["identity"], {"run_id": obs.get("run_id")})
        self._finish(
            scenario_id, codes, obs, facts=list(obs["facts"]), checked=judgement["verified_facts_checked"],
            mismatched=judgement["verified_facts_mismatched"], hard=judgement["hard_failures"], gaps=judgement["known_gaps"], runs=1,
        )

    # -- scenario 3

    def undated(self) -> None:
        scenario_id = "S3_time_request"
        show_scenario(3, "没给时间范围：服务端追问时间，再恢复", UNDATED_QUESTION)
        code, body = self._http("a-requester", "/queries", method="POST", body={"question": UNDATED_QUESTION})
        self._remember(scenario_id, "a-requester", body)
        run_id = body.get("run_id")
        reviews = smoke._clarification_reviews(self.state_path, run_id) if isinstance(run_id, str) else []
        pending_is_catalog = body.get("status") == "WAITING_USER" and body.get("pending_question") in smoke._catalog_questions()
        show_response("第一次", code, body)
        first = {"http_status": code, "status": body.get("status"), "error_code": demo_run._error_code(body)}
        resumed_body: dict = {}
        resumed_code = 0
        basis = window_ok = has_gross = False
        if code == 202 and body.get("status") == "WAITING_USER" and not pending_is_catalog:
            _say(f"  用户回答：{UNDATED_ANSWER}")
            resumed_code, resumed_body = self._http("a-requester", f"/runs/{run_id}/resume", method="POST", body={"answer": UNDATED_ANSWER})
            show_response("恢复后", resumed_code, resumed_body)
            answer = str(resumed_body.get("answer") or "")
            basis = "口径：支付订单总额（gross_fen）" in answer and "问题中提到‘支付金额’" in answer
            resumed_facts = smoke._facts(resumed_body)
            has_gross = "gross_fen" in {str(item.get("metric_id")) for item in resumed_facts}
            september = {"start": "2026-09-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
            window_ok = bool(resumed_facts) and all(
                isinstance(item.get("time_window"), Mapping)
                and {key: item["time_window"].get(key) for key in ("start", "end")} == september
                for item in resumed_facts
            )
            _say(f"  回答写明口径：{basis}；时间窗是 2026 年 9 月：{window_ok}")
        # The answer the scenario ends with: the resumed one, or the first one when the model guessed the time
        # window and the run succeeded at once.  Its verified facts are compared on EVERY path (R3).
        final_body = resumed_body or body
        final_facts = smoke._facts(final_body)
        checked, mismatched = demo_run.verified_fact_counts({"identity": "a-requester"}, {"facts": final_facts})
        judgement = judge_time_scenario(
            first, reviews, pending_is_catalog,
            {"http_status": resumed_code, "status": resumed_body.get("status"), "answer_status": resumed_body.get("answer_status")} if resumed_code else None,
            basis, window_ok, has_gross,
        )
        hard = list(judgement["hard_failures"]) + (["verified_fact_mismatch"] if mismatched else [])
        self._finish(
            scenario_id, [code] + ([resumed_code] if resumed_code else []), final_body, facts=final_facts, checked=checked,
            mismatched=mismatched, hard=hard, gaps=judgement["known_gaps"], runs=1,
        )

    # -- scenario 6

    def async_run(self) -> None:
        scenario_id = "S6_async"
        show_scenario(6, "异步提交：提交、查状态、取结果", ASYNC_QUESTION)
        code, accepted = self._http("a-requester", "/queries", method="POST", body={"question": ASYNC_QUESTION}, headers={"Prefer": "respond-async"})
        self._remember(scenario_id, "a-requester", accepted)
        _say(f"提交：HTTP {code} 状态={accepted.get('status')}")
        run_id = accepted.get("run_id")
        final: dict = {}
        if code == 202 and isinstance(run_id, str):
            deadline = time.monotonic() + 120
            polls = 0
            printed_status = None
            while time.monotonic() < deadline:
                status_code, final = self._http("a-requester", f"/runs/{run_id}")
                polls += 1
                # One line per CHANGE of state, not one per poll (a real model can take dozens of polls).
                if final.get("status") != printed_status:
                    printed_status = final.get("status")
                    _say(f"查状态：HTTP {status_code} 状态={printed_status}")
                if final.get("status") in TERMINAL | {"WAITING_USER", "WAITING_APPROVAL"}:
                    break
                time.sleep(0.5)
            _say(f"  一共查了 {polls} 次状态")
            if final.get("status") == "SUCCEEDED":
                result_code, final = self._http("a-requester", f"/runs/{run_id}/result")
                show_response("取结果", result_code, final)
        obs = demo_run._observe(int(code), final or accepted, self.state_path, self.no_data_reply)
        checked, mismatched = demo_run.verified_fact_counts({"identity": "a-requester"}, obs)
        judgement = judge_async_scenario(int(code), accepted.get("status"), obs)
        hard = list(judgement["hard_failures"]) + (["verified_fact_mismatch"] if mismatched else [])
        self._finish(scenario_id, [int(code)], obs, facts=list(obs["facts"]), checked=checked, mismatched=mismatched, hard=hard, gaps=[], runs=1)

    # -- scenario 7

    def approval(self) -> None:
        scenario_id = "S7_approval"
        show_scenario(7, "敏感查询：等待审批，先看被拒的尝试，再由本租户审批人批准", APPROVAL_QUESTION)
        code, body = self._http("a-requester", "/queries", method="POST", body={"question": APPROVAL_QUESTION})
        self._remember(scenario_id, "a-requester", body)
        show_response("发起（本租户请求人）", code, body)
        step: dict[str, object] = {
            "request_http": code,
            "request_status": body.get("status"),
        }
        if not (code == 202 and body.get("status") == "WAITING_APPROVAL"):
            judgement = judge_approval_scenario(step)
            self._finish(scenario_id, [code], body, facts=smoke._facts(body), checked=0, mismatched=0, hard=judgement["hard_failures"], gaps=[], runs=1)
            return
        run_id, approval_id = body["run_id"], body["approval_id"]
        _, approver_view = self._http("a-approver", f"/runs/{run_id}")
        step["approval_permission_bound"] = smoke.approval_permission_bound(approver_view)
        approval = approver_view.get("approval") if isinstance(approver_view, Mapping) else None
        if isinstance(approval, Mapping):
            action = approval.get("action") if isinstance(approval.get("action"), Mapping) else {}
            _say(
                f"  待审批：状态={approval.get('status')} 权限来源={action.get('permission_source_id')} "
                f"权限版本={action.get('permission_version')}"
            )
        decision = {"approval_id": approval_id, "decision": "approve"}

        # B tenant's approver: the run does not exist for them.
        code_b, body_b = self._http("b-approver", f"/runs/{run_id}/approval", method="POST", body=decision)
        step["cross_tenant_http"], step["cross_tenant_error"] = code_b, demo_run._error_code(body_b)
        _say(f"另一租户（B）的审批人批准：HTTP {code_b} 错误码={step['cross_tenant_error']}  —— 对方租户看不到这条任务")
        # The requester is not an approver.
        code_r, body_r = self._http("a-requester", f"/runs/{run_id}/approval", method="POST", body=decision)
        step["requester_http"], step["requester_error"] = code_r, demo_run._error_code(body_r)
        _say(f"本租户请求人自己批准：HTTP {code_r} 错误码={step['requester_error']}  —— 只有同租户审批人能批准")
        # The right approver.
        code_a, approved = self._http("a-approver", f"/runs/{run_id}/approval", method="POST", body=decision)
        step["approve_http"], step["approve_status"] = code_a, approved.get("status")
        show_response("本租户审批人批准", code_a, approved)
        _, result = self._http("a-requester", f"/runs/{run_id}/result")
        rows = [row for row in ((result.get("result") or {}).get("rows") or []) if isinstance(row, Mapping)]
        names = [str(row.get("name")) for row in rows if row.get("name")]
        answer = str(result.get("answer") or "")
        step["row_count"] = len(rows)
        step["answer_contains_row_values"] = any(name in answer for name in names)
        step["answer_status"] = result.get("answer_status")
        step["fact_count"] = len(smoke._facts(result))
        show_response("请求人取回结果", 200, {key: value for key, value in result.items() if key != "result"})
        _say(f"  行集（{ROWSET_LABEL}）：共 {len(rows)} 行，前 5 行：")
        for row in rows[:5]:
            _say(f"    {dict(row)}")
        judgement = judge_approval_scenario(step)
        self._finish(scenario_id, [code, code_b, code_r, code_a], result, facts=smoke._facts(result), checked=0, mismatched=0, hard=judgement["hard_failures"], gaps=[], runs=1)

    # -- run

    def run(self) -> dict:
        self.demo_question("S1_verified_sync", 1, "已核实的数据回答（同步）", DEMO_QUESTION_IDS["sync"])
        self.demo_question("S2_clarify_resume", 2, "含糊的“销售额”：追问并恢复，回答写明口径", DEMO_QUESTION_IDS["clarify"])
        self.undated()
        self.demo_question("S4_no_data", 4, "不需要数据的问题：服务端固定回答", DEMO_QUESTION_IDS["no_data"])
        self.demo_question("S5_knowledge", 5, "定义题：知识回答，带来源", DEMO_QUESTION_IDS["knowledge"])
        self.async_run()
        self.approval()
        return self.finish()

    def finish(self) -> dict:
        types_by_run: dict[str, list[str]] = {}
        for run_id, token in self.run_tokens.items():
            types_by_run[run_id] = event_types(self.base, run_id, token)
        mcp_runs = runs_with_metadata_session(types_by_run)
        _say()
        _say("=== 元数据工具走 MCP 的 run（metadata_session 事件）===")
        if mcp_runs:
            for run_id in mcp_runs:
                _say(f"  {self.run_scenario.get(run_id, '?')}")
        else:
            _say("  无（元数据工具在本地）")
        extra = metadata_expectation_failures(self.metadata_tools, len(mcp_runs))
        hard = sorted({item for record in self.records for item in record["hard_failures"]} | set(extra))
        gaps = sorted({item for record in self.records for item in record["known_gaps"]})
        return {
            "mode": self.mode,
            "metadata_tools_expectation": self.metadata_tools,
            "status": "pass" if not hard else "fail",
            "hard_failures": hard,
            "known_gaps": gaps,
            "scenarios": self.records,
            "mcp_run_count": len(mcp_runs),
            "mcp_scenarios": sorted({self.run_scenario.get(run_id, "unknown") for run_id in mcp_runs}),
            "verified_fact_check": {
                "checked": sum(item["verified_facts_checked"] for item in self.records),
                "mismatched": sum(item["verified_facts_mismatched"] for item in self.records),
            },
            "note": "ids, status codes, terminal states, counts and numbers only; no question or answer text, customer names, URLs or credentials",
        }


# --- main ------------------------------------------------------------------------


def read_tokens(environ: Mapping[str, str]) -> dict[str, str]:
    tokens, problems = demo_run.tokens_from_environment(environ)
    if tokens is None:
        raise Blocked("missing or repeated token variables: " + ", ".join(problems))
    return tokens


def check_state_path(value: str | None) -> Path:
    if not value:
        raise Blocked(f"the service state store is needed for the action trace: set {STATE_PATH_ENV} or give --state-path")
    path = Path(value)
    if not path.is_file() or not os.access(path, os.R_OK):
        raise Blocked("the service state store cannot be read here (run this inside the service container, or give --state-path)")
    return path


def check_service(base: str) -> None:
    try:
        with urlopen(base + "/health", timeout=5) as response:
            if response.status == 200:
                return
    except (URLError, OSError, ValueError):
        pass
    raise Blocked("the service did not answer /health")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--mode", choices=("fake", "real"), default="fake")
    parser.add_argument("--metadata-tools", choices=METADATA_CHOICES, default="any")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--state-path", default=os.environ.get(STATE_PATH_ENV))
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")
    try:
        tokens = read_tokens(os.environ)
        state_path = check_state_path(args.state_path)
        check_service(base)
    except Blocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False))
        return 2
    walkthrough = Walkthrough(base, tokens, state_path, args.mode, args.metadata_tools)
    try:
        summary = walkthrough.run()
    except smoke._StepStopped as stop:
        summary = {
            "mode": args.mode, "status": "fail", "hard_failures": [f"{stop.kind}"], "known_gaps": [],
            "scenarios": walkthrough.records, "note": "a request did not complete",
        }
    if args.evidence_dir:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        (args.evidence_dir / "b4b-walkthrough-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: summary[key] for key in ("status", "hard_failures", "known_gaps")}, ensure_ascii=False))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
