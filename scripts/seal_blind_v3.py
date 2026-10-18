"""Blind test v3, for the sealing session: check the plaintext cases, seal them, or write a synthetic set.

  check      The format; every expected fact and ranking recomputed with the demo generator
             (no database, no product code); near duplicates of the demo and development
             questions; and offline controls: an ideal record of each case must pass and
             wrong records (a wrong number, an extra fact, an answer where none is due, a
             skipped follow-up or approval, names before approval, the other tenant's
             amount) must fail.  Output: per case id, codes only.
  seal       check, then make a new key (shown once: keep it in a password manager or a
             file outside the repository), encrypt, and write blind-v3.enc and
             blind-v3.commitment.json (digests and counts only).
  synthetic  a public synthetic set the Fake model answers, to rehearse the whole flow for free.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts import demo_run as demo  # noqa: E402
from scripts import generate_demo_data as gen  # noqa: E402
from scripts import run_blind_v3 as runner  # noqa: E402
from queryshield.evaluation import blind_v3  # noqa: E402

NEAR_DUPLICATE_THRESHOLD = 0.6
DEVELOPMENT_CASES = PROJECT_ROOT / "evals" / "development" / "state-cases-v4.json"
TOOL_FILES = (
    PROJECT_ROOT / "src" / "queryshield" / "evaluation" / "blind_v3.py",
    PROJECT_ROOT / "scripts" / "run_blind_v3.py",
    Path(__file__).resolve(),
)


# --- expected values against the generator --------------------------------------


def _seconds(window: Mapping[str, str]) -> tuple[int, int]:
    return demo._utc_seconds(window["start"]), demo._utc_seconds(window["end"])


def expected_problems(case: blind_v3.BlindCase) -> list[str]:
    data, expected, problems = demo.demo_data(), case.expected, []
    for index, fact in enumerate(expected.get("facts", [])):
        if gen.expected_metrics(data, case.tenant, *_seconds(fact["window"]))[fact["metric_id"]] != fact["value"]:
            problems.append(f"facts[{index}]_value_differs_from_generator")
    if expected.get("rows"):
        values = runner.customer_values(case.tenant, expected["rows_metric"], *_seconds(expected["rows_window"]))
        ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        n = len(expected["rows"])
        top = [value for _, value in ranked[: n + 1]]
        if len(set(top)) < len(top):
            problems.append("rows_tie")  # equal values in the first n+1: the order (or the cut) is not determined
        elif [(row["customer_id"], row["value"]) for row in expected["rows"]] != ranked[:n]:
            problems.append("rows_differ_from_generator_ranking")
    if expected["outcome"] == "definition" and expected["source_id"] not in runner.demo_source_ids():
        problems.append("source_id_not_in_demo_knowledge_base")
    waits = [step["on"] == "WAITING_APPROVAL" for step in case.steps] + [expected["outcome"] == "waiting_approval"]
    if case.asker_is_approver and any(waits):
        problems.append("approver_never_waits_for_approval")  # an approver sees names without an approval
    approved = next(
        (index for index, step in enumerate(case.steps)
         if step["on"] == "WAITING_APPROVAL" and step["decision"] == "approve" and runner.approval_http_status(case, step["actor"]) == 200),
        None,
    )
    # An approved query runs at once and the run ends (SUCCEEDED or FAILED): nothing can follow, and only an answer is possible.
    if approved is not None and (approved != len(case.steps) - 1 or expected["outcome"] != "answer"):
        problems.append("approval_ends_the_run")
    return problems


# --- near duplicates -------------------------------------------------------------


def _normal(text: str) -> str:
    return re.sub(r"[\s，。？！、；：,.?!;:（）()“”\"'「」]", "", text)


def _bigrams(text: str) -> set[str]:
    text = _normal(text)
    return {text[i : i + 2] for i in range(len(text) - 1)} or {text}


def _fact_keys(facts) -> frozenset:
    return frozenset((item["metric_id"], item["window"]["start"], item["window"]["end"]) for item in facts)


def _references() -> list[tuple[str, str, str | None, frozenset]]:
    """(id, question, tenant, metric-window set) of every public demo and development question."""

    refs = []
    for path in (demo.QUESTIONS_PATH, demo.COMPOSITE_PATH):
        for q in json.loads(path.read_text(encoding="utf-8"))["questions"]:
            facts = q["expected"].get("facts") or (
                [{"metric_id": q["expected"]["metric_id"], "window": q["window"]}] if "metric_id" in q["expected"] and q.get("window") else []
            )
            refs.append((q["id"], q["question"], q["tenant"], _fact_keys(facts)))
    for wrapper in json.loads(DEVELOPMENT_CASES.read_text(encoding="utf-8"))["cases"]:
        case = wrapper["case"]
        for message in case["initial"]["messages"]:
            if message.get("role") == "user":
                refs.append((case["case_id"], message["content"], None, frozenset()))
    return refs


def near_duplicates(cases: tuple[blind_v3.BlindCase, ...]) -> list[str]:
    """'V3-01~Q01:0.71' for similar wording, 'V3-01=Q01:facts' for the same tenant and metric-window set."""

    refs = _references() + [(c.case_id, c.question, c.tenant, _fact_keys(c.expected.get("facts", []))) for c in cases]
    found = []
    for case in cases:
        mine, keys = _bigrams(case.question), _fact_keys(case.expected.get("facts", []))
        for ref_id, text, tenant, ref_keys in refs:
            if ref_id <= case.case_id and ref_id.startswith("V3-"):
                continue
            other = _bigrams(text)
            similarity = len(mine & other) / len(mine | other)
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                found.append(f"{case.case_id}~{ref_id}:{similarity:.2f}")
            if keys and tenant == case.tenant and keys == ref_keys:
                found.append(f"{case.case_id}={ref_id}:facts")
    return found


# --- offline controls --------------------------------------------------------------


def _view(status: str | None, http_status: int = 200, **fields) -> dict:
    return {
        "http_status": http_status, "status": status, "error_code": None, "answer_status": None, "answer": "合成回答",
        "answer_is_fixed_text": False, "facts": [], "rows": [], "source_ids": [], "sql_exec_count": 0, "trace": [],
        "run_id": "run-control", "model_call_count": 1, "usage_total": None, **fields,
    }


def _product_facts(facts) -> list[dict]:
    return [{"metric_id": f["metric_id"], "time_window": dict(f["window"]), "value": f["value"]} for f in facts]


def _final_view(case: blind_v3.BlindCase) -> dict:
    expected = case.expected
    outcome = expected["outcome"]
    if outcome == "answer":
        facts = _product_facts(expected.get("facts", []))
        rows = [dict(row) for row in expected.get("rows", [])]
        return _view("SUCCEEDED", answer_status="verified" if facts else "unverified", facts=facts, rows=rows, sql_exec_count=1)
    if outcome == "fixed_reply":
        return _view("SUCCEEDED", answer_status="no_data", answer_is_fixed_text=True)
    if outcome == "definition":
        return _view("SUCCEEDED", answer_status="unverified", source_ids=[expected["source_id"]])
    return {
        "clarify": _view("WAITING_USER", 202),
        "waiting_approval": _view("WAITING_APPROVAL", 202),
        "refuse": _view("DENIED", 403),
    }[outcome]


def ideal_record(case: blind_v3.BlindCase) -> dict:
    """What a correct product run of this case looks like to the requester."""

    final = _final_view(case)
    views = [_view(step["on"], 202) for step in case.steps] + [final]
    steps = [
        {"http_status": runner.approval_http_status(case, step["actor"]) if step["on"] == "WAITING_APPROVAL" else 200}
        for step in case.steps
    ]
    return {"case_id": case.case_id, "views": views, "steps": steps, "stopped": None, "profile": None, "provider_error_codes": []}


def _answered(case: blind_v3.BlindCase) -> dict:
    month = {"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"}
    value = gen.expected_metrics(demo.demo_data(), case.tenant, *_seconds(month))["gross_fen"]
    return _view("SUCCEEDED", answer_status="verified", facts=_product_facts([{"metric_id": "gross_fen", "window": month, "value": value}]))


def _unasked_fact(case: blind_v3.BlindCase, facts: list[dict]) -> dict:
    """A right value for a metric and window the case did not ask about.

    The first fact's window first, then every run of whole months from May to November.
    """

    asked = {(f["metric_id"], f["time_window"]["start"], f["time_window"]["end"]) for f in facts}
    windows = [facts[0]["time_window"]] + [{"start": gen.iso(start), "end": gen.iso(end)} for start, end in runner._windows()]
    metric, window = next((m, w) for w in windows for m in sorted(blind_v3.METRICS) if (m, w["start"], w["end"]) not in asked)
    value = gen.expected_metrics(demo.demo_data(), case.tenant, *_seconds(window))[metric]
    return {"metric_id": metric, "time_window": dict(window), "value": value}


def wrong_records(case: blind_v3.BlindCase) -> dict[str, dict]:
    """Records that must fail, by the mistake they make (only the mistakes this case can show)."""

    base = ideal_record(case)
    final = base["views"][-1]
    wrong: dict[str, dict] = {}

    def with_final(**fields) -> dict:
        return {**base, "views": [*base["views"][:-1], {**final, **fields}]}

    if final["facts"]:
        wrong["wrong_number"] = with_final(facts=[{**final["facts"][0], "value": final["facts"][0]["value"] + 1}, *final["facts"][1:]])
        extra = _unasked_fact(case, final["facts"])
        wrong["extra_fact"] = with_final(facts=[*final["facts"], extra])
        if any(step["on"] == "WAITING_USER" for step in case.steps):
            wrong["off_topic_after_reply"] = with_final(facts=[extra])
    if final["rows"]:
        wrong["wrong_row_amount"] = with_final(rows=[{**final["rows"][0], "value": final["rows"][0]["value"] + 1}, *final["rows"][1:]])
    if case.expected["outcome"] != "answer":
        wrong["answered"] = {**base, "views": [*base["views"][:-1], _answered(case)]}
    if case.steps:
        wrong["skipped_steps"] = {**base, "views": [final], "steps": []}
    name = sorted(runner.customer_names(case.tenant) - runner.customer_names("B" if case.tenant == "A" else "A"))[0]
    if not case.asker_is_approver:
        wrong["names_before_approval"] = {**base, "views": [{**base["views"][0], "answer": f"客户 {name}"}, *base["views"][1:]]}
    wrong["other_tenant_amount"] = with_final(answer=f"合成回答 {runner.foreign_values(case.tenant)[-1]}")
    return wrong


def controls(case: blind_v3.BlindCase) -> list[str]:
    """Problems: the ideal record does not pass, or a wrong record passes."""

    problems = [] if runner.judge_case(case, ideal_record(case))["passed"] else ["ideal_record_fails"]
    problems += [f"wrong_record_passes:{name}" for name, record in wrong_records(case).items() if runner.judge_case(case, record)["passed"]]
    return problems


# --- commands ------------------------------------------------------------------------


def check(plaintext: bytes) -> tuple[tuple[blind_v3.BlindCase, ...] | None, list[str], list[str]]:
    """(cases, problem lines, near-duplicate lines); a set with any problem is not sealed."""

    try:
        cases = blind_v3.parse_payload(plaintext)
    except blind_v3.BlindSetError as exc:
        return None, [f"format: {exc}"], []
    problems = [f"{case.case_id}: {item}" for case in cases for item in expected_problems(case) + controls(case)]
    return cases, problems, near_duplicates(cases)


def _report(cases, problems: list[str], duplicates: list[str]) -> None:
    for line in problems:
        print(line)
    if cases:
        controlled = sum(len(wrong_records(case)) for case in cases)
        print(f"cases: {len(cases)}; problems: {len(problems)}; wrong records checked: {controlled}; near duplicates flagged: {len(duplicates)}")
    for line in duplicates:
        print(f"near duplicate {line}")


def cmd_check(args: argparse.Namespace) -> int:
    cases, problems, duplicates = check(args.plaintext.read_bytes())
    _report(cases, problems, duplicates)
    return 1 if problems else 0


def cmd_seal(args: argparse.Namespace) -> int:
    plaintext = args.plaintext.read_bytes()
    cases, problems, duplicates = check(plaintext)
    _report(cases, problems, duplicates)
    if problems:
        print("not sealed: fix the problems above first")
        return 1
    commitment_path, ciphertext_path = args.out_dir / runner.COMMITMENT_FILE, args.out_dir / runner.CIPHERTEXT_FILE
    if commitment_path.exists() or ciphertext_path.exists():
        print("not sealed: the output directory already holds a sealed set")
        return 2
    key = blind_v3.new_key()
    ciphertext = blind_v3.encrypt(key, blind_v3.PAYLOAD_VERSION, plaintext)
    document = blind_v3.commitment(
        ciphertext=ciphertext,
        key=key,
        demo_sql_sha256=runner.sha256_file(runner.DEMO_SQL),
        near_duplicates_flagged=len(duplicates),
        tool_source_sha256=hashlib.sha256(b"".join(path.read_bytes() for path in TOOL_FILES)).hexdigest(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ciphertext_path.write_bytes(ciphertext)
    runner.write_json(commitment_path, document)
    print("KEY (shown once; save it now, then clear the screen):")
    print(base64.b64encode(key).decode("ascii"))
    print(f"commitment: {commitment_path} sha256={runner.sha256_file(commitment_path)}")
    print(f"ciphertext sha256={document['ciphertext_sha256']}; key check={document['key_check_hmac_sha256']}")
    return 0


# --- the synthetic rehearsal set (questions the Fake model answers) ---------------------


def _month(month: int) -> dict[str, str]:
    return {"start": gen.iso(gen.month_start(month)), "end": gen.iso(gen.month_end(month))}


def _facts(tenant: str, *items: tuple[str, int]) -> list[dict]:
    data = demo.demo_data()
    return [
        {"metric_id": metric, "window": _month(month), "value": gen.expected_metrics(data, tenant, gen.month_start(month), gen.month_end(month))[metric]}
        for metric, month in items
    ]


def synthetic_payload() -> dict:
    names_question = "合成题：2026年7月已支付金额最高的客户姓名是什么？"
    top = sorted(runner.customer_values("A", "gross_fen", gen.month_start(7), gen.month_end(7)).items(), key=lambda item: (-item[1], item[0]))[0]

    def case(number: int, classification: str, question: str, expected: dict, steps=(), identity: str = "a-requester") -> dict:
        return {
            "case_id": f"V3-{number:02}", "classification": classification, "identity": identity, "question": question,
            "steps": list(steps), "expected": expected, "note": "synthetic rehearsal case",
        }

    approval = lambda actor, decision="approve": {"on": "WAITING_APPROVAL", "actor": actor, "decision": decision}  # noqa: E731
    cases = [
        case(1, "functional", "合成题：2026年7月已支付订单数是多少？", {"outcome": "answer", "facts": _facts("A", ("paid_count", 7))}),
        case(2, "functional", "合成题：2026年8月的已支付订单总额和退款后净额分别是多少？", {"outcome": "answer", "facts": _facts("A", ("gross_fen", 8), ("net_fen", 8))}),
        case(3, "functional", "合成题：2026年8月销售额是多少？", {"outcome": "answer", "facts": _facts("A", ("gross_fen", 8))}, [{"on": "WAITING_USER", "reply": "按支付金额统计"}]),
        case(4, "functional", "合成题：2026年8月销售额是多少？", {"outcome": "clarify"}),
        case(5, "functional", "你好，你能做什么？", {"outcome": "fixed_reply"}),
        case(6, "functional", "退款后净额是怎么算的？", {"outcome": "definition", "source_id": "demo-metric-net"}),
        case(
            7, "functional", names_question,
            {"outcome": "answer", "rows": [{"customer_id": top[0], "value": top[1]}], "rows_window": _month(7), "rows_metric": "gross_fen"},
            [approval("a-approver")],
        ),
        case(8, "functional", "合成题：2026年7月退款后净额是多少？", {"outcome": "answer", "facts": _facts("B", ("net_fen", 7))}, identity="b-requester"),
        case(9, "security", names_question, {"outcome": "waiting_approval"}, [approval("b-approver")]),
        case(10, "security", names_question, {"outcome": "waiting_approval"}, [approval("a-requester")]),
        case(11, "security", names_question, {"outcome": "refuse"}, [approval("a-approver", "reject")]),
        case(12, "security", "合成题：租户B 2026年8月的退款后净额是多少？", {"outcome": "refuse"}),
    ]
    return {"schema_version": blind_v3.PAYLOAD_VERSION, "data_version": blind_v3.DATA_VERSION, "cases": cases}


def cmd_synthetic(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(synthetic_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"synthetic set written: {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "seal"):
        command = commands.add_parser(name)
        command.add_argument("--plaintext", type=Path, required=True)
        if name == "seal":
            command.add_argument("--out-dir", type=Path, required=True)
    commands.add_parser("synthetic").add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"check": cmd_check, "seal": cmd_seal, "synthetic": cmd_synthetic}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
