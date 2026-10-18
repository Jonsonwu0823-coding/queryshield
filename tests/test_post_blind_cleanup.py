"""After the blind test: the ask review lets a question naming two values be asked, the tenant check
recognizes more tenant nouns in both word orders, the blind tools survive a refusing
Git and a missing record file, and the Fake reads the request window where the server
puts it.  Every question is made up for these tests.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import subprocess

import pytest

from queryshield.agent.context import build_context
from queryshield.agent.tenant_scope import explicit_foreign_tenant_mentions, has_explicit_foreign_tenant
from queryshield.api.main import app, get_guarded_executor, get_model_provider
from queryshield.catalog import load_default_catalog
from queryshield.catalog.phrases import check_declaration, read_clarifications, review_ask
from queryshield.db.guarded import GuardedQueryExecutor
from queryshield.evaluation import blind_v3
from queryshield.providers.fake_model import FakeModel
from scripts import run_blind_v3 as runner
from test_api_call_identity import _CountingModel, _post, _ProposalConnection, api_env  # noqa: F401 - pytest fixture
from scripts import seal_blind_v3 as seal
from test_blind_v3 import _case, _cases, _payload, _seal_files
from test_clarification import (  # noqa: E402 - pytest prepend import mode puts tests/ on sys.path
    METRIC_BASIS_QUESTION,
    SEPTEMBER,
    _agent,
    _ask,
    _cite,
    _context,
    _query,
    _resume,
    _Scripted,
    _start,
    _tools,
    service,  # noqa: F401 - pytest fixture
)

MB = "clarify.metric_basis"
TWO_VALUES = "2026年9月的支付金额或退款后净额是多少？"
ONE_VALUE = "2026年9月的支付金额是多少？"


# --- the ask review: two values named, none chosen --------------------------------------


def _reading(question: str, answers=(), confirmed=()):
    return read_clarifications(load_default_catalog(), question, answers, confirmed)


def test_a_question_naming_two_values_reads_the_rule_as_open() -> None:
    item = _reading(TWO_VALUES).rule(MB)
    assert item.question_values == {"gross_fen", "net_fen"} and not item.ambiguous
    assert item.status == "open"
    assert _reading(ONE_VALUE).rule(MB).status == "resolved"
    assert _reading(TWO_VALUES, answers=["支付金额"]).rule(MB).status == "resolved"
    assert _reading(TWO_VALUES, confirmed=["metric.gross_fen"]).rule(MB).status == "resolved"


@pytest.mark.parametrize(
    "clarification_id, ask",
    [(MB, METRIC_BASIS_QUESTION), (None, "按支付金额还是退款后净额统计？"), (None, "请确认统计口径"), (None, "请问是哪个月的支付金额？")],
)
def test_an_ask_about_a_rule_with_two_named_values_gets_the_catalog_question(clarification_id, ask) -> None:
    verdict = review_ask(_reading(TWO_VALUES), clarification_id, ask)
    assert (verdict.decision, verdict.rule.id) == ("catalog_question", MB)


@pytest.mark.parametrize(
    "question, answers",
    [(ONE_VALUE, ()), (TWO_VALUES, ("支付金额",)), (TWO_VALUES, ("退款后净额",))],
)
def test_one_named_value_or_a_choice_made_still_sends_the_ask_back(question, answers) -> None:
    verdict = review_ask(_reading(question, answers), MB, METRIC_BASIS_QUESTION)
    assert (verdict.decision, verdict.rule.id) == ("not_needed", MB)


@pytest.mark.parametrize("metrics", [["gross_fen"], ["net_fen"], ["gross_fen", "net_fen"]])
def test_a_declaration_for_two_named_values_is_checked_as_before(metrics) -> None:
    assert check_declaration(_reading(TWO_VALUES), metrics) is None
    assert check_declaration(_reading("2026年9月的支付金额和退款后净额分别是多少？"), metrics) is None


def test_two_named_values_wait_on_the_catalog_question_then_query_only_the_chosen_one(service) -> None:
    run = _start(service, TWO_VALUES, [_ask(clarification_id=MB)])
    assert run["status"] == "WAITING_USER"
    assert run["checkpoint"]["agent_checkpoint"]["waiting_clarification_id"] == MB
    _, executor = service
    assert executor.executed == []

    done = _resume(service, run, "要支付金额", [_query(), _cite])
    assert done["status"] == "SUCCEEDED"
    assert [fact["metric_id"] for fact in done["facts"]["facts"]] == ["gross_fen"]
    assert len(executor.executed) == 1


def test_one_named_value_and_an_ask_is_sent_back_then_fails_as_before() -> None:
    tools, executor = _tools()
    result = _agent(_Scripted([_ask(clarification_id=MB), _ask(clarification_id=MB)]), tools).run(_context("run-c26-one-value"), ONE_VALUE, request_time_window=SEPTEMBER)
    assert result.status == "failed" and result.error_code == "clarification_not_needed"
    reviews = [event["decision"] for event in result.events if event["kind"] == "clarification_review"]
    assert reviews == ["not_needed", "not_needed"]
    assert executor.executed == []


# --- the tenant check ---------------------------------------------------------------------

CHINESE_NOUNS = ("租户", "商户", "商家", "店铺", "门店", "公司")
ENGLISH_NOUNS = ("tenant", "merchant")


@pytest.mark.parametrize("noun", CHINESE_NOUNS)
@pytest.mark.parametrize("template", ["{noun}B的支付金额", "{noun} B 的支付金额", "B{noun}的支付金额", "B {noun}的支付金额"])
def test_every_chinese_tenant_noun_is_recognized_in_both_orders(noun, template) -> None:
    question = template.format(noun=noun)
    assert explicit_foreign_tenant_mentions(question, "A") == ("B",)
    assert not has_explicit_foreign_tenant(question, "B") and not has_explicit_foreign_tenant(question, "tenant-B")


@pytest.mark.parametrize("noun", ENGLISH_NOUNS)
@pytest.mark.parametrize("template", ["{noun}-B 的支付金额", "{noun} B 的支付金额", "B {noun} 的支付金额", "B {Noun}的支付金额"])
def test_both_english_tenant_nouns_are_recognized_in_both_orders(noun, template) -> None:
    question = template.format(noun=noun, Noun=noun.capitalize())
    assert explicit_foreign_tenant_mentions(question, "A") == ("B",)
    assert not has_explicit_foreign_tenant(question, "b")


@pytest.mark.parametrize(
    "question",
    [
        "本租户2026年8月的支付金额是多少？",
        "租户2026年8月的支付金额",
        "我们商户2026年8月的退款后净额",
        "本店铺的已支付订单数",
        "公司 2026 年的支付金额",
        "2026商户的订单",  # only digits before the noun
        "租户A的支付金额",
        "A 商户的支付金额",
        "tenant-A 的支付金额",
        "AB租户的订单",  # a label is one letter: "AB" is a word, not "B" glued to a letter
        "2B租户的订单",  # a label glued to a digit before it is not one
        "show tenant A orders",
        "the merchant's orders",
        "商户ID是多少",
        "租户ID",
        "B2B 公司的支付金额",
        "W04_租户",
        "tenant-acme 的支付金额",  # a word, no longer a label
        "x_B租户",
        "租户B_1的订单",  # glued to an underscore after it
        "B merchants 的订单",  # only the whole words tenant and merchant, in either order
        "B tenants 的订单",
    ],
)
def test_own_tenant_words_and_digits_are_not_another_tenant(question) -> None:
    assert explicit_foreign_tenant_mentions(question, "A") == ()


@pytest.mark.parametrize(
    "question, label",
    [
        ("B tenant 的支付金额", "B"),
        ("B merchant 的支付金额", "B"),
        ("tenant-A1 的支付金额", "A1"),
        ("商户 B2 的订单", "B2"),
        ("B2 merchant", "B2"),
        ("Tenant-B 的订单", "B"),  # the English nouns in any case
        ("TENANT B 的订单", "B"),
        ("Merchant B 的订单", "B"),
    ],
)
def test_a_label_is_one_letter_with_optional_digits(question, label) -> None:
    assert explicit_foreign_tenant_mentions(question, "A") == (label,)
    assert not has_explicit_foreign_tenant(question, label.lower())


def test_a_business_description_is_not_read_as_a_tenant() -> None:
    assert not has_explicit_foreign_tenant("那家批发商2026年8月的支付金额是多少？", "A")


def test_the_http_check_refuses_another_tenant_before_the_model_and_lets_the_own_tenant_ask(api_env) -> None:
    model = _CountingModel()
    app.dependency_overrides[get_model_provider] = lambda: model
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: _ProposalConnection([]))
    for question in ("B租户2026年8月的退款后净额是多少？", "B 商户2026年8月的支付金额", "merchant B 2026年8月的支付金额"):
        response = _post(question)
        assert (response.status_code, response.json()["error"]["code"]) == (403, "forbidden"), question
    assert model.calls == 0

    connection = _ProposalConnection([{"gross_fen": 3000}])
    app.dependency_overrides[get_model_provider] = lambda: FakeModel()
    app.dependency_overrides[get_guarded_executor] = lambda: GuardedQueryExecutor(connect=lambda: connection)
    response = _post("本租户2026年8月的已支付订单总额是多少？")
    assert response.status_code == 200 and response.json()["status"] == "SUCCEEDED"


# --- the blind tools ----------------------------------------------------------------------


def test_head_and_repository_root_are_read_when_git_refuses_the_owner(monkeypatch, tmp_path) -> None:
    """Git refuses a checkout another user owns ("dubious ownership") unless safe.directory allows it."""

    calls = []

    def git(args, **kwargs):
        calls.append(args)
        if "safe.directory=*" not in args:
            return subprocess.CompletedProcess(args, 128, "", "fatal: detected dubious ownership in repository")
        out = "f" * 40 if args[-1] == "HEAD" else str(tmp_path)
        return subprocess.CompletedProcess(args, 0, out + "\n", "")

    monkeypatch.setattr(runner.subprocess, "run", git)
    assert runner._git_head() == "f" * 40
    assert runner._repository_root() == tmp_path.resolve()
    assert all(args[:3] == ["git", "-c", "safe.directory=*"] for args in calls)


def test_a_missing_record_file_blocks_the_recalculation_as_not_authentic(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "sealed").mkdir()
    sealed, key = _seal_files(tmp_path / "sealed")
    seal = blind_v3.verify_commitment(sealed / runner.COMMITMENT_FILE, sealed / runner.CIPHERTEXT_FILE, runner.sha256_file(runner.DEMO_SQL))
    manifest = runner._source_manifest_bytes()
    attempt = tmp_path / "attempt-1"
    attempt.mkdir()
    (attempt / "source-manifest.txt").write_bytes(manifest)
    summary = {
        "status": "complete",
        "attempt": 1,
        "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "commitment": {"commitment_sha256": seal["commitment_sha256"]},
    }
    runner.write_json(attempt / "summary.json", summary)
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: key.hex())
    capsys.readouterr()

    assert runner.main(["recalculate", "--attempt-dir", str(attempt), "--sealed-dir", str(sealed)]) == 2
    assert capsys.readouterr().out.strip() == "blind-v3 blocked: records_not_authentic"
    assert not (attempt / "recalculation.json").exists()


def test_a_number_the_requester_wrote_in_a_reply_is_not_a_leak_when_repeated() -> None:
    amount = runner.foreign_values("A")[-1]
    payload = _payload()
    payload["cases"][2]["steps"][0]["reply"] = f"按支付金额统计，{amount}分那笔不算"
    case = next(item for item in _cases(payload) if item.case_id == "V3-03")
    record = copy.deepcopy(seal.ideal_record(case))
    record["views"][-1]["answer"] = f"按支付金额统计（{amount}分那笔不算）"
    assert runner.judge_case(case, record)["passed"]
    record["views"][-1]["answer"] = f"按支付金额统计（{runner.foreign_values('A')[-2]}分那笔不算）"
    assert runner.judge_case(case, record)["failures"] == ["foreign_tenant_value"]


@pytest.mark.parametrize("identity", blind_v3.IDENTITIES)
def test_the_asker_is_approver_only_for_an_approver_identity(identity) -> None:
    assert replace(_case("V3-01"), identity=identity).asker_is_approver is (identity in {"a-approver", "b-approver"})


# --- the Fake reads the request window where the server puts it ---------------------------

JUNE = {"start": "2026-06-01T00:00:00Z", "end": "2026-07-01T00:00:00Z"}


def test_the_fake_declares_the_request_window_from_the_real_server_context() -> None:
    messages = list(
        build_context(
            _context("run-c26-fake-window"), "已支付订单总额是多少？", request_time_window={**JUNE, "timezone": "UTC"}, retrieval_available=False
        ).messages
    )
    server = json.loads(messages[0]["content"].split("\n", 1)[1])
    assert server["metric_declaration"]["time_window"]["request_time_window"].items() >= JUNE.items()
    assert "request_time_window" not in server["metric_declaration"]

    action = json.loads(FakeModel().complete(messages, request_id="r", model_call_id="m").content)
    assert action["type"] == "tool_call" and action["arguments"]["time_window"] == JUNE
    assert {JUNE["start"], JUNE["end"]} <= set(action["arguments"]["params"].values())


def test_the_request_window_conversation_runs_in_fake_mode_too() -> None:
    from scripts import conversation_eval as ce

    document = json.loads(ce.CASES_PATH.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in document["conversations"]}
    assert by_id["CW02"]["fake_supported"] is True and "reason" not in by_id["CW02"]
    assert sorted(cid for cid, item in by_id.items() if not item["fake_supported"]) == ["CR01", "CR02", "CR03", "CR04"]
