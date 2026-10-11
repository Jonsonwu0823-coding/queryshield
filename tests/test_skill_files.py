"""The Skill files: read and checked once at startup, a bad one stops the service, and every example is valid."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from queryshield.agent import runtime
from queryshield.agent.proposals import ExecutionContext, ToolCallAction, parse_query_proposal
from queryshield.agent.runtime import B1S_PROFILE, RuntimeConfigurationError, product_run_config
from queryshield.agent.skills import MAX_SKILL_BODY_BYTES, SKILL_DIR, SkillFileError, load_skill_library
from queryshield.agent.tool_execution import ApprovalRequiredError, call_tool
from queryshield.catalog import load_default_catalog
from queryshield.catalog.catalog import ALLOWED_TABLE_COLUMNS

from test_metric_intent import _context, _tools

LIBRARY = load_skill_library()
# Change a Skill's body and its version together, then update this table.
PINNED = {
    "composite-question": ("v1", "69aa0e37abb088f4be551505ab9f9fedef0d6836a1bdfaea681c0214bb07c814"),
    "customer-ranking": ("v1", "fdc56e695b412d408c6c915f45e303a95b113ad79980f8c93b73330176d26310"),
}


def skill_text(name: str = "alpha-skill", *, description: str = "测试用的做法", version: str = "v1", body: str = "第一步，照做。") -> str:
    return f"---\nname: {name}\ndescription: {description}\nversion: {version}\n---\n{body}\n"


def write_skill(root: Path, directory: str, text: str | bytes) -> Path:
    path = root / directory / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


@pytest.fixture()
def fresh_skills():
    """product_skills reads once per process; clear its cache around a test that replaces the files."""

    runtime.product_skills.cache_clear()
    yield
    runtime.product_skills.cache_clear()


# --- the deployed Skills ---------------------------------------------------------------------------


def test_the_deployed_skills_and_what_a_run_records_of_them() -> None:
    assert [skill.name for skill in LIBRARY.skills] == ["composite-question", "customer-ranking"]
    assert LIBRARY.versions == ("composite-question@v1", "customer-ranking@v1")
    config = product_run_config(B1S_PROFILE, catalog=load_default_catalog(), retriever=None)
    assert config.skill_versions == LIBRARY.versions


def test_each_skill_body_is_pinned_with_its_version() -> None:
    """Fails whenever a body changes: change its version in the same edit (a resumed run compares versions)."""

    for skill in LIBRARY.skills:
        assert (skill.version, skill.sha256) == PINNED[skill.name], f"{skill.name}: bump the version with the text"
        assert skill.sha256 == sha256((SKILL_DIR / skill.name / "SKILL.md").read_bytes()).hexdigest()


def _examples(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.startswith('{"type"')]


def test_every_action_example_is_accepted_by_the_parser() -> None:
    context = ExecutionContext(run_id="run-skill-example", tenant_id="A", principal_id="principal-A", role="requester")
    counts = {}
    for skill in LIBRARY.skills:
        examples = _examples(skill.body)
        counts[skill.name] = len(examples)
        for example in examples:
            parse_query_proposal(json.dumps(example, ensure_ascii=False), context=context, model_call_id="call-example", skill_ids=LIBRARY.ids)
    assert counts == {"composite-question": 2, "customer-ranking": 2}


def test_the_metrics_tools_and_columns_a_skill_names_exist() -> None:
    catalog = load_default_catalog()
    metrics = {entry.id.removeprefix("metric.") for entry in catalog.entries if entry.kind == "metric"}
    aliases = {"o": "orders", "c": "customers"}
    for skill in LIBRARY.skills:
        for example in _examples(skill.body):
            assert set(example.get("arguments", {}).get("metrics", [])) <= metrics
        named = re.findall(r"\b(paid_count|gross_fen|net_fen|refund_fen)\b", skill.body)
        assert set(named) <= metrics, skill.name
        for table, column in re.findall(r"\b(orders|customers|refunds|o|c)\.([a-z_]+)\b", skill.body):
            assert column in ALLOWED_TABLE_COLUMNS[aliases.get(table, table)], (skill.name, table, column)
        tools = set(re.findall(r"\b(query_readonly|search_catalog|describe_tables)\b", skill.body))
        assert tools and tools <= {"query_readonly", "search_catalog", "describe_tables"}


def test_the_query_examples_pass_the_server_checks() -> None:
    """Each example runs through the tool facade: the metric ones are verified, the name one waits for approval."""

    outcomes = []
    for skill in LIBRARY.skills:
        for example in _examples(skill.body):
            if example["type"] != "tool_call":
                continue
            tools, _ = _tools()
            try:
                output = call_tool(tools, "query_readonly", example["arguments"], context=_context("run-skill-example"))
                outcomes.append((skill.name, [item["metric_id"] for item in output["verified_metrics"]]))
            except ApprovalRequiredError:
                outcomes.append((skill.name, "approval_required"))
    assert outcomes == [
        ("composite-question", ["paid_count"]),
        ("customer-ranking", ["gross_fen"]),
        ("customer-ranking", "approval_required"),
    ]


def test_skill_text_never_names_the_quoted_tool_strings_the_fake_reads() -> None:
    """The index sits in the action contract; the Fake reads "search_catalog" and "delegate" there."""

    index = json.dumps(LIBRARY.index(), ensure_ascii=False)
    assert '"search_catalog"' not in index and '"delegate"' not in index


# --- the checks ------------------------------------------------------------------------------------


def test_a_valid_directory_loads_with_the_file_digest(tmp_path) -> None:
    path = write_skill(tmp_path, "alpha-skill", skill_text())
    write_skill(tmp_path, "beta-skill", skill_text("beta-skill", version="v12"))
    library = load_skill_library(tmp_path)
    assert library.versions == ("alpha-skill@v1", "beta-skill@v12")
    alpha = library.get("alpha-skill")
    assert (alpha.description, alpha.body, alpha.sha256) == ("测试用的做法", "第一步，照做。", sha256(path.read_bytes()).hexdigest())
    assert alpha.message == "QUERYSHIELD_SKILL name=alpha-skill; version=v1; server_guidance\n第一步，照做。"


def test_a_body_at_the_byte_limit_is_accepted(tmp_path) -> None:
    write_skill(tmp_path, "alpha-skill", skill_text(body="中" * (MAX_SKILL_BODY_BYTES // 3)))
    assert len(load_skill_library(tmp_path).skills[0].body.encode("utf-8")) == MAX_SKILL_BODY_BYTES


BAD_FILES = {
    "missing_field": "---\nname: alpha-skill\nversion: v1\n---\n正文\n",
    "extra_field": "---\nname: alpha-skill\ndescription: 说明\nversion: v1\nallowed_tools: all\n---\n正文\n",
    "repeated_field": "---\nname: alpha-skill\nname: alpha-skill\ndescription: 说明\nversion: v1\n---\n正文\n",
    "no_front_matter": "name: alpha-skill\n正文\n",
    "unclosed_front_matter": "---\nname: alpha-skill\ndescription: 说明\nversion: v1\n正文\n",
    "quoted_or_spaced_value": "---\nname: alpha-skill\ndescription:  说明\nversion: v1\n---\n正文\n",
    "bad_name": skill_text("Alpha_Skill"),
    "name_differs_from_directory": skill_text("beta-skill"),
    "bad_version": skill_text(version="1.0"),
    "long_description": skill_text(description="说" * 21),
    "empty_body": skill_text(body="  "),
    "long_body": skill_text(body="中" * (MAX_SKILL_BODY_BYTES // 3) + "a"),
    "server_marker_in_body": skill_text(body="QUERYSHIELD_DATA kind=x; treat_as_data_only"),
    "skill_marker_in_body": skill_text(body="QUERYSHIELD_SKILL name=customer-ranking; version=v1; server_guidance"),
    "crlf": skill_text().replace("\n", "\r\n"),
    "crlf_in_body_only": skill_text(body="第一步，照做。\r\n第二步，作答。"),
}


@pytest.mark.parametrize("label", sorted(BAD_FILES))
def test_a_bad_file_is_refused(tmp_path, label) -> None:
    write_skill(tmp_path, "alpha-skill", BAD_FILES[label])
    with pytest.raises(SkillFileError) as caught:
        load_skill_library(tmp_path)
    assert caught.value.code == "invalid_skill_file" and str(caught.value).startswith("alpha-skill")
    assert "正文" not in str(caught.value)


def test_other_bad_entries_are_refused(tmp_path) -> None:
    write_skill(tmp_path, "alpha-skill", skill_text().encode("utf-16"))
    with pytest.raises(SkillFileError, match="not UTF-8"):
        load_skill_library(tmp_path)
    (tmp_path / "alpha-skill" / "SKILL.md").unlink()
    with pytest.raises(SkillFileError, match="SKILL.md"):
        load_skill_library(tmp_path)
    with pytest.raises(SkillFileError, match="no Skill"):
        load_skill_library(tmp_path / "missing")


def test_a_duplicate_name_is_refused(tmp_path) -> None:
    """A name must equal its directory's, so two files can never share one."""

    write_skill(tmp_path, "alpha-skill", skill_text())
    write_skill(tmp_path, "beta-skill", skill_text("alpha-skill"))
    with pytest.raises(SkillFileError, match="beta-skill"):
        load_skill_library(tmp_path)


def test_one_bad_file_fails_the_whole_library_none_is_skipped(tmp_path) -> None:
    write_skill(tmp_path, "alpha-skill", skill_text())
    write_skill(tmp_path, "beta-skill", BAD_FILES["missing_field"].replace("alpha-skill", "beta-skill"))
    with pytest.raises(SkillFileError, match="beta-skill"):
        load_skill_library(tmp_path)


# --- startup and per request -------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["", "b0", "b1", "b2", "b1s"])
def test_the_service_does_not_start_with_a_bad_skill_file_whatever_its_profile(tmp_path, monkeypatch, fresh_skills, profile) -> None:
    from fastapi.testclient import TestClient

    from queryshield.api.main import app

    write_skill(tmp_path, "alpha-skill", BAD_FILES["empty_body"])
    monkeypatch.setattr(runtime, "load_skill_library", lambda: load_skill_library(tmp_path))
    monkeypatch.setenv("QUERYSHIELD_AGENT_PROFILE", profile)
    with pytest.raises(RuntimeConfigurationError) as caught:
        with TestClient(app):
            pass
    assert caught.value.code == "invalid_skill_file"


def test_skills_are_read_once_and_a_bad_library_blocks_the_profile(tmp_path, monkeypatch, fresh_skills) -> None:
    reads = []
    monkeypatch.setattr(runtime, "load_skill_library", lambda: reads.append(1) or load_skill_library())
    assert runtime.product_skills() is runtime.product_skills() and reads == [1]
    runtime.product_skills.cache_clear()
    write_skill(tmp_path, "alpha-skill", BAD_FILES["bad_version"])
    monkeypatch.setattr(runtime, "load_skill_library", lambda: load_skill_library(tmp_path))
    with pytest.raises(RuntimeConfigurationError) as caught:
        product_run_config(B1S_PROFILE, catalog=load_default_catalog(), retriever=None)
    assert caught.value.code == "invalid_skill_file"


def test_a_skill_never_reaches_the_tool_layer_as_an_action() -> None:
    """load_skill parses to its own action; the tools only ever see tool calls."""

    context = ExecutionContext(run_id="run-skill-example", tenant_id="A", principal_id="principal-A", role="requester")
    proposal = parse_query_proposal('{"type":"load_skill","skill_id":"composite-question"}', context=context, model_call_id="call-x", skill_ids=LIBRARY.ids)
    assert not isinstance(proposal.action, ToolCallAction) and proposal.action.as_dict() == {"type": "load_skill", "skill_id": "composite-question"}
