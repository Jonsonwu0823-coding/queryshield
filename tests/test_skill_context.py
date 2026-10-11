"""What the skills profile sends the model: the index in the server context, loaded Skills as system messages.

B0, B1 and B2 contexts are pinned elsewhere (tests/test_shared_runtime.py, tests/test_native_runtime.py,
tests/agent_core_pins.json); here they only must not mention Skills.
"""

from __future__ import annotations

import itertools
import json

import pytest

import test_metric_intent
from queryshield.agent.context import MAX_CONTEXT_BYTES, MAX_SERVER_CONTEXT_CHARS, build_context
from queryshield.agent.proposals import ExecutionContext
from queryshield.agent.runtime import B1S_PROFILE, B2_PROFILE, RuntimeConfigurationError, product_run_config, product_skills
from queryshield.agent.skills import MAX_SKILL_BODY_BYTES, Skill, SkillLibrary
from queryshield.agent.config import NATIVE_VERSIONS, RunConfig
from queryshield.catalog import load_default_catalog

from test_delegate_action import SEP, _http_identity

CATALOG = load_default_catalog()
SKILLS = product_skills()
CONFIG = product_run_config(B1S_PROFILE, catalog=CATALOG, retriever=None)
SERVER_PREFIX = "QUERYSHIELD_SERVER_CONTEXT\n"


def _contract(messages) -> dict:
    return json.loads(messages[0]["content"][len(SERVER_PREFIX):])["action_contract"]


def skills_context(*, loaded=(), **options):
    return build_context(_http_identity(), "q", run_config=CONFIG, parallel_available=False, skills=SKILLS, loaded_skills=loaded, **options)


def system_messages_are_trusted(messages, library: SkillLibrary) -> bool:
    """Every system message is the server context or, byte for byte, one registered Skill's message; none carries data."""

    known = {skill.message for skill in library.skills}
    systems = [message["content"] for message in messages if message["role"] == "system"]
    return (
        bool(systems)
        and systems[0].startswith(SERVER_PREFIX)
        and all(text in known for text in systems[1:])
        and not any("QUERYSHIELD_DATA" in text for text in systems)
        and [message["role"] for message in messages[: len(systems)]] == ["system"] * len(systems)
    )


def test_only_the_skills_profile_is_told_about_skills() -> None:
    contract = _contract(skills_context().messages)
    assert contract["actions"]["load_skill"] == {
        "required_fields": ["type", "skill_id"],
        "skills": {"composite-question": "一题问多个指标或多个时间段", "customer-ranking": "按客户汇总、排名或要客户姓名"},
        "rule": "At most 2 per run, each once.",
    }
    assert "deny or load_skill;" in contract["actions"]["tool_call"]["critical_wire_rules"][0]
    runtime_versions = json.loads(skills_context().messages[0]["content"][len(SERVER_PREFIX):])["runtime_versions"]
    assert runtime_versions["skill_versions"] == ["composite-question@v1", "customer-ranking@v1"]
    b2 = product_run_config(B2_PROFILE, catalog=CATALOG, retriever=None)
    others = [
        build_context(_http_identity(), "q"),
        build_context(_http_identity(), "q", run_config=b2, parallel_available=False, delegate_available=True),
        build_context(_http_identity(), "q", run_config=RunConfig(**NATIVE_VERSIONS)),
    ]
    for built in others:
        text = json.dumps(built.messages, ensure_ascii=False)
        assert "load_skill" not in text and "QUERYSHIELD_SKILL" not in text and "composite-question" not in text


ROUTING_RULE = "Load the Skill fitting the question before querying; follow its text."


def test_only_the_skills_profile_is_told_to_load_a_fitting_skill_first() -> None:
    """The index alone was never used by a real model: the routing rule leads the skills profile's workflow."""

    assert _contract(skills_context().messages)["workflow"][0] == ROUTING_RULE
    assert _contract(skills_context(retrieval_available=False).messages)["workflow"][0] == ROUTING_RULE
    b2 = product_run_config(B2_PROFILE, catalog=CATALOG, retriever=None)
    for built in (
        build_context(_http_identity(), "q"),
        build_context(_http_identity(), "q", parallel_available=False),
        build_context(_http_identity(), "q", run_config=b2, parallel_available=False, delegate_available=True),
        build_context(_http_identity(), "q", run_config=RunConfig(**NATIVE_VERSIONS)),
    ):
        assert ROUTING_RULE not in json.dumps(built.messages, ensure_ascii=False)


def test_the_skills_profiles_worst_server_message_stays_within_the_policy_limit() -> None:
    """HTTP only (uuid run ids), never parallel, every other length parameter at its worst, as B2's test."""

    december = {"start": "2026-12-01T00:00:00Z", "end": "2027-01-01T00:00:00Z", "timezone": "UTC"}
    sizes = []
    for retrieval, window, count, confirmed in itertools.product(
        (True, False), (None, {**SEP, "timezone": "UTC"}, december), range(4), (None, "paid_count", "gross_fen", "net_fen")
    ):
        bindings = [
            {"metric_id": m, "result_position": m, "unit": "CNY_fen", "time_window": window or {**SEP, "timezone": "UTC"}}
            for m in ("paid_count", "gross_fen", "net_fen")[:count]
        ]
        built = skills_context(
            request_time_window=window, metric_bindings=bindings, confirmed_metric=confirmed, time_window=window, retrieval_available=retrieval
        )
        sizes.append(len(built.messages[0]["content"]))
    assert len(sizes) == 2 * 3 * 4 * 4
    assert max(sizes) <= MAX_SERVER_CONTEXT_CHARS - 500 == 11_500, max(sizes)


def _with_skills(loaded):
    """The B1 whole-context worst case (tests/test_metric_intent.py), rebuilt under the skills profile over HTTP."""

    built = []

    def build(context, question, **options):
        options.update(run_config=CONFIG, parallel_available=False, skills=SKILLS, loaded_skills=loaded)
        built.append(build_context(_http_identity(), question, **options))
        return built[-1]

    return build, built


@pytest.mark.parametrize("which", ["deployed", "at_the_limit"])
def test_the_whole_worst_context_with_two_loaded_skills_fits_without_dropping(monkeypatch, which) -> None:
    if which == "deployed":
        loaded = tuple(sorted(SKILLS.skills, key=lambda skill: len(skill.message.encode()), reverse=True)[:2])
    else:
        body = "中" * (MAX_SKILL_BODY_BYTES // 3)
        loaded = tuple(Skill(f"limit-skill-{n}x", "说" * 20, "v999", body, "0" * 64) for n in range(2))
    build, built = _with_skills(loaded)
    monkeypatch.setattr(test_metric_intent, "build_context", build)
    test_metric_intent.test_worst_case_b1_context_fits_total_byte_budget()  # asserts <= 24,000 bytes and nothing dropped
    (result,) = built
    assert result.serialized_bytes <= MAX_CONTEXT_BYTES and result.dropped_optional_ids == ()
    assert [message["role"] for message in result.messages[:3]] == ["system", "system", "system"]
    assert system_messages_are_trusted(result.messages, SkillLibrary(loaded))


def test_loaded_skills_follow_the_server_context_and_are_never_trimmed() -> None:
    first, second = SKILLS.skills
    big = [{"tool_name": "query_readonly", "status": "succeeded", "output": {"rows": [{"x": "y" * 7_000}]}} for _ in range(4)]
    built = skills_context(loaded=(second, first), tool_results=big)
    assert built.messages[1] == {"role": "system", "content": second.message}
    assert built.messages[2] == {"role": "system", "content": first.message}
    assert built.messages[3]["role"] == "user" and built.messages[3]["content"] == "q"
    assert built.hard_message_ids[:3] == ("server-context", f"skill-{second.name}", f"skill-{first.name}")
    assert built.dropped_optional_ids  # tool results go first; the Skills stay
    assert second.message.startswith("QUERYSHIELD_SKILL name=customer-ranking; version=v1; server_guidance\n")
    assert system_messages_are_trusted(built.messages, SKILLS)


def test_the_trust_check_refuses_a_tool_result_as_a_system_message() -> None:
    built = skills_context(loaded=SKILLS.skills[:1], tool_results=[{"tool_name": "query_readonly", "status": "succeeded"}])
    messages = [dict(message) for message in built.messages]
    assert system_messages_are_trusted(messages, SKILLS)
    data = next(message for message in messages if message["content"].startswith("QUERYSHIELD_DATA"))
    data["role"] = "system"
    assert not system_messages_are_trusted(messages, SKILLS)


def test_the_skills_profiles_configuration(monkeypatch) -> None:
    assert (CONFIG.profile, CONFIG.prompt_version, CONFIG.action_schema_version) == (B1S_PROFILE, "qs-skills-prompt-v2", "qs-skills-schema-v1")
    assert CONFIG.model_protocol == "json" and CONFIG.skill_versions == SKILLS.versions
    assert RunConfig.from_dict(CONFIG.as_dict()) == CONFIG
    monkeypatch.setenv("QUERYSHIELD_MODEL_PROTOCOL", "native")
    with pytest.raises(RuntimeConfigurationError) as caught:
        product_run_config(B1S_PROFILE, catalog=CATALOG, retriever=None)
    assert caught.value.code == "invalid_model_protocol"


def test_the_identity_in_a_skills_context_is_the_servers() -> None:
    identity = ExecutionContext(run_id="run-skill-identity", tenant_id="A", principal_id="principal-A", role="requester")
    built = build_context(identity, "q", run_config=CONFIG, parallel_available=False, skills=SKILLS, loaded_skills=SKILLS.skills)
    payload = json.loads(built.messages[0]["content"][len(SERVER_PREFIX):])
    assert payload["authenticated_execution_context"] == {"run_id": "run-skill-identity", "tenant_id": "A", "principal_id": "principal-A", "role": "requester"}
