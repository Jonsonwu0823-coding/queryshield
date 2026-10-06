"""Static checks of the public documentation (README and the docs a reader is pointed to).

The docs go into a public repository whose root is this directory, so every link and path they
use must exist here, they must not point at anything outside it, and they must not carry local
paths, credentials, private workflow vocabulary or acceptance tags that the evidence document
does not explain.  The README's quick start must be the operations document's quick start.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
DOCS = PROJECT_ROOT / "docs"
EVIDENCE = DOCS / "evidence.md"
OPERATIONS = DOCS / "operations.md"
PUBLIC_DOCS = (
    README,
    DOCS / "architecture.md",
    EVIDENCE,
    DOCS / "retrospective.md",
    OPERATIONS,
    DOCS / "mcp.md",
    DOCS / "demo-data.md",
)
DOC_IDS = [path.relative_to(PROJECT_ROOT).as_posix() for path in PUBLIC_DOCS]

# Private workflow vocabulary and tool names that must not appear in the public docs.
INTERNAL_TERMS = (
    "总控", "实现会话", "验收会话", "周目录", "learner", "LEARNING_STATE", "REVIEW.md", "ACTIVE_WORK", "找实习",
    "Claude", "Codex",
)
# The one placeholder a Windows command example may use; any other drive path is a local path.
WINDOWS_PLACEHOLDER = "C:\\path\\to\\"
LOCAL_PATHS = (
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"[A-Za-z]:/Users", re.IGNORECASE),
    re.compile(r"/Users/"),
    re.compile(r"/root/"),
    re.compile(r"/home/"),
)
WORKSPACE_ID = re.compile(r"\bws-[A-Za-z0-9]{4,}")  # only placeholders such as ws-<…> or ws-… are allowed
SECRET_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
# scheme://user:password@host -- the password may only be an angle-bracket placeholder such as <管理员密码>.
CREDENTIAL_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:'\"`]+:([^\s@'\"`]+)@")
PLACEHOLDER = re.compile(r"<[^<>]+>")
LINK = re.compile(r"\]\(([^)\s]+)\)")
# Repository paths written in backticks.  Package-relative module paths (`approval/service.py`)
# are resolved under src/queryshield/.
REPO_PATH = re.compile(r"`((?:src|scripts|tests|docs|migrations|fixtures|evals|deploy)/[^`\s]*)`")
PACKAGES = ("agent", "api", "approval", "auth", "catalog", "db", "evaluation", "facts", "knowledge",
            "mcp_metadata", "memory", "models", "policy", "providers", "tools")
MODULE_PATH = re.compile(r"`((?:" + "|".join(PACKAGES) + r")/[A-Za-z0-9_/]+\.py)`")
# "`path` 的 `symbol`" and "、`symbol`" continuations: the symbol must be in that file.
PATH_SYMBOLS = re.compile(r"`([^`\s]+\.(?:py|ps1))` 的 `([A-Za-z_][\w.]*)`((?:、`[A-Za-z_][\w.]*`)*)")
# "验收 H7", "实现 D-2", "验收 G1、G3".
_TAG = r"[A-Z]-?\d+(?![A-Za-z0-9-])"
ACCEPTANCE_TAG = re.compile(r"(?:验收|实现)\s*(" + _TAG + r")((?:\s*[、，,和及]\s*" + _TAG + r")*)")
TAG = re.compile(_TAG)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _doc(doc_id: str) -> Path:
    return PROJECT_ROOT / doc_id


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return False
    return True


def _section(text: str, heading: str) -> str:
    """The body of a `## ` section, up to the next `## ` heading."""

    start = text.index(heading)
    following = text.find("\n## ", start + len(heading))
    return text[start: following if following != -1 else len(text)]


def link_problems(path: Path, text: str | None = None) -> list[str]:
    problems = []
    for target in LINK.findall(_text(path) if text is None else text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        file_part = target.split("#", 1)[0]
        resolved = (path.parent / file_part)
        if not _inside_project(resolved):
            problems.append(f"{target}: leaves the project directory")
        elif resolved.name.startswith("w03-"):
            problems.append(f"{target}: the early design records are not linked from the public docs")
        elif not resolved.exists():
            problems.append(f"{target}: does not exist")
    return problems


def content_problems(text: str) -> list[str]:
    problems = []
    if "control/" in text:
        problems.append("mentions control/")
    if ".cloud/" in text:
        problems.append("mentions .cloud/")
    for password in CREDENTIAL_URL.findall(text):
        if not PLACEHOLDER.fullmatch(password):
            problems.append("a connection string with a literal password")
    for term in INTERNAL_TERMS:
        if term.casefold() in text.casefold():
            problems.append(f"internal term {term!r}")
    without_placeholder = text.replace(WINDOWS_PLACEHOLDER, "")
    for pattern in LOCAL_PATHS:
        match = pattern.search(without_placeholder)
        if match:
            problems.append(f"local path {match.group(0)!r}")
    for pattern, label in ((WORKSPACE_ID, "workspace id"), (SECRET_KEY, "key-shaped string")):
        match = pattern.search(text)
        if match:
            problems.append(f"{label} {match.group(0)!r}")
    return problems


def _resolve_repo_path(written: str) -> Path:
    candidate = written.rstrip("。，、；：,.;:)）")
    return PROJECT_ROOT / candidate


def _expand_braces(written: str) -> list[str]:
    """`a.{sql,md}` -> [`a.sql`, `a.md`] (one brace group, as the docs write it)."""

    match = re.search(r"\{([^{}]+)\}", written)
    if match is None:
        return [written]
    return [written[: match.start()] + option + written[match.end():] for option in match.group(1).split(",")]


def path_problems(text: str) -> list[str]:
    problems = []
    for written in REPO_PATH.findall(text):
        for expanded in _expand_braces(written):
            path = _resolve_repo_path(expanded)
            found = any(PROJECT_ROOT.glob(path.relative_to(PROJECT_ROOT).as_posix())) if "*" in expanded else path.exists()
            if not found:
                problems.append(f"{expanded}: does not exist")
    for written in MODULE_PATH.findall(text):
        if not (PROJECT_ROOT / "src" / "queryshield" / written).is_file():
            problems.append(f"{written}: no such module under src/queryshield/")
    for written, first, rest in PATH_SYMBOLS.findall(text):
        module = PROJECT_ROOT / written if (PROJECT_ROOT / written).is_file() else PROJECT_ROOT / "src" / "queryshield" / written
        if not module.is_file():
            continue  # reported above
        source = _text(module)
        for symbol in (first, *re.findall(r"`([^`]+)`", rest)):
            if symbol.split(".")[-1] not in source:
                problems.append(f"{symbol}: not found in {written}")
    return problems


def cited_tags(text: str) -> set[str]:
    tags: set[str] = set()
    for first, rest in ACCEPTANCE_TAG.findall(text):
        tags.add(first)
        tags.update(TAG.findall(rest))
    return tags


def mapped_tags() -> set[str]:
    table = _section(_text(EVIDENCE), "## 编号对照")
    return {match.group(1) for match in re.finditer(r"^\| ([A-Z]-?\d+) \|", table, re.MULTILINE)}


# --- the checks ---------------------------------------------------------------------------------


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_every_relative_link_points_to_a_file_in_the_project(doc_id: str) -> None:
    assert link_problems(_doc(doc_id)) == []


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_no_private_paths_terms_or_credentials(doc_id: str) -> None:
    assert content_problems(_text(_doc(doc_id))) == []


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_every_repository_path_and_cited_symbol_exists(doc_id: str) -> None:
    assert path_problems(_text(_doc(doc_id))) == []


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_every_cited_acceptance_tag_is_explained_in_the_evidence_mapping(doc_id: str) -> None:
    missing = cited_tags(_text(_doc(doc_id))) - mapped_tags()
    assert missing == set(), f"tags not in docs/evidence.md 编号对照: {sorted(missing)}"


def test_the_mapping_covers_the_operations_limits_at_least() -> None:
    assert {"K6", "H7", "H2", "G6", "N6", "D-2", "M11"} <= mapped_tags()


def test_readme_quick_start_is_the_operations_quick_start() -> None:
    readme = _text(README)
    for command in ("python scripts/new_env.py", "docker compose up -d --build", "demo_walkthrough.py"):
        assert command in readme
    block = re.search(r"```bash\n(.*?)```", _section(_text(OPERATIONS), "## 1. 用 Compose 跑起来"), re.DOTALL)
    assert block is not None
    for line in block.group(1).splitlines():
        command = line.split("#", 1)[0].strip()
        if command:
            assert command in readme, f"{command!r} from operations §1 is not in the README"


def test_operations_points_the_acceptance_tags_to_the_evidence_mapping() -> None:
    head = "\n".join(_text(OPERATIONS).splitlines()[:6])
    assert "evidence.md#编号对照" in head


# --- the checkers catch what they are meant to catch ---------------------------------------------------


def test_the_content_checker_flags_each_kind_of_problem() -> None:
    assert content_problems("见 control/STATE.md")
    assert content_problems("由总控决定")
    assert content_problems("written by Claude")
    assert content_problems("D:\\work\\queryshield")
    assert content_problems("/Users/someone/queryshield")
    assert content_problems("https://ws-abc123def.cn-beijing.maas.aliyuncs.com")
    assert content_problems("QUERYSHIELD_MODEL_API_KEY=sk-0123456789abcdef")
    assert content_problems("'C:\\path\\to\\x' 之外还有 C:\\Users\\me") != []
    assert content_problems("'C:\\path\\to\\evidence' 和 https://ws-<编号>.example 和 ws-…") == []
    assert content_problems("export URL='postgresql://queryshield:s3cret-pw@127.0.0.1:5433/queryshield_demo'")
    assert content_problems("source ../.cloud/env.sh")
    assert content_problems("export URL='postgresql://queryshield_ro:<只读角色密码>@127.0.0.1:5433/queryshield_demo'") == []


def test_the_path_and_tag_checkers_flag_unknown_targets() -> None:
    assert path_problems("`src/queryshield/not_a_module.py`")
    assert path_problems("`approval/not_here.py`")
    assert path_problems("`fixtures/demo/commerce-demo-v1.{sql,txt}`") == ["fixtures/demo/commerce-demo-v1.txt: does not exist"]
    assert path_problems("`src/queryshield/approval/service.py` 的 `no_such_symbol_xyz`")
    assert path_problems("`src/queryshield/approval/service.py` 的 `_approval_lock`、`build_pending_action`") == []
    assert cited_tags("验收 H7，实现 D-2，验收 G1、G3") == {"H7", "D-2", "G1", "G3"}


def test_the_link_checker_flags_missing_and_escaping_links() -> None:
    sample = "[a](missing.md) [b](../../outside.md) [c](operations.md#2-环境变量) [d](w03-a04-versioned-runtime.md)"
    problems = link_problems(DOCS / "sample.md", sample)
    assert len(problems) == 3
    assert any("does not exist" in item for item in problems)
    assert any("leaves the project" in item for item in problems)
    assert any("early design records" in item for item in problems)
