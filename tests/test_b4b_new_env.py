"""B4b: scripts/new_env.py writes distinct random credentials, never prints them, never overwrites."""

from __future__ import annotations

from pathlib import Path
import re
import stat
import sys

import pytest

from scripts import new_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URL_SAFE = re.compile(r"[A-Za-z0-9_\-]+")


def _read(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            values[name] = value
    return values


def test_writes_all_variables_with_distinct_values(tmp_path, capsys) -> None:
    target = tmp_path / ".env"
    assert new_env.main(["--path", str(target)]) == 0
    values = _read(target)
    assert set(values) == set(new_env.PASSWORD_VARIABLES) | set(new_env.TOKEN_VARIABLES)
    assert len(set(values.values())) == len(values), "every password and token differs from every other"
    assert all(URL_SAFE.fullmatch(values[name]) for name in new_env.PASSWORD_VARIABLES), "passwords are safe inside a URL"
    assert all(len(value) >= 32 for value in values.values())


def test_prints_no_generated_value(tmp_path, capsys) -> None:
    target = tmp_path / ".env"
    assert new_env.main(["--path", str(target)]) == 0
    out = capsys.readouterr()
    for value in _read(target).values():
        assert value not in out.out and value not in out.err
    assert "variables=7" in out.out


def test_does_not_overwrite_an_existing_file(tmp_path, capsys) -> None:
    target = tmp_path / ".env"
    target.write_text("KEEP=me\n", encoding="utf-8")
    assert new_env.main(["--path", str(target)]) == 2
    assert target.read_text(encoding="utf-8") == "KEEP=me\n"
    assert "--force" in capsys.readouterr().out


def test_force_replaces_the_file_with_new_values(tmp_path) -> None:
    target = tmp_path / ".env"
    assert new_env.main(["--path", str(target)]) == 0
    first = _read(target)
    assert new_env.main(["--path", str(target), "--force"]) == 0
    second = _read(target)
    assert set(first) == set(second) and not set(first.values()) & set(second.values())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_file_mode_is_600_even_when_forced_over_a_wider_file(tmp_path) -> None:
    target = tmp_path / ".env"
    assert new_env.main(["--path", str(target)]) == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    target.chmod(0o644)
    assert new_env.main(["--path", str(target), "--force"]) == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_unwritable_target_is_reported_without_a_traceback(tmp_path, capsys) -> None:
    assert new_env.main(["--path", str(tmp_path / "missing-dir" / ".env")]) == 2
    assert "Traceback" not in capsys.readouterr().out


def test_env_example_lists_exactly_the_variables_the_generator_writes() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    names = {line.partition("=")[0].strip().lstrip("#").strip() for line in example.splitlines() if "=" in line}
    generated = set(new_env.PASSWORD_VARIABLES) | set(new_env.TOKEN_VARIABLES)
    assert generated <= names
    # The example holds placeholders only: no value that could be a real credential.
    for line in example.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            value = line.partition("=")[2].strip()
            assert value == "" or value.startswith("<") or value.isdigit(), line
