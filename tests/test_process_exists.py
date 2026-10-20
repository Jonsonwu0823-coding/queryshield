"""process_exists on the host: a running process exists, an exited one does not, a zombie has exited."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from queryshield.mcp_metadata.process import process_exists


def test_this_process_exists() -> None:
    assert process_exists(os.getpid()) is True


def test_a_reaped_child_is_gone() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert process_exists(child.pid) is False


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="needs os.waitid to leave a zombie (POSIX)")
def test_an_exited_child_not_yet_reaped_is_gone() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)  # exited, still a zombie
        assert process_exists(child.pid) is False
    finally:
        child.wait()


@pytest.mark.parametrize("pid", [0, -1, "1", 1.0, True])
def test_not_a_pid(pid: object) -> None:
    assert process_exists(pid) is None  # type: ignore[arg-type]
