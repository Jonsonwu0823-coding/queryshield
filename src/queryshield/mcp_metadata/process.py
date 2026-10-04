"""Whether a process the MCP server reported (its own pid) still exists.

No signals: ``os.kill(pid, 0)`` ends the process on Windows.  Linux reads
``/proc``; Windows opens the process with query rights and waits zero
milliseconds.  Other platforms answer ``None`` (unverified).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time


def process_exists(pid: int) -> bool | None:
    if type(pid) is not int or pid <= 0:
        return None
    if sys.platform.startswith("linux"):
        return _linux_exists(pid)
    if sys.platform == "win32":
        return _windows_exists(pid)
    return None


def wait_until_gone(pid: int, timeout: float) -> bool | None:
    """True once the process is gone, False if it is still there at the deadline."""

    deadline = time.monotonic() + timeout
    while True:
        exists = process_exists(pid)
        if exists is None:
            return None
        if not exists:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _linux_exists(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError:
        return True
    # The state follows the parenthesised command name; a zombie has exited.
    state = stat.rpartition(")")[2].split()
    return not (state and state[0] in {"Z", "X"})


def _windows_exists(pid: int) -> bool:  # pragma: no cover - exercised on the controller's Windows host
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    synchronize = 0x00100000
    wait_object_0 = 0x0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(process_query_limited_information | synchronize, False, pid)
    if not handle:
        # No such process (or no access to one that already ended).
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: it exists, owned by someone else
    try:
        return kernel32.WaitForSingleObject(handle, 0) != wait_object_0
    finally:
        kernel32.CloseHandle(handle)


__all__ = ["process_exists", "wait_until_gone"]
