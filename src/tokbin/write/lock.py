"""``.lock``: one writer per partial directory (spec 5.4, 8.3).

The lock file is created with ``O_CREAT | O_EXCL``, so of two processes only one gets
it. It records the owner's pid, host and start time. A lock whose owner is gone (the
process crashed or was killed) is stale and is taken over; a lock held by a live
process, or by a process on another host whose liveness cannot be checked, is refused.

Checking a pid never sends a signal: ``os.kill(pid, 0)`` on POSIX, ``OpenProcess`` on
Windows (where ``os.kill`` would terminate the process).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import socket
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tokbin import _fs, codes
from tokbin.errors import ResumeError
from tokbin.format import naming
from tokbin.format._time import utc_timestamp

__all__ = ["LockInfo", "WriteLock", "pid_alive", "read_lock"]

#: Locks held by live writer objects of this process, by token. A lock that names this
#: process but is not here was abandoned (its writer failed without cleaning up).
_HELD: weakref.WeakValueDictionary[str, WriteLock] = weakref.WeakValueDictionary()

#: A lock file that cannot be parsed is treated as being created right now by another
#: process until it is this old: its owner writes the content right after creating it.
_UNREADABLE_GRACE_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True, kw_only=True)
class LockInfo:
    pid: int
    host: str
    started_at: str
    token: str

    @property
    def local(self) -> bool:
        return self.host == socket.gethostname()

    @property
    def alive(self) -> bool:
        """The owner is running, or may be (another host cannot be checked)."""
        if not self.local:
            return True
        if self.pid == os.getpid():
            return self.token in _HELD
        return pid_alive(self.pid)


def pid_alive(pid: int) -> bool:
    """Is a process with this pid running on this machine?"""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, but belongs to another user
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    # Windows-only ctypes parts are looked up dynamically so type checking works anywhere.
    ctypes = importlib.import_module("ctypes")
    query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except (AttributeError, OSError):
        return True  # cannot tell: assume alive, never steal a lock by mistake
    handle = kernel32.OpenProcess(query_limited_information, False, pid)
    if not handle:
        return bool(ctypes.get_last_error() == error_access_denied)
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return bool(code.value == still_active)
    finally:
        kernel32.CloseHandle(handle)


def _parse(raw: bytes) -> LockInfo | None:
    try:
        data = json.loads(raw.decode("utf-8"))
        return LockInfo(
            pid=int(data["pid"]),
            host=str(data["host"]),
            started_at=str(data["started_at"]),
            token=str(data["token"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


def read_lock(partial: Path) -> LockInfo | None:
    """The owner recorded in ``partial/.lock``; ``None`` without a readable lock."""
    try:
        raw = (partial / naming.LOCK).read_bytes()
    except OSError:
        return None
    return _parse(raw)


def _busy(partial: Path, owner: LockInfo | None) -> ResumeError:
    if owner is None:
        why = "another process is creating the lock right now"
    elif owner.local:
        why = f"process {owner.pid} has been writing it since {owner.started_at}"
    else:
        why = (
            f"process {owner.pid} on host {owner.host!r} has been writing it since "
            f"{owner.started_at}; a process on another host cannot be checked"
        )
    return ResumeError(
        codes.WRITER_ACTIVE,
        str(partial),
        why=why,
        fix=f"wait for that write to finish; if no such process is running, delete "
        f"{partial / naming.LOCK}",
    )


class WriteLock:
    """The lock of one partial directory, held by this process."""

    __slots__ = ("__weakref__", "_info", "_path")

    def __init__(self, partial: Path) -> None:
        self._path = partial / naming.LOCK
        self._info = LockInfo(
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=utc_timestamp(),
            token=uuid.uuid4().hex,
        )

    def _content(self) -> bytes:
        info = self._info
        data = {
            "pid": info.pid,
            "host": info.host,
            "started_at": info.started_at,
            "token": info.token,
        }
        return json.dumps(data).encode("utf-8")

    def acquire(self) -> None:
        """Take the lock, taking over a stale one; ``ResumeError`` if it is held."""
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            self._take_over()
            return
        try:
            os.write(fd, self._content())
            os.fsync(fd)
        finally:
            os.close(fd)
        _HELD[self._info.token] = self

    def _take_over(self) -> None:
        partial = self._path.parent
        try:
            raw = self._path.read_bytes()
            age = time.time() - self._path.stat().st_mtime
        except FileNotFoundError:  # released in the meantime
            self.acquire()
            return
        owner = _parse(raw)
        if owner is None:
            if age < _UNREADABLE_GRACE_SECONDS:
                raise _busy(partial, None)
        elif owner.alive:
            raise _busy(partial, owner)
        # Stale: replace atomically, then make sure no other process replaced it too.
        _fs.atomic_write_bytes(self._path, self._content())
        if read_lock(partial) != self._info:
            raise _busy(partial, read_lock(partial))
        _HELD[self._info.token] = self

    def release(self) -> None:
        """Remove the lock if it is still ours."""
        _HELD.pop(self._info.token, None)
        if read_lock(self._path.parent) == self._info:
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()
