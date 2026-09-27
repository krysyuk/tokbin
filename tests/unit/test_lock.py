from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tokbin import ResumeError, codes
from tokbin.write.lock import WriteLock, pid_alive, read_lock


def test_pid_alive() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(-5)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not pid_alive(proc.pid)


def test_acquire_and_release(tmp_path: Path) -> None:
    lock = WriteLock(tmp_path)
    lock.acquire()
    owner = read_lock(tmp_path)
    assert owner is not None
    assert owner.pid == os.getpid()
    assert owner.alive
    with pytest.raises(ResumeError) as err:
        WriteLock(tmp_path).acquire()
    assert err.value.code is codes.WRITER_ACTIVE
    lock.release()
    assert not (tmp_path / ".lock").exists()
    lock.release()  # idempotent


def test_release_keeps_a_lock_taken_over_by_someone_else(tmp_path: Path) -> None:
    lock = WriteLock(tmp_path)
    lock.acquire()
    (tmp_path / ".lock").write_text('{"pid": 1, "host": "h", "started_at": "x", "token": "t"}')
    lock.release()
    assert (tmp_path / ".lock").exists()


def test_owner_that_is_another_process(tmp_path: Path) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        import json
        import socket

        data = {"pid": child.pid, "host": socket.gethostname(), "started_at": "x", "token": "t"}
        (tmp_path / ".lock").write_text(json.dumps(data))
        with pytest.raises(ResumeError):
            WriteLock(tmp_path).acquire()
    finally:
        child.kill()
        child.wait()
    lock = WriteLock(tmp_path)
    lock.acquire()  # the owner died: stale
    lock.release()
