"""Durable disk writes: fsync of files and directories, atomic replacement."""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from typing import BinaryIO

__all__ = ["atomic_write_bytes", "free_bytes", "fsync_dir", "fsync_file"]


def fsync_file(f: BinaryIO) -> None:
    """Flush Python and OS buffers of an open file."""
    f.flush()
    os.fsync(f.fileno())


def fsync_dir(path: Path) -> None:
    """Persist a directory entry (new names, renames).

    On Windows a directory cannot be opened for fsync; NTFS journals renames anyway,
    so the step is skipped there.
    """
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write a file completely or not at all.

    The data goes to a temporary file next to the target, is flushed to disk and then
    atomically replaces the target. A reader sees either the old version or the new one.
    """
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            fsync_file(f)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    fsync_dir(path.parent)


def free_bytes(path: Path) -> int:
    """Free space on the file system that holds ``path``."""
    return shutil.disk_usage(path).free
