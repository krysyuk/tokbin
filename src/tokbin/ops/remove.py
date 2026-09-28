"""Removing a source (``tokbin rm``, spec 14.2).

Only a tokbin source is removed: a directory with ``meta.json`` or ``dataset.json``,
not a symlink, not a corpus. The source first disappears under a hidden name in one
rename, so it never exists half-deleted; then the files are deleted. Its unfinished
write, if any, goes too, unless a live process is writing it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import ConfigError
from tokbin.format import naming
from tokbin.ops.inspect import _target_of
from tokbin.write.lock import WriteLock
from tokbin.write.partial import partial_path

__all__ = ["RemoveResult", "remove_source"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoveResult:
    path: Path
    removed: tuple[Path, ...]
    n_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "removed": [str(p) for p in self.removed],
            "n_bytes": self.n_bytes,
        }


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def source_size(path: str | Path) -> int:
    """Bytes a removal would free (the source and its unfinished write)."""
    target = _target_of(Path(path))
    return sum(_size(p) for p in (target, partial_path(target)) if p.is_dir())


@public_api
def remove_source(path: str | Path) -> RemoveResult:
    """Delete the source at ``path`` and its unfinished write. There is no undo."""
    target = _target_of(Path(path))
    naming.check_source_name(target.name)
    partial = partial_path(target)
    if target.is_symlink():
        raise ConfigError(
            codes.NOT_A_SOURCE,
            str(target),
            why="it is a symbolic link; removing through links is refused",
            fix="remove the link itself, or pass the real directory",
        )
    is_source = (target / naming.META_JSON).is_file() or (target / naming.DATASET_JSON).is_file()
    if target.exists() and not is_source:
        raise ConfigError(
            codes.NOT_A_SOURCE,
            str(target),
            why=f"it has no {naming.META_JSON}; only tokbin sources are removed",
            fix="check the path; to remove a corpus, remove its sources one by one",
        )
    if not target.exists() and not partial.is_dir():
        raise ConfigError(
            codes.SOURCE_NOT_FOUND,
            str(target),
            why="neither the source nor an unfinished write of it exists",
            fix="check the path",
        )

    removed: list[Path] = []
    n_bytes = 0
    lock: WriteLock | None = None
    if partial.is_dir():
        lock = WriteLock(partial)
        lock.acquire()  # a live writer blocks the whole removal
    try:
        if target.exists():
            n_bytes += _size(target)
            doomed = target.with_name(f".{target.name}.rm-{os.getpid()}")
            target.rename(doomed)  # the source disappears at once
            shutil.rmtree(doomed)
            removed.append(target)
        if partial.is_dir():
            n_bytes += _size(partial)
            shutil.rmtree(partial)
            removed.append(partial)
    finally:
        if lock is not None:
            lock.release()
    return RemoveResult(path=target, removed=tuple(removed), n_bytes=n_bytes)
