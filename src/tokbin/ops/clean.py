"""Removal of unfinished writes (``tokbin clean``, spec 14.2).

Only what tokbin itself left behind is removed: the ``<name>.partial/`` directory of an
interrupted write and hidden ``.<name>.old-<pid>`` directories of a replacement that
was interrupted after the new version was already in place. A partial directory that a
live process is writing is never touched.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from tokbin._boundary import public_api
from tokbin.format import naming
from tokbin.ops.inspect import _target_of
from tokbin.write.lock import WriteLock
from tokbin.write.partial import partial_path

__all__ = ["CleanResult", "clean_source"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CleanResult:
    """What was removed. ``removed`` is empty when there was nothing to clean."""

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
    """Bytes of data; the lock file taken for the removal itself does not count."""
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name != naming.LOCK
    )


@public_api
def clean_source(path: str | Path) -> CleanResult:
    """Remove the unfinished write of the source at ``path`` (or ``path.partial``).

    ``ResumeError`` if another process is writing it.
    """
    target = _target_of(Path(path))
    naming.check_source_name(target.name)
    removed: list[Path] = []
    n_bytes = 0

    partial = partial_path(target)
    if partial.is_dir():
        lock = WriteLock(partial)
        lock.acquire()  # refuses a live writer; takes over a stale lock
        n_bytes += _size(partial)
        try:
            shutil.rmtree(partial)
        finally:
            lock.release()
        removed.append(partial)

    if target.is_dir():  # the new version is in place: old copies are garbage
        for old in sorted(target.parent.glob(f".{target.name}.old-*")):
            if old.is_dir() and not old.is_symlink():
                n_bytes += _size(old)
                shutil.rmtree(old)
                removed.append(old)
    return CleanResult(path=target, removed=tuple(removed), n_bytes=n_bytes)
