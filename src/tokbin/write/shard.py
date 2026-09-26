"""A shard file being written (spec 8.2).

The shard is written as ``<name>.bin.open``. Closing flushes and fsyncs it, renames it
to ``<name>.bin`` and fsyncs the directory, so a ``.bin`` file is always complete.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tokbin import _fs
from tokbin.format import naming
from tokbin.format.meta import ShardInfo

__all__ = ["ShardFile"]


class ShardFile:
    __slots__ = ("_dir", "_f", "_hash", "_itemsize", "index", "n_items", "split")

    def __init__(self, directory: Path, split: str, index: int, itemsize: int) -> None:
        self._dir = directory
        self.split = split
        self.index = index
        self._itemsize = itemsize
        self.n_items = 0
        self._hash = hashlib.sha256()
        # "x": never silently overwrite a file that is already there.
        self._f = (directory / naming.open_shard_name(split, index)).open("xb")

    @property
    def name(self) -> str:
        return naming.shard_name(self.split, self.index)

    def write(self, items: npt.NDArray[np.unsignedinteger]) -> None:
        """Append items; the array must already have the little-endian storage dtype."""
        data = memoryview(np.ascontiguousarray(items)).cast("B")
        self._f.write(data)
        self._hash.update(data)
        self.n_items += items.size

    def close(self) -> ShardInfo:
        """Make the shard durable and give it its final name."""
        _fs.fsync_file(self._f)
        self._f.close()
        open_path = self._dir / naming.open_shard_name(self.split, self.index)
        open_path.replace(self._dir / self.name)
        _fs.fsync_dir(self._dir)
        return ShardInfo(
            name=self.name,
            n_items=self.n_items,
            n_bytes=self.n_items * self._itemsize,
            sha256=self._hash.hexdigest(),
        )

    def discard(self) -> None:
        """Drop an unfinished shard."""
        self._f.close()
        with contextlib.suppress(FileNotFoundError):
            (self._dir / naming.open_shard_name(self.split, self.index)).unlink()
