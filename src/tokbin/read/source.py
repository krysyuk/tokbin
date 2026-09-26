"""``read_source``: random access to one split of a source (spec 17).

::

    src = read_source("corpus/code")
    len(src)                                # items
    src.window(start, length)               # a window across shard boundaries
    src.sample_windows(n, block_size, rng)  # (n, block_size)
    src.doc(i)                              # a whole document
    src.id_of_doc(i)                        # its name, O(1)

Shards are memory-mapped lazily; nothing proportional to the corpus is loaded. Finding
the shard of a position is a binary search over shard starts: ``O(log n_shards)``.

A ``Source`` can be pickled (DataLoader workers): only the path and the split travel,
and the receiving process opens and checks the files again.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any, overload

import numpy as np
import numpy.typing as npt

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import ConfigError, IntegrityError, OutOfRangeError
from tokbin.format import naming
from tokbin.format.meta import Meta, SplitMeta, read_meta
from tokbin.format.schema import source_kind
from tokbin.read._index import corrupt_index, open_index

__all__ = ["Source", "read_source"]

TokenArray = npt.NDArray[np.unsignedinteger[Any]]

_FIX_SHARD = "download the shard again and run `tokbin verify {root}`"


@public_api
def read_source(path: str | Path, split: str = "train") -> Source:
    """Open one split of a single-stream source for reading."""
    return Source(path, split)


def _out_of_range(what: str, why: str) -> OutOfRangeError:
    return OutOfRangeError(
        codes.OUT_OF_RANGE, what, why=why, fix="check the position against len() / n_docs"
    )


def _as_index(value: object, name: str) -> int:
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise ConfigError(
            codes.CONFIG_VALUE_INVALID,
            f"{name}={value!r}",
            why=f"{name} must be an integer",
            fix=f"pass an int as {name}",
        ) from None


class Source:
    """One split of a single-stream source, opened for reading."""

    __slots__ = (
        "_cum",
        "_ids_idx",
        "_ids_map",
        "_maps",
        "_offsets",
        "_shard_paths",
        "meta",
        "path",
        "split",
    )

    def __init__(self, path: str | Path, split: str = "train") -> None:
        self._open(Path(path), split)

    def _open(self, root: Path, split: str) -> None:
        self.path = root
        self.split = split
        source_kind(root)
        self.meta: Meta = read_meta(root)
        info = self._split_meta()

        self._shard_paths: tuple[Path, ...] = tuple(
            self._check_shard(naming.resolve_inside(root, s.name), s.n_bytes) for s in info.shards
        )
        self._cum: npt.NDArray[np.int64] = np.cumsum(
            [0, *(s.n_items for s in info.shards)], dtype=np.int64
        )
        self._maps: list[np.memmap[Any, Any] | None] = [None] * len(info.shards)

        self._offsets = open_index(
            naming.resolve_inside(root, naming.offsets_name(split)),
            length=info.n_docs + 1,
            first=0,
            last=info.n_items,
        )
        ids_path = naming.resolve_inside(root, naming.ids_name(split))
        ids_size = ids_path.stat().st_size if ids_path.is_file() else -1
        self._ids_idx = open_index(
            naming.resolve_inside(root, naming.ids_idx_name(split)),
            length=info.n_docs + 1,
            first=0,
            last=ids_size,
        )
        self._ids_map: np.memmap[Any, Any] | None = None

    def _split_meta(self) -> SplitMeta:
        try:
            return self.meta.split(self.split)
        except KeyError:
            available = ", ".join(self.meta.split_names)
            raise ConfigError(
                codes.SPLIT_NOT_FOUND,
                f"{self.path} [{self.split}]",
                why=f"the source has the splits: {available}",
                fix="pass one of the available splits",
            ) from None

    def _check_shard(self, path: Path, n_bytes: int) -> Path:
        n_shards = len(self._split_meta().shards)
        if not path.is_file():
            raise IntegrityError(
                codes.SHARD_MISSING,
                str(path),
                why=f"meta.json lists {n_shards} shards for {self.split!r}; "
                "this one is not on disk",
                fix=_FIX_SHARD.format(root=self.path),
            )
        size = path.stat().st_size
        if size != n_bytes:
            raise IntegrityError(
                codes.SHARD_WRONG_SIZE,
                str(path),
                why=f"meta.json expects {n_bytes} bytes, the file has {size}",
                fix=_FIX_SHARD.format(root=self.path),
            )
        return path

    # --- pickling ---------------------------------------------------------------------

    def __getstate__(self) -> dict[str, object]:
        return {"path": str(self.path), "split": self.split}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._open(Path(state["path"]), state["split"])

    def __repr__(self) -> str:
        return f"Source({str(self.path)!r}, split={self.split!r}, n_items={len(self)})"

    # --- sizes ------------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of items in the split."""
        return int(self._cum[-1])

    @property
    def n_docs(self) -> int:
        return int(self._offsets.shape[0]) - 1

    @property
    def n_shards(self) -> int:
        return len(self._shard_paths)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.meta.np_dtype

    # --- items ------------------------------------------------------------------------

    def _shard(self, i: int) -> np.memmap[Any, Any]:
        mapped = self._maps[i]
        if mapped is None:
            mapped = np.memmap(self._shard_paths[i], dtype=self.dtype, mode="r")
            self._maps[i] = mapped
        return mapped

    def _read(self, start: int, length: int) -> TokenArray:
        """Items ``[start, start + length)``; bounds already checked."""
        if length == 0:
            return np.empty(0, dtype=self.dtype)
        first = int(np.searchsorted(self._cum, start, side="right")) - 1
        local = start - int(self._cum[first])
        shard = self._shard(first)
        if local + length <= shard.shape[0]:
            # Inside one shard: a view, no copy.
            return np.asarray(shard[local : local + length])
        out = np.empty(length, dtype=self.dtype)
        filled, i = 0, first
        while filled < length:
            shard = self._shard(i)
            take = min(shard.shape[0] - local, length - filled)
            out[filled : filled + take] = shard[local : local + take]
            filled += take
            i, local = i + 1, 0
        return out

    @public_api
    def window(self, start: int, length: int) -> TokenArray:
        """``length`` items from ``start``, across shard boundaries if needed."""
        start = _as_index(start, "start")
        length = _as_index(length, "length")
        n = len(self)
        if start < 0 or length < 0 or start + length > n:
            raise _out_of_range(
                f"window({start}, {length})",
                f"the split {self.split!r} has {n} items; a window must lie in [0, {n}]",
            )
        return self._read(start, length)

    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> TokenArray: ...
    def __getitem__(self, key: int | slice) -> int | TokenArray:
        """``src[i]`` is one item; ``src[a:b]`` materializes a slice (it may be large)."""
        n = len(self)
        if isinstance(key, slice):
            positions = range(*key.indices(n))
            if not positions:
                return self._read(0, 0)
            lo, hi = min(positions[0], positions[-1]), max(positions[0], positions[-1])
            block = self._read(lo, hi - lo + 1)
            if positions.step == 1:
                return block
            if positions.step > 0:
                return block[:: positions.step]
            return block[::-1][:: -positions.step]
        index = _as_index(key, "index")
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise _out_of_range(f"[{key}]", f"the split {self.split!r} has {n} items")
        return int(self._read(index, 1)[0])

    @public_api
    def sample_windows(
        self, n: int, block_size: int, rng: np.random.Generator
    ) -> npt.NDArray[np.unsignedinteger[Any]]:
        """``n`` windows of ``block_size`` items at uniform random starts: ``(n, block_size)``.

        Randomness comes only from ``rng``; the global numpy generator is never used.
        """
        if not isinstance(rng, np.random.Generator):
            raise ConfigError(
                codes.RNG_REQUIRED,
                type(rng).__qualname__,
                why="sampling takes its randomness from an explicit generator",
                fix="pass rng=np.random.default_rng(seed)",
            )
        n = _as_index(n, "n")
        block_size = _as_index(block_size, "block_size")
        if n < 0 or block_size < 1:
            raise ConfigError(
                codes.CONFIG_VALUE_INVALID,
                f"n={n}, block_size={block_size}",
                why="n must be >= 0 and block_size >= 1",
                fix="pass positive sizes",
            )
        total = len(self)
        if block_size > total:
            raise ConfigError(
                codes.WINDOW_TOO_LARGE,
                f"block_size={block_size}",
                why=f"the split {self.split!r} has only {total} items",
                fix="use a smaller block_size",
            )
        starts = rng.integers(0, total - block_size + 1, size=n)
        out = np.empty((n, block_size), dtype=self.dtype)
        for row, start in enumerate(starts):
            out[row] = self._read(int(start), block_size)
        return out

    # --- documents --------------------------------------------------------------------

    def _doc_index(self, i: object) -> int:
        index = _as_index(i, "i")
        n = self.n_docs
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise _out_of_range(f"document {i}", f"the split {self.split!r} has {n} documents")
        return index

    @public_api
    def doc(self, i: int) -> TokenArray:
        """All items of document ``i`` (separators such as EOS included)."""
        index = self._doc_index(i)
        start, end = int(self._offsets[index]), int(self._offsets[index + 1])
        if not 0 <= start <= end <= len(self):
            raise corrupt_index(
                self.path / naming.offsets_name(self.split),
                f"document {index} spans [{start}, {end}) outside the split",
            )
        return self._read(start, end - start)

    @public_api
    def id_of_doc(self, i: int) -> str | None:
        """The name of document ``i``, or ``None`` if documents were written without ids."""
        index = self._doc_index(i)
        start, end = int(self._ids_idx[index]), int(self._ids_idx[index + 1])
        path = self.path / naming.ids_name(self.split)
        if self._ids_map is None:
            self._ids_map = np.memmap(path, dtype=np.uint8, mode="r")
        if not 0 <= start < end <= self._ids_map.shape[0]:
            raise corrupt_index(path, f"line {index} spans bytes [{start}, {end})")
        try:
            record = json.loads(self._ids_map[start:end].tobytes())
            doc_id = record["id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise corrupt_index(path, f"line {index} is not an id record: {exc}") from exc
        if doc_id is not None and not isinstance(doc_id, str):
            raise corrupt_index(path, f"line {index}: id must be a string or null")
        return doc_id
