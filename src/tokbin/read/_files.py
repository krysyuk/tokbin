"""Cheap checks of the files a source consists of (spec 8.7).

They look at presence, sizes and index headers, never at the full content: the result
does not depend on the size of the corpus. Full sha256 checks are done by ``verify``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from tokbin import codes
from tokbin.errors import IntegrityError, TokbinError
from tokbin.format import naming
from tokbin.format.meta import Meta, ShardInfo, SplitMeta
from tokbin.read._index import open_index

__all__ = ["check_shard", "check_source_files", "check_tokenizer_copy", "open_split_indexes"]

_FIX_SHARD = "download the shard again and run `tokbin verify {root}`"
_FIX_TOKENIZER = (
    "copy tokenizer.json of the tokenizer used for writing into {path}; "
    "without it the tokens cannot be decoded"
)


def check_shard(root: Path, split: SplitMeta, shard: ShardInfo) -> IntegrityError | None:
    """``None`` if the shard file is present with the size from ``meta.json``."""
    path = naming.resolve_inside(root, shard.name)
    if not path.is_file():
        return IntegrityError(
            codes.SHARD_MISSING,
            str(path),
            why=f"meta.json lists {len(split.shards)} shards for {split.name!r}; "
            "this one is not on disk",
            fix=_FIX_SHARD.format(root=root),
        )
    size = path.stat().st_size
    if size != shard.n_bytes:
        return IntegrityError(
            codes.SHARD_WRONG_SIZE,
            str(path),
            why=f"meta.json expects {shard.n_bytes} bytes, the file has {size}",
            fix=_FIX_SHARD.format(root=root),
        )
    return None


def open_split_indexes(
    root: Path, split: SplitMeta
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Map ``offsets.npy`` and ``ids.idx.npy`` of a split, checking their headers."""
    offsets = open_index(
        naming.resolve_inside(root, naming.offsets_name(split.name)),
        length=split.n_docs + 1,
        first=0,
        last=split.n_items,
    )
    ids_path = naming.resolve_inside(root, naming.ids_name(split.name))
    ids_size = ids_path.stat().st_size if ids_path.is_file() else -1
    ids_idx = open_index(
        naming.resolve_inside(root, naming.ids_idx_name(split.name)),
        length=split.n_docs + 1,
        first=0,
        last=ids_size,
    )
    return offsets, ids_idx


def check_tokenizer_copy(root: Path) -> IntegrityError | None:
    path = root / naming.TOKENIZER_DIR / naming.TOKENIZER_JSON
    if path.is_file():
        return None
    return IntegrityError(
        codes.TOKENIZER_COPY_MISSING,
        str(path),
        why="a source keeps a copy of its tokenizer next to the data",
        fix=_FIX_TOKENIZER.format(path=path.parent),
    )


def check_source_files(root: Path, meta: Meta) -> list[TokbinError]:
    """All problems found by the cheap checks, in a stable order."""
    problems: list[TokbinError] = []
    for split in meta.splits:
        for shard in split.shards:
            try:
                shard_err = check_shard(root, split, shard)
            except TokbinError as exc:  # a shard name escaping the root
                problems.append(exc)
                continue
            if shard_err is not None:
                problems.append(shard_err)
        try:
            open_split_indexes(root, split)
        except TokbinError as exc:
            problems.append(exc)
    err = check_tokenizer_copy(root)
    if err is not None:
        problems.append(err)
    return problems
