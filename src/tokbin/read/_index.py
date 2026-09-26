"""Memory-mapped int64 index files: ``offsets.npy`` and ``ids.idx.npy``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from tokbin import codes
from tokbin.errors import IntegrityError

__all__ = ["corrupt_index", "open_index"]

_FIX_INDEX = "the source is damaged; restore it from the original or run `tokbin verify`"


def corrupt_index(path: Path, why: str) -> IntegrityError:
    return IntegrityError(codes.INDEX_CORRUPT, str(path), why=why, fix=_FIX_INDEX)


def open_index(path: Path, *, length: int, first: int, last: int) -> npt.NDArray[np.int64]:
    """Map an int64 index and check its shape and end points.

    Only the header and two values are read, so opening does not depend on the number
    of documents. ``allow_pickle=False``: an ``.npy`` with pickled objects would
    execute code on load.
    """
    if not path.is_file():
        raise IntegrityError(
            codes.INDEX_MISSING,
            str(path),
            why="the file listed by the source layout is not on disk",
            fix=_FIX_INDEX,
        )
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as exc:
        raise corrupt_index(path, f"not a valid .npy file: {exc}") from exc
    if arr.dtype != np.dtype("<i8") or arr.ndim != 1:
        raise corrupt_index(path, f"expected a 1-D int64 array, got {arr.dtype} {arr.shape}")
    if arr.shape[0] != length:
        raise corrupt_index(path, f"expected {length} entries, found {arr.shape[0]}")
    if int(arr[0]) != first or int(arr[-1]) != last:
        raise corrupt_index(
            path,
            f"expected values from {first} to {last}, found {int(arr[0])} to {int(arr[-1])}",
        )
    result: npt.NDArray[np.int64] = arr
    return result
