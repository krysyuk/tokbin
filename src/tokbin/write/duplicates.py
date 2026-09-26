"""Duplicate document ids, found with 8 bytes of memory per document.

Every id is reduced to a 64-bit hash stored in a growing ``uint64`` buffer; duplicates
are found once, at the end of the write, by sorting. A Python set of ids would cost
around 80 bytes per document (8 GB for 100 million documents).

Hash matches are confirmed by comparing the real id strings, so a hash collision
never produces a false warning.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["DuplicateReport", "IdHashes"]

_CHUNK: Final = 1 << 16
#: Repeats confirmed against the real ids.
_MAX_CONFIRM: Final = 1000


class DuplicateReport:
    __slots__ = ("count", "example")

    def __init__(self, count: int, example: str | None) -> None:
        #: Documents whose id hash already appeared earlier.
        self.count = count
        #: A confirmed duplicate id, if one was found among the first groups.
        self.example = example


class IdHashes:
    """Hashes of document ids in write order: position ``i`` is document ``i``."""

    __slots__ = ("_chunks", "_current", "_fill")

    def __init__(self) -> None:
        self._chunks: list[npt.NDArray[np.uint64]] = []
        self._current: npt.NDArray[np.uint64] = np.empty(_CHUNK, dtype=np.uint64)
        self._fill = 0

    def __len__(self) -> int:
        return len(self._chunks) * _CHUNK + self._fill

    def add(self, doc_id: str) -> None:
        digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
        self._current[self._fill] = int.from_bytes(digest, "little")
        self._fill += 1
        if self._fill == _CHUNK:
            self._chunks.append(self._current)
            self._current = np.empty(_CHUNK, dtype=np.uint64)
            self._fill = 0

    def report(self, id_of: Callable[[int], str | None]) -> DuplicateReport:
        """Count duplicates; ``id_of(i)`` reads back the id of document ``i``."""
        hashes = np.concatenate([*self._chunks, self._current[: self._fill]])
        if hashes.size < 2:
            return DuplicateReport(0, None)
        order = np.argsort(hashes, kind="stable")
        ranked = hashes[order]
        repeat = ranked[1:] == ranked[:-1]
        count = int(np.count_nonzero(repeat))
        if count == 0:
            return DuplicateReport(0, None)

        # Each repeat is a pair of equal hashes adjacent in sorted order. Up to
        # _MAX_CONFIRM pairs are confirmed against the real ids, which makes the count
        # exact and immune to hash collisions; beyond that the hash count is used (a
        # 64-bit collision among real ids is vanishingly unlikely).
        pairs = np.flatnonzero(repeat)
        confirmed: list[str] = []
        for pos in pairs[:_MAX_CONFIRM]:
            doc_id = id_of(int(order[pos]))
            if doc_id is not None and doc_id == id_of(int(order[pos + 1])):
                confirmed.append(doc_id)
        if pairs.size <= _MAX_CONFIRM:
            count = len(confirmed)
        return DuplicateReport(count, confirmed[0] if confirmed else None)
