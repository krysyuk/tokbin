"""Full verification of a source (spec 14.2): sha256 of every shard, completeness, sizes.

Unlike opening a source, verification reads everything: every shard is hashed and the
index files are checked value by value. Its cost is proportional to the corpus size;
the ``progress`` callback reports the bytes hashed so far.

Problems with the data are collected into :class:`VerifyReport` instead of being
raised, so that one run lists every damaged file. Errors that make verification itself
impossible (no ``meta.json`` at all, a schema newer than this tokbin) are raised.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import (
    FormatError,
    IntegrityError,
    SchemaVersionError,
    TokbinError,
    UnsupportedFeatureError,
)
from tokbin.format import naming
from tokbin.format.meta import Meta, ShardInfo, SplitMeta, read_meta
from tokbin.format.schema import source_kind
from tokbin.ops.inspect import Problem
from tokbin.read._files import check_shard, check_tokenizer_copy, open_split_indexes
from tokbin.read._index import corrupt_index

__all__ = ["ShardCheck", "ShardStatus", "VerifyProgress", "VerifyReport", "verify_source"]

ShardStatus = Literal["ok", "missing", "wrong_size", "corrupt"]

_CHUNK_BYTES = 8 * 1024**2
#: Index entries checked per step: bounds memory for any number of documents.
_INDEX_CHUNK = 1 << 20
_NEWLINE = 0x0A


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyProgress:
    """Passed to the ``progress`` callback while shards are hashed."""

    done_bytes: int
    total_bytes: int
    #: The shard being hashed.
    file: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ShardCheck:
    """The result for one shard."""

    split: str
    name: str
    n_bytes: int
    status: ShardStatus
    expected_sha256: str
    #: ``None`` when the file could not be hashed (missing or of the wrong size).
    actual_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "name": self.name,
            "n_bytes": self.n_bytes,
            "status": self.status,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyReport:
    """Everything ``verify`` found. ``ok`` means no problems at all."""

    path: Path
    shards: tuple[ShardCheck, ...]
    problems: tuple[Problem, ...]
    #: Bytes of shard data hashed.
    n_bytes_checked: int

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def n_bad_shards(self) -> int:
        return sum(1 for s in self.shards if s.status != "ok")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "ok": self.ok,
            "n_bytes_checked": self.n_bytes_checked,
            "shards": [s.to_dict() for s in self.shards],
            "problems": [p.to_dict() for p in self.problems],
        }


ProgressCallback = Callable[[VerifyProgress], None]


def _sha256(path: Path, on_chunk: Callable[[int], None]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            digest.update(chunk)
            on_chunk(len(chunk))
    return digest.hexdigest()


class _Verifier:
    def __init__(self, root: Path, meta: Meta, progress: ProgressCallback | None) -> None:
        self.root = root
        self.meta = meta
        self.progress = progress
        self.total = sum(sh.n_bytes for s in meta.splits for sh in s.shards)
        self.done = 0
        self.hashed = 0
        self.shards: list[ShardCheck] = []
        self.problems: list[TokbinError] = []

    def run(self) -> None:
        for split in self.meta.splits:
            for shard in split.shards:
                self._shard(split, shard)
            self._indexes(split)
            self._skipped(split)
        err = check_tokenizer_copy(self.root)
        if err is not None:
            self.problems.append(err)

    def _shard(self, split: SplitMeta, shard: ShardInfo) -> None:
        def check(status: ShardStatus, actual: str | None) -> None:
            self.shards.append(
                ShardCheck(
                    split=split.name,
                    name=shard.name,
                    n_bytes=shard.n_bytes,
                    status=status,
                    expected_sha256=shard.sha256,
                    actual_sha256=actual,
                )
            )

        err = check_shard(self.root, split, shard)
        if err is not None:
            self.problems.append(err)
            check("missing" if err.code is codes.SHARD_MISSING else "wrong_size", None)
            self.done += shard.n_bytes
            self._report(shard.name)
            return

        def on_chunk(n: int) -> None:
            self.done += n
            self.hashed += n
            self._report(shard.name)

        path = self.root / shard.name
        actual = _sha256(path, on_chunk)
        if actual != shard.sha256:
            self.problems.append(
                IntegrityError(
                    codes.SHARD_CORRUPT,
                    str(path),
                    why=f"sha256 {actual[:12]}... differs from {shard.sha256[:12]}... in meta.json",
                    fix=f"download the shard again and repeat `tokbin verify {self.root}`",
                )
            )
            check("corrupt", actual)
        else:
            check("ok", actual)

    def _report(self, file: str) -> None:
        if self.progress is not None:
            self.progress(VerifyProgress(done_bytes=self.done, total_bytes=self.total, file=file))

    def _indexes(self, split: SplitMeta) -> None:
        try:
            offsets, ids_idx = open_split_indexes(self.root, split)
        except TokbinError as exc:
            self.problems.append(exc)
            return
        offsets_path = self.root / naming.offsets_name(split.name)
        if not _nondecreasing(offsets):
            self.problems.append(corrupt_index(offsets_path, "document offsets decrease"))
        ids_path = self.root / naming.ids_name(split.name)
        idx_path = self.root / naming.ids_idx_name(split.name)
        if not _increasing(ids_idx):
            self.problems.append(corrupt_index(idx_path, "line offsets do not increase"))
            return
        if split.n_docs and not _lines_start_at(ids_path, ids_idx):
            self.problems.append(
                corrupt_index(ids_path, f"lines do not match the offsets in {idx_path.name}")
            )

    def _skipped(self, split: SplitMeta) -> None:
        path = self.root / naming.skipped_name(split.name)
        if not path.is_file():
            self.problems.append(
                IntegrityError(
                    codes.INDEX_MISSING,
                    str(path),
                    why="every split keeps a list of skipped documents, even an empty one",
                    fix="restore the file from the original dataset",
                )
            )
            return
        lines = _count_lines(path)
        if lines != split.n_skipped:
            self.problems.append(
                corrupt_index(path, f"meta.json records {split.n_skipped} skipped, found {lines}")
            )


def _chunks(arr: npt.NDArray[np.int64]) -> range:
    # Chunks overlap by one element so that differences across chunk edges are seen.
    return range(0, max(arr.shape[0] - 1, 1), _INDEX_CHUNK)


def _nondecreasing(arr: npt.NDArray[np.int64]) -> bool:
    for start in _chunks(arr):
        part = np.asarray(arr[start : start + _INDEX_CHUNK + 1])
        if part.size > 1 and bool(np.any(part[1:] < part[:-1])):
            return False
    return True


def _increasing(arr: npt.NDArray[np.int64]) -> bool:
    for start in _chunks(arr):
        part = np.asarray(arr[start : start + _INDEX_CHUNK + 1])
        if part.size > 1 and bool(np.any(part[1:] <= part[:-1])):
            return False
    return True


def _lines_start_at(ids_path: Path, idx: npt.NDArray[np.int64]) -> bool:
    """Every line ends with a newline exactly where the next one starts."""
    data = np.memmap(ids_path, dtype=np.uint8, mode="r")
    ends = idx[1:]
    for start in range(0, ends.shape[0], _INDEX_CHUNK):
        part = np.asarray(ends[start : start + _INDEX_CHUNK])
        if not bool(np.all(data[part - 1] == _NEWLINE)):
            return False
    return True


def _count_lines(path: Path) -> int:
    count = 0
    last = b"\n"
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count if last == b"\n" else count + 1


@public_api
def verify_source(path: str | Path, *, progress: ProgressCallback | None = None) -> VerifyReport:
    """Check every file of the source at ``path`` against its ``meta.json``."""
    root = Path(path)
    source_kind(root)
    try:
        meta = read_meta(root)
    except (SchemaVersionError, UnsupportedFeatureError):
        raise
    except FormatError as exc:
        if exc.code is codes.SCHEMA_TOO_OLD:
            raise
        return VerifyReport(
            path=root, shards=(), problems=(Problem.from_error(exc),), n_bytes_checked=0
        )
    verifier = _Verifier(root, meta, progress)
    verifier.run()
    return VerifyReport(
        path=root,
        shards=tuple(verifier.shards),
        problems=tuple(Problem.from_error(p) for p in verifier.problems),
        n_bytes_checked=verifier.hashed,
    )
