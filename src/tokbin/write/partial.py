"""The ``<name>.partial/`` directory of an unfinished write and its publication (spec 8.1).

The partial directory sits next to the target, not in a system temp directory: the
final rename is atomic only within one file system.

Publication. A new source is published by renaming ``<name>.partial`` to ``<name>``.
When a split is added to an existing source, the other files of that source are
hard-linked (or copied, where links are unavailable) into the partial directory, the
merged ``meta.json`` is written, and the directories are swapped. Until the swap the
existing source is untouched; no finished source ever references a missing file.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import struct
from collections.abc import Sequence
from json.encoder import encode_basestring
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import numpy.typing as npt

from tokbin import _fs, codes
from tokbin.errors import InternalError, ResumeError
from tokbin.format import naming
from tokbin.format.checkpoint import SideFileLengths
from tokbin.format.meta import Meta, write_meta

__all__ = [
    "SideFiles",
    "create_partial",
    "discard_open_shards",
    "discard_stale_outputs",
    "partial_exists_error",
    "partial_path",
    "publish",
    "swap_in",
]

_INT64 = struct.Struct("<q")


def partial_path(target: Path) -> Path:
    return target.with_name(target.name + naming.PARTIAL_SUFFIX)


def partial_exists_error(partial: Path) -> ResumeError:
    return ResumeError(
        codes.PARTIAL_EXISTS,
        str(partial),
        why="a previous write of this source did not finish",
        fix=f"pass resume=True to continue it, or delete it with `tokbin clean {partial}` "
        "to start over",
    )


def create_partial(target: Path) -> Path:
    """Create an empty partial directory for ``target``."""
    partial = partial_path(target)
    try:
        partial.mkdir(parents=False)
    except FileExistsError:
        raise partial_exists_error(partial) from None
    return partial


class SideFiles:
    """Append-only side files of one split in a partial directory.

    - ``{split}-offsets.i64``: document boundaries, int64 items, starting with 0;
    - ``{split}-ids.jsonl``: one ``{"id": ...}`` line per written document;
    - ``{split}-ids.idx.i64``: byte offset of every line start, plus the end;
    - ``{split}-skipped.jsonl``: skipped documents with reasons, no content.

    The ``.i64`` files become ``.npy`` files in :meth:`finish`.
    """

    __slots__ = (
        "_dir",
        "_ids",
        "_ids_pos",
        "_idx",
        "_n_docs",
        "_offsets",
        "_skipped",
        "_skipped_pos",
        "split",
    )

    def __init__(
        self, directory: Path, split: str, *, resume: SideFileLengths | None = None
    ) -> None:
        """Create the files, or with ``resume`` reopen them cut back to those lengths."""
        self._dir = directory
        self.split = split
        paths = self._paths()
        if resume is None:
            mode: Literal["xb", "ab"] = "xb"
            self._ids_pos = self._skipped_pos = self._n_docs = 0
        else:
            _truncate_all(paths, resume)
            mode = "ab"
            self._ids_pos, self._skipped_pos = resume.ids, resume.skipped
            self._n_docs = resume.offsets // _INT64.size - 1
        self._offsets: BinaryIO = paths[0].open(mode)
        self._ids: BinaryIO = paths[1].open(mode)
        self._idx: BinaryIO = paths[2].open(mode)
        self._skipped: BinaryIO = paths[3].open(mode)
        if resume is None:
            self._offsets.write(_INT64.pack(0))
            self._idx.write(_INT64.pack(0))

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        d, split = self._dir, self.split
        return (
            d / naming.offsets_raw_name(split),
            d / naming.ids_name(split),
            d / naming.ids_idx_raw_name(split),
            d / naming.skipped_name(split),
        )

    def lengths(self) -> SideFileLengths:
        """Make everything written so far durable and report the file lengths."""
        self.sync()
        entries = (self._n_docs + 1) * _INT64.size
        return SideFileLengths(
            ids=self._ids_pos, ids_idx=entries, offsets=entries, skipped=self._skipped_pos
        )

    def add_docs(self, doc_ids: Sequence[str | None], ends: npt.NDArray[np.int64]) -> None:
        """Record written documents; document ``k`` ends at global item ``ends[k]``."""
        if not doc_ids:
            return
        # Same bytes as json.dumps({"id": doc_id}, ensure_ascii=False) per line, with the
        # C string encoder that json.dumps itself uses, minus its generic overhead.
        lines = [
            b'{"id": '
            + (b"null" if doc_id is None else encode_basestring(doc_id).encode("utf-8"))
            + b"}\n"
            for doc_id in doc_ids
        ]
        positions = self._ids_pos + np.cumsum(
            np.fromiter(map(len, lines), dtype=np.int64, count=len(lines))
        )
        self._ids.write(b"".join(lines))
        self._idx.write(positions.astype("<i8").tobytes())
        self._offsets.write(ends.astype("<i8").tobytes())
        self._ids_pos = int(positions[-1])
        self._n_docs += len(doc_ids)

    def add_skipped(self, input_index: int, doc_id: str | None, code: str, reason: str) -> None:
        record = {"input_index": input_index, "id": doc_id, "code": code, "reason": reason}
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        self._skipped.write(line)
        self._skipped_pos += len(line)

    def sync(self) -> None:
        for f in (self._offsets, self._ids, self._idx, self._skipped):
            _fs.fsync_file(f)

    def close(self) -> None:
        for f in (self._offsets, self._ids, self._idx, self._skipped):
            f.close()

    def finish(self) -> None:
        """Sync, close, and write the raw int64 files as ``.npy``.

        The raw files stay until publication: a write interrupted between here and the
        final rename can still be resumed from its checkpoint.
        """
        self.sync()
        self.close()
        _raw_to_npy(
            self._dir / naming.offsets_raw_name(self.split),
            self._dir / naming.offsets_name(self.split),
        )
        _raw_to_npy(
            self._dir / naming.ids_idx_raw_name(self.split),
            self._dir / naming.ids_idx_name(self.split),
        )


def _truncate_all(paths: tuple[Path, Path, Path, Path], lengths: SideFileLengths) -> None:
    """Cut the side files back to a checkpoint, dropping what was written after it."""
    wanted = (lengths.offsets, lengths.ids, lengths.ids_idx, lengths.skipped)
    for path, length in zip(paths, wanted, strict=True):
        size = path.stat().st_size if path.is_file() else -1
        if size < length:
            raise ResumeError(
                codes.PARTIAL_DAMAGED,
                str(path),
                why=f"the checkpoint expects at least {length} bytes, the file has "
                f"{max(size, 0)}" + ("" if size >= 0 else " (it is missing)"),
                fix=f"delete the unfinished write (`tokbin clean {path.parent}`) and "
                "start the write again",
            )
    for path, length in zip(paths, wanted, strict=True):
        os.truncate(path, length)


def _raw_to_npy(raw: Path, npy: Path) -> None:
    """Prefix a raw little-endian int64 file with an ``.npy`` header, streaming."""
    n = raw.stat().st_size // _INT64.size
    header = {"descr": "<i8", "fortran_order": False, "shape": (n,)}
    with npy.open("xb") as out:
        np.lib.format.write_array_header_1_0(out, header)
        with raw.open("rb") as src:
            shutil.copyfileobj(src, out, 1024**2)
        _fs.fsync_file(out)


def _split_owned(name: str, split: str) -> bool:
    """Does the file ``name`` belong to ``split`` of a finished source?"""
    if name in naming.split_files(split):
        return True
    return re.fullmatch(rf"{re.escape(split)}-\d{{5,}}\.bin", name) is not None


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _carry_over(existing: Path, partial: Path, split: str) -> None:
    """Bring files of the other splits (and anything else) from the existing source."""
    for entry in existing.iterdir():
        name = entry.name
        if name == naming.META_JSON or name == naming.TOKENIZER_DIR or _split_owned(name, split):
            continue
        dst = partial / name
        # Left over from a publication that was interrupted and then resumed.
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        elif dst.exists() or dst.is_symlink():
            dst.unlink()
        if entry.is_dir() and not entry.is_symlink():
            shutil.copytree(entry, dst, copy_function=_link_or_copy)
        else:
            _link_or_copy(str(entry), str(dst))


def _check_complete(root: Path, meta: Meta) -> None:
    """Every shard listed in meta is present with the listed size."""
    for split in meta.splits:
        for shard in split.shards:
            path = root / shard.name
            size = path.stat().st_size if path.is_file() else -1
            if size != shard.n_bytes:
                raise InternalError.wrap(
                    RuntimeError(
                        f"{path}: expected {shard.n_bytes} bytes, found {size} before publishing"
                    ),
                    where="tokbin.write.partial.publish",
                )


def publish(partial: Path, target: Path, meta: Meta, *, replace_split: str | None) -> None:
    """Make the partial write visible as ``target``.

    ``replace_split`` is the split written into ``partial`` when ``target`` already
    exists as a source; its other files are carried over.
    """
    if replace_split is not None:
        _carry_over(target, partial, replace_split)
    write_meta(partial, meta)
    _check_complete(partial, meta)
    # Up to here the write can be resumed from its checkpoint.
    _drop_work_files(partial)
    _fs.fsync_dir(partial)

    swap_in(partial, target)


def swap_in(ready: Path, target: Path) -> None:
    """Rename the finished directory ``ready`` to ``target``, replacing an existing one.

    An existing target is moved aside first and deleted only after the new one is in
    place; if the process dies in between, ``tokbin clean`` removes the old copy.
    """
    parent = target.parent
    if target.exists():
        old = target.with_name(f".{target.name}.old-{os.getpid()}")
        target.rename(old)
        ready.rename(target)
        _fs.fsync_dir(parent)
        shutil.rmtree(old)
    else:
        ready.rename(target)
        _fs.fsync_dir(parent)


def _drop_work_files(partial: Path) -> None:
    """Files of an unfinished write that a finished source never contains.

    The lock goes last: until then no other process may take the directory.
    """
    work = (naming.CHECKPOINT, naming.BUILD_RECIPE, naming.LOCK)
    for path in (*partial.glob("*.i64"), *(partial / name for name in work)):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def discard_stale_outputs(partial: Path, split: str) -> None:
    """Remove ``.npy`` files made by a finish that never got published."""
    for name in (naming.offsets_name(split), naming.ids_idx_name(split)):
        with contextlib.suppress(FileNotFoundError):
            (partial / name).unlink()


def discard_open_shards(partial: Path) -> None:
    """Remove ``*.bin.open`` files: an open shard is never trusted."""
    for path in partial.glob("*.bin.open"):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
