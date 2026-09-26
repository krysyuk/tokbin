"""``checkpoint.json``: state of an unfinished write after the last closed shard (spec 6.2).

The checkpoint lives only inside ``<name>.partial/`` and is deleted when the write is
finalized; a finished dataset never contains one.

A shard may close in the middle of a document, so the checkpoint also records how
many items of the next document are already in closed shards.

Besides the counters, it records the byte lengths of the append-only side files at the
moment the shard was closed. On resume those files are truncated back to these
lengths, dropping entries written after the checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokbin import codes
from tokbin.errors import FormatError
from tokbin.format import naming
from tokbin.format._fields import Fields
from tokbin.format._json import read_json_bounded, write_json_atomic
from tokbin.format._time import is_timestamp
from tokbin.format.dtypes import SUPPORTED_DTYPES
from tokbin.format.meta import ShardInfo
from tokbin.format.schema import check_schema

__all__ = ["Checkpoint", "SideFileLengths", "read_checkpoint", "write_checkpoint"]

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_HASH_RE = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True, slots=True, kw_only=True)
class SideFileLengths:
    """Byte lengths of the append-only files of a partial write."""

    ids: int
    ids_idx: int
    offsets: int
    skipped: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ids": self.ids,
            "ids_idx": self.ids_idx,
            "offsets": self.offsets,
            "skipped": self.skipped,
        }

    @classmethod
    def from_fields(cls, f: Fields) -> SideFileLengths:
        return cls(
            ids=f.get_int("ids", minimum=0),
            ids_idx=f.get_int("ids_idx", minimum=0),
            offsets=f.get_int("offsets", minimum=0),
            skipped=f.get_int("skipped", minimum=0),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """Everything needed to resume a write from the next shard."""

    schema_version: int
    split: str
    dtype: str
    config_hash: str
    tokenizer_hash: str
    #: Documents fully processed (written or skipped). Resume skips this many.
    n_input_consumed: int
    #: Items of the next input document that already sit in closed shards: a shard
    #: may close in the middle of a document. On resume that document is tokenized
    #: again and its first ``pending_doc_items`` items are not written twice.
    pending_doc_items: int
    #: Id of the last consumed document, if documents come with ids.
    last_input_id: str | None
    n_items: int
    n_docs: int
    n_skipped: int
    has_split_docs: bool
    closed_shards: tuple[ShardInfo, ...]
    side_files: SideFileLengths
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "dtype": self.dtype,
            "config_hash": self.config_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "n_input_consumed": self.n_input_consumed,
            "pending_doc_items": self.pending_doc_items,
            "last_input_id": self.last_input_id,
            "n_items": self.n_items,
            "n_docs": self.n_docs,
            "n_skipped": self.n_skipped,
            "has_split_docs": self.has_split_docs,
            "closed_shards": [s.to_dict() for s in self.closed_shards],
            "side_files": self.side_files.to_dict(),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: object, where: str) -> Checkpoint:
        f = Fields(data, where)
        check_schema(f.get_int("schema_version", minimum=1), where)

        split = f.get_str("split")
        if split not in naming.SPLITS:
            raise f.error("split", f"expected one of {', '.join(naming.SPLITS)}")
        dtype = f.get_str("dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise f.error("dtype", f"expected one of {', '.join(SUPPORTED_DTYPES)}")
        config_hash = f.get_str("config_hash")
        if not _HASH_RE.match(config_hash):
            raise f.error("config_hash", "expected 64 lowercase hex digits")
        tokenizer_hash = f.get_str("tokenizer_hash")
        if not _TOKENIZER_HASH_RE.match(tokenizer_hash):
            raise f.error("tokenizer_hash", "expected 16 lowercase hex digits")
        updated_at = f.get_str("updated_at")
        if not is_timestamp(updated_at):
            raise f.error("updated_at", "expected a UTC timestamp like 2026-09-26T10:22:31Z")

        shards = tuple(
            ShardInfo.from_fields(Fields(item, f"{where}: closed_shards[{i}]"))
            for i, item in enumerate(f.get_list("closed_shards"))
        )
        checkpoint = cls(
            schema_version=f.get_int("schema_version"),
            split=split,
            dtype=dtype,
            config_hash=config_hash,
            tokenizer_hash=tokenizer_hash,
            n_input_consumed=f.get_int("n_input_consumed", minimum=0),
            pending_doc_items=f.get_int("pending_doc_items", minimum=0),
            last_input_id=f.get_optional_str("last_input_id"),
            n_items=f.get_int("n_items", minimum=0),
            n_docs=f.get_int("n_docs", minimum=0),
            n_skipped=f.get_int("n_skipped", minimum=0),
            has_split_docs=f.get_bool("has_split_docs"),
            closed_shards=shards,
            side_files=SideFileLengths.from_fields(f.get_object("side_files")),
            updated_at=updated_at,
        )
        _check_consistency(checkpoint, where)
        return checkpoint


def _check_consistency(cp: Checkpoint, where: str) -> None:
    def fail(why: str) -> FormatError:
        return FormatError(
            codes.METADATA_INCONSISTENT,
            where,
            why=why,
            fix="the checkpoint is damaged; delete the partial write (`tokbin clean`) "
            "and start the write again",
        )

    itemsize = np.dtype(cp.dtype).itemsize
    for i, shard in enumerate(cp.closed_shards):
        expected = naming.shard_name(cp.split, i)
        if shard.name != expected:
            raise fail(f"shard #{i} is named {shard.name!r}, expected {expected!r}")
        if shard.n_bytes != shard.n_items * itemsize:
            raise fail(f"{shard.name}: n_bytes does not match n_items of {cp.dtype}")
    if sum(s.n_items for s in cp.closed_shards) != cp.n_items:
        raise fail("closed shards do not add up to n_items")
    if cp.n_docs + cp.n_skipped != cp.n_input_consumed:
        raise fail("written and skipped documents do not add up to n_input_consumed")
    if cp.pending_doc_items > cp.n_items:
        raise fail("pending_doc_items exceeds the items in closed shards")


def read_checkpoint(partial_root: Path) -> Checkpoint:
    path = partial_root / naming.CHECKPOINT
    return Checkpoint.from_dict(read_json_bounded(path), where=str(path))


def write_checkpoint(partial_root: Path, checkpoint: Checkpoint) -> None:
    write_json_atomic(partial_root / naming.CHECKPOINT, checkpoint.to_dict())
