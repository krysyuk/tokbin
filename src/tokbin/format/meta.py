"""``meta.json``: the self-describing root of a single-stream source (spec 6.1).

Layout::

    {
      "schema_version": 1,
      "tokbin_version": "0.1.0",
      "modality": "text",
      "packing": "stream",
      "dtype": "uint16",
      "item_shape": [],
      "byteorder": "little",
      "vocab_size": 50257,
      "tokenizer_id": "gpt2",
      "tokenizer_hash": "a3f2c81b9e04d5a7",
      "eos_id": 50256,
      "bos_id": null,
      "splits": {
        "train": {
          "created_at": "2026-09-26T10:22:31Z",
          "n_items": 847392014,
          "n_docs": 1204881,
          "n_skipped": 412,
          "has_split_docs": true,
          "shards": [{"name": "train-00000.bin", "n_items": ..., "n_bytes": ..., "sha256": "..."}]
        }
      }
    }

Fields shared by all splits (dtype, tokenizer) live at the top level; counters and
shards live per split. Parsing is strict: a successfully parsed :class:`Meta` is
internally consistent (shard names, sizes and totals agree).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from tokbin import codes
from tokbin.errors import FormatError, InternalError, UnsupportedFeatureError
from tokbin.format import naming
from tokbin.format._fields import Fields
from tokbin.format._json import read_json_bounded, write_json_atomic
from tokbin.format._time import is_timestamp
from tokbin.format.dtypes import SUPPORTED_DTYPES
from tokbin.format.schema import check_schema

__all__ = ["Meta", "ShardInfo", "SplitMeta", "read_meta", "write_meta"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_HASH_RE = re.compile(r"^[0-9a-f]{16}$")
_FIX = "the file is corrupted or was not written by tokbin; restore it from the original"


@dataclass(frozen=True, slots=True, kw_only=True)
class ShardInfo:
    """One closed shard file. ``n_items`` is the actual count, not a planned one."""

    name: str
    n_items: int
    n_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_items": self.n_items,
            "n_bytes": self.n_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_fields(cls, f: Fields) -> ShardInfo:
        sha256 = f.get_str("sha256")
        if not _SHA256_RE.match(sha256):
            raise f.error("sha256", "expected 64 lowercase hex digits")
        return cls(
            name=f.get_str("name"),
            n_items=f.get_int("n_items", minimum=1),
            n_bytes=f.get_int("n_bytes", minimum=1),
            sha256=sha256,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitMeta:
    """Counters and shards of one split."""

    name: str
    created_at: str
    n_items: int
    n_docs: int
    n_skipped: int
    has_split_docs: bool
    shards: tuple[ShardInfo, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "n_items": self.n_items,
            "n_docs": self.n_docs,
            "n_skipped": self.n_skipped,
            "has_split_docs": self.has_split_docs,
            "shards": [s.to_dict() for s in self.shards],
        }

    @classmethod
    def from_fields(cls, name: str, f: Fields) -> SplitMeta:
        created_at = f.get_str("created_at")
        if not is_timestamp(created_at):
            raise f.error("created_at", "expected a UTC timestamp like 2026-09-26T10:22:31Z")
        shards = tuple(
            ShardInfo.from_fields(Fields(item, f"{f.where}: shards[{i}]"))
            for i, item in enumerate(f.get_list("shards"))
        )
        return cls(
            name=name,
            created_at=created_at,
            n_items=f.get_int("n_items", minimum=0),
            n_docs=f.get_int("n_docs", minimum=0),
            n_skipped=f.get_int("n_skipped", minimum=0),
            has_split_docs=f.get_bool("has_split_docs"),
            shards=shards,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Meta:
    """Parsed ``meta.json`` of a single-stream source."""

    schema_version: int
    tokbin_version: str
    modality: Literal["text"] = "text"
    packing: Literal["stream"] = "stream"
    dtype: str
    item_shape: tuple[int, ...] = ()
    byteorder: Literal["little"] = "little"
    vocab_size: int
    tokenizer_id: str | None
    tokenizer_hash: str
    eos_id: int | None
    bos_id: int | None
    #: Splits in canonical order (train, valid, test).
    splits: tuple[SplitMeta, ...]

    @property
    def np_dtype(self) -> np.dtype[np.unsignedinteger]:
        return np.dtype(self.dtype).newbyteorder("<")

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.splits)

    def split(self, name: str) -> SplitMeta:
        """The split ``name``; ``KeyError`` if the source has no such split."""
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(name)

    def with_split(self, split: SplitMeta) -> Meta:
        """A copy with ``split`` added or replaced, keeping canonical split order."""
        others = {s.name: s for s in self.splits if s.name != split.name}
        others[split.name] = split
        ordered = tuple(others[n] for n in naming.SPLITS if n in others)
        return replace(self, splits=ordered)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tokbin_version": self.tokbin_version,
            "modality": self.modality,
            "packing": self.packing,
            "dtype": self.dtype,
            "item_shape": list(self.item_shape),
            "byteorder": self.byteorder,
            "vocab_size": self.vocab_size,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_hash": self.tokenizer_hash,
            "eos_id": self.eos_id,
            "bos_id": self.bos_id,
            "splits": {s.name: s.to_dict() for s in self.splits},
        }

    @classmethod
    def from_dict(cls, data: object, where: str, *, warn_outdated: bool = True) -> Meta:
        """Parse and validate. ``where`` names the file for messages."""
        f = Fields(data, where)
        # The schema version comes first: a newer schema must produce
        # SchemaVersionError, not a confusing error about some field.
        check_schema(f.get_int("schema_version", minimum=1), where, warn_outdated=warn_outdated)

        _require_supported(f, "modality", "text", "only text sources are supported")
        _require_supported(f, "packing", "stream", 'packing "record" is planned for 2.0')
        _require_supported(f, "byteorder", "little", "only little-endian data is supported")
        if f.get_list("item_shape"):
            raise _unsupported(f, "item_shape", "only scalar elements are supported")

        dtype = f.get_str("dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise f.error("dtype", f"expected one of {', '.join(SUPPORTED_DTYPES)}")
        tokenizer_hash = f.get_str("tokenizer_hash")
        if not _TOKENIZER_HASH_RE.match(tokenizer_hash):
            raise f.error("tokenizer_hash", "expected 16 lowercase hex digits")

        splits_f = f.get_object("splits")
        unknown = [k for k in splits_f.field_names() if k not in naming.SPLITS]
        if unknown:
            raise f.error("splits", f"unknown split {unknown[0]!r}")
        splits = tuple(
            SplitMeta.from_fields(name, splits_f.get_object(name))
            for name in naming.SPLITS
            if name in splits_f.field_names()
        )

        meta = cls(
            schema_version=f.get_int("schema_version"),
            tokbin_version=f.get_str("tokbin_version"),
            dtype=dtype,
            vocab_size=f.get_int("vocab_size", minimum=1),
            tokenizer_id=f.get_optional_str("tokenizer_id"),
            tokenizer_hash=tokenizer_hash,
            eos_id=f.get_optional_int("eos_id", minimum=0),
            bos_id=f.get_optional_int("bos_id", minimum=0),
            splits=splits,
        )
        _check_consistency(meta, where)
        return meta


def _unsupported(f: Fields, key: str, why: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        codes.FEATURE_UNSUPPORTED,
        f"{f.where}: {key}",
        why=why,
        fix="read this dataset with a tokbin release that supports this feature",
    )


def _require_supported(f: Fields, key: str, supported: str, why: str) -> None:
    value = f.get_str(key)
    if value != supported:
        raise _unsupported(f, key, f"{why}; got {value!r}")


def _inconsistent(where: str, why: str) -> FormatError:
    return FormatError(codes.METADATA_INCONSISTENT, where, why=why, fix=_FIX)


def _check_consistency(meta: Meta, where: str) -> None:
    """Cross-field checks that single-field parsing cannot express."""
    max_ids = int(np.iinfo(np.dtype(meta.dtype)).max) + 1
    if meta.vocab_size > max_ids:
        raise _inconsistent(where, f"vocab_size {meta.vocab_size} does not fit dtype {meta.dtype}")
    for key, value in (("eos_id", meta.eos_id), ("bos_id", meta.bos_id)):
        if value is not None and value >= meta.vocab_size:
            raise _inconsistent(
                where, f"{key} {value} is outside the vocabulary of {meta.vocab_size} tokens"
            )
    if not meta.splits:
        raise _inconsistent(where, "the source has no splits")

    for split in meta.splits:
        at = f"{where}: splits.{split.name}"
        for i, shard in enumerate(split.shards):
            expected = naming.shard_name(split.name, i)
            if shard.name != expected:
                raise _inconsistent(
                    at, f"shard #{i} is named {shard.name!r}, expected {expected!r}"
                )
            if shard.n_bytes != shard.n_items * meta.itemsize:
                raise _inconsistent(
                    at,
                    f"{shard.name}: {shard.n_items} items of {meta.dtype} take "
                    f"{shard.n_items * meta.itemsize} bytes, meta says {shard.n_bytes}",
                )
        total = sum(s.n_items for s in split.shards)
        if total != split.n_items:
            raise _inconsistent(
                at, f"shards hold {total} items in total, but n_items is {split.n_items}"
            )


def read_meta(root: Path, *, warn_outdated: bool = True) -> Meta:
    """Read and validate ``meta.json`` of the source at ``root``."""
    path = root / naming.META_JSON
    return Meta.from_dict(read_json_bounded(path), where=str(path), warn_outdated=warn_outdated)


def write_meta(root: Path, meta: Meta) -> None:
    """Write ``meta.json`` atomically.

    The document is validated by parsing it back before writing: tokbin never writes
    metadata that it would refuse to read.
    """
    data = meta.to_dict()
    path = root / naming.META_JSON
    try:
        Meta.from_dict(data, where=str(path))
    except FormatError as exc:
        raise InternalError.wrap(exc, where="tokbin.format.meta.write_meta") from exc
    write_json_atomic(path, data)
