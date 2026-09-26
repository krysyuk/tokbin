"""Writer settings (spec 3.3, 8.4, 10.1).

Settings are frozen: a configuration cannot change in the middle of a write. Derived
configurations are made with ``dataclasses.replace``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tokbin import codes
from tokbin.errors import ConfigError
from tokbin.format import naming
from tokbin.format._json import canonical_dumps
from tokbin.format.dtypes import parse_dtype
from tokbin.format.schema import SCHEMA_VERSION

__all__ = ["MIN_SHARD_BYTES", "ErrorPolicy", "WriterConfig", "config_hash"]

#: Smallest allowed shard. Tiny shards are legal (tests use them), just inefficient.
MIN_SHARD_BYTES = 16


def _invalid(field: str, value: object, why: str, fix: str) -> ConfigError:
    return ConfigError(codes.CONFIG_VALUE_INVALID, f"{field}={value!r}", why=why, fix=fix)


@dataclass(frozen=True, slots=True, kw_only=True)
class WriterConfig:
    """How documents become shards."""

    #: Which split to write: ``train``, ``valid`` or ``test``.
    split: str = "train"
    #: Target shard size in bytes; the last shard of a split is the remainder.
    shard_bytes: int = 512 * 1024**2
    #: Append the tokenizer's EOS token after every document.
    append_eos: bool = True
    #: Prepend the tokenizer's BOS token before every document.
    prepend_bos: bool = False
    #: Storage dtype; ``None`` derives the narrowest one from the vocabulary size.
    dtype: str | None = None
    #: EOS token string; ``None`` detects a well-known one (``<|endoftext|>``, ``</s>``...).
    eos_token: str | None = None
    #: BOS token string; ``None`` detects a well-known one (``<s>``, ``<bos>``...).
    bos_token: str | None = None
    #: Documents tokenized together (``encode_batch``). Affects speed, not the output.
    batch_docs: int = 1024

    def __post_init__(self) -> None:
        naming.check_split(self.split)
        if type(self.shard_bytes) is not int or self.shard_bytes < MIN_SHARD_BYTES:
            raise _invalid(
                "shard_bytes",
                self.shard_bytes,
                f"shard_bytes must be an integer of at least {MIN_SHARD_BYTES}",
                "use the default (512 MB) or a size in the recommended 256 MB - 2 GB range",
            )
        for name in ("append_eos", "prepend_bos"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise _invalid(name, value, f"{name} must be True or False", "pass True or False")
        if type(self.batch_docs) is not int or self.batch_docs < 1:
            raise _invalid(
                "batch_docs",
                self.batch_docs,
                "batch_docs must be a positive integer",
                "use the default (1024); 1 tokenizes documents one by one",
            )
        if self.dtype is not None:
            parse_dtype(self.dtype)
        for name in ("eos_token", "bos_token"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise _invalid(
                    name,
                    value,
                    f"{name} must be a non-empty string or None",
                    "pass the token text, e.g. '</s>', or None to detect it",
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorPolicy:
    """What to do with documents that cannot be written (spec 8.4)."""

    #: ``skip``: record the document in ``skipped.jsonl`` and go on; ``raise``: stop.
    on_data_error: Literal["skip", "raise"] = "skip"
    #: Stop the write when the share of skipped documents exceeds this ratio...
    max_skip_ratio: float = 0.01
    #: ...but only after this many input documents, so early noise does not trip it.
    min_docs_for_ratio: int = 1000

    def __post_init__(self) -> None:
        if self.on_data_error not in ("skip", "raise"):
            raise _invalid(
                "on_data_error",
                self.on_data_error,
                "expected 'skip' or 'raise'",
                "use on_data_error='skip' or 'raise'",
            )
        ratio = self.max_skip_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
            raise _invalid(
                "max_skip_ratio",
                ratio,
                "expected a number between 0 and 1",
                "e.g. max_skip_ratio=0.01",
            )
        if type(self.min_docs_for_ratio) is not int or self.min_docs_for_ratio < 1:
            raise _invalid(
                "min_docs_for_ratio",
                self.min_docs_for_ratio,
                "expected a positive integer",
                "e.g. min_docs_for_ratio=1000",
            )


def config_hash(
    config: WriterConfig,
    *,
    dtype: np.dtype[np.unsignedinteger],
    eos_id: int | None,
    bos_id: int | None,
) -> str:
    """sha256 of everything in the configuration that affects the written bytes.

    Used to refuse resuming a write with different settings. ``ErrorPolicy`` is not
    part of it: the policy may change between runs.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "packing": "stream",
        "split": config.split,
        "shard_bytes": config.shard_bytes,
        "append_eos": config.append_eos,
        "prepend_bos": config.prepend_bos,
        "dtype": dtype.name,
        "eos_id": eos_id if config.append_eos else None,
        "bos_id": bos_id if config.prepend_bos else None,
    }
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
