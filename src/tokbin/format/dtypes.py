"""Element dtypes and the explicit range check before narrowing (spec 6.1, 8.6).

numpy cannot be trusted to catch overflow: ``astype`` silently wraps (70000 becomes
4464 in ``uint16``), and ``np.asarray`` on a list quietly turns ``[-1, 2**63]`` into
float64. Every conversion to the storage dtype goes through :func:`to_dtype_checked`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from tokbin import codes
from tokbin.errors import ConfigError, ContractError, DataError

__all__ = [
    "SUPPORTED_DTYPES",
    "TokenArray",
    "dtype_for_vocab",
    "parse_dtype",
    "resolve_dtype",
    "to_dtype_checked",
]

#: Storage dtypes, narrowest first. Byte order is always little-endian.
SUPPORTED_DTYPES: Final = ("uint8", "uint16", "uint32")

TokenArray = npt.NDArray[np.unsignedinteger]


def _max_vocab(name: str) -> int:
    return int(np.iinfo(np.dtype(name)).max) + 1


def dtype_for_vocab(vocab_size: int) -> np.dtype[np.unsignedinteger]:
    """The narrowest dtype that holds every id of a vocabulary of ``vocab_size`` tokens."""
    for name in SUPPORTED_DTYPES:
        if 0 < vocab_size <= _max_vocab(name):
            return np.dtype(name)
    raise ConfigError(
        codes.VOCAB_SIZE_OUT_OF_RANGE,
        str(vocab_size),
        why=f"the vocabulary size must be between 1 and 2**32, got {vocab_size}",
        fix="check the tokenizer; vocabularies above 2**32 tokens are not supported",
    )


def parse_dtype(name: str) -> np.dtype[np.unsignedinteger]:
    """Validate a dtype name given by the user."""
    if name not in SUPPORTED_DTYPES:
        raise ConfigError(
            codes.DTYPE_UNSUPPORTED,
            repr(name),
            why=f"supported dtypes are {', '.join(SUPPORTED_DTYPES)}",
            fix="use one of the supported dtypes, or omit it to derive it from the vocabulary",
        )
    return np.dtype(name)


def resolve_dtype(vocab_size: int, explicit: str | None = None) -> np.dtype[np.unsignedinteger]:
    """The storage dtype: derived from the vocabulary, or an explicit one that is wide enough.

    An explicit dtype narrower than the derived one is an error, not a silent upgrade:
    the user asked for something that cannot hold the data.
    """
    required = dtype_for_vocab(vocab_size)
    if explicit is None:
        return required
    chosen = parse_dtype(explicit)
    if chosen.itemsize < required.itemsize:
        raise ConfigError(
            codes.DTYPE_TOO_NARROW,
            explicit,
            why=f"a vocabulary of {vocab_size} tokens needs at least {required.name}",
            fix=f"use dtype={required.name!r} or wider, or omit dtype",
        )
    return chosen


def _out_of_range(low: int, high: int, dtype: np.dtype[np.unsignedinteger]) -> DataError:
    top = int(np.iinfo(dtype).max)
    bad = low if low < 0 else high
    return DataError(
        codes.TOKEN_ID_OUT_OF_RANGE,
        str(bad),
        why=f"token ids stored as {dtype.name} must be within 0..{top}",
        fix="check that the tokenizer matches the dataset vocabulary and dtype",
    )


def _invalid_ids(why: str) -> ContractError:
    return ContractError(
        codes.TOKEN_IDS_INVALID,
        why=why,
        fix="the tokenizer must return a flat sequence of non-negative integers",
    )


def to_dtype_checked(
    ids: Sequence[int] | npt.NDArray[np.generic],
    dtype: np.dtype[np.unsignedinteger],
) -> TokenArray:
    """Convert token ids to ``dtype``, refusing any value that does not fit."""
    arr = np.asarray(ids)
    if arr.size == 0:
        return np.empty(0, dtype=dtype)
    if arr.ndim != 1:
        raise _invalid_ids(f"expected a flat sequence, got an array of shape {arr.shape}")

    kind = arr.dtype.kind
    top = int(np.iinfo(dtype).max)
    if kind in "iu":
        low, high = int(arr.min()), int(arr.max())
    elif kind == "O" and all(type(v) is int for v in arr):
        # numpy falls back to object arrays for integers beyond 64 bits.
        low, high = min(arr), max(arr)
    elif kind == "f":
        # Floats appear either from genuine non-integer ids or from mixing negative
        # values with values above int64 max. Report the range problem if there is one.
        low_f, high_f = float(arr.min()), float(arr.max())
        if low_f < 0 or high_f > top:
            raise _out_of_range(int(low_f), int(high_f), dtype)
        raise _invalid_ids("expected integers, got floating point values")
    else:
        raise _invalid_ids(f"expected integers, got values of numpy dtype {arr.dtype}")

    if low < 0 or high > top:
        raise _out_of_range(low, high, dtype)
    return arr.astype(dtype)
