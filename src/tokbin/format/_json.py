"""JSON input and output for metadata files.

Metadata comes from untrusted sources (datasets downloaded from elsewhere), so reading
is bounded in size and strict: duplicate keys, ``NaN`` and ``Infinity`` are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokbin import _fs, codes
from tokbin.errors import FormatError

__all__ = [
    "MAX_METADATA_BYTES",
    "canonical_dumps",
    "parse_json_bounded",
    "read_json_bounded",
    "write_json_atomic",
]

#: Upper bound for any metadata file (spec 9.3).
MAX_METADATA_BYTES = 16 * 1024**2

_CORRUPT_FIX = "the file is corrupted or was not written by tokbin; restore it from the original"


def canonical_dumps(obj: object) -> str:
    """Serialize to a canonical form: sorted keys, no whitespace, UTF-8 kept as is.

    Used for hashing. The output must never change for the same input, otherwise
    hashes stored in existing datasets stop matching.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _reject_constant(name: str) -> object:
    raise ValueError(f"non-standard JSON constant {name}")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def read_json_bounded(path: Path, *, limit: int = MAX_METADATA_BYTES) -> object:
    """Read and parse a metadata file of at most ``limit`` bytes."""
    try:
        with path.open("rb") as f:
            raw = f.read(limit + 1)
    except FileNotFoundError:
        raise FormatError(
            codes.METADATA_MISSING,
            str(path),
            why="the file does not exist",
            fix="check the path; an incomplete or foreign directory is not a tokbin dataset",
        ) from None
    return parse_json_bounded(raw, str(path), limit=limit)


def parse_json_bounded(raw: bytes, where: str, *, limit: int = MAX_METADATA_BYTES) -> object:
    """Parse a metadata document already read into memory (for example from an archive).

    ``raw`` may be one byte longer than ``limit``, which is how oversized input is seen.
    """
    if len(raw) > limit:
        raise FormatError(
            codes.METADATA_TOO_LARGE,
            where,
            why=f"the file is larger than the {limit // 1024**2} MB limit for metadata",
            fix=_CORRUPT_FIX,
        )
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise FormatError(
            codes.METADATA_NOT_JSON,
            where,
            why=str(exc),
            fix=_CORRUPT_FIX,
        ) from exc


def write_json_atomic(path: Path, obj: object) -> None:
    """Write a human-readable JSON document atomically (spec 8.2)."""
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _fs.atomic_write_bytes(path, text.encode("utf-8"))
