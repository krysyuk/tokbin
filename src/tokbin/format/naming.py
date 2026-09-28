"""File names inside a source directory (spec 5.5) and path safety (spec 9.3).

The directory name is the source name and the key in ``mix.json``; metadata never
stores it separately.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from tokbin import codes
from tokbin.errors import ConfigError, FormatError

__all__ = [
    "BUILD_RECIPE",
    "CHECKPOINT",
    "DATASET_JSON",
    "LOCK",
    "META_JSON",
    "MIX_JSON",
    "PARTIAL_SUFFIX",
    "SPLITS",
    "TOKENIZER_CONFIG_JSON",
    "TOKENIZER_DIR",
    "TOKENIZER_JSON",
    "check_source_name",
    "check_split",
    "ids_idx_name",
    "ids_idx_raw_name",
    "ids_name",
    "is_valid_source_name",
    "offsets_name",
    "offsets_raw_name",
    "open_shard_name",
    "resolve_inside",
    "shard_name",
    "skipped_name",
    "split_files",
]

SPLITS: Final = ("train", "valid", "test")

META_JSON: Final = "meta.json"
DATASET_JSON: Final = "dataset.json"
MIX_JSON: Final = "mix.json"
CHECKPOINT: Final = "checkpoint.json"
#: Settings of a `tokbin build`, kept in the partial directory for `--resume`.
BUILD_RECIPE: Final = "build.json"
LOCK: Final = ".lock"
TOKENIZER_DIR: Final = "tokenizer"
TOKENIZER_JSON: Final = "tokenizer.json"
TOKENIZER_CONFIG_JSON: Final = "tokenizer_config.json"
PARTIAL_SUFFIX: Final = ".partial"

_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def shard_name(split: str, index: int) -> str:
    return f"{split}-{index:05d}.bin"


def offsets_name(split: str) -> str:
    return f"{split}-offsets.npy"


def ids_name(split: str) -> str:
    return f"{split}-ids.jsonl"


def ids_idx_name(split: str) -> str:
    return f"{split}-ids.idx.npy"


def skipped_name(split: str) -> str:
    return f"{split}-skipped.jsonl"


def offsets_raw_name(split: str) -> str:
    """Append-only int64 offsets of a partial write; becomes ``offsets.npy`` at the end."""
    return f"{split}-offsets.i64"


def ids_idx_raw_name(split: str) -> str:
    """Append-only int64 line offsets of a partial write; becomes ``ids.idx.npy``."""
    return f"{split}-ids.idx.i64"


def open_shard_name(split: str, index: int) -> str:
    """A shard being written; renamed to :func:`shard_name` once closed and synced."""
    return shard_name(split, index) + ".open"


def split_files(split: str) -> tuple[str, ...]:
    """Fixed-name files that belong to ``split`` in a finished source (shards excluded)."""
    return (offsets_name(split), ids_name(split), ids_idx_name(split), skipped_name(split))


def check_split(split: str) -> str:
    """Validate a split name passed by the user."""
    if split not in SPLITS:
        raise ConfigError(
            codes.SPLIT_INVALID,
            repr(split),
            why=f"supported splits are {', '.join(SPLITS)}",
            fix="use one of the supported split names",
        )
    return split


def is_valid_source_name(name: str) -> bool:
    """A source name is a single safe path component that is not a partial write."""
    return bool(_SOURCE_NAME_RE.match(name)) and not name.endswith(PARTIAL_SUFFIX)


def check_source_name(name: str) -> str:
    """Validate a source name passed by the user."""
    if not is_valid_source_name(name):
        raise ConfigError(
            codes.SOURCE_NAME_INVALID,
            repr(name),
            why="a source name is a directory name: letters, digits, '.', '_' and '-', "
            f"starting with a letter or digit and not ending with {PARTIAL_SUFFIX!r}",
            fix="choose a simple name such as 'web' or 'code-v2'",
        )
    return name


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve a path taken from metadata, refusing anything that leaves ``root``.

    Rejects absolute paths (POSIX and Windows forms), ``..`` components, and
    symlinks that point outside the root.
    """
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise _escapes(root, relative)
    candidate = root.joinpath(*posix.parts)
    real_root = root.resolve()
    if not candidate.resolve().is_relative_to(real_root):
        raise _escapes(root, relative)
    return candidate


def _escapes(root: Path, relative: str) -> FormatError:
    return FormatError(
        codes.PATH_ESCAPES_ROOT,
        repr(relative),
        why=f"metadata refers to a path outside the dataset root {root}",
        fix="the dataset is corrupted or malicious; do not use it",
    )
