"""Format schema versions (spec 13.3) and recognition of the source kind (spec 5.3).

``schema_version`` is independent of the package version and changes only when the
on-disk format changes. The writer writes only :data:`SCHEMA_VERSION`; the reader
accepts every version in :data:`SUPPORTED_SCHEMAS`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final, Literal

from tokbin import codes
from tokbin.errors import FormatError, SchemaVersionError, UnsupportedFeatureError
from tokbin.format import naming

__all__ = ["SCHEMA_VERSION", "SUPPORTED_SCHEMAS", "SourceKind", "check_schema", "source_kind"]

SCHEMA_VERSION: Final = 1
SUPPORTED_SCHEMAS: Final = frozenset({1})

SourceKind = Literal["single"]


def check_schema(version: int, where: str, *, warn_outdated: bool = True) -> None:
    """Accept a supported schema; warn about an outdated one, refuse the rest.

    ``where`` names the file for messages. Inspection passes ``warn_outdated=False``:
    it reports an outdated schema as a status instead of a warning.
    """
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            codes.SCHEMA_TOO_NEW,
            f"{where}: schema_version {version}",
            why=f"this tokbin understands schema versions up to {SCHEMA_VERSION}",
            fix="upgrade tokbin: pip install -U tokbin",
        )
    if version not in SUPPORTED_SCHEMAS:
        raise FormatError(
            codes.SCHEMA_TOO_OLD,
            f"{where}: schema_version {version}",
            why=f"supported schema versions are {sorted(SUPPORTED_SCHEMAS)}",
            fix="convert the dataset with an older tokbin release (`tokbin migrate`)",
        )
    if version < SCHEMA_VERSION and warn_outdated:
        warnings.warn(
            f"{where} uses format schema {version}; the current schema is {SCHEMA_VERSION}. "
            "Reading still works; convert the dataset with `tokbin migrate`.",
            FutureWarning,
            stacklevel=3,
        )


def source_kind(root: Path) -> SourceKind:
    """Recognize a source directory by its root file.

    ``meta.json`` means a single-stream source. ``dataset.json`` is a multi-stream
    source; the format is recognized but not supported before 2.x.
    """
    if (root / naming.DATASET_JSON).is_file():
        raise UnsupportedFeatureError(
            codes.FEATURE_UNSUPPORTED,
            str(root / naming.DATASET_JSON),
            why="multi-stream (multimodal) sources are not supported by this version",
            fix="read this dataset with a tokbin release that supports multimodal sources",
        )
    if (root / naming.META_JSON).is_file():
        return "single"
    raise FormatError(
        codes.METADATA_MISSING,
        str(root / naming.META_JSON),
        why=f"{root} contains neither {naming.META_JSON} nor {naming.DATASET_JSON}",
        fix="check the path; if the write was interrupted, see `tokbin status`",
    )
