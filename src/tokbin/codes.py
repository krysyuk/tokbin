"""Registry of error and warning codes.

A code is stable: once released, it never changes meaning and is never reused, even
after the error itself is removed (such codes move to ``RETIRED``). The code table in
the documentation is generated from this module.

Format: ``TB-<category><three digits>``. Categories:

====  ==================  =========================================================
 C    configuration       ``ConfigError``
 K    contract            ``ContractError``
 D    data                ``DataError``, ``DataQualityWarning``
 F    format              ``FormatError``, ``SchemaVersionError``,
                          ``UnsupportedFeatureError``
 I    integrity           ``IntegrityError``
 M    compatibility       ``CompatibilityError``, ``CompatibilityWarning``
 R    resume              ``ResumeError``
 P    dependencies        ``DependencyError``
 S    sharding            ``ShardingWarning``
 X    internal            ``InternalError``
====  ==================  =========================================================
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["CODES", "RETIRED", "Code"]

_CODE_RE = re.compile(r"^TB-[CKDFIMRPSX]\d{3}$")


@dataclass(frozen=True, slots=True)
class Code:
    """A stable diagnostic code and its short title ("what happened")."""

    id: str
    title: str

    @property
    def category(self) -> str:
        """Category letter: ``"I"`` for ``TB-I301``."""
        return self.id[3]

    def __str__(self) -> str:
        return self.id


_registry: dict[str, Code] = {}

#: Codes that are no longer issued. They must never be registered again.
RETIRED: frozenset[str] = frozenset()


def _register(code_id: str, title: str) -> Code:
    if not _CODE_RE.match(code_id):
        raise ValueError(f"malformed code: {code_id!r}")
    if code_id in _registry:
        raise ValueError(f"code registered twice: {code_id}")
    if code_id in RETIRED:
        raise ValueError(f"code is retired and cannot be reused: {code_id}")
    code = Code(code_id, title)
    _registry[code_id] = code
    return code


# Numbering inside a category: 1xx on-disk format, 2xx writing, 3xx reading and
# verification.

# --- C: configuration ---------------------------------------------------------------

VOCAB_SIZE_OUT_OF_RANGE = _register("TB-C201", "Vocabulary size is out of range")
DTYPE_TOO_NARROW = _register("TB-C202", "Explicit dtype is too narrow for the vocabulary")
DTYPE_UNSUPPORTED = _register("TB-C203", "Unsupported dtype")
SPLIT_INVALID = _register("TB-C204", "Invalid split name")
SOURCE_NAME_INVALID = _register("TB-C205", "Invalid source name")

# --- K: contract --------------------------------------------------------------------

TOKEN_IDS_INVALID = _register("TB-K201", "Tokenizer returned invalid token ids")

# --- D: data ------------------------------------------------------------------------

TOKEN_ID_OUT_OF_RANGE = _register("TB-D202", "Token id is out of range for the dtype")

# --- F: format ----------------------------------------------------------------------

METADATA_TOO_LARGE = _register("TB-F101", "Metadata file is too large")
METADATA_NOT_JSON = _register("TB-F102", "Metadata file is not valid JSON")
METADATA_FIELD_INVALID = _register("TB-F103", "Invalid metadata field")
METADATA_MISSING = _register("TB-F104", "Metadata file is missing")
METADATA_INCONSISTENT = _register("TB-F105", "Metadata is inconsistent")
PATH_ESCAPES_ROOT = _register("TB-F106", "Path escapes the dataset root")
SCHEMA_TOO_NEW = _register("TB-F107", "Format schema is newer than this tokbin supports")
SCHEMA_TOO_OLD = _register("TB-F108", "Format schema is no longer supported")
FEATURE_UNSUPPORTED = _register("TB-F109", "Feature is not supported by this tokbin version")

# --- P: dependencies ----------------------------------------------------------------

DEPENDENCY_MISSING = _register("TB-P001", "Required package is not installed")

# --- X: internal --------------------------------------------------------------------

INTERNAL = _register("TB-X001", "Internal tokbin error")


#: All active codes, keyed by id.
CODES: Mapping[str, Code] = MappingProxyType(_registry)
