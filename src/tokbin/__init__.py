"""tokbin: tokenized corpora as binary shards, read through memmap.

``import tokbin`` imports nothing but the standard library and numpy, and does
nothing but declarations.
"""

from __future__ import annotations

from tokbin.errors import (
    CompatibilityError,
    CompatibilityWarning,
    ConfigError,
    ContractError,
    DataError,
    DataQualityWarning,
    DependencyError,
    FormatError,
    IntegrityError,
    InternalError,
    ResumeError,
    SchemaVersionError,
    ShardingWarning,
    TokbinError,
    TokbinWarning,
    UnsupportedFeatureError,
)

__all__ = [
    "CompatibilityError",
    "CompatibilityWarning",
    "ConfigError",
    "ContractError",
    "DataError",
    "DataQualityWarning",
    "DependencyError",
    "FormatError",
    "IntegrityError",
    "InternalError",
    "ResumeError",
    "SchemaVersionError",
    "ShardingWarning",
    "TokbinError",
    "TokbinWarning",
    "UnsupportedFeatureError",
    "__version__",
]


def __getattr__(name: str) -> str:
    # The version is read from metadata lazily: importing must not compute anything.
    if name == "__version__":
        from tokbin._version import get_version

        return get_version()
    raise AttributeError(f"module 'tokbin' has no attribute {name!r}")
