"""tokbin: tokenized corpora as binary shards, read through memmap.

``import tokbin`` imports nothing but the standard library and numpy, and does
nothing but declarations.
"""

from __future__ import annotations

from tokbin.dataset import Dataset
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
    OutOfRangeError,
    ResumeError,
    SchemaVersionError,
    ShardingWarning,
    TokbinError,
    TokbinWarning,
    UnsupportedFeatureError,
)
from tokbin.read.source import Source, read_source
from tokbin.write.config import ErrorPolicy, WriterConfig
from tokbin.write.documents import Segment
from tokbin.write.result import Issue, WriteResult, WriteStats
from tokbin.write.stream_writer import StreamWriter

__all__ = [
    "CompatibilityError",
    "CompatibilityWarning",
    "ConfigError",
    "ContractError",
    "DataError",
    "DataQualityWarning",
    "Dataset",
    "DependencyError",
    "ErrorPolicy",
    "FormatError",
    "IntegrityError",
    "InternalError",
    "Issue",
    "OutOfRangeError",
    "ResumeError",
    "SchemaVersionError",
    "Segment",
    "ShardingWarning",
    "Source",
    "StreamWriter",
    "TokbinError",
    "TokbinWarning",
    "UnsupportedFeatureError",
    "WriteResult",
    "WriteStats",
    "WriterConfig",
    "__version__",
    "read_source",
]


def __getattr__(name: str) -> str:
    # The version is read from metadata lazily: importing must not compute anything.
    if name == "__version__":
        from tokbin._version import get_version

        return get_version()
    raise AttributeError(f"module 'tokbin' has no attribute {name!r}")
