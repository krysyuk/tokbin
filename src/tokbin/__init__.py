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
from tokbin.ops.clean import CleanResult, clean_source
from tokbin.ops.doctor import DoctorReport, doctor
from tokbin.ops.inspect import (
    CorpusInfo,
    PartialInfo,
    Problem,
    SourceInfo,
    SplitInfo,
    inspect_corpus,
    inspect_source,
)
from tokbin.ops.pack import PackResult, UnpackResult, pack_source, unpack_pack
from tokbin.ops.verify import ShardCheck, VerifyProgress, VerifyReport, verify_source
from tokbin.read.mixture import Mixture
from tokbin.read.source import Source, read_source
from tokbin.write.config import ErrorPolicy, WriterConfig
from tokbin.write.documents import Segment
from tokbin.write.result import Issue, WriteResult, WriteStats
from tokbin.write.stream_writer import StreamWriter

__all__ = [
    "CleanResult",
    "CompatibilityError",
    "CompatibilityWarning",
    "ConfigError",
    "ContractError",
    "CorpusInfo",
    "DataError",
    "DataQualityWarning",
    "Dataset",
    "DependencyError",
    "DoctorReport",
    "ErrorPolicy",
    "FormatError",
    "IntegrityError",
    "InternalError",
    "Issue",
    "Mixture",
    "OutOfRangeError",
    "PackResult",
    "PartialInfo",
    "Problem",
    "ResumeError",
    "SchemaVersionError",
    "Segment",
    "ShardCheck",
    "ShardingWarning",
    "Source",
    "SourceInfo",
    "SplitInfo",
    "StreamWriter",
    "TokbinError",
    "TokbinWarning",
    "UnpackResult",
    "UnsupportedFeatureError",
    "VerifyProgress",
    "VerifyReport",
    "WriteResult",
    "WriteStats",
    "WriterConfig",
    "__version__",
    "clean_source",
    "doctor",
    "inspect_corpus",
    "inspect_source",
    "pack_source",
    "read_source",
    "unpack_pack",
    "verify_source",
]


def __getattr__(name: str) -> str:
    # The version is read from metadata lazily: importing must not compute anything.
    if name == "__version__":
        from tokbin._version import get_version

        return get_version()
    raise AttributeError(f"module 'tokbin' has no attribute {name!r}")
