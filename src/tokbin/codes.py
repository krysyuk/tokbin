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
SPLIT_EXISTS = _register("TB-C206", "Split already exists in the source")
TARGET_NOT_A_SOURCE = _register("TB-C207", "Target directory exists and is not a tokbin source")
NO_EOS_TOKEN = _register("TB-C208", "Tokenizer has no end-of-sequence token")
NO_BOS_TOKEN = _register("TB-C209", "Tokenizer has no beginning-of-sequence token")
CONFIG_VALUE_INVALID = _register("TB-C210", "Invalid configuration value")
SPECIAL_TOKEN_NOT_FOUND = _register("TB-C211", "Special token is not in the tokenizer vocabulary")
TOKENIZER_FILE_INVALID = _register("TB-C212", "Tokenizer file cannot be loaded")
TOKENIZER_LIMITS_DISABLED = _register("TB-C213", "Tokenizer truncation and padding were disabled")
WRITER_NOT_FINISHED = _register("TB-C214", "Writer is not finished")
SPECIAL_TOKENS_DETECTED = _register("TB-C215", "Special tokens were detected automatically")
DISK_SPACE_LOW = _register("TB-C216", "Not enough free disk space for the next shard")
SPLIT_NOT_FOUND = _register("TB-C301", "Split is not in the source")
OUT_OF_RANGE = _register("TB-C302", "Position is out of range")
RNG_REQUIRED = _register("TB-C303", "A numpy random Generator is required")
WINDOW_TOO_LARGE = _register("TB-C304", "Window is larger than the split")
SOURCE_NOT_FOUND = _register("TB-C305", "Dataset path not found")
NOT_A_SOURCE = _register("TB-C306", "Path is not a tokbin source")
NOT_A_CORPUS = _register("TB-C307", "Path is a source, not a corpus")
OUTPUT_EXISTS = _register("TB-C308", "Output already exists")
SOURCE_NOT_FINISHED = _register("TB-C309", "Source is not finished")
MIX_WEIGHTS_INVALID = _register("TB-C310", "Invalid mixture weights")

# --- K: contract --------------------------------------------------------------------

TOKEN_IDS_INVALID = _register("TB-K201", "Tokenizer returned invalid token ids")
DOCUMENT_FORMS_MIXED = _register("TB-K202", "Documents mix different forms")
DOCUMENT_FORM_INVALID = _register("TB-K203", "Document has an unsupported form")
TOO_MANY_SKIPPED = _register("TB-K204", "Too many documents were skipped")
TOKENIZER_UNSUPPORTED = _register("TB-K205", "Unsupported tokenizer object")

# --- D: data ------------------------------------------------------------------------

DOCUMENT_NOT_UTF8 = _register("TB-D201", "Document is not valid UTF-8")
TOKEN_ID_OUT_OF_RANGE = _register("TB-D202", "Token id is out of range for the dtype")
DOCUMENT_EMPTY = _register("TB-D203", "Document is empty")
TOKENIZER_FAILED = _register("TB-D204", "Tokenizer failed on the document")
DUPLICATE_DOC_ID = _register("TB-D205", "Duplicate document id")
DOCS_WERE_SKIPPED = _register("TB-D301", "Documents were skipped when the source was written")

# --- S: sharding --------------------------------------------------------------------

DOC_LARGER_THAN_SHARD = _register("TB-S201", "Document is larger than a whole shard")
DOC_LARGE_FOR_SHARD = _register("TB-S202", "Document is larger than 5% of a shard")
DOCS_SPLIT_ACROSS_SHARDS = _register("TB-S203", "Documents are split across shards")

# --- I: integrity -------------------------------------------------------------------

SHARD_MISSING = _register("TB-I301", "Shard is missing")
SHARD_CORRUPT = _register("TB-I302", "Shard is corrupted")
SHARD_WRONG_SIZE = _register("TB-I303", "Shard has the wrong size")
INDEX_MISSING = _register("TB-I304", "Index file is missing")
INDEX_CORRUPT = _register("TB-I305", "Index file is corrupted")
TOKENIZER_COPY_MISSING = _register("TB-I306", "Tokenizer copy is missing")
PACKED_FILE_CORRUPT = _register("TB-I307", "Packed file is corrupted")
PACKED_FILE_TOO_LARGE = _register("TB-I308", "Packed file expands beyond its declared size")
PACKED_FILE_MISSING = _register("TB-I309", "Packed file is missing")

# --- M: compatibility ---------------------------------------------------------------

TOKENIZER_MISMATCH = _register("TB-M201", "Existing source uses a different tokenizer")
DTYPE_MISMATCH = _register("TB-M202", "Existing source uses a different dtype")
TOKENIZERS_DIFFER = _register("TB-M301", "Sources of the corpus use different tokenizers")

# --- R: resume ----------------------------------------------------------------------

PARTIAL_EXISTS = _register("TB-R201", "An unfinished write already exists")
WRITER_ACTIVE = _register("TB-R202", "Another process is writing this source")
RESUME_INPUT_MISMATCH = _register("TB-R203", "Documents differ from the interrupted write")
RESUME_ORDER_UNCHECKED = _register("TB-R204", "Document order cannot be checked on resume")
RESUME_SETTINGS_MISMATCH = _register("TB-R205", "Settings differ from the interrupted write")
PARTIAL_DAMAGED = _register("TB-R206", "Unfinished write cannot be resumed")
RESUMED = _register("TB-R207", "Write resumed from a checkpoint")

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
MIX_SOURCE_MISSING = _register("TB-F110", "mix.json refers to a missing source")
SCHEMA_OUTDATED = _register("TB-F111", "Format schema is outdated")
ARCHIVE_ENTRY_UNSAFE = _register("TB-F112", "Archive contains an unsafe entry")
NOT_A_PACK = _register("TB-F113", "Path is not a tokbin pack")

# --- P: dependencies ----------------------------------------------------------------

DEPENDENCY_MISSING = _register("TB-P001", "Required package is not installed")
WRITE_UNAVAILABLE = _register("TB-P002", "Writing is not available in tokbin-core")
SLOW_COMPRESSION = _register("TB-P003", "zstd is not available; lzma is used instead")

# --- X: internal --------------------------------------------------------------------

INTERNAL = _register("TB-X001", "Internal tokbin error")


#: All active codes, keyed by id.
CODES: Mapping[str, Code] = MappingProxyType(_registry)
