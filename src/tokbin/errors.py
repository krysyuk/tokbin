"""Exception and warning hierarchy of tokbin.

Every exception carries a stable code and a three-part message: what happened, why,
and what to do about it. The text is rendered as::

    [TB-I301] Shard is missing: corpus/web/train-00003.bin
      where: tokbin.read.source.read_source
      cause: meta.json lists 13 shards, 12 found on disk
      fix:   download the shard and run `tokbin verify corpus/web`
"""

from __future__ import annotations

from typing import ClassVar

from tokbin import codes
from tokbin.codes import Code

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
]

_ISSUE_HINT = (
    "this is a bug in tokbin, not in your data; please open an issue with the full "
    "error text and the output of `tokbin doctor`"
)


class TokbinError(Exception):
    """Base class of all tokbin errors."""

    #: Code categories (the letter after ``TB-``) allowed for this class.
    categories: ClassVar[frozenset[str]] = frozenset()

    code: Code
    detail: str | None
    why: str
    fix: str
    where: str | None

    def __init__(
        self,
        code: Code,
        detail: str | None = None,
        *,
        why: str,
        fix: str,
        where: str | None = None,
    ) -> None:
        if code.category not in type(self).categories:
            raise TypeError(
                f"code {code.id} does not belong to {type(self).__name__} "
                f"(allowed categories: {sorted(type(self).categories)})"
            )
        self.code = code
        self.detail = detail
        self.why = why
        self.fix = fix
        self.where = where
        super().__init__(self._render())

    @property
    def what(self) -> str:
        """First line: the code title and, if present, the detail (usually a path)."""
        if self.detail:
            return f"{self.code.title}: {self.detail}"
        return self.code.title

    def _render(self) -> str:
        lines = [f"[{self.code.id}] {self.what}"]
        if self.where:
            lines.append(f"  where: {self.where}")
        lines.append(f"  cause: {self.why}")
        lines.append(f"  fix:   {self.fix}")
        return "\n".join(lines)

    def __str__(self) -> str:
        # ``where`` may be filled in later, at the public API boundary.
        return self._render()

    def __reduce__(self) -> tuple[object, ...]:
        # Exceptions must survive pickling: they cross process boundaries
        # (DataLoader workers, multiprocessing).
        return (
            _rebuild,
            (type(self), self.code, self.detail, self.why, self.fix, self.where),
        )


def _rebuild(
    cls: type[TokbinError],
    code: Code,
    detail: str | None,
    why: str,
    fix: str,
    where: str | None,
) -> TokbinError:
    return cls(code, detail, why=why, fix=fix, where=where)


class ConfigError(TokbinError):
    """Invalid parameters."""

    categories = frozenset("C")


class ContractError(TokbinError):
    """The document generator or the tokenizer violates its contract."""

    categories = frozenset("K")


class DataError(TokbinError):
    """A problem with a single document; such a document may be skipped."""

    categories = frozenset("D")


class FormatError(TokbinError):
    """Corrupted or unrecognized metadata."""

    categories = frozenset("F")


class SchemaVersionError(FormatError):
    """The on-disk format is newer than this version of the library understands."""


class IntegrityError(TokbinError):
    """sha256 mismatch, missing shard or wrong size."""

    categories = frozenset("I")


class CompatibilityError(TokbinError):
    """Sources are incompatible (for example, different tokenizers in a mixture)."""

    categories = frozenset("M")


class ResumeError(TokbinError):
    """Resuming a write is impossible or unsafe."""

    categories = frozenset("R")


class DependencyError(TokbinError, ImportError):
    """An optional package is not installed.

    Inherits ``ImportError`` so that existing user handlers (``except ImportError``)
    keep working.
    """

    categories = frozenset("P")


class UnsupportedFeatureError(TokbinError):
    """The format is recognized but not supported by this version of the library."""

    categories = frozenset("F")


class InternalError(TokbinError):
    """A bug inside the library. The message asks the user to open an issue."""

    categories = frozenset("X")

    @classmethod
    def wrap(cls, exc: BaseException, *, where: str | None = None) -> InternalError:
        """Wrap an exception that is not part of the tokbin hierarchy."""
        return cls(
            codes.INTERNAL,
            f"{type(exc).__name__}: {exc}",
            why=_ISSUE_HINT,
            fix="retry the operation; if the error persists, please report it",
            where=where,
        )


# --- warnings -----------------------------------------------------------------------


class TokbinWarning(UserWarning):
    """Base class of tokbin warnings: the result exists, but with a caveat."""


class DataQualityWarning(TokbinWarning):
    """Skipped documents, duplicate ids and other remarks about the data."""


class ShardingWarning(TokbinWarning):
    """Documents split across shards, oversized documents."""


class CompatibilityWarning(TokbinWarning):
    """Old format schema and other compatibility concerns."""
