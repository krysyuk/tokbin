"""What a write reports back: statistics, issues and the final status (spec 7.3, 12.1)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tokbin.codes import Code
from tokbin.errors import TokbinWarning

__all__ = ["Issue", "IssueCollector", "IssueLevel", "WriteResult", "WriteStats"]

IssueLevel = Literal["warning", "info"]
WriteStatus = Literal["complete", "complete_with_issues"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Issue:
    """Something worth knowing about the result.

    ``warning``: the result exists, with a caveat. ``info``: a useful fact.
    """

    code: str
    level: IssueLevel
    #: Title of the code and the detail of the first occurrence.
    message: str
    #: How many times it happened.
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteStats:
    """A snapshot of writer counters."""

    #: Documents taken from the input, skipped ones included.
    n_input: int
    n_docs: int
    n_skipped: int
    n_items: int
    #: Closed shards.
    n_shards: int
    #: Documents whose items continue from one shard into the next.
    n_split_docs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteResult:
    """A finished write. Failure is always an exception, never a status."""

    path: Path
    split: str
    status: WriteStatus
    stats: WriteStats
    issues: tuple[Issue, ...]


class IssueCollector:
    """Counts issues by code; warns once per code, on first occurrence."""

    __slots__ = ("_issues",)

    def __init__(self) -> None:
        self._issues: dict[str, Issue] = {}

    def add(
        self,
        code: Code,
        detail: str,
        *,
        level: IssueLevel = "warning",
        category: type[TokbinWarning] | None = None,
        count: int = 1,
    ) -> None:
        known = self._issues.get(code.id)
        if known is not None:
            self._issues[code.id] = Issue(
                code=known.code,
                level=known.level,
                message=known.message,
                count=known.count + count,
            )
            return
        message = f"{code.title}: {detail}" if detail else code.title
        self._issues[code.id] = Issue(code=code.id, level=level, message=message, count=count)
        if level == "warning" and category is not None:
            warnings.warn(f"[{code.id}] {message}", category, stacklevel=4)

    def set_count(self, code: Code, detail: str, count: int, *, level: IssueLevel) -> None:
        """Record an aggregate fact known only at the end (no warning is emitted)."""
        if count <= 0:
            return
        message = f"{code.title}: {detail}" if detail else code.title
        self._issues[code.id] = Issue(code=code.id, level=level, message=message, count=count)

    def snapshot(self) -> tuple[Issue, ...]:
        return tuple(self._issues.values())

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self._issues.values())
