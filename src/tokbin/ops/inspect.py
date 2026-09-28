"""What is on disk: the state of a source and an overview of a corpus (spec 12.1, 14.2).

States of a source:

==============  ================================================================
``complete``    ready for reading
``partial``     only an unfinished write exists (``<name>.partial/``)
``corrupt``     files are missing, have wrong sizes, or metadata is damaged
``outdated``    written in an older format schema; ``tokbin migrate`` converts it
``unsupported`` a newer schema or a feature this version cannot read
==============  ================================================================

A complete source may have an unfinished write next to it (a split being added); it
is reported in :attr:`SourceInfo.partial` and as a warning.

Inspection only runs the cheap checks of spec 8.7: presence and sizes of files and
index headers. It never reads the data; the full sha256 check is ``verify``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import (
    ConfigError,
    FormatError,
    SchemaVersionError,
    TokbinError,
    UnsupportedFeatureError,
)
from tokbin.format import naming
from tokbin.format.checkpoint import read_checkpoint
from tokbin.format.meta import Meta, read_meta
from tokbin.format.mix import read_mix
from tokbin.format.schema import SCHEMA_VERSION
from tokbin.read._files import check_source_files
from tokbin.write.lock import read_lock
from tokbin.write.result import Issue

__all__ = [
    "CorpusInfo",
    "PartialInfo",
    "Problem",
    "SourceInfo",
    "SourceState",
    "SplitInfo",
    "describe",
    "inspect_corpus",
    "inspect_source",
]

SourceState = Literal["complete", "partial", "corrupt", "outdated", "unsupported"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Problem:
    """An error found while inspecting: the same four parts as an exception message."""

    code: str
    what: str
    why: str
    fix: str

    @classmethod
    def from_error(cls, err: TokbinError) -> Problem:
        return cls(code=err.code.id, what=err.what, why=err.why, fix=err.fix)

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "what": self.what, "why": self.why, "fix": self.fix}


def _issue_dict(issue: Issue) -> dict[str, object]:
    return {
        "code": issue.code,
        "level": issue.level,
        "message": issue.message,
        "count": issue.count,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitInfo:
    """Counters of one split, as recorded in ``meta.json``."""

    name: str
    n_items: int
    n_docs: int
    n_skipped: int
    n_shards: int
    n_bytes: int
    has_split_docs: bool
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_items": self.n_items,
            "n_docs": self.n_docs,
            "n_skipped": self.n_skipped,
            "n_shards": self.n_shards,
            "n_bytes": self.n_bytes,
            "has_split_docs": self.has_split_docs,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PartialInfo:
    """An unfinished write: ``<name>.partial/``.

    The counters come from ``checkpoint.json`` and are ``None`` without one (the write
    stopped before its first shard closed). ``resumable`` is true when a valid
    checkpoint exists.
    """

    path: Path
    split: str | None
    n_closed_shards: int
    n_items: int | None
    n_docs: int | None
    n_input_consumed: int | None
    updated_at: str
    resumable: bool
    problem: Problem | None
    #: The process writing it right now, as ``(pid, host)``; ``None`` if nobody is.
    writer: tuple[int, str] | None = None
    #: It was started by ``tokbin build``, whose settings are saved: ``--resume`` alone
    #: continues it.
    build: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "split": self.split,
            "n_closed_shards": self.n_closed_shards,
            "n_items": self.n_items,
            "n_docs": self.n_docs,
            "n_input_consumed": self.n_input_consumed,
            "updated_at": self.updated_at,
            "resumable": self.resumable,
            "problem": None if self.problem is None else self.problem.to_dict(),
            "writer": None
            if self.writer is None
            else {"pid": self.writer[0], "host": self.writer[1]},
            "build": self.build,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceInfo:
    """The state of one source directory.

    Fields taken from ``meta.json`` are ``None`` when it cannot be read.
    """

    path: Path
    name: str
    state: SourceState
    schema_version: int | None
    tokbin_version: str | None
    dtype: str | None
    vocab_size: int | None
    tokenizer_id: str | None
    tokenizer_hash: str | None
    eos_id: int | None
    bos_id: int | None
    splits: tuple[SplitInfo, ...]
    partial: PartialInfo | None
    #: Warnings and facts worth knowing.
    issues: tuple[Issue, ...]
    #: Errors that make the source unusable (empty for complete sources).
    problems: tuple[Problem, ...]

    @property
    def n_items(self) -> int:
        """Items in all splits."""
        return sum(s.n_items for s in self.splits)

    @property
    def n_docs(self) -> int:
        return sum(s.n_docs for s in self.splits)

    @property
    def n_shards(self) -> int:
        return sum(s.n_shards for s in self.splits)

    @property
    def n_bytes(self) -> int:
        return sum(s.n_bytes for s in self.splits)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "name": self.name,
            "state": self.state,
            "schema_version": self.schema_version,
            "tokbin_version": self.tokbin_version,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_hash": self.tokenizer_hash,
            "eos_id": self.eos_id,
            "bos_id": self.bos_id,
            "n_items": self.n_items,
            "n_docs": self.n_docs,
            "n_shards": self.n_shards,
            "n_bytes": self.n_bytes,
            "splits": [s.to_dict() for s in self.splits],
            "partial": None if self.partial is None else self.partial.to_dict(),
            "issues": [_issue_dict(i) for i in self.issues],
            "problems": [p.to_dict() for p in self.problems],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CorpusInfo:
    """An overview of a corpus directory: its sources, tokenizer and mixture."""

    path: Path
    sources: tuple[SourceInfo, ...]
    #: Normalized weights from ``mix.json``, or ``None`` without one.
    mix: tuple[tuple[str, float], ...] | None
    #: The tokenizer shared by all readable sources; ``None`` if they differ or none.
    tokenizer_id: str | None
    tokenizer_hash: str | None
    issues: tuple[Issue, ...]
    problems: tuple[Problem, ...]

    @property
    def n_items(self) -> int:
        return sum(s.n_items for s in self.sources)

    @property
    def n_bytes(self) -> int:
        return sum(s.n_bytes for s in self.sources)

    @property
    def n_warnings(self) -> int:
        own = sum(1 for i in self.issues if i.level == "warning")
        return own + sum(s.n_warnings for s in self.sources)

    @property
    def ready(self) -> bool:
        """Every source is complete and the corpus itself has no problems."""
        return (
            bool(self.sources)
            and not self.problems
            and all(s.state == "complete" for s in self.sources)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "ready": self.ready,
            "n_items": self.n_items,
            "n_bytes": self.n_bytes,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_hash": self.tokenizer_hash,
            "mix": None if self.mix is None else dict(self.mix),
            "sources": [s.to_dict() for s in self.sources],
            "issues": [_issue_dict(i) for i in self.issues],
            "problems": [p.to_dict() for p in self.problems],
        }


# --- sources --------------------------------------------------------------------------


def _is_source_dir(path: Path) -> bool:
    return (path / naming.META_JSON).is_file() or (path / naming.DATASET_JSON).is_file()


def _target_of(path: Path) -> Path:
    """``corpus/web.partial`` and ``corpus/web`` both mean the source ``corpus/web``."""
    if path.name.endswith(naming.PARTIAL_SUFFIX) and len(path.name) > len(naming.PARTIAL_SUFFIX):
        return path.with_name(path.name[: -len(naming.PARTIAL_SUFFIX)])
    return path


def _partial_dir(target: Path) -> Path:
    return target.with_name(target.name + naming.PARTIAL_SUFFIX)


def _mtime(path: Path) -> str:
    stamp = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _inspect_partial(partial: Path) -> PartialInfo:
    info = _inspect_partial_files(partial)
    if (partial / naming.BUILD_RECIPE).is_file():
        info = replace(info, build=True)
    owner = read_lock(partial)
    if owner is not None and owner.alive:
        return replace(info, writer=(owner.pid, owner.host))
    return info


def _inspect_partial_files(partial: Path) -> PartialInfo:
    n_closed = sum(1 for p in partial.glob("*.bin") if p.is_file())
    split = next(
        (s for s in naming.SPLITS if (partial / naming.offsets_raw_name(s)).exists()), None
    )
    if not (partial / naming.CHECKPOINT).is_file():
        return PartialInfo(
            path=partial,
            split=split,
            n_closed_shards=n_closed,
            n_items=None,
            n_docs=None,
            n_input_consumed=None,
            updated_at=_mtime(partial),
            resumable=False,
            problem=None,
        )
    try:
        cp = read_checkpoint(partial)
    except TokbinError as exc:
        return PartialInfo(
            path=partial,
            split=split,
            n_closed_shards=n_closed,
            n_items=None,
            n_docs=None,
            n_input_consumed=None,
            updated_at=_mtime(partial),
            resumable=False,
            problem=Problem.from_error(exc),
        )
    return PartialInfo(
        path=partial,
        split=cp.split,
        n_closed_shards=len(cp.closed_shards),
        n_items=cp.n_items,
        n_docs=cp.n_docs,
        n_input_consumed=cp.n_input_consumed,
        updated_at=cp.updated_at,
        resumable=True,
        problem=None,
    )


def _split_infos(meta: Meta) -> tuple[SplitInfo, ...]:
    return tuple(
        SplitInfo(
            name=s.name,
            n_items=s.n_items,
            n_docs=s.n_docs,
            n_skipped=s.n_skipped,
            n_shards=len(s.shards),
            n_bytes=sum(sh.n_bytes for sh in s.shards),
            has_split_docs=s.has_split_docs,
            created_at=s.created_at,
        )
        for s in meta.splits
    )


def _meta_issues(root: Path, meta: Meta) -> list[Issue]:
    issues: list[Issue] = []
    if meta.schema_version < SCHEMA_VERSION:
        issues.append(
            Issue(
                code=codes.SCHEMA_OUTDATED.id,
                level="warning",
                message=f"{codes.SCHEMA_OUTDATED.title}: schema {meta.schema_version}, "
                f"current {SCHEMA_VERSION}; convert it with `tokbin migrate {root}`",
                count=1,
            )
        )
    skipped = [s for s in meta.splits if s.n_skipped]
    if skipped:
        where = ", ".join(naming.skipped_name(s.name) for s in skipped)
        issues.append(
            Issue(
                code=codes.DOCS_WERE_SKIPPED.id,
                level="info",
                message=f"{codes.DOCS_WERE_SKIPPED.title}: the reasons are in {where}",
                count=sum(s.n_skipped for s in skipped),
            )
        )
    split_docs = [s.name for s in meta.splits if s.has_split_docs]
    if split_docs:
        issues.append(
            Issue(
                code=codes.DOCS_SPLIT_ACROSS_SHARDS.id,
                level="info",
                message=f"{codes.DOCS_SPLIT_ACROSS_SHARDS.title}: in {', '.join(split_docs)}; "
                "readers handle this transparently",
                count=1,
            )
        )
    return issues


def _state_of_error(exc: TokbinError) -> SourceState:
    if isinstance(exc, (SchemaVersionError, UnsupportedFeatureError)):
        return "unsupported"
    if exc.code is codes.SCHEMA_TOO_OLD:
        return "outdated"
    return "corrupt"


def _not_found(path: Path) -> ConfigError:
    return ConfigError(
        codes.SOURCE_NOT_FOUND,
        str(path),
        why="neither the source directory nor an unfinished write of it exists",
        fix="check the path; `tokbin ls <corpus>` lists the sources of a corpus",
    )


def _not_a_source(path: Path) -> ConfigError:
    names = sorted(e.name for e in path.iterdir() if e.is_dir() and _is_source_dir(e))
    hint = f"; it contains the sources {', '.join(names)}" if names else ""
    return ConfigError(
        codes.NOT_A_SOURCE,
        str(path),
        why=f"the directory has no {naming.META_JSON}{hint}",
        fix="pass a source directory, or use `tokbin info` on a corpus",
    )


@public_api
def inspect_source(path: str | Path) -> SourceInfo:
    """The state of the source at ``path`` (or of its unfinished write)."""
    target = _target_of(Path(path))
    partial_dir = _partial_dir(target)
    partial = _inspect_partial(partial_dir) if partial_dir.is_dir() else None
    empty = SourceInfo(
        path=target,
        name=target.name,
        state="partial",
        schema_version=None,
        tokbin_version=None,
        dtype=None,
        vocab_size=None,
        tokenizer_id=None,
        tokenizer_hash=None,
        eos_id=None,
        bos_id=None,
        splits=(),
        partial=partial,
        issues=(),
        problems=(),
    )
    if not target.is_dir():
        if partial is None:
            raise _not_found(target)
        return empty
    if not _is_source_dir(target):
        raise _not_a_source(target)

    issues: list[Issue] = []
    if partial is not None:
        issues.append(
            Issue(
                code=codes.PARTIAL_EXISTS.id,
                level="warning",
                message=f"{codes.PARTIAL_EXISTS.title}: {partial.path}",
                count=1,
            )
        )
    if (target / naming.DATASET_JSON).is_file():
        exc: TokbinError = UnsupportedFeatureError(
            codes.FEATURE_UNSUPPORTED,
            str(target / naming.DATASET_JSON),
            why="multi-stream (multimodal) sources are not supported by this version",
            fix="read this dataset with a tokbin release that supports multimodal sources",
        )
        return _replace_state(empty, "unsupported", issues, [exc])
    try:
        meta = read_meta(target, warn_outdated=False)
    except FormatError as exc:
        return _replace_state(empty, _state_of_error(exc), issues, [exc])
    except UnsupportedFeatureError as exc:
        return _replace_state(empty, "unsupported", issues, [exc])

    issues = _meta_issues(target, meta) + issues
    problems = check_source_files(target, meta)
    state: SourceState = "complete"
    if problems:
        state = "corrupt"
    elif meta.schema_version < SCHEMA_VERSION:
        state = "outdated"
    return SourceInfo(
        path=target,
        name=target.name,
        state=state,
        schema_version=meta.schema_version,
        tokbin_version=meta.tokbin_version,
        dtype=meta.dtype,
        vocab_size=meta.vocab_size,
        tokenizer_id=meta.tokenizer_id,
        tokenizer_hash=meta.tokenizer_hash,
        eos_id=meta.eos_id,
        bos_id=meta.bos_id,
        splits=_split_infos(meta),
        partial=partial,
        issues=tuple(issues),
        problems=tuple(Problem.from_error(p) for p in problems),
    )


def _replace_state(
    base: SourceInfo, state: SourceState, issues: list[Issue], errors: list[TokbinError]
) -> SourceInfo:
    return SourceInfo(
        path=base.path,
        name=base.name,
        state=state,
        schema_version=None,
        tokbin_version=None,
        dtype=None,
        vocab_size=None,
        tokenizer_id=None,
        tokenizer_hash=None,
        eos_id=None,
        bos_id=None,
        splits=(),
        partial=base.partial,
        issues=tuple(issues),
        problems=tuple(Problem.from_error(e) for e in errors),
    )


# --- corpora --------------------------------------------------------------------------


def _source_names(root: Path) -> list[str]:
    names: set[str] = set()
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        target = _target_of(entry)
        if not naming.is_valid_source_name(target.name):
            continue
        if target is not entry or _is_source_dir(entry):
            names.add(target.name)
    return sorted(names)


@public_api
def inspect_corpus(root: str | Path) -> CorpusInfo:
    """All sources of the corpus at ``root``, their tokenizer and the mixture."""
    root = Path(root)
    if not root.is_dir():
        raise ConfigError(
            codes.SOURCE_NOT_FOUND,
            str(root),
            why="the corpus directory does not exist",
            fix="check the path",
        )
    if _is_source_dir(root):
        raise ConfigError(
            codes.NOT_A_CORPUS,
            str(root),
            why=f"the directory has its own {naming.META_JSON}: it is a single source",
            fix=f"use `tokbin info {root}` or Dataset({str(root.parent)!r}).status({root.name!r})",
        )
    sources = tuple(inspect_source(root / name) for name in _source_names(root))
    issues: list[Issue] = []
    problems: list[Problem] = []

    hashes = {s.tokenizer_hash: s for s in sources if s.tokenizer_hash is not None}
    tokenizer_id = tokenizer_hash = None
    if len(hashes) == 1:
        ((tokenizer_hash, first),) = hashes.items()
        tokenizer_id = first.tokenizer_id
    elif len(hashes) > 1:
        detail = ", ".join(
            f"{s.name}={s.tokenizer_hash}" for s in sources if s.tokenizer_hash is not None
        )
        issues.append(
            Issue(
                code=codes.TOKENIZERS_DIFFER.id,
                level="warning",
                message=f"{codes.TOKENIZERS_DIFFER.title}: {detail}; they cannot be mixed",
                count=len(hashes),
            )
        )

    mix: tuple[tuple[str, float], ...] | None = None
    if (root / naming.MIX_JSON).is_file():
        try:
            mix = read_mix(root).normalized()
        except TokbinError as exc:
            problems.append(Problem.from_error(exc))
        else:
            known = {s.name for s in sources}
            missing = [name for name, _ in mix if name not in known]
            if missing:
                issues.append(
                    Issue(
                        code=codes.MIX_SOURCE_MISSING.id,
                        level="warning",
                        message=f"{codes.MIX_SOURCE_MISSING.title}: {', '.join(missing)}",
                        count=len(missing),
                    )
                )
    return CorpusInfo(
        path=root,
        sources=sources,
        mix=mix,
        tokenizer_id=tokenizer_id,
        tokenizer_hash=tokenizer_hash,
        issues=tuple(issues),
        problems=tuple(problems),
    )


@public_api
def describe(path: str | Path) -> SourceInfo | CorpusInfo:
    """A source if ``path`` is one (or an unfinished write of one), otherwise a corpus."""
    path = Path(path)
    target = _target_of(path)
    if (
        target is not path
        or not path.is_dir()
        or _is_source_dir(path)
        or _partial_dir(path).is_dir()
    ):
        return inspect_source(path)
    return inspect_corpus(path)
