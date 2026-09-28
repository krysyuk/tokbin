"""Standard inputs: directories of text files and JSON Lines (spec 11, 12.3).

::

    ds.write("code", txt_dir("data/code"), tokenizer)
    ds.write("web", jsonl("data/web", field="text"), tokenizer)

Both yield ``(doc_id, text)`` in a deterministic order: files sorted by their relative
path written with ``/``, the same on every operating system. Resuming a write depends
on that order.

Text files are yielded as ``bytes``, so a file with broken UTF-8 becomes a skipped
document instead of a crashed generator. A JSON line that cannot be parsed, or has no text in
``field``, is yielded as a :class:`~tokbin.SkipDocument` and recorded in
``skipped.jsonl`` with the reason. Blank lines are not records and are ignored.

Both are iterables that can be iterated more than once, and they report progress in
bytes of input read (``bytes_read`` of ``total_bytes``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import ConfigError
from tokbin.write.documents import SkipDocument

__all__ = ["JsonlInput", "TxtDirInput", "jsonl", "txt_dir"]

InputDoc = tuple[str, bytes] | tuple[str, str] | SkipDocument


def _files(root: Path, pattern: str, what: str) -> list[tuple[str, Path]]:
    """``(relative posix path, path)`` of matching files, in a portable order."""
    if root.is_file():
        return [(root.name, root)]
    if not root.is_dir():
        raise ConfigError(
            codes.NO_INPUT_FILES,
            str(root),
            why="the path does not exist",
            fix=f"pass a directory of {what} files (or one such file)",
        )
    found = [(p.relative_to(root).as_posix(), p) for p in root.rglob(pattern) if p.is_file()]
    if not found:
        raise ConfigError(
            codes.NO_INPUT_FILES,
            str(root),
            why=f"no files matching {pattern!r} were found in it or below it",
            fix=f"check the path, or pass a directory of {what} files",
        )
    # Sort by code points of the relative path: Path ordering is case-insensitive on
    # Windows, and resume needs the same order everywhere.
    return sorted(found, key=lambda item: item[0])


class _FileInput:
    """Common part: the file list, sizes and progress counters."""

    def __init__(self, root: str | Path, pattern: str, what: str) -> None:
        self.root = Path(root)
        self.files = _files(self.root, pattern, what)
        self.total_bytes = sum(p.stat().st_size for _, p in self.files)
        #: Bytes of input consumed by the current iteration.
        self.bytes_read = 0
        #: The file being read.
        self.current = ""

    def __iter__(self) -> Iterator[InputDoc]:
        self.bytes_read = 0
        return self._iterate()

    def _iterate(self) -> Iterator[InputDoc]:  # pragma: no cover - overridden
        raise NotImplementedError


class TxtDirInput(_FileInput):
    """Every ``*.txt`` file below a directory is one document; its id is the path."""

    def __init__(self, root: str | Path, *, pattern: str = "*.txt") -> None:
        super().__init__(root, pattern, "text")

    def _iterate(self) -> Iterator[InputDoc]:
        for rel, path in self.files:
            self.current = rel
            data = path.read_bytes()
            self.bytes_read += len(data)
            yield rel, data


class JsonlInput(_FileInput):
    """Every line of ``*.jsonl`` files is one document; its id is ``<path>:<line>``.

    Lines are numbered from 0 and blank lines keep their numbers, so an id points at
    the line in the file.
    """

    def __init__(self, root: str | Path, *, field: str = "text", pattern: str = "*.jsonl") -> None:
        if not isinstance(field, str) or not field:
            raise ConfigError(
                codes.CONFIG_VALUE_INVALID,
                f"field={field!r}",
                why="field must be the name of a JSON key",
                fix='pass the key that holds the text, e.g. field="text"',
            )
        super().__init__(root, pattern, "JSON Lines")
        self.field = field

    def _iterate(self) -> Iterator[InputDoc]:
        for rel, path in self.files:
            self.current = rel
            with path.open("rb") as f:
                for i, line in enumerate(f):
                    self.bytes_read += len(line)
                    if not line.strip():
                        continue
                    yield self._record(f"{rel}:{i}", line)

    def _record(self, doc_id: str, line: bytes) -> InputDoc:
        try:
            record = json.loads(line)
        except (ValueError, RecursionError) as exc:
            # ValueError covers invalid JSON and invalid UTF-8 alike.
            return SkipDocument(doc_id, f"not a JSON line: {exc}")
        if not isinstance(record, dict):
            return SkipDocument(doc_id, f"expected a JSON object, got {type(record).__name__}")
        if self.field not in record:
            return SkipDocument(doc_id, f"no {self.field!r} field")
        text = record[self.field]
        if not isinstance(text, str):
            return SkipDocument(
                doc_id, f"{self.field!r} is {type(text).__name__}, expected a string"
            )
        return doc_id, text


@public_api
def txt_dir(root: str | Path, *, pattern: str = "*.txt") -> TxtDirInput:
    """Documents from text files below ``root`` (recursively), one per file."""
    return TxtDirInput(root, pattern=pattern)


@public_api
def jsonl(root: str | Path, field: str = "text", *, pattern: str = "*.jsonl") -> JsonlInput:
    """Documents from the ``field`` of every line of JSON Lines files below ``root``.

    ``root`` may also be a single ``.jsonl`` file.
    """
    return JsonlInput(root, field=field, pattern=pattern)
