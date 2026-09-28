"""Converting a source to the current format schema (spec 13.3).

Migrations form a chain: :data:`MIGRATIONS` maps schema ``N`` to the step that turns
an ``N`` source into an ``N + 1`` one. ``migrate_source`` applies the steps in order.

By default the result is a new copy next to the original (``<name>-v<schema>``); the
original is replaced only with ``in_place=True``. Either way the steps run on a copy in
``<name>.partial/``, the result is checked like a freshly written source, and only then
is it renamed into place: user data is never rewritten silently or left half-converted.

The copy is made of hard links where possible, so it costs no space. A migration step
must therefore never modify a file in place: it writes a new file and replaces the old
name, which leaves the original source untouched.

Schema 1 is the first and current schema, so the chain is empty for now; the machinery
is in place (and tested) for the first format change.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import ConfigError, FormatError, SchemaVersionError
from tokbin.format import naming, schema
from tokbin.format._fields import Fields
from tokbin.format._json import read_json_bounded
from tokbin.format.meta import read_meta
from tokbin.read._files import check_source_files
from tokbin.write.lock import WriteLock
from tokbin.write.partial import create_partial, swap_in

__all__ = ["MIGRATIONS", "MigrateResult", "Migration", "migrate_source"]

#: A step: converts the source in the given directory from schema N to N + 1.
Migration = Callable[[Path], None]

#: Registered steps, keyed by the schema they start from.
MIGRATIONS: Mapping[int, Migration] = MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class MigrateResult:
    path: Path
    from_version: int
    to_version: int

    @property
    def migrated(self) -> bool:
        return self.from_version != self.to_version

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "from_version": self.from_version,
            "to_version": self.to_version,
            "migrated": self.migrated,
        }


def _schema_of(root: Path) -> int:
    """Only the version field: an old ``meta.json`` may not parse as the current one."""
    path = root / naming.META_JSON
    return Fields(read_json_bounded(path), str(path)).get_int("schema_version", minimum=1)


def _link_tree(src: Path, dst: Path) -> None:
    def link_or_copy(a: str, b: str) -> None:
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)

    shutil.copytree(src, dst, copy_function=link_or_copy, dirs_exist_ok=True)


@public_api
def migrate_source(
    path: str | Path, out: str | Path | None = None, *, in_place: bool = False
) -> MigrateResult:
    """Convert the source at ``path`` to the current schema.

    Writes ``out`` (default ``<name>-v<schema>`` next to it), or replaces the source
    itself with ``in_place=True``. A source already at the current schema is left alone.
    """
    if in_place and out is not None:
        raise ConfigError(
            codes.CONFIG_VALUE_INVALID,
            "out and in_place",
            why="a migration either writes a copy or replaces the source",
            fix="pass out=... or in_place=True, not both",
        )
    root = Path(path)
    if not (root / naming.META_JSON).is_file():
        schema.source_kind(root)  # the precise error: missing, multimodal...
    current = schema.SCHEMA_VERSION
    version = _schema_of(root)
    if version > current:
        raise SchemaVersionError(
            codes.SCHEMA_TOO_NEW,
            f"{root}: schema_version {version}",
            why=f"this tokbin knows schema versions up to {current}",
            fix="upgrade tokbin: pip install -U tokbin",
        )
    if version == current:
        return MigrateResult(path=root, from_version=version, to_version=version)
    steps = [MIGRATIONS.get(v) for v in range(version, current)]
    if any(step is None for step in steps):
        raise FormatError(
            codes.SCHEMA_TOO_OLD,
            f"{root}: schema_version {version}",
            why=f"there is no conversion from schema {version} in this tokbin",
            fix="convert it with an older tokbin release first",
        )
    target = (
        root
        if in_place
        else Path(out)
        if out is not None
        else root.with_name(f"{root.name}-v{current}")
    )
    naming.check_source_name(target.name)
    if not in_place and target.exists():
        raise ConfigError(
            codes.OUTPUT_EXISTS,
            str(target),
            why="refusing to overwrite it",
            fix="choose another output, or remove it first",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = create_partial(target)
    lock = WriteLock(partial)
    lock.acquire()
    try:
        _link_tree(root, partial)
        for step in steps:
            step(partial)  # type: ignore[misc]
        meta = read_meta(partial)
        if meta.schema_version != current:
            raise FormatError(
                codes.METADATA_INCONSISTENT,
                str(partial / naming.META_JSON),
                why=f"after migration the schema is {meta.schema_version}, not {current}",
                fix="this is a bug in a migration step; please report it",
            )
        problems = check_source_files(partial, meta)
        if problems:
            raise problems[0]
        lock.release()
        (partial / naming.LOCK).unlink(missing_ok=True)
        swap_in(partial, target)
    except BaseException:
        lock.release()
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return MigrateResult(path=target, from_version=version, to_version=current)
