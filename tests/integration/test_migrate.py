"""migrate: the machinery, exercised with a pretend schema 2 (spec 13.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokbin import (
    ConfigError,
    FormatError,
    SchemaVersionError,
    codes,
    inspect_source,
    migrate_source,
    read_source,
)
from tokbin.format import schema
from tokbin.format._json import write_json_atomic
from tokbin.ops import inspect as inspect_mod
from tokbin.ops import migrate as migrate_mod

from support import FIXTURE, copy_fixture


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return copy_fixture(tmp_path / "web")


def test_current_schema_is_left_alone(source: Path) -> None:
    before = {p.name: p.stat().st_mtime_ns for p in source.iterdir()}
    result = migrate_source(source)
    assert not result.migrated
    assert (result.from_version, result.to_version) == (1, 1)
    assert {p.name: p.stat().st_mtime_ns for p in source.iterdir()} == before
    assert not (source.parent / "web-v1").exists()


def _step_1_to_2(root: Path) -> None:
    """A sample step: bump the version (writes a new file, never edits in place)."""
    meta = json.loads((root / "meta.json").read_text())
    meta["schema_version"] = 2
    (root / "meta.json").unlink()  # the name may be a hard link to the original
    write_json_atomic(root / "meta.json", meta)


@pytest.fixture
def schema_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(schema, "SUPPORTED_SCHEMAS", frozenset({1, 2}))
    monkeypatch.setattr(inspect_mod, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(migrate_mod, "MIGRATIONS", {1: _step_1_to_2})


@pytest.mark.usefixtures("schema_2")
def test_copy_by_default(source: Path) -> None:
    original = (source / "meta.json").read_bytes()
    assert inspect_source(source).state == "outdated"
    result = migrate_source(source)
    assert result.migrated
    assert result.path == source.parent / "web-v2"
    assert (source / "meta.json").read_bytes() == original  # the original is untouched
    assert json.loads((result.path / "meta.json").read_text())["schema_version"] == 2
    assert inspect_source(result.path).state == "complete"
    with pytest.warns(FutureWarning, match="tokbin migrate"):  # the fixture is schema 1
        original_items = read_source(FIXTURE)[:]
    assert (read_source(result.path)[:] == original_items).all()
    assert not list(source.parent.glob("*.partial"))

    with pytest.raises(ConfigError) as err:
        migrate_source(source)  # the copy exists
    assert err.value.code is codes.OUTPUT_EXISTS


@pytest.mark.usefixtures("schema_2")
def test_in_place(source: Path) -> None:
    result = migrate_source(source, in_place=True)
    assert result.path == source
    assert json.loads((source / "meta.json").read_text())["schema_version"] == 2
    assert not list(source.parent.glob(".web.old-*"))
    with pytest.raises(ConfigError):
        migrate_source(source, source.parent / "x", in_place=True)


@pytest.mark.usefixtures("schema_2")
def test_failed_step_leaves_nothing(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(root: Path) -> None:
        (root / "train-00000.bin").unlink()
        _step_1_to_2(root)

    monkeypatch.setattr(migrate_mod, "MIGRATIONS", {1: broken})
    with pytest.raises(Exception, match="TB-I301"):
        migrate_source(source, in_place=True)
    assert (source / "train-00000.bin").exists()  # the original is intact
    assert json.loads((source / "meta.json").read_text())["schema_version"] == 1
    assert not list(source.parent.glob("*.partial"))


def test_unknown_versions(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meta = json.loads((source / "meta.json").read_text())
    meta["schema_version"] = 7
    (source / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SchemaVersionError):
        migrate_source(source)
    monkeypatch.setattr(schema, "SCHEMA_VERSION", 9)
    with pytest.raises(FormatError) as err:
        migrate_source(source)  # no chain from 7
    assert err.value.code is codes.SCHEMA_TOO_OLD


def test_not_a_source(tmp_path: Path) -> None:
    with pytest.raises(FormatError):
        migrate_source(tmp_path)
