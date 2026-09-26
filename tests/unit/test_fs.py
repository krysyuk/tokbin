from __future__ import annotations

from pathlib import Path

import pytest

from tokbin import _fs


def test_atomic_write_creates_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / "meta.json"
    _fs.atomic_write_bytes(target, b"one")
    assert target.read_bytes() == b"one"
    _fs.atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json"]


def test_atomic_write_failure_keeps_old_and_leaves_no_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "meta.json"
    target.write_bytes(b"old")

    def fail(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(_fs, "fsync_file", fail)
    with pytest.raises(OSError, match="disk full"):
        _fs.atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json"]


def test_fsync_dir_and_free_bytes(tmp_path: Path) -> None:
    _fs.fsync_dir(tmp_path)
    assert _fs.free_bytes(tmp_path) > 0
