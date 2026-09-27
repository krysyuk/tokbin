"""Full verification: every shard is hashed, every index checked value by value."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import pytest

from tokbin import (
    Dataset,
    FormatError,
    SchemaVersionError,
    VerifyProgress,
    codes,
    verify_source,
)
from tokbin.ops import verify as verify_mod

from support import copy_fixture


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return copy_fixture(tmp_path / "corpus" / "web")


def _codes(report: object) -> list[str]:
    return [p.code for p in report.problems]  # type: ignore[attr-defined]


def test_intact_source(
    source: Path, no_side_effects: Callable[[], AbstractContextManager[None]]
) -> None:
    with no_side_effects():
        report = verify_source(source)
    assert report.ok
    assert report.n_bytes_checked == 720
    assert [(s.split, s.name, s.status) for s in report.shards] == [
        ("train", "train-00000.bin", "ok"),
        ("train", "train-00001.bin", "ok"),
        ("train", "train-00002.bin", "ok"),
        ("valid", "valid-00000.bin", "ok"),
    ]
    for shard in report.shards:
        digest = hashlib.sha256((source / shard.name).read_bytes()).hexdigest()
        assert shard.actual_sha256 == shard.expected_sha256 == digest
    assert Dataset(source.parent).verify("web") == report


def test_flipped_byte_is_found(source: Path) -> None:
    path = source / "train-00001.bin"
    data = bytearray(path.read_bytes())
    data[10] ^= 0xFF
    path.write_bytes(bytes(data))
    report = verify_source(source)
    assert not report.ok
    assert report.n_bad_shards == 1
    bad = report.shards[1]
    assert bad.status == "corrupt"
    assert bad.actual_sha256 == hashlib.sha256(bytes(data)).hexdigest()
    assert _codes(report) == ["TB-I302"]
    assert str(path) in report.problems[0].what


def test_every_problem_is_listed_in_one_run(source: Path) -> None:
    (source / "train-00000.bin").unlink()
    with (source / "valid-00000.bin").open("ab") as f:
        f.write(b"\0\0")
    (source / "tokenizer" / "tokenizer.json").unlink()
    report = verify_source(source)
    assert [s.status for s in report.shards] == ["missing", "ok", "ok", "wrong_size"]
    assert _codes(report) == ["TB-I301", "TB-I303", "TB-I306"]
    assert report.n_bytes_checked == 256 + 94  # only the files that could be hashed


def _rewrite_index(path: Path, change: Callable[[np.ndarray], None]) -> None:
    arr = np.load(path, allow_pickle=False).copy()
    change(arr)
    np.save(path, arr, allow_pickle=False)


def test_decreasing_offsets(source: Path) -> None:
    def swap(arr: np.ndarray) -> None:
        arr[5], arr[6] = arr[6], arr[5]

    _rewrite_index(source / "train-offsets.npy", swap)
    report = verify_source(source)
    assert _codes(report) == ["TB-I305"]
    assert "offsets decrease" in report.problems[0].why


def test_ids_index_not_at_line_starts(source: Path) -> None:
    def shift(arr: np.ndarray) -> None:
        arr[3] += 1

    _rewrite_index(source / "train-ids.idx.npy", shift)
    report = verify_source(source)
    assert _codes(report) == ["TB-I305"]
    assert "lines do not match" in report.problems[0].why


def test_ids_index_not_increasing(source: Path) -> None:
    def repeat(arr: np.ndarray) -> None:
        arr[3] = arr[2]

    _rewrite_index(source / "train-ids.idx.npy", repeat)
    report = verify_source(source)
    assert "do not increase" in report.problems[0].why


def test_skipped_list(source: Path) -> None:
    (source / "valid-skipped.jsonl").write_text('{"input_index": 0}\n')
    (source / "train-skipped.jsonl").unlink()
    report = verify_source(source)
    assert _codes(report) == ["TB-I304", "TB-I305"]


def test_index_checks_across_chunks(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_mod, "_INDEX_CHUNK", 3)
    assert verify_source(source).ok

    def swap(arr: np.ndarray) -> None:  # exactly at a chunk edge
        arr[3], arr[4] = arr[4], arr[3]

    _rewrite_index(source / "train-offsets.npy", swap)
    assert _codes(verify_source(source)) == ["TB-I305"]


def test_damaged_meta_is_a_problem(source: Path) -> None:
    (source / "meta.json").write_text("[]")
    report = verify_source(source)
    assert report.shards == ()
    assert _codes(report) == ["TB-F103"]


def test_errors_that_prevent_verification(source: Path, tmp_path: Path) -> None:
    meta = json.loads((source / "meta.json").read_text())
    meta["schema_version"] = 99
    (source / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SchemaVersionError):
        verify_source(source)
    with pytest.raises(FormatError) as err:
        verify_source(tmp_path)
    assert err.value.code is codes.METADATA_MISSING


def test_progress(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_mod, "_CHUNK_BYTES", 100)
    (source / "train-00002.bin").unlink()
    seen: list[VerifyProgress] = []
    verify_source(source, progress=seen.append)
    done = [p.done_bytes for p in seen]
    assert done == sorted(done)
    assert all(p.total_bytes == 720 for p in seen)
    assert done[-1] == 720  # a missing shard still counts, so progress reaches the end
    assert [p.file for p in seen][:3] == ["train-00000.bin"] * 3
