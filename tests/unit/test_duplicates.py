from __future__ import annotations

import hashlib

import pytest

from tokbin.write import duplicates
from tokbin.write.duplicates import IdHashes


def _collect(ids: list[str]) -> tuple[IdHashes, dict[int, str]]:
    hashes = IdHashes()
    for doc_id in ids:
        hashes.add(doc_id)
    return hashes, dict(enumerate(ids))


def test_no_duplicates() -> None:
    hashes, by_index = _collect([f"doc-{i}" for i in range(1000)])
    report = hashes.report(by_index.__getitem__)
    assert report.count == 0
    assert report.example is None


def test_counts_every_repeat_and_names_one() -> None:
    ids = ["a", "b", "a", "c", "a", "b"]
    hashes, by_index = _collect(ids)
    report = hashes.report(by_index.__getitem__)
    assert report.count == 3  # "a" twice more, "b" once more
    assert report.example in {"a", "b"}


def test_spans_several_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(duplicates, "_CHUNK", 8)
    ids = [f"doc-{i % 20}" for i in range(50)]
    hashes, by_index = _collect(ids)
    assert len(hashes) == 50
    assert hashes.report(by_index.__getitem__).count == 30


def test_hash_collision_is_not_a_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    class Constant:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def digest(self) -> bytes:
            return b"\x00" * 8

    monkeypatch.setattr(hashlib, "blake2b", Constant)
    hashes, by_index = _collect(["x", "y", "z"])
    report = hashes.report(by_index.__getitem__)
    assert report.count == 0
    assert report.example is None


def test_many_repeats_use_the_hash_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(duplicates, "_MAX_CONFIRM", 3)
    hashes, by_index = _collect(["same"] * 10)
    report = hashes.report(by_index.__getitem__)
    assert report.count == 9
    assert report.example == "same"
