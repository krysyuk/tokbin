"""Standard inputs: deterministic order, unreadable records become SkipDocument."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokbin import ConfigError, SkipDocument, codes, jsonl, txt_dir


def test_txt_dir_order_and_ids(tmp_path: Path) -> None:
    for name in ("b.txt", "A.txt", "sub/a.txt", "sub/Z/x.txt", "notes.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    docs = list(txt_dir(tmp_path))
    # Code point order of the relative posix path, the same on every OS.
    assert [d[0] for d in docs] == ["A.txt", "b.txt", "sub/Z/x.txt", "sub/a.txt"]
    assert docs[0] == ("A.txt", b"A.txt")


def test_txt_dir_is_reiterable_and_reports_progress(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"12345")
    (tmp_path / "b.txt").write_bytes(b"678")
    source = txt_dir(tmp_path)
    assert source.total_bytes == 8
    # Read the counters while iterating: they follow the document just yielded.
    seen = [(source.bytes_read, source.current) for _ in source]
    assert seen == [(5, "a.txt"), (8, "b.txt")]
    assert list(source) == list(source)


def test_single_file_and_errors(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_bytes(b"x")
    assert list(txt_dir(tmp_path / "one.txt")) == [("one.txt", b"x")]
    with pytest.raises(ConfigError) as err:
        txt_dir(tmp_path / "missing")
    assert err.value.code is codes.NO_INPUT_FILES
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError) as err:
        jsonl(tmp_path / "empty")
    assert err.value.code is codes.NO_INPUT_FILES
    with pytest.raises(ConfigError):
        jsonl(tmp_path, field="")


def test_jsonl_records(tmp_path: Path) -> None:
    lines = [
        b'{"text": "hello", "id": 1}',
        b"",
        b"   ",
        b"not json",
        b'{"body": "x"}',
        b'{"text": 5}',
        b"[1, 2]",
        b'{"text": "caf\\u00e9"}',
        b'{"text": "\xff"}',
    ]
    (tmp_path / "b.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    (tmp_path / "a.jsonl").write_bytes(b'{"text": "first"}\n{"text": ""}')  # no final newline
    docs = list(jsonl(tmp_path))
    assert docs[:2] == [("a.jsonl:0", "first"), ("a.jsonl:1", "")]
    assert docs[2] == ("b.jsonl:0", "hello")
    kinds = [(d[0], type(d).__name__) for d in docs[3:]]
    assert kinds == [
        ("b.jsonl:3", "SkipDocument"),
        ("b.jsonl:4", "SkipDocument"),
        ("b.jsonl:5", "SkipDocument"),
        ("b.jsonl:6", "SkipDocument"),
        ("b.jsonl:7", "tuple"),
        ("b.jsonl:8", "SkipDocument"),
    ]
    reasons = [d.reason for d in docs if isinstance(d, SkipDocument)]
    assert reasons[1] == "no 'text' field"
    assert reasons[2] == "'text' is int, expected a string"
    assert reasons[3] == "expected a JSON object, got list"
    assert docs[7] == ("b.jsonl:7", "café")


def test_jsonl_other_field_and_pattern(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text('{"content": "a"}\n')
    assert list(jsonl(tmp_path, field="content", pattern="*.json")) == [("x.json:0", "a")]
