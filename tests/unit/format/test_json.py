from __future__ import annotations

from pathlib import Path

import pytest

from tokbin import codes
from tokbin.errors import FormatError
from tokbin.format._json import canonical_dumps, read_json_bounded, write_json_atomic


def test_canonical_dumps_is_order_independent_and_stable() -> None:
    # Non-ASCII text is kept as is, not escaped.
    a = canonical_dumps({"b": 1, "a": [1, {"d": "\u00fc", "c": None}]})
    b = canonical_dumps({"a": [1, {"c": None, "d": "\u00fc"}], "b": 1})
    assert a == b == '{"a":[1,{"c":null,"d":"\u00fc"}],"b":1}'


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    write_json_atomic(path, {"k": [1, 2], "s": "\u00fc"})
    assert read_json_bounded(path) == {"k": [1, 2], "s": "\u00fc"}
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FormatError) as info:
        read_json_bounded(tmp_path / "nope.json")
    assert info.value.code is codes.METADATA_MISSING


def test_too_large(tmp_path: Path) -> None:
    path = tmp_path / "big.json"
    path.write_text('{"a": "' + "x" * 200 + '"}')
    with pytest.raises(FormatError) as info:
        read_json_bounded(path, limit=100)
    assert info.value.code is codes.METADATA_TOO_LARGE


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b'{"a": 1, "a": 2}',
        b'{"a": NaN}',
        b'{"a": Infinity}',
        b"\xff\xfe garbage",
    ],
)
def test_rejects_malformed(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(FormatError) as info:
        read_json_bounded(path)
    assert info.value.code is codes.METADATA_NOT_JSON


def test_write_refuses_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "x.json", {"a": float("nan")})
    assert not (tmp_path / "x.json").exists()
