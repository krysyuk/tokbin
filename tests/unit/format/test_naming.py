from __future__ import annotations

import os
from pathlib import Path

import pytest

from tokbin import codes
from tokbin.errors import ConfigError, FormatError
from tokbin.format import naming


def test_file_names_follow_spec() -> None:
    assert naming.shard_name("train", 0) == "train-00000.bin"
    assert naming.shard_name("valid", 12) == "valid-00012.bin"
    assert naming.offsets_name("train") == "train-offsets.npy"
    assert naming.ids_name("train") == "train-ids.jsonl"
    assert naming.ids_idx_name("train") == "train-ids.idx.npy"
    assert naming.skipped_name("test") == "test-skipped.jsonl"


def test_check_split() -> None:
    assert naming.check_split("valid") == "valid"
    with pytest.raises(ConfigError) as info:
        naming.check_split("dev")
    assert info.value.code is codes.SPLIT_INVALID


@pytest.mark.parametrize("name", ["web", "code-v2", "books_ru", "a.b", "0"])
def test_valid_source_names(name: str) -> None:
    assert naming.check_source_name(name) == name


@pytest.mark.parametrize(
    "name", ["", ".", "..", ".hidden", "a/b", "a\\b", "web.partial", "-x", "\u0441"]
)
def test_invalid_source_names(name: str) -> None:
    with pytest.raises(ConfigError) as info:
        naming.check_source_name(name)
    assert info.value.code is codes.SOURCE_NAME_INVALID


def test_resolve_inside_accepts_nested(tmp_path: Path) -> None:
    assert naming.resolve_inside(tmp_path, "tokenizer/tokenizer.json") == (
        tmp_path / "tokenizer" / "tokenizer.json"
    )


@pytest.mark.parametrize(
    "rel",
    ["", "/etc/passwd", "../x", "a/../../x", "C:\\Windows", "C:x", "..\\x", "\\\\server\\s"],
)
def test_resolve_inside_rejects_escapes(tmp_path: Path, rel: str) -> None:
    with pytest.raises(FormatError) as info:
        naming.resolve_inside(tmp_path, rel)
    assert info.value.code is codes.PATH_ESCAPES_ROOT


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_resolve_inside_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    (root / "train-00000.bin").symlink_to(outside)
    with pytest.raises(FormatError):
        naming.resolve_inside(root, "train-00000.bin")
