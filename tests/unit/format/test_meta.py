from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tokbin import codes
from tokbin.errors import (
    FormatError,
    InternalError,
    SchemaVersionError,
    UnsupportedFeatureError,
)
from tokbin.format.meta import Meta, SplitMeta, read_meta, write_meta


def _parse(d: dict[str, Any]) -> Meta:
    return Meta.from_dict(d, where="meta.json")


def _field_error(d: dict[str, Any]) -> FormatError:
    with pytest.raises(FormatError) as info:
        _parse(d)
    return info.value


def test_parse_valid(meta_dict: dict[str, Any]) -> None:
    meta = _parse(meta_dict)
    assert meta.dtype == "uint16"
    assert meta.np_dtype == np.dtype("<u2")
    assert meta.itemsize == 2
    assert meta.split_names == ("train", "valid")
    train = meta.split("train")
    assert train.n_items == 150
    assert [s.name for s in train.shards] == ["train-00000.bin", "train-00001.bin"]
    with pytest.raises(KeyError):
        meta.split("test")


def test_roundtrip_dict(meta_dict: dict[str, Any]) -> None:
    assert _parse(meta_dict).to_dict() == meta_dict


def test_splits_are_kept_in_canonical_order(meta_dict: dict[str, Any]) -> None:
    splits = meta_dict["splits"]
    meta_dict["splits"] = {"valid": splits["valid"], "train": splits["train"]}
    assert _parse(meta_dict).split_names == ("train", "valid")


def test_with_split_adds_and_replaces(meta_dict: dict[str, Any]) -> None:
    del meta_dict["splits"]["valid"]
    meta = _parse(meta_dict)
    empty = SplitMeta(
        name="test",
        created_at="2026-09-26T12:00:00Z",
        n_items=0,
        n_docs=0,
        n_skipped=0,
        has_split_docs=False,
        shards=(),
    )
    added = meta.with_split(empty)
    assert added.split_names == ("train", "test")
    replaced = added.with_split(replace(empty, n_skipped=5))
    assert replaced.split_names == ("train", "test")
    assert replaced.split("test").n_skipped == 5


def test_file_roundtrip(
    tmp_path: Path,
    meta_dict: dict[str, Any],
    no_side_effects: Callable[[], AbstractContextManager[None]],
) -> None:
    meta = _parse(meta_dict)
    with no_side_effects():
        write_meta(tmp_path, meta)
        assert read_meta(tmp_path) == meta
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json"]


def test_write_refuses_inconsistent_meta(tmp_path: Path, meta_dict: dict[str, Any]) -> None:
    meta = _parse(meta_dict)
    broken = replace(meta, vocab_size=10**6)
    with pytest.raises(InternalError):
        write_meta(tmp_path, broken)
    assert not (tmp_path / "meta.json").exists()


# --- field-level validation -------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vocab_size", "50257"),
        ("vocab_size", True),
        ("vocab_size", 0),
        ("vocab_size", 1.5),
        ("dtype", "int16"),
        ("dtype", None),
        ("tokenizer_hash", "XYZ"),
        ("tokenizer_hash", "a3f2c81b9e04d5a"),
        ("eos_id", -1),
        ("eos_id", "1"),
        ("tokbin_version", 3),
        ("splits", []),
        ("item_shape", "[]"),
    ],
)
def test_invalid_top_level_field(meta_dict: dict[str, Any], key: str, value: object) -> None:
    meta_dict[key] = value
    err = _field_error(meta_dict)
    assert err.code is codes.METADATA_FIELD_INVALID
    assert key in str(err)


@pytest.mark.parametrize(
    "key", ["schema_version", "dtype", "vocab_size", "tokenizer_hash", "eos_id", "splits"]
)
def test_missing_field(meta_dict: dict[str, Any], key: str) -> None:
    del meta_dict[key]
    err = _field_error(meta_dict)
    assert err.code is codes.METADATA_FIELD_INVALID
    assert "missing" in str(err)


def test_not_an_object() -> None:
    err = _field_error([1, 2])  # type: ignore[arg-type]
    assert err.code is codes.METADATA_FIELD_INVALID


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("n_items", -1),
        ("has_split_docs", 1),
        ("created_at", "2026-09-26 10:22:31"),
        ("shards", {}),
    ],
)
def test_invalid_split_field(meta_dict: dict[str, Any], key: str, value: object) -> None:
    meta_dict["splits"]["train"][key] = value
    assert _field_error(meta_dict).code is codes.METADATA_FIELD_INVALID


@pytest.mark.parametrize(
    ("key", "value"),
    [("sha256", "abc"), ("sha256", "A" * 64), ("n_items", 0), ("name", 5)],
)
def test_invalid_shard_field(meta_dict: dict[str, Any], key: str, value: object) -> None:
    meta_dict["splits"]["train"]["shards"][0][key] = value
    assert _field_error(meta_dict).code is codes.METADATA_FIELD_INVALID


def test_unknown_split(meta_dict: dict[str, Any]) -> None:
    meta_dict["splits"]["dev"] = meta_dict["splits"]["valid"]
    assert _field_error(meta_dict).code is codes.METADATA_FIELD_INVALID


def test_unknown_top_level_keys_are_ignored(meta_dict: dict[str, Any]) -> None:
    meta_dict["comment"] = "added by hand"
    _parse(meta_dict)


# --- schema and unsupported features --------------------------------------------------


def test_newer_schema_wins_over_field_errors(meta_dict: dict[str, Any]) -> None:
    meta_dict["schema_version"] = 99
    meta_dict["dtype"] = "garbage"
    with pytest.raises(SchemaVersionError):
        _parse(meta_dict)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("modality", "image"),
        ("packing", "record"),
        ("byteorder", "big"),
        ("item_shape", [768]),
    ],
)
def test_unsupported_features(meta_dict: dict[str, Any], key: str, value: object) -> None:
    meta_dict[key] = value
    with pytest.raises(UnsupportedFeatureError) as info:
        _parse(meta_dict)
    assert info.value.code is codes.FEATURE_UNSUPPORTED


# --- cross-field consistency ----------------------------------------------------------


def _inconsistent(meta_dict: dict[str, Any]) -> FormatError:
    err = _field_error(meta_dict)
    assert err.code is codes.METADATA_INCONSISTENT
    return err


def test_vocab_must_fit_dtype(meta_dict: dict[str, Any]) -> None:
    meta_dict["vocab_size"] = 70000
    _inconsistent(meta_dict)


def test_explicit_wider_dtype_is_consistent(meta_dict: dict[str, Any]) -> None:
    meta_dict["dtype"] = "uint32"
    for split in meta_dict["splits"].values():
        for shard in split["shards"]:
            shard["n_bytes"] = shard["n_items"] * 4
    assert _parse(meta_dict).itemsize == 4


def test_eos_must_be_in_vocab(meta_dict: dict[str, Any]) -> None:
    meta_dict["eos_id"] = 50257
    _inconsistent(meta_dict)


def test_no_splits(meta_dict: dict[str, Any]) -> None:
    meta_dict["splits"] = {}
    _inconsistent(meta_dict)


def test_shard_names_must_be_sequential(meta_dict: dict[str, Any]) -> None:
    meta_dict["splits"]["train"]["shards"][1]["name"] = "train-00002.bin"
    _inconsistent(meta_dict)


@pytest.mark.parametrize("name", ["../../etc/passwd", "/abs/train-00001.bin", "valid-00001.bin"])
def test_shard_names_cannot_point_elsewhere(meta_dict: dict[str, Any], name: str) -> None:
    meta_dict["splits"]["train"]["shards"][1]["name"] = name
    _inconsistent(meta_dict)


def test_shard_bytes_must_match_items(meta_dict: dict[str, Any]) -> None:
    meta_dict["splits"]["train"]["shards"][0]["n_bytes"] = 201
    _inconsistent(meta_dict)


def test_split_total_must_match_shards(meta_dict: dict[str, Any]) -> None:
    meta_dict["splits"]["train"]["n_items"] = 151
    _inconsistent(meta_dict)


def test_empty_split_is_consistent(meta_dict: dict[str, Any]) -> None:
    meta_dict["splits"]["valid"].update(n_items=0, n_docs=0, shards=[])
    assert _parse(meta_dict).split("valid").shards == ()


def test_read_meta_from_disk_rejects_garbage(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(FormatError) as info:
        read_meta(tmp_path)
    assert str(tmp_path / "meta.json") in str(info.value)
