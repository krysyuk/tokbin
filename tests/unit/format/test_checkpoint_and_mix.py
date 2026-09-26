from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tokbin import codes
from tokbin.errors import FormatError
from tokbin.format.checkpoint import Checkpoint, read_checkpoint, write_checkpoint
from tokbin.format.mix import Mix, read_mix, write_mix

# --- checkpoint -----------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path, checkpoint_dict: dict[str, Any]) -> None:
    cp = Checkpoint.from_dict(checkpoint_dict, where="checkpoint.json")
    assert cp.to_dict() == checkpoint_dict
    write_checkpoint(tmp_path, cp)
    assert read_checkpoint(tmp_path) == cp


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("split", "dev"),
        ("dtype", "int8"),
        ("config_hash", "5d1e"),
        ("tokenizer_hash", "a" * 64),
        ("pending_doc_items", -1),
        ("updated_at", "yesterday"),
        ("last_input_id", 5),
    ],
)
def test_checkpoint_invalid_field(checkpoint_dict: dict[str, Any], key: str, value: object) -> None:
    checkpoint_dict[key] = value
    with pytest.raises(FormatError) as info:
        Checkpoint.from_dict(checkpoint_dict, where="checkpoint.json")
    assert info.value.code is codes.METADATA_FIELD_INVALID


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(n_items=99),
        lambda d: d.update(n_docs=9),
        lambda d: d.update(pending_doc_items=101),
        lambda d: d["closed_shards"][0].update(name="train-00001.bin"),
        lambda d: d["closed_shards"][0].update(n_bytes=100),
    ],
)
def test_checkpoint_inconsistent(checkpoint_dict: dict[str, Any], mutate: Any) -> None:
    mutate(checkpoint_dict)
    with pytest.raises(FormatError) as info:
        Checkpoint.from_dict(checkpoint_dict, where="checkpoint.json")
    assert info.value.code is codes.METADATA_INCONSISTENT
    assert "tokbin clean" in str(info.value)


def test_checkpoint_without_ids(checkpoint_dict: dict[str, Any]) -> None:
    checkpoint_dict["last_input_id"] = None
    assert Checkpoint.from_dict(checkpoint_dict, where="c").last_input_id is None


# --- mix ------------------------------------------------------------------------------


def test_mix_roundtrip_and_normalization(tmp_path: Path) -> None:
    mix = Mix.from_dict({"web": 3, "code": 1.0, "books": 0}, where="mix.json")
    assert mix.names == ("web", "code", "books")
    assert mix.normalized() == (("web", 0.75), ("code", 0.25), ("books", 0.0))
    write_mix(tmp_path, mix)
    assert read_mix(tmp_path) == mix


@pytest.mark.parametrize(
    "data",
    [
        {"web": -1},
        {"web": "0.5"},
        {"web": True},
        {"../evil": 1},
        {"web.partial": 1},
        [["web", 1]],
    ],
)
def test_mix_invalid(data: object) -> None:
    with pytest.raises(FormatError) as info:
        Mix.from_dict(data, where="mix.json")
    assert info.value.code is codes.METADATA_FIELD_INVALID


@pytest.mark.parametrize("data", [{}, {"web": 0, "code": 0}])
def test_mix_without_positive_weight(data: object) -> None:
    with pytest.raises(FormatError) as info:
        Mix.from_dict(data, where="mix.json")
    assert info.value.code is codes.METADATA_INCONSISTENT
