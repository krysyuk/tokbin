from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from tokbin import ConfigError, DependencyError, ErrorPolicy, StreamWriter, WriterConfig, codes
from tokbin.write.config import config_hash


def test_defaults_follow_spec() -> None:
    cfg = WriterConfig()
    assert cfg.split == "train"
    assert cfg.shard_bytes == 512 * 1024**2
    assert cfg.append_eos is True
    assert cfg.prepend_bos is False
    policy = ErrorPolicy()
    assert (policy.on_data_error, policy.max_skip_ratio, policy.min_docs_for_ratio) == (
        "skip",
        0.01,
        1000,
    )


def test_config_is_frozen_slotted_and_keyword_only() -> None:
    cfg = WriterConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.shard_bytes = 1  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        cfg.shard_byte = 1  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        WriterConfig("valid")  # type: ignore[misc]
    assert dataclasses.replace(cfg, split="valid").split == "valid"
    assert hash(cfg) == hash(WriterConfig())


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"split": "dev"}, codes.SPLIT_INVALID),
        ({"shard_bytes": 0}, codes.CONFIG_VALUE_INVALID),
        ({"shard_bytes": 1.5}, codes.CONFIG_VALUE_INVALID),
        ({"shard_bytes": True}, codes.CONFIG_VALUE_INVALID),
        ({"append_eos": 1}, codes.CONFIG_VALUE_INVALID),
        ({"dtype": "int16"}, codes.DTYPE_UNSUPPORTED),
        ({"eos_token": ""}, codes.CONFIG_VALUE_INVALID),
    ],
)
def test_writer_config_validation(kwargs: dict[str, object], code: object) -> None:
    with pytest.raises(ConfigError) as info:
        WriterConfig(**kwargs)  # type: ignore[arg-type]
    assert info.value.code is code


@pytest.mark.parametrize(
    "kwargs",
    [
        {"on_data_error": "ignore"},
        {"max_skip_ratio": 1.5},
        {"max_skip_ratio": -0.1},
        {"max_skip_ratio": True},
        {"min_docs_for_ratio": 0},
    ],
)
def test_error_policy_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        ErrorPolicy(**kwargs)  # type: ignore[arg-type]


def test_config_hash_tracks_output_affecting_fields() -> None:
    u16 = np.dtype("uint16")
    base = config_hash(WriterConfig(), dtype=u16, eos_id=1, bos_id=2)
    assert len(base) == 64
    assert base == config_hash(WriterConfig(), dtype=u16, eos_id=1, bos_id=2)
    assert base != config_hash(WriterConfig(shard_bytes=1024), dtype=u16, eos_id=1, bos_id=2)
    assert base != config_hash(WriterConfig(), dtype=np.dtype("uint32"), eos_id=1, bos_id=2)
    assert base != config_hash(WriterConfig(), dtype=u16, eos_id=7, bos_id=2)
    # bos is unused unless prepend_bos: changing it does not change the output.
    assert base == config_hash(WriterConfig(), dtype=u16, eos_id=1, bos_id=9)


def test_writing_in_core_mode_explains_the_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(DependencyError) as info:
        StreamWriter(tmp_path / "code", "tokenizer.json")
    err = info.value
    assert err.code is codes.WRITE_UNAVAILABLE
    assert isinstance(err, ImportError)
    assert "pip install tokbin" in str(err)
    assert "tokbin-core" in str(err)
    assert list(tmp_path.iterdir()) == []
