from __future__ import annotations

import numpy as np
import pytest

from tokbin import codes
from tokbin.errors import ConfigError, ContractError, DataError
from tokbin.format.dtypes import dtype_for_vocab, resolve_dtype, to_dtype_checked


@pytest.mark.parametrize(
    ("vocab", "expected"),
    [
        (1, "uint8"),
        (256, "uint8"),
        (257, "uint16"),
        (50257, "uint16"),
        (65536, "uint16"),
        (65537, "uint32"),
        (2**32, "uint32"),
    ],
)
def test_dtype_for_vocab(vocab: int, expected: str) -> None:
    assert dtype_for_vocab(vocab) == np.dtype(expected)


@pytest.mark.parametrize("vocab", [0, -1, 2**32 + 1])
def test_dtype_for_vocab_out_of_range(vocab: int) -> None:
    with pytest.raises(ConfigError) as info:
        dtype_for_vocab(vocab)
    assert info.value.code is codes.VOCAB_SIZE_OUT_OF_RANGE


def test_resolve_dtype_explicit_wider_is_allowed() -> None:
    assert resolve_dtype(1000, "uint32") == np.dtype("uint32")
    assert resolve_dtype(1000) == np.dtype("uint16")


def test_resolve_dtype_explicit_narrower_is_refused() -> None:
    with pytest.raises(ConfigError) as info:
        resolve_dtype(50257, "uint8")
    assert info.value.code is codes.DTYPE_TOO_NARROW


@pytest.mark.parametrize("name", ["int16", "uint64", "float32", "u2"])
def test_resolve_dtype_unsupported(name: str) -> None:
    with pytest.raises(ConfigError) as info:
        resolve_dtype(100, name)
    assert info.value.code is codes.DTYPE_UNSUPPORTED


def test_checked_conversion_keeps_values() -> None:
    out = to_dtype_checked([0, 1, 65535], np.dtype("uint16"))
    assert out.dtype == np.uint16
    assert out.tolist() == [0, 1, 65535]


def test_empty_input() -> None:
    out = to_dtype_checked([], np.dtype("uint16"))
    assert out.dtype == np.uint16
    assert out.size == 0


def test_overflow_is_detected_not_wrapped() -> None:
    # astype would silently turn 70000 into 4464.
    assert np.array([70000]).astype(np.uint16)[0] == 4464
    with pytest.raises(DataError) as info:
        to_dtype_checked([1, 70000], np.dtype("uint16"))
    assert info.value.code is codes.TOKEN_ID_OUT_OF_RANGE
    assert "70000" in str(info.value)


@pytest.mark.parametrize(
    "ids",
    [
        [-1, 5],
        [2**63],  # numpy: uint64
        [2**70],  # numpy: object
        [-1, 2**63],  # numpy: float64, silently
        np.array([2**64 - 1], dtype=np.uint64),
        np.array([-3], dtype=np.int8),
    ],
)
def test_out_of_range_inputs(ids: object) -> None:
    with pytest.raises(DataError):
        to_dtype_checked(ids, np.dtype("uint32"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ids",
    [
        [1.5, 2.0],
        np.array([True, False]),
        [[1, 2], [3, 4]],
        ["1", "2"],
    ],
)
def test_non_integer_inputs_violate_contract(ids: object) -> None:
    with pytest.raises(ContractError) as info:
        to_dtype_checked(ids, np.dtype("uint32"))  # type: ignore[arg-type]
    assert info.value.code is codes.TOKEN_IDS_INVALID


def test_numpy_int_array_input() -> None:
    out = to_dtype_checked(np.array([1, 2, 255], dtype=np.int64), np.dtype("uint8"))
    assert out.tolist() == [1, 2, 255]
    assert out.dtype == np.uint8
