from __future__ import annotations

import io
import os

import pytest

from tokbin import DependencyError, IntegrityError, codes
from tokbin.ops import compression

DATA = os.urandom(5000) + bytes(2_000_000) + os.urandom(100)


def _pack(method: compression.Method, data: bytes = DATA) -> bytes:
    out = io.BytesIO()
    compression.compress_stream(io.BytesIO(data), out, method, level=3)
    return out.getvalue()


@pytest.mark.parametrize("method", compression.METHODS)
def test_roundtrip(method: compression.Method) -> None:
    import hashlib

    out = io.BytesIO()
    seen: list[int] = []
    n, sha, stored_n, stored_sha = compression.compress_stream(
        io.BytesIO(DATA), out, method, level=3, on_bytes=seen.append
    )
    stored = out.getvalue()
    assert (n, sha) == (len(DATA), hashlib.sha256(DATA).hexdigest())
    assert (stored_n, stored_sha) == (len(stored), hashlib.sha256(stored).hexdigest())
    assert stored_n < n / 10
    assert sum(seen) == len(DATA)

    back = io.BytesIO()
    result = compression.decompress_stream(
        io.BytesIO(stored), back, method, expected_bytes=len(DATA), name="x"
    )
    assert back.getvalue() == DATA
    assert result == (sha, stored_n, stored_sha)


@pytest.mark.parametrize("method", compression.METHODS)
def test_empty_input(method: compression.Method) -> None:
    stored = _pack(method, b"")
    back = io.BytesIO()
    compression.decompress_stream(io.BytesIO(stored), back, method, expected_bytes=0, name="x")
    assert back.getvalue() == b""


@pytest.mark.parametrize("method", compression.METHODS)
def test_bomb_is_stopped_at_the_declared_size(method: compression.Method) -> None:
    stored = _pack(method, bytes(50_000_000))  # 50 MB of zeros, a few KB compressed
    out = io.BytesIO()
    with pytest.raises(IntegrityError) as err:
        compression.decompress_stream(
            io.BytesIO(stored), out, method, expected_bytes=1000, name="bomb"
        )
    assert err.value.code is codes.PACKED_FILE_TOO_LARGE
    assert len(out.getvalue()) <= 1000  # nothing beyond the declared size was written


@pytest.mark.parametrize("method", compression.METHODS)
@pytest.mark.parametrize("damage", ["truncate", "garbage", "shorter_than_declared"])
def test_damaged_input(method: compression.Method, damage: str) -> None:
    stored = _pack(method)
    expected = len(DATA)
    if damage == "truncate":
        stored = stored[: len(stored) // 2]
    elif damage == "garbage":
        stored = b"not compressed at all" * 10
    else:
        expected += 1
    with pytest.raises(IntegrityError) as err:
        compression.decompress_stream(
            io.BytesIO(stored), io.BytesIO(), method, expected_bytes=expected, name="x"
        )
    assert err.value.code is codes.PACKED_FILE_CORRUPT


def test_trailing_bytes_are_hashed(monkeypatch: pytest.MonkeyPatch) -> None:
    stored = _pack("zstd")
    _, n_clean, _ = compression.decompress_stream(
        io.BytesIO(stored), io.BytesIO(), "zstd", expected_bytes=len(DATA), name="x"
    )
    assert n_clean == len(stored)


def _no_zstd(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokbin import _deps

    real = _deps.probe
    monkeypatch.setattr(compression, "_stdlib_zstd", lambda: None)
    monkeypatch.setattr(
        _deps,
        "probe",
        lambda name: (
            _deps.DependencyStatus(dependency=_deps.KNOWN[name], installed=False, version=None)
            if name == "zstandard"
            else real(name)
        ),
    )


def test_fallback_to_lzma(monkeypatch: pytest.MonkeyPatch) -> None:
    assert compression.choose_method() == "zstd"
    _no_zstd(monkeypatch)
    assert not compression.zstd_available()
    assert compression.choose_method() == "lzma"


def test_zstd_pack_without_a_zstd_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    stored = _pack("zstd")
    _no_zstd(monkeypatch)
    with pytest.raises(DependencyError) as err:
        compression.decompress_stream(
            io.BytesIO(stored), io.BytesIO(), "zstd", expected_bytes=len(DATA), name="x"
        )
    assert "tokbin-core[zstd]" in err.value.fix


def test_backends_are_interchangeable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame from zstandard decodes with compression.zstd and the other way round."""
    if compression._stdlib_zstd() is None:
        pytest.skip("compression.zstd needs Python 3.14")
    pytest.importorskip("zstandard")
    from_stdlib = _pack("zstd")
    stdlib = compression._stdlib_zstd
    monkeypatch.setattr(compression, "_stdlib_zstd", lambda: None)
    from_zstandard = _pack("zstd")
    for stored in (from_stdlib, from_zstandard):
        for backend in (None, stdlib):
            monkeypatch.setattr(compression, "_stdlib_zstd", lambda b=backend: b() if b else None)
            back = io.BytesIO()
            compression.decompress_stream(
                io.BytesIO(stored), back, "zstd", expected_bytes=len(DATA), name="x"
            )
            assert back.getvalue() == DATA
