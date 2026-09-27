"""Compression backends for pack / unpack (spec 15).

The chain: ``compression.zstd`` (Python 3.14+) -> ``zstandard`` (the ``zstd`` extra) ->
``lzma`` from the standard library, slower, with a warning. Both zstd backends produce
and read the same standard zstd frames, so a pack made with one unpacks with the other.
The method is recorded in the manifest and ``unpack`` picks the decoder by it.

Decompression is always bounded by the size declared in the manifest: a crafted file
that expands beyond it (a decompression bomb) is refused as soon as it does, without
writing more than the declared size.
"""

from __future__ import annotations

import hashlib
import importlib
import lzma
from collections.abc import Callable
from typing import IO, Final, Literal

from tokbin import _deps, codes
from tokbin.errors import DependencyError, IntegrityError

__all__ = [
    "METHODS",
    "Method",
    "choose_method",
    "compress_stream",
    "decompress_stream",
    "zstd_available",
]

Method = Literal["zstd", "lzma"]
METHODS: Final = ("zstd", "lzma")

#: Default levels: zstd 3 is fast and compresses token ids well; lzma preset 3 keeps
#: the fallback usable on big files.
DEFAULT_LEVEL: Final[dict[str, int]] = {"zstd": 3, "lzma": 3}

_CHUNK: Final = 1024**2

OnBytes = Callable[[int], None]


def _stdlib_zstd() -> object | None:
    if not _deps.has_stdlib_zstd():
        return None
    try:
        return importlib.import_module("compression.zstd")
    except ImportError:  # a Python built without zstd
        return None


def zstd_available() -> bool:
    return _stdlib_zstd() is not None or _deps.probe("zstandard").installed


def choose_method() -> Method:
    """The best method available here: zstd if possible, lzma otherwise."""
    return "zstd" if zstd_available() else "lzma"


def _no_zstd() -> DependencyError:
    return DependencyError(
        codes.DEPENDENCY_MISSING,
        "zstandard",
        why="the pack is compressed with zstd; this Python has no compression.zstd "
        "(it arrives in 3.14) and the zstandard package is not installed",
        fix=f"install it: {_deps.KNOWN['zstandard'].install}",
    )


def _writer(method: Method, out: IO[bytes], level: int) -> IO[bytes]:
    if method == "lzma":
        return lzma.LZMAFile(out, "wb", preset=level)
    stdlib = _stdlib_zstd()
    if stdlib is not None:
        writer: IO[bytes] = stdlib.ZstdFile(out, "wb", level=level)  # type: ignore[attr-defined]
        return writer
    if not _deps.probe("zstandard").installed:
        raise _no_zstd()
    zstandard = _deps.require("zstandard")
    compressor = zstandard.ZstdCompressor(level=level, write_content_size=False)
    stream: IO[bytes] = compressor.stream_writer(out, closefd=False)
    return stream


def _reader(method: Method, src: IO[bytes]) -> IO[bytes]:
    if method == "lzma":
        return lzma.LZMAFile(src, "rb")
    stdlib = _stdlib_zstd()
    if stdlib is not None:
        reader: IO[bytes] = stdlib.ZstdFile(src, "rb")  # type: ignore[attr-defined]
        return reader
    if not _deps.probe("zstandard").installed:
        raise _no_zstd()
    zstandard = _deps.require("zstandard")
    # read_across_frames: a stream may consist of several frames; closefd=False: the
    # caller owns ``src``.
    stream: IO[bytes] = zstandard.ZstdDecompressor().stream_reader(
        src, read_across_frames=True, closefd=False
    )
    return stream


class _HashingWriter:
    """Counts and hashes what the compressor writes."""

    def __init__(self, out: IO[bytes]) -> None:
        self._out = out
        self.hash = hashlib.sha256()
        self.n_bytes = 0

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        self._out.write(view)
        self.hash.update(view)
        self.n_bytes += len(view)
        return len(view)

    def flush(self) -> None:
        self._out.flush()

    # File-like protocol bits the compressors look for.
    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:  # the caller closes the real file
        pass


class _HashingReader:
    """Counts and hashes what the decompressor reads."""

    def __init__(self, src: IO[bytes]) -> None:
        self._src = src
        self.hash = hashlib.sha256()
        self.n_bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self._src.read(size)
        self.hash.update(data)
        self.n_bytes += len(data)
        return data

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        pass


def compress_stream(
    src: IO[bytes],
    out: IO[bytes],
    method: Method,
    *,
    level: int,
    on_bytes: OnBytes | None = None,
) -> tuple[int, str, int, str]:
    """Compress ``src`` into ``out``.

    Returns ``(n_bytes, sha256, stored_bytes, stored_sha256)``: size and hash of the
    original data and of the compressed data.
    """
    sink = _HashingWriter(out)
    raw_hash = hashlib.sha256()
    n_bytes = 0
    writer = _writer(method, sink, level)  # type: ignore[arg-type]
    try:
        while chunk := src.read(_CHUNK):
            raw_hash.update(chunk)
            writer.write(chunk)
            n_bytes += len(chunk)
            if on_bytes is not None:
                on_bytes(len(chunk))
    finally:
        writer.close()
    if sink.n_bytes == 0:
        # compression.zstd writes no frame at all for empty input, which no decoder
        # accepts; an explicit empty frame keeps every backend able to read it.
        sink.write(_empty_frame(method, level))
    return n_bytes, raw_hash.hexdigest(), sink.n_bytes, sink.hash.hexdigest()


def _empty_frame(method: Method, level: int) -> bytes:
    if method == "lzma":
        return lzma.compress(b"", preset=level)
    stdlib = _stdlib_zstd()
    if stdlib is not None:
        frame: bytes = stdlib.compress(b"", level=level)  # type: ignore[attr-defined]
        return frame
    zstandard = _deps.require("zstandard")
    data: bytes = zstandard.ZstdCompressor(level=level).compress(b"")
    return data


def decompress_stream(
    src: IO[bytes],
    out: IO[bytes],
    method: Method,
    *,
    expected_bytes: int,
    name: str,
    on_bytes: OnBytes | None = None,
) -> tuple[str, int, str]:
    """Decompress ``src`` into ``out``, writing at most ``expected_bytes``.

    Returns ``(sha256, stored_bytes, stored_sha256)``: the hash of the output, and the
    size and hash of the compressed input that was consumed. ``IntegrityError`` if the
    data expands beyond ``expected_bytes`` or cannot be decoded.
    """
    counted = _HashingReader(src)
    digest = hashlib.sha256()
    written = 0
    try:
        reader = _reader(method, counted)  # type: ignore[arg-type]
        try:
            while True:
                # One byte more than allowed is enough to detect an oversized file.
                chunk = reader.read(min(_CHUNK, expected_bytes - written + 1))
                if not chunk:
                    break
                if written + len(chunk) > expected_bytes:
                    raise IntegrityError(
                        codes.PACKED_FILE_TOO_LARGE,
                        name,
                        why=f"it expands beyond the {expected_bytes} bytes declared in "
                        "the manifest",
                        fix="the pack is damaged or crafted; do not use it, get the "
                        "dataset from its original source",
                    )
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if on_bytes is not None:
                    on_bytes(len(chunk))
        finally:
            reader.close()
        # Hash the whole stored file, trailing bytes after the last frame included.
        while counted.read(_CHUNK):
            pass
    except (lzma.LZMAError, EOFError, ValueError, OSError) as exc:
        if isinstance(exc, (FileNotFoundError, PermissionError)):
            raise
        raise _corrupt(name, f"it cannot be decompressed: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        # Both zstd backends raise their own ZstdError (not an OSError subclass).
        if type(exc).__name__ == "ZstdError":
            raise _corrupt(name, f"it cannot be decompressed: {exc}") from exc
        raise
    if written != expected_bytes:
        raise _corrupt(
            name, f"it decompresses to {written} bytes, the manifest says {expected_bytes}"
        )
    return digest.hexdigest(), counted.n_bytes, counted.hash.hexdigest()


def _corrupt(name: str, why: str) -> IntegrityError:
    return IntegrityError(
        codes.PACKED_FILE_CORRUPT,
        name,
        why=why,
        fix="copy the pack again from its origin and repeat `tokbin unpack`",
    )
