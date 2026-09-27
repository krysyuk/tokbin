"""Packing a source for transfer and unpacking it (spec 15).

A pack is a directory ``<name>.tbpack/`` (or one ``.tar`` of it) with every file of the
source compressed separately and a ``manifest.json``::

    {
      "format": "tokbin-pack",
      "format_version": 1,
      "tokbin_version": "0.4.0",
      "created_at": "2026-09-27T10:00:00Z",
      "source": "web",
      "method": "zstd",
      "level": 3,
      "files": [
        {"path": "train-00000.bin", "stored": "train-00000.bin.zst",
         "n_bytes": 536870912, "sha256": "...",
         "stored_bytes": 161061273, "stored_sha256": "..."}
      ]
    }

Packing refuses a source that is not complete, and a shard whose sha256 differs from
``meta.json``: damaged data is never packed. Unpacking writes into ``<name>.partial/``,
checks the size and sha256 of every compressed and every decompressed file, checks the
result against its own ``meta.json``, and only then renames it into place (spec 8.1).
A failed unpack leaves nothing behind.
"""

from __future__ import annotations

import re
import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

from tokbin import _fs, codes
from tokbin._boundary import public_api
from tokbin._version import get_version
from tokbin.errors import (
    CompatibilityWarning,
    ConfigError,
    FormatError,
    IntegrityError,
    SchemaVersionError,
    TokbinError,
)
from tokbin.format import naming
from tokbin.format._fields import Fields
from tokbin.format._json import MAX_METADATA_BYTES, parse_json_bounded, write_json_atomic
from tokbin.format._time import utc_timestamp
from tokbin.format.meta import read_meta
from tokbin.ops.compression import (
    DEFAULT_LEVEL,
    METHODS,
    Method,
    choose_method,
    compress_stream,
    decompress_stream,
)
from tokbin.ops.inspect import inspect_source
from tokbin.ops.safe_tar import PACK_SUFFIX, TarPack, check_relative_path, write_tar
from tokbin.read._files import check_source_files
from tokbin.write.lock import WriteLock
from tokbin.write.partial import create_partial, swap_in
from tokbin.write.result import Issue

__all__ = ["PackFile", "PackManifest", "PackResult", "UnpackResult", "pack_source", "unpack_pack"]

FORMAT: Final = "tokbin-pack"
FORMAT_VERSION: Final = 1
MANIFEST: Final = "manifest.json"
SUFFIX: Final[dict[str, str]] = {"zstd": ".zst", "lzma": ".xz"}
TAR_SUFFIX: Final = ".tar"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Called with the number of bytes processed so far and the total.
ProgressCallback = Callable[[int, int, str], None]


# --- manifest -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PackFile:
    path: str
    stored: str
    n_bytes: int
    sha256: str
    stored_bytes: int
    stored_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "stored": self.stored,
            "n_bytes": self.n_bytes,
            "sha256": self.sha256,
            "stored_bytes": self.stored_bytes,
            "stored_sha256": self.stored_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PackManifest:
    tokbin_version: str
    created_at: str
    source: str
    method: Method
    level: int
    files: tuple[PackFile, ...]

    @property
    def n_bytes(self) -> int:
        return sum(f.n_bytes for f in self.files)

    @property
    def stored_bytes(self) -> int:
        return sum(f.stored_bytes for f in self.files)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "tokbin_version": self.tokbin_version,
            "created_at": self.created_at,
            "source": self.source,
            "method": self.method,
            "level": self.level,
            "files": [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: object, where: str) -> PackManifest:
        f = Fields(data, where)
        if f.get_str("format") != FORMAT:
            raise FormatError(
                codes.NOT_A_PACK,
                where,
                why=f'"format" is not {FORMAT!r}',
                fix="pass a pack made by `tokbin pack`",
            )
        version = f.get_int("format_version", minimum=1)
        if version > FORMAT_VERSION:
            raise SchemaVersionError(
                codes.SCHEMA_TOO_NEW,
                f"{where}: format_version {version}",
                why=f"this tokbin reads pack format versions up to {FORMAT_VERSION}",
                fix="upgrade tokbin: pip install -U tokbin",
            )
        method = f.get_str("method")
        if method not in METHODS:
            raise f.error("method", f"expected one of {', '.join(METHODS)}")
        source = f.get_str("source")
        if not naming.is_valid_source_name(source):
            raise f.error("source", "not a valid source name")
        files: list[PackFile] = []
        seen: set[str] = set()
        for i, item in enumerate(f.get_list("files")):
            ff = Fields(item, f"{where}: files[{i}]")
            path = ff.get_str("path")
            check_relative_path(path, ff.where)
            stored = ff.get_str("stored")
            if stored != path + SUFFIX[method]:
                raise ff.error("stored", f"expected {path + SUFFIX[method]!r}")
            if path in seen or path == MANIFEST:
                raise ff.error("path", "listed twice or reserved")
            seen.add(path)
            for key in ("sha256", "stored_sha256"):
                if not _SHA256_RE.match(ff.get_str(key)):
                    raise ff.error(key, "expected 64 lowercase hex digits")
            files.append(
                PackFile(
                    path=path,
                    stored=stored,
                    n_bytes=ff.get_int("n_bytes", minimum=0),
                    sha256=ff.get_str("sha256"),
                    stored_bytes=ff.get_int("stored_bytes", minimum=0),
                    stored_sha256=ff.get_str("stored_sha256"),
                )
            )
        if naming.META_JSON not in seen:
            raise f.error("files", f"the pack has no {naming.META_JSON}")
        return cls(
            tokbin_version=f.get_str("tokbin_version"),
            created_at=f.get_str("created_at"),
            source=source,
            method=method,  # type: ignore[arg-type]
            level=f.get_int("level"),
            files=tuple(files),
        )


# --- results --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PackResult:
    path: Path
    manifest: PackManifest
    issues: tuple[Issue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "manifest": self.manifest.to_dict(),
            "n_bytes": self.manifest.n_bytes,
            "stored_bytes": self.manifest.stored_bytes,
            "issues": [
                {"code": i.code, "level": i.level, "message": i.message, "count": i.count}
                for i in self.issues
            ],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class UnpackResult:
    path: Path
    manifest: PackManifest

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "source": self.manifest.source,
            "method": self.manifest.method,
            "n_files": len(self.manifest.files),
            "n_bytes": self.manifest.n_bytes,
            "stored_bytes": self.manifest.stored_bytes,
        }


# --- pack -----------------------------------------------------------------------------


def _output_exists(path: Path) -> ConfigError:
    return ConfigError(
        codes.OUTPUT_EXISTS,
        str(path),
        why="refusing to overwrite it",
        fix="pass overwrite=True (--overwrite), or choose another output directory",
    )


def _source_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file()]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


@public_api
def pack_source(
    path: str | Path,
    out_dir: str | Path | None = None,
    *,
    method: Method | None = None,
    level: int | None = None,
    tar: bool = False,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> PackResult:
    """Compress every file of a complete source into ``<name>.tbpack`` (or ``.tbpack.tar``).

    ``method`` defaults to zstd when available, otherwise lzma (with a warning).
    """
    root = Path(path)
    info = inspect_source(root)
    if info.state != "complete":
        detail = f": {info.problems[0].what}" if info.problems else ""
        raise ConfigError(
            codes.SOURCE_NOT_FINISHED,
            str(root),
            why=f"only complete sources are packed; this one is {info.state}{detail}",
            fix=f"see `tokbin status {root}` and `tokbin verify {root}`",
        )
    issues: list[Issue] = []
    if method is None:
        method = choose_method()
        if method == "lzma":
            message = (
                f"{codes.SLOW_COMPRESSION.title}; install 'tokbin-core[zstd]' for "
                "faster packing and unpacking"
            )
            warnings.warn(
                f"[{codes.SLOW_COMPRESSION.id}] {message}", CompatibilityWarning, stacklevel=3
            )
            issues.append(
                Issue(code=codes.SLOW_COMPRESSION.id, level="warning", message=message, count=1)
            )
    elif method not in METHODS:
        raise ConfigError(
            codes.CONFIG_VALUE_INVALID,
            f"method={method!r}",
            why=f"supported methods are {', '.join(METHODS)}",
            fix="pass method='zstd' or method='lzma', or omit it",
        )
    if level is None:
        level = DEFAULT_LEVEL[method]

    out_root = Path(out_dir) if out_dir is not None else root.parent
    out_root.mkdir(parents=True, exist_ok=True)
    pack_dir = out_root / (root.name + PACK_SUFFIX)
    final = pack_dir.with_name(pack_dir.name + TAR_SUFFIX) if tar else pack_dir
    if final.exists() and not overwrite:
        raise _output_exists(final)
    work = pack_dir.with_name(pack_dir.name + naming.PARTIAL_SUFFIX)
    _remove(work)  # a leftover of an interrupted pack: packing is cheap to redo
    work.mkdir()
    try:
        manifest = _pack_into(root, work, method, level, progress)
        if tar:
            ready = work.with_name(final.name + naming.PARTIAL_SUFFIX)
            try:
                write_tar(work, ready, top=pack_dir.name)
                ready.replace(final)
            finally:
                _remove(ready)
            _fs.fsync_dir(out_root)
        else:
            swap_in(work, final)
    finally:
        _remove(work)
    return PackResult(path=final, manifest=manifest, issues=tuple(issues))


def _pack_into(
    root: Path, work: Path, method: Method, level: int, progress: ProgressCallback | None
) -> PackManifest:
    meta = read_meta(root)
    shard_sha = {s.name: s.sha256 for split in meta.splits for s in split.shards}
    sources = _source_files(root)
    total = sum(p.stat().st_size for p in sources)
    done = 0
    files: list[PackFile] = []
    for src_path in sources:
        rel = src_path.relative_to(root).as_posix()
        stored = rel + SUFFIX[method]
        out_path = work / stored
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def on_bytes(n: int, name: str = rel) -> None:
            nonlocal done
            done += n
            if progress is not None:
                progress(done, total, name)

        with src_path.open("rb") as src, out_path.open("xb") as out:
            n_bytes, sha, stored_bytes, stored_sha = compress_stream(
                src, out, method, level=level, on_bytes=on_bytes
            )
            _fs.fsync_file(out)
        expected = shard_sha.get(rel)
        if expected is not None and expected != sha:
            raise IntegrityError(
                codes.SHARD_CORRUPT,
                str(src_path),
                why=f"sha256 {sha[:12]}... differs from {expected[:12]}... in meta.json; "
                "damaged data is not packed",
                fix=f"restore the shard and run `tokbin verify {root}`",
            )
        files.append(
            PackFile(
                path=rel,
                stored=stored,
                n_bytes=n_bytes,
                sha256=sha,
                stored_bytes=stored_bytes,
                stored_sha256=stored_sha,
            )
        )
    manifest = PackManifest(
        tokbin_version=get_version(),
        created_at=utc_timestamp(),
        source=root.name,
        method=method,
        level=level,
        files=tuple(files),
    )
    write_json_atomic(work / MANIFEST, manifest.to_dict())
    _fs.fsync_dir(work)
    return manifest


# --- unpack ---------------------------------------------------------------------------


class _DirPack:
    def __init__(self, path: Path) -> None:
        self.path = path

    def names(self) -> set[str]:
        return {p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()}

    def size(self, name: str) -> int:
        return (self.path / name).stat().st_size

    def open(self, name: str) -> IO[bytes]:
        return naming.resolve_inside(self.path, name).open("rb")

    def close(self) -> None:
        pass


def _open_pack(path: Path) -> _DirPack | TarPack:
    if path.is_dir():
        if not (path / MANIFEST).is_file():
            raise FormatError(
                codes.NOT_A_PACK,
                str(path),
                why=f"the directory has no {MANIFEST}",
                fix="pass a .tbpack directory or a .tar made by `tokbin pack --tar`",
            )
        return _DirPack(path)
    if path.is_file():
        return TarPack(path)
    raise ConfigError(
        codes.SOURCE_NOT_FOUND,
        str(path),
        why="the pack does not exist",
        fix="check the path",
    )


def _read_manifest(pack: _DirPack | TarPack) -> PackManifest:
    where = f"{pack.path}: {MANIFEST}"
    if MANIFEST not in pack.names():
        raise FormatError(
            codes.NOT_A_PACK,
            str(pack.path),
            why=f"it has no {MANIFEST}",
            fix="pass a pack made by `tokbin pack`",
        )
    with pack.open(MANIFEST) as f:
        raw = f.read(MAX_METADATA_BYTES + 1)
    return PackManifest.from_dict(parse_json_bounded(raw, where), where)


@public_api
def unpack_pack(
    path: str | Path,
    into: str | Path | None = None,
    *,
    name: str | None = None,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> UnpackResult:
    """Unpack and verify a pack into ``into/<name>`` (the source name by default)."""
    pack_path = Path(path)
    pack = _open_pack(pack_path)
    try:
        manifest = _read_manifest(pack)
        target_root = Path(into) if into is not None else pack_path.parent
        target = target_root / naming.check_source_name(name or manifest.source)
        if target.exists() and not overwrite:
            raise _output_exists(target)
        target_root.mkdir(parents=True, exist_ok=True)
        partial = create_partial(target)
        lock = WriteLock(partial)
        lock.acquire()
        try:
            _unpack_into(pack, manifest, partial, progress)
            _check_result(partial, manifest)
            lock.release()
            swap_in(partial, target)
        except BaseException:
            lock.release()
            shutil.rmtree(partial, ignore_errors=True)
            raise
    finally:
        pack.close()
    return UnpackResult(path=target, manifest=manifest)


def _unpack_into(
    pack: _DirPack | TarPack,
    manifest: PackManifest,
    partial: Path,
    progress: ProgressCallback | None,
) -> None:
    available = pack.names()
    total = manifest.n_bytes
    done = 0
    for entry in manifest.files:
        where = f"{pack.path}: {entry.stored}"
        if entry.stored not in available:
            raise IntegrityError(
                codes.PACKED_FILE_MISSING,
                where,
                why="the manifest lists it, the pack does not contain it",
                fix="copy the pack again from its origin",
            )
        if pack.size(entry.stored) != entry.stored_bytes:
            raise IntegrityError(
                codes.PACKED_FILE_CORRUPT,
                where,
                why=f"it has {pack.size(entry.stored)} bytes, the manifest says "
                f"{entry.stored_bytes}",
                fix="copy the pack again from its origin and repeat `tokbin unpack`",
            )
        out_path = naming.resolve_inside(partial, entry.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def on_bytes(n: int, label: str = entry.path) -> None:
            nonlocal done
            done += n
            if progress is not None:
                progress(done, total, label)

        with pack.open(entry.stored) as src, out_path.open("xb") as out:
            sha, _, stored_sha = decompress_stream(
                src,
                out,
                manifest.method,
                expected_bytes=entry.n_bytes,
                name=where,
                on_bytes=on_bytes,
            )
            _fs.fsync_file(out)
        if stored_sha != entry.stored_sha256:
            raise IntegrityError(
                codes.PACKED_FILE_CORRUPT,
                where,
                why="the sha256 of the compressed file differs from the manifest",
                fix="copy the pack again from its origin and repeat `tokbin unpack`",
            )
        if sha != entry.sha256:
            raise IntegrityError(
                codes.PACKED_FILE_CORRUPT,
                where,
                why="the sha256 of the unpacked data differs from the manifest",
                fix="copy the pack again from its origin and repeat `tokbin unpack`",
            )


def _check_result(partial: Path, manifest: PackManifest) -> None:
    """The unpacked files form a complete source that agrees with the manifest."""
    meta = read_meta(partial)
    by_path = {f.path: f for f in manifest.files}
    for split in meta.splits:
        for shard in split.shards:
            entry = by_path.get(shard.name)
            if entry is None or entry.sha256 != shard.sha256:
                raise IntegrityError(
                    codes.PACKED_FILE_CORRUPT,
                    shard.name,
                    why="meta.json and the manifest disagree about this shard",
                    fix="the pack is inconsistent; get the dataset from its original source",
                )
    problems: list[TokbinError] = check_source_files(partial, meta)
    if problems:
        raise problems[0]
