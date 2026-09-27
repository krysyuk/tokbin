"""Hostile packs: path traversal, links, bombs and tampering are refused (spec 9.3, 19)."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pytest

from tokbin import (
    ConfigError,
    FormatError,
    IntegrityError,
    ResumeError,
    codes,
    pack_source,
    unpack_pack,
)

from support import copy_fixture


@pytest.fixture
def packed(tmp_path: Path) -> Path:
    source = copy_fixture(tmp_path / "src" / "web")
    return pack_source(source, tmp_path / "packs").path


def _tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    with tarfile.open(path, "w") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


def _file(name: str, data: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _special(name: str, kind: bytes, target: str = "") -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
    return info, None


@pytest.mark.parametrize(
    "members",
    [
        [_file("/etc/evil")],
        [_file("web.tbpack/../../evil")],
        [_file("web.tbpack/./x")],
        [_file("web.tbpack\\..\\evil")],
        [_file("C:/evil")],
        [_special("web.tbpack/link", tarfile.SYMTYPE, "/etc/passwd")],
        [_special("web.tbpack/hard", tarfile.LNKTYPE, "web.tbpack/manifest.json")],
        [_special("web.tbpack/fifo", tarfile.FIFOTYPE)],
        [_special("web.tbpack/dev", tarfile.CHRTYPE)],
        [_file("web.tbpack/a"), _file("web.tbpack/a")],
        [_file("web.tbpack/a"), _file("other.tbpack/b")],
        [_file("notapack/manifest.json")],
        [_file("web.tbpack")],
    ],
)
def test_unsafe_tar_entries(tmp_path: Path, members: list[Any]) -> None:
    archive = _tar(tmp_path / "evil.tar", members)
    with pytest.raises(FormatError) as err:
        unpack_pack(archive, tmp_path / "out")
    assert err.value.code is codes.ARCHIVE_ENTRY_UNSAFE
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    assert not Path("/etc/evil").exists()


def test_not_a_tar_or_empty(tmp_path: Path) -> None:
    (tmp_path / "x.tar").write_bytes(b"hello")
    with pytest.raises(FormatError) as err:
        unpack_pack(tmp_path / "x.tar", tmp_path)
    assert err.value.code is codes.NOT_A_PACK
    with pytest.raises(FormatError) as err:
        unpack_pack(_tar(tmp_path / "empty.tar", []), tmp_path)
    assert err.value.code is codes.NOT_A_PACK
    with pytest.raises(ConfigError):
        unpack_pack(tmp_path / "missing.tar", tmp_path)


def _manifest(pack: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((pack / "manifest.json").read_text())
    return data


def _save(pack: Path, manifest: dict[str, Any]) -> None:
    (pack / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "path", ["../evil", "/abs", "tokenizer/../../evil", "a\\b", "manifest.json"]
)
def test_manifest_paths_are_confined(tmp_path: Path, packed: Path, path: str) -> None:
    manifest = _manifest(packed)
    entry = manifest["files"][1]
    entry["path"] = path
    entry["stored"] = path + ".zst"
    _save(packed, manifest)
    with pytest.raises(FormatError):
        unpack_pack(packed, tmp_path / "out")
    assert not (tmp_path / "evil").exists()


def _first_shard(manifest: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = next(f for f in manifest["files"] if f["path"].endswith(".bin"))
    return entry


def test_bomb(tmp_path: Path, packed: Path) -> None:
    manifest = _manifest(packed)
    _first_shard(manifest)["n_bytes"] = 10  # the real content is larger
    _save(packed, manifest)
    with pytest.raises(IntegrityError) as err:
        unpack_pack(packed, tmp_path / "out")
    assert err.value.code is codes.PACKED_FILE_TOO_LARGE


def test_tampered_compressed_file(tmp_path: Path, packed: Path) -> None:
    stored = packed / _first_shard(_manifest(packed))["stored"]
    data = bytearray(stored.read_bytes())
    data[-3] ^= 0xFF
    stored.write_bytes(bytes(data))
    with pytest.raises(IntegrityError) as err:
        unpack_pack(packed, tmp_path / "out")
    assert err.value.code is codes.PACKED_FILE_CORRUPT


@pytest.mark.parametrize("field", ["sha256", "stored_sha256", "stored_bytes"])
def test_tampered_manifest(tmp_path: Path, packed: Path, field: str) -> None:
    manifest = _manifest(packed)
    entry = _first_shard(manifest)
    entry[field] = "0" * 64 if field != "stored_bytes" else entry[field] + 1
    _save(packed, manifest)
    with pytest.raises(IntegrityError) as err:
        unpack_pack(packed, tmp_path / "out")
    assert err.value.code is codes.PACKED_FILE_CORRUPT


def test_missing_stored_file(tmp_path: Path, packed: Path) -> None:
    (packed / _first_shard(_manifest(packed))["stored"]).unlink()
    with pytest.raises(IntegrityError) as err:
        unpack_pack(packed, tmp_path / "out")
    assert err.value.code is codes.PACKED_FILE_MISSING


def test_meta_and_manifest_must_agree(tmp_path: Path, packed: Path) -> None:
    """A consistent-looking pack whose meta.json names shards the manifest lacks."""
    manifest = _manifest(packed)
    shard = _first_shard(manifest)
    manifest["files"].remove(shard)
    _save(packed, manifest)
    with pytest.raises(IntegrityError):
        unpack_pack(packed, tmp_path / "out")


def test_failure_leaves_nothing_behind(tmp_path: Path, packed: Path) -> None:
    stored = packed / _first_shard(_manifest(packed))["stored"]
    stored.write_bytes(stored.read_bytes()[:-5])
    with pytest.raises(IntegrityError):
        unpack_pack(packed, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_newer_pack_format(tmp_path: Path, packed: Path) -> None:
    from tokbin import SchemaVersionError

    manifest = _manifest(packed)
    manifest["format_version"] = 99
    _save(packed, manifest)
    with pytest.raises(SchemaVersionError):
        unpack_pack(packed, tmp_path / "out")


def test_unpack_does_not_trample_an_unfinished_write(tmp_path: Path, packed: Path) -> None:
    (tmp_path / "out" / "web.partial").mkdir(parents=True)
    with pytest.raises(ResumeError):
        unpack_pack(packed, tmp_path / "out")
    assert (tmp_path / "out" / "web.partial").is_dir()


def test_tar_roundtrip_through_the_safe_reader(tmp_path: Path) -> None:
    source = copy_fixture(tmp_path / "src" / "web")
    archive = pack_source(source, tmp_path / "packs", tar=True).path
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert names[0] == "web.tbpack"
        assert all(n.startswith("web.tbpack") for n in names)
        assert {m.uid for m in tar.getmembers()} == {0}
    result = unpack_pack(archive, tmp_path / "out")
    assert (result.path / "meta.json").read_bytes() == (source / "meta.json").read_bytes()
    shutil.rmtree(result.path)
