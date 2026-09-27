"""pack -> unpack -> sha256 roundtrips (spec 15, 19)."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import pytest

from tokbin import (
    CompatibilityWarning,
    ConfigError,
    IntegrityError,
    codes,
    pack_source,
    unpack_pack,
    verify_source,
)
from tokbin.cli import main as cli_main
from tokbin.cli.main import main
from tokbin.ops import compression

from support import copy_fixture


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return copy_fixture(tmp_path / "corpus" / "web")


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("method", ["zstd", "lzma"])
@pytest.mark.parametrize("tar", [False, True])
def test_roundtrip(tmp_path: Path, source: Path, method: compression.Method, tar: bool) -> None:
    seen: list[tuple[int, int, str]] = []
    result = pack_source(
        source, tmp_path / "packs", method=method, tar=tar, progress=lambda *a: seen.append(a)
    )
    assert result.path.name == ("web.tbpack.tar" if tar else "web.tbpack")
    manifest = result.manifest
    assert (manifest.source, manifest.method, manifest.level) == ("web", method, 3)
    assert {f.path for f in manifest.files} == set(_tree(source))
    assert manifest.stored_bytes < manifest.n_bytes
    assert seen[-1][0] == seen[-1][1] == manifest.n_bytes
    assert not list((tmp_path / "packs").glob("*.partial"))

    unpacked = unpack_pack(result.path, tmp_path / "remote")
    assert unpacked.path == tmp_path / "remote" / "web"
    assert _tree(unpacked.path) == _tree(source)
    assert verify_source(unpacked.path).ok
    assert not list((tmp_path / "remote").glob("*.partial"))


def test_pack_dir_layout(tmp_path: Path, source: Path) -> None:
    pack = pack_source(source, tmp_path / "packs").path
    names = set(_tree(pack))
    assert "manifest.json" in names
    assert "tokenizer/tokenizer.json.zst" in names
    assert "train-00000.bin.zst" in names
    manifest = json.loads((pack / "manifest.json").read_text())
    assert manifest["format"] == "tokbin-pack"
    shard = next(f for f in manifest["files"] if f["path"] == "train-00000.bin")
    meta = json.loads((source / "meta.json").read_text())
    assert shard["sha256"] == meta["splits"]["train"]["shards"][0]["sha256"]


def test_default_output_places(tmp_path: Path, source: Path) -> None:
    pack = pack_source(source).path
    assert pack == source.parent / "web.tbpack"
    result = unpack_pack(pack, tmp_path / "other", name="web2")
    assert result.path == tmp_path / "other" / "web2"


def test_existing_outputs(tmp_path: Path, source: Path) -> None:
    pack_source(source, tmp_path / "p")
    with pytest.raises(ConfigError) as err:
        pack_source(source, tmp_path / "p")
    assert err.value.code is codes.OUTPUT_EXISTS
    again = pack_source(source, tmp_path / "p", overwrite=True)

    unpack_pack(again.path, tmp_path / "u")
    with pytest.raises(ConfigError) as err:
        unpack_pack(again.path, tmp_path / "u")
    assert err.value.code is codes.OUTPUT_EXISTS
    (tmp_path / "u" / "web" / "extra.txt").write_text("old")
    unpack_pack(again.path, tmp_path / "u", overwrite=True)
    assert not (tmp_path / "u" / "web" / "extra.txt").exists()
    assert not list((tmp_path / "u").glob(".web.old-*"))


def test_damaged_source_is_not_packed(tmp_path: Path, source: Path) -> None:
    path = source / "train-00001.bin"
    path.write_bytes(b"\xff" + path.read_bytes()[1:])
    with pytest.raises(IntegrityError) as err:
        pack_source(source, tmp_path / "p")
    assert err.value.code is codes.SHARD_CORRUPT
    assert not list((tmp_path / "p").iterdir())

    (source / "train-00001.bin").unlink()
    with pytest.raises(ConfigError) as err2:
        pack_source(source, tmp_path / "p")
    assert err2.value.code is codes.SOURCE_NOT_FINISHED
    assert "corrupt" in err2.value.why


def test_unfinished_source_is_not_packed(tmp_path: Path) -> None:
    (tmp_path / "news.partial").mkdir()
    with pytest.raises(ConfigError) as err:
        pack_source(tmp_path / "news")
    assert err.value.code is codes.SOURCE_NOT_FINISHED


def test_lzma_fallback_warns(tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tokbin.ops import pack as pack_mod

    monkeypatch.setattr(pack_mod, "choose_method", lambda: "lzma")
    with pytest.warns(CompatibilityWarning, match="TB-P003"):
        result = pack_source(source, tmp_path / "p")
    assert result.manifest.method == "lzma"
    assert [i.code for i in result.issues] == ["TB-P003"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pack_source(source, tmp_path / "p2", method="lzma")  # asked for: no warning


def test_invalid_method(tmp_path: Path, source: Path) -> None:
    with pytest.raises(ConfigError):
        pack_source(source, tmp_path / "p", method="gzip")  # type: ignore[arg-type]


def test_cli(
    tmp_path: Path,
    source: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "detect_mode", lambda: "full")
    assert main(["pack", "corpus/web", "--out", "packs", "--tar"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "✔ packed corpus/web → packs/web.tbpack.tar"
    assert lines[2].startswith("✔ Status: 14 files · 9.8 KiB → ")
    assert lines[2].endswith(" · zstd-3")

    assert main(["unpack", "packs/web.tbpack.tar", "--into", "remote"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == ("✔ Status: 14 files · 9.8 KiB · sha256 ok")
    assert main(["unpack", "packs/web.tbpack.tar", "--into", "remote"]) == 1
    capsys.readouterr()

    main(["pack", "corpus/web", "--out", "dirs"])
    capsys.readouterr()
    stored = tmp_path / "dirs" / "web.tbpack" / "train-00000.bin.zst"
    stored.write_bytes(stored.read_bytes()[:-4])
    assert main(["unpack", "dirs/web.tbpack", "--into", "bad", "--json"]) == 3
    document = json.loads(capsys.readouterr().out)
    assert document["error"]["code"] == "TB-I307"
