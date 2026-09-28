"""rm: only tokbin sources, never through links, never under a live writer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tokbin import ConfigError, ResumeError, codes, remove_source
from tokbin.cli import main as cli_main
from tokbin.cli.main import main
from tokbin.write.lock import WriteLock

from support import copy_fixture


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    copy_fixture(tmp_path / "web")
    return tmp_path


def test_removes_the_source_and_its_partial(corpus: Path) -> None:
    (corpus / "web.partial").mkdir()
    (corpus / "web.partial" / "x.bin").write_bytes(b"12345")
    result = remove_source(corpus / "web")
    assert result.removed == (corpus / "web", corpus / "web.partial")
    assert result.n_bytes > 5
    assert sorted(p.name for p in corpus.iterdir()) == []


def test_refuses_what_is_not_a_source(corpus: Path) -> None:
    (corpus / "notes").mkdir()
    (corpus / "notes" / "a.txt").write_text("keep me")
    with pytest.raises(ConfigError) as err:
        remove_source(corpus / "notes")
    assert err.value.code is codes.NOT_A_SOURCE
    assert (corpus / "notes" / "a.txt").exists()
    with pytest.raises(ConfigError) as err:
        remove_source(corpus / "missing")
    assert err.value.code is codes.SOURCE_NOT_FOUND
    with pytest.raises(ConfigError):
        remove_source(corpus / "..")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_refuses_a_symlink(corpus: Path) -> None:
    (corpus / "alias").symlink_to(corpus / "web")
    with pytest.raises(ConfigError):
        remove_source(corpus / "alias")
    assert (corpus / "web" / "meta.json").exists()


def test_refuses_a_live_writer(corpus: Path) -> None:
    (corpus / "web.partial").mkdir()
    lock = WriteLock(corpus / "web.partial")
    lock.acquire()
    try:
        with pytest.raises(ResumeError):
            remove_source(corpus / "web")
        assert (corpus / "web" / "meta.json").exists()
    finally:
        lock.release()


def test_cli_requires_yes(
    corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(corpus)
    monkeypatch.setattr(cli_main, "detect_mode", lambda: "full")
    assert main(["rm", "web"]) == 2
    assert "repeat with --yes" in capsys.readouterr().err
    assert (corpus / "web").exists()
    assert main(["rm", "web", "--yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["removed"] == ["web"]
    assert not (corpus / "web").exists()
