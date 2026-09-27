"""``clean``: removing unfinished writes, never a live one or a finished source."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from tokbin import Dataset, ResumeError, clean_source, codes, inspect_source
from tokbin.cli.main import main
from tokbin.write.lock import WriteLock

from support import copy_fixture


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    copy_fixture(tmp_path / "web")
    partial = tmp_path / "web.partial"
    partial.mkdir()
    (partial / "train-00000.bin").write_bytes(b"\0" * 100)
    return tmp_path


def test_removes_only_the_partial(corpus: Path) -> None:
    result = Dataset(corpus).clean("web")
    assert result.removed == (corpus / "web.partial",)
    assert result.n_bytes == 100
    assert not (corpus / "web.partial").exists()
    assert inspect_source(corpus / "web").state == "complete"
    assert Dataset(corpus).clean("web").removed == ()  # nothing left


def test_accepts_the_partial_path_and_a_partial_only_source(corpus: Path) -> None:
    (corpus / "news.partial").mkdir()
    assert clean_source(corpus / "news.partial").removed == (corpus / "news.partial",)
    assert clean_source(corpus / "missing").removed == ()


def test_refuses_a_live_writer(corpus: Path) -> None:
    lock = WriteLock(corpus / "web.partial")
    lock.acquire()
    try:
        info = inspect_source(corpus / "web")
        assert info.partial is not None
        assert info.partial.writer == (os.getpid(), socket.gethostname())
        with pytest.raises(ResumeError) as err:
            clean_source(corpus / "web")
        assert err.value.code is codes.WRITER_ACTIVE
        assert (corpus / "web.partial").exists()
    finally:
        lock.release()


def test_takes_over_a_stale_lock(corpus: Path) -> None:
    data = {"pid": 2**22 + 12345, "host": socket.gethostname(), "started_at": "x", "token": "t"}
    (corpus / "web.partial" / ".lock").write_text(json.dumps(data))
    info = inspect_source(corpus / "web")
    assert info.partial is not None and info.partial.writer is None
    assert clean_source(corpus / "web").removed


def test_old_copies_of_a_replaced_source(corpus: Path) -> None:
    (corpus / ".web.old-4242").mkdir()
    (corpus / ".web.old-4242" / "x").write_bytes(b"12")
    result = clean_source(corpus / "web")
    assert set(result.removed) == {corpus / "web.partial", corpus / ".web.old-4242"}
    assert result.n_bytes == 102


def test_invalid_names(corpus: Path) -> None:
    from tokbin import ConfigError

    with pytest.raises(ConfigError):
        Dataset(corpus).clean("..")


def test_cli(
    corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokbin.cli import main as cli_main

    monkeypatch.chdir(corpus)
    monkeypatch.setattr(cli_main, "detect_mode", lambda: "full")
    assert main(["status", "web"]) == 0
    out = capsys.readouterr().out
    assert "tokbin clean web" in out
    assert main(["clean", "web"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "✔ removed web.partial",
        "",
        "✔ Status: 100 B freed",
    ]
    assert main(["clean", "web"]) == 0
    assert "nothing to clean" in capsys.readouterr().out
    assert main(["clean", "web", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["removed"] == []
