"""The CLI: output snapshots (plain text and JSON), exit codes, global flags."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from tokbin import DependencyError, IntegrityError, codes
from tokbin._deps import DependencyStatus
from tokbin.cli import main as cli_main
from tokbin.cli.commands import info as info_cmd
from tokbin.cli.main import main
from tokbin.ops.inspect import inspect_corpus

from support import copy_fixture, native


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``corpus/web`` in the current directory: paths in the output are relative."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "detect_mode", lambda: "full")
    copy_fixture(tmp_path / "corpus" / "web")
    return Path("corpus")


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code, out, err = run(capsys, *argv, "--json")
    assert err == ""
    document: dict[str, Any] = json.loads(out)
    assert document["exit_code"] == code
    return code, document


SPLIT_NOTE = (
    "└ [TB-S203] Documents are split across shards: in train; readers handle this transparently"
)


def test_info_source(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "info", "corpus/web")
    assert (code, err) == (0, "")
    assert out == "\n".join(
        [
            "corpus/web/   360 tokens · complete",
            "",
            "  dtype       uint16 · vocab 303",
            "  tokenizer   unnamed · 5b6c9ce0 · eos 1 · bos 2",
            "  format      schema 1 · written by tokbin 0.1.0.dev0",
            "",
            "  split   tokens   docs   shards    size   skipped",
            "  train      303     40        3   606 B         0",
            "  valid       57      5        1   114 B         0",
            "",
            f"  {SPLIT_NOTE}",
            "",
            "✔ Status: ready for training",
            "",
        ]
    )


def test_info_corpus(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    copy_fixture(corpus / "code")
    (corpus / "mix.json").write_text('{"web": 3, "code": 1}')
    code, out, _ = run(capsys, "info", "corpus")
    assert code == 0
    assert out == "\n".join(
        [
            "corpus/   720 tokens · 2 sources",
            "",
            "  ✔ code   360 tokens   45 docs   4 shards   720 B",
            f"    {SPLIT_NOTE}",
            "  ✔ web    360 tokens   45 docs   4 shards   720 B",
            f"    {SPLIT_NOTE}",
            "",
            "  tokenizer   unnamed · 5b6c9ce0 · same in all sources",
            "  mix         web 0.75 · code 0.25",
            "",
            "✔ Status: ready for training",
            "",
        ]
    )


def test_ls_with_problems(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    copy_fixture(corpus / "code")
    (corpus / "code" / "train-00002.bin").unlink()
    (corpus / "news.partial").mkdir()
    code, out, _ = run(capsys, "ls", "corpus")
    assert code == 3  # a damaged source
    lines = out.splitlines()
    assert lines[0] == "corpus/   3 sources · 720 tokens · 1.4 KiB"
    assert lines[2] == "  ✖ code   corrupt    360 tokens   45 docs   4 shards   720 B"
    assert lines[3].startswith("  ! news   partial    0 shards closed · updated ")
    assert lines[4] == "  ✔ web    complete   360 tokens   45 docs   4 shards   720 B"


def test_ls_defaults_to_the_current_directory(
    corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(corpus)
    code, out, _ = run(capsys, "ls")
    assert code == 0
    assert out.startswith("./   1 source")


def test_status_with_unfinished_write(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    partial = corpus / "web.partial"
    partial.mkdir()
    (partial / "valid-offsets.i64").touch()
    code, out, _ = run(capsys, "status", "corpus/web")
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "corpus/web/   complete · train, valid"
    assert lines[2] == native("  ! unfinished write   corpus/web.partial")
    assert lines[3].startswith(" " * 23 + "split valid · 0 shards closed · updated ")
    assert lines[-1] == "! Status: complete · 1 warning"

    code, _, err = run(capsys, "status", "corpus/web", "--strict")
    assert code == 1
    assert "--strict" in err


def test_verify(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "verify", "corpus/web")
    assert code == 0
    assert out == "\n".join(
        [
            "corpus/web/",
            "",
            "  ✔ train-00000.bin   256 B   sha256 ok",
            "  ✔ train-00001.bin   256 B   sha256 ok",
            "  ✔ train-00002.bin    94 B   sha256 ok",
            "  ✔ valid-00000.bin   114 B   sha256 ok",
            "",
            "✔ Status: intact · 4 shards · 720 B checked",
            "",
        ]
    )


def test_verify_damaged(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = corpus / "web" / "train-00001.bin"
    path.write_bytes(b"\xff" + path.read_bytes()[1:])
    code, out, _ = run(capsys, "verify", "corpus/web")
    assert code == 3
    lines = out.splitlines()
    assert lines[3] == "  ✖ train-00001.bin   256 B   sha256 mismatch"
    assert lines[7] == native("[TB-I302] Shard is corrupted: corpus/web/train-00001.bin")
    assert lines[8].startswith("  cause: sha256 ")
    fix = "  fix:   download the shard again and repeat `tokbin verify corpus/web`"
    assert lines[9] == native(fix)
    assert lines[-1] == "✖ Status: damaged · 1 of 4 shards"


def test_verify_corpus(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    copy_fixture(corpus / "code")
    (corpus / "code" / "valid-skipped.jsonl").write_text("{}\n")
    (corpus / "news.partial").mkdir()
    code, out, _ = run(capsys, "verify", "corpus")
    assert code == 3
    assert "  \u2013 news   unfinished write, nothing to verify" in out
    assert out.splitlines()[-1] == ("✖ Status: damaged · 1 other problem")

    code, document = run_json(capsys, "verify", "corpus")
    assert code == 3
    assert document["result"]["skipped"] == ["news"]
    assert [s["ok"] for s in document["result"]["sources"]] == [False, True]


@pytest.mark.parametrize(
    "argv", [["info", "corpus"], ["info", "corpus/web"], ["ls", "corpus"], ["status", "corpus"]]
)
def test_json_matches_the_library(
    corpus: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    code, document = run_json(capsys, *argv)
    assert code == 0
    assert set(document) == {
        "format",
        "tokbin",
        "command",
        "mode",
        "exit_code",
        "result",
        "warnings",
        "error",
    }
    assert (document["format"], document["command"], document["mode"]) == (1, argv[0], "full")
    assert document["error"] is None
    if argv[1] == "corpus":
        expected = inspect_corpus("corpus").to_dict()
        assert document["result"] == json.loads(json.dumps(expected))
    else:
        assert document["result"]["state"] == "complete"


def test_json_error(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, document = run_json(capsys, "info", "corpus/missing")
    assert code == 1
    assert document["result"] is None
    assert document["error"] == {
        "type": "ConfigError",
        "code": "TB-C305",
        "what": native("Dataset path not found: corpus/missing"),
        "why": "neither the source directory nor an unfinished write of it exists",
        "fix": "check the path; `tokbin ls <corpus>` lists the sources of a corpus",
        "where": "tokbin.ops.inspect.inspect_source",
    }


def test_error_text(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "info", "corpus/missing")
    assert (code, out) == (1, "")
    assert err.splitlines() == [
        native("✖ [TB-C305] Dataset path not found: corpus/missing"),
        "  where: tokbin.ops.inspect.inspect_source",
        "  cause: neither the source directory nor an unfinished write of it exists",
        "  fix:   check the path; `tokbin ls <corpus>` lists the sources of a corpus",
    ]


@pytest.mark.parametrize(
    ("argv", "lines"),
    [
        (["nope"], ["error: unrecognized command 'nope'", "Usage: tokbin [OPTIONS] <COMMAND>"]),
        (["bild"], ["  tip: a similar command exists: 'build'"]),
        (["help", "bild"], ["  tip: a similar command exists: 'build'"]),
        (
            ["info"],
            [
                "error: the following required arguments were not provided:",
                "  <PATH>",
                "Usage: tokbin info [OPTIONS] <PATH>",
                "For more information, try 'tokbin info --help'.",
            ],
        ),
        (["verify", "a", "b"], ["error: unexpected argument 'b' found"]),
        (["info", "x", "--bad"], ["error: unexpected argument '--bad' found"]),
        (
            ["build", "x", "--split", "nope"],
            [
                "error: invalid value 'nope' for '--split <SPLIT>'",
                "  [possible values: train, valid, test]",
            ],
        ),
        (
            ["build", "x", "--from-txt", "a", "--from-jsonl", "b"],
            ["error: the argument '--from-jsonl <PATH>' cannot be used with '--from-txt <DIR>'"],
        ),
        (
            ["build", "x", "--tokenizer"],
            ["error: a value is required for '--tokenizer <PATH>' but none was supplied"],
        ),
    ],
)
def test_usage_errors(
    argv: list[str], lines: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    out, err = capsys.readouterr()
    assert out == ""
    for line in lines:
        assert line in err.splitlines()


@pytest.mark.parametrize("argv", [[], ["--help"], ["-h"], ["help"]])
def test_help(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 0
    out, err = capsys.readouterr()
    assert err == ""
    lines = out.splitlines()
    assert lines[0].startswith("tokbin ")
    assert "Usage: tokbin [OPTIONS] <COMMAND>" in lines
    for header in ("Write:", "Inspect:", "Transfer:", "Maintenance:", "Global options:"):
        assert header in lines
    for name in ("build", "ls", "info", "status", "verify", "pack", "rm", "doctor"):
        assert any(line.startswith(f"  {name} ") for line in lines)
    assert "  -V, --version   Print version" in lines
    assert "Examples:" in lines


@pytest.mark.parametrize("argv", [["build", "--help"], ["help", "build"]])
def test_command_help(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Tokenize text files or JSON Lines into a source"
    assert "Usage: tokbin build [OPTIONS] <TARGET>" in lines
    # Sections in order, global options last.
    headers = [line for line in lines if line.endswith(":") and not line.startswith(" ")]
    assert headers == ["Arguments:", "Input:", "Tokenizer:", "Output:", "Global options:"]
    assert any(line.startswith("      --shard-size <SIZE>") for line in lines)
    assert "[possible values:" in "\n".join(lines)


def test_help_colors(capsys: pytest.CaptureFixture[str]) -> None:
    """Without a terminal the help has no escape codes; with one, the accent is used."""
    from tokbin.cli.help import Parser
    from tokbin.cli.render import Style

    assert main(["--help"]) == 0
    assert "\x1b[" not in capsys.readouterr().out
    parser = Parser(prog="tokbin", description="x", add_help=False)
    parser.add_argument("--flag", help="A flag [default: off]")
    true_lines = "\n".join(parser.help_lines(Style(color=True, unicode=True, truecolor=True)))
    assert "\x1b[1;38;2;242;196;170m--flag\x1b[0m" in true_lines
    assert "\x1b[2m[default:\x1b[0m" in true_lines
    lines_256 = "\n".join(parser.help_lines(Style(color=True, unicode=True)))
    assert "\x1b[1;38;5;223m--flag\x1b[0m" in lines_256


def _raising(exc: BaseException) -> Callable[..., Any]:
    def fail(*args: object) -> Any:
        raise exc

    return fail


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            DependencyError(codes.DEPENDENCY_MISSING, "torch", why="needed", fix="install it"),
            4,
        ),
        (IntegrityError(codes.SHARD_MISSING, "x", why="gone", fix="restore"), 3),
        (KeyboardInterrupt(), 130),
        (PermissionError(13, "Permission denied", "corpus/web"), 1),
        (ValueError("boom"), 1),
    ],
)
def test_exit_codes(
    corpus: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    exc: BaseException,
    expected: int,
) -> None:
    monkeypatch.setattr(info_cmd, "run", _raising(exc))
    code, _, err = run(capsys, "info", "corpus")
    assert code == expected
    first = err.splitlines()[0]
    if isinstance(exc, ValueError):
        assert first == "✖ [TB-X001] Internal tokbin error: ValueError: boom"
    elif isinstance(exc, KeyboardInterrupt):
        assert first == "✖ interrupted"
    elif isinstance(exc, OSError):
        assert first == "✖ PermissionError: [Errno 13] Permission denied: 'corpus/web'"
    code, document = run_json(capsys, "info", "corpus")
    assert code == expected
    assert document["error"]["type"] in {
        "DependencyError",
        "IntegrityError",
        "KeyboardInterrupt",
        "PermissionError",
        "InternalError",
    }


def test_library_warnings_are_shown_and_counted(
    corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = info_cmd.run

    def warn_then_run(args: Any, ctx: Any) -> Any:
        warnings.warn("old schema", FutureWarning, stacklevel=1)
        return real(args, ctx)

    monkeypatch.setattr(info_cmd, "run", warn_then_run)
    code, out, err = run(capsys, "info", "corpus/web")
    assert code == 0
    assert out.startswith("corpus/web/")
    assert err == "! old schema\n"
    assert run(capsys, "--strict", "info", "corpus/web")[0] == 1
    code, document = run_json(capsys, "info", "corpus/web")
    assert document["warnings"] == ["old schema"]


def test_core_mode_header(
    corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_main, "detect_mode", lambda: "core")
    _, out, _ = run(capsys, "ls", "corpus")
    assert out.splitlines()[0].startswith("tokbin ")
    assert "mode core" in out.splitlines()[0]
    _, document = run_json(capsys, "ls", "corpus")
    assert document["mode"] == "core"


def test_doctor_in_core_mode(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokbin import _deps

    real = _deps.probe

    def probe(name: str) -> DependencyStatus:
        if name == "tokenizers":
            return DependencyStatus(dependency=_deps.KNOWN[name], installed=False, version=None)
        return real(name)

    monkeypatch.setattr(_deps, "probe", probe)
    code, out, err = run(capsys, "doctor")
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert " · mode core · Python " in lines[0]
    assert lines[1].startswith("  ✔ numpy ")
    tokenizers = next(line for line in lines if "tokenizers" in line)
    assert tokenizers.startswith("  ✖ tokenizers ")
    assert "not installed" in tokenizers
    assert "writing datasets: pip install tokbin" in tokenizers
    assert lines[-1] == "To write datasets: pip install tokbin"

    code, document = run_json(capsys, "doctor")
    names = [p["name"] for p in document["result"]["packages"]]
    assert names == ["numpy", "tokenizers", "zstd", "torch", "huggingface-hub"]
    assert document["result"]["packages"][1]["state"] == "missing"


def test_cli_has_no_side_effects(
    corpus: Path,
    capsys: pytest.CaptureFixture[str],
    no_side_effects: Callable[[], AbstractContextManager[None]],
) -> None:
    with no_side_effects():
        for argv in (["info", "corpus"], ["verify", "corpus/web"], ["doctor"]):
            assert main(argv) == 0
    capsys.readouterr()


def test_entry_point_exit_code_and_plain_pipe_output(corpus: Path) -> None:
    (corpus / "web" / "train-00000.bin").unlink()
    proc = subprocess.run(
        [sys.executable, "-m", "tokbin.cli.main", "info", "corpus"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 3
    assert "\x1b[" not in proc.stdout  # not a terminal: no escape codes
    assert "[TB-I301]" in proc.stdout
