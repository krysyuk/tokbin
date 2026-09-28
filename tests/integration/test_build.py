"""build from standard inputs, including an interrupted build resumed from its settings."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tokbin import (
    BuildRecipe,
    ConfigError,
    Dataset,
    ResumeError,
    WriterConfig,
    build_source,
    codes,
    inspect_source,
    jsonl,
    read_source,
)
from tokbin.cli import main as cli_main
from tokbin.cli.main import main

from support import EOS, encode, make_docs, read_ids, read_skipped

pytestmark = [
    pytest.mark.requires_tokenizers,
    pytest.mark.filterwarnings("ignore::tokbin.ShardingWarning"),
    pytest.mark.filterwarnings("ignore::tokbin.DataQualityWarning"),
]


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    docs = make_docs(60, seed=9)
    lines = [json.dumps({"text": t}) for _, t in docs[:30]]
    lines.insert(5, "broken")
    (root / "a.jsonl").write_text("\n".join(lines) + "\n")
    (root / "b.jsonl").write_text("\n".join(json.dumps({"text": t}) for _, t in docs[30:]) + "\n")
    return root


def _recipe(inputs: Path, tokenizer_path: Path, **config: Any) -> BuildRecipe:
    return BuildRecipe(
        input_kind="jsonl",
        input_path=inputs,
        tokenizer=tokenizer_path,
        config=WriterConfig(**config),
    )


def test_build_equals_a_write_of_the_same_input(
    tmp_path: Path, inputs: Path, tokenizer_path: Path
) -> None:
    result = build_source(tmp_path / "c" / "web", _recipe(inputs, tokenizer_path, shard_bytes=64))
    assert result.stats.n_docs == 60
    assert result.stats.n_skipped == 1
    Dataset(tmp_path / "d").write(
        "web", jsonl(inputs), tokenizer_path, config=WriterConfig(shard_bytes=64)
    )
    a, b = read_source(tmp_path / "c" / "web"), read_source(tmp_path / "d" / "web")
    assert (a[:] == b[:]).all()
    assert read_ids(tmp_path / "c" / "web")[:2] == ["a.jsonl:0", "a.jsonl:1"]
    assert read_skipped(tmp_path / "c" / "web")[0]["id"] == "a.jsonl:5"
    assert read_skipped(tmp_path / "c" / "web")[0]["code"] == "TB-D206"
    assert not (tmp_path / "c" / "web" / "build.json").exists()


def test_txt_build(tmp_path: Path, tokenizer_path: Path) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "x.txt").write_text("t1 t2")
    recipe = BuildRecipe(input_kind="txt", input_path=tmp_path / "t", tokenizer=tokenizer_path)
    build_source(tmp_path / "web", recipe)
    assert read_source(tmp_path / "web")[:].tolist() == [*encode("t1 t2"), EOS]


class Stop(BaseException):
    pass


def test_resume_with_saved_settings(
    tmp_path: Path, inputs: Path, tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe(inputs, tokenizer_path, shard_bytes=64, batch_docs=4)
    target = tmp_path / "c" / "web"
    calls = [0]

    def stop(done: int, total: int, name: str) -> None:
        calls[0] += 1
        if calls[0] == 40:
            raise Stop

    with pytest.raises(Stop):
        build_source(target, recipe, progress=stop)
    saved = json.loads((tmp_path / "c" / "web.partial" / "build.json").read_text())
    assert saved["input"]["kind"] == "jsonl"
    assert saved["config"]["shard_bytes"] == 64
    info = inspect_source(target)
    assert info.partial is not None and info.partial.build

    with pytest.raises(ConfigError) as err:
        build_source(target)  # no input, no resume
    assert err.value.code is codes.BUILD_RECIPE_MISSING
    build_source(target, resume=True)  # settings from build.json
    build_source(tmp_path / "ref" / "web", recipe)
    for name in ("train-offsets.npy", "train-ids.jsonl", "train-00003.bin"):
        assert (target / name).read_bytes() == (tmp_path / "ref" / "web" / name).read_bytes()


def test_resume_without_anything_to_resume(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as err:
        build_source(tmp_path / "web", resume=True)
    assert err.value.code is codes.BUILD_RECIPE_MISSING
    assert "build.json" in err.value.why


def test_changed_input_is_refused_on_resume(
    tmp_path: Path, inputs: Path, tokenizer_path: Path
) -> None:
    recipe = _recipe(inputs, tokenizer_path, shard_bytes=64, batch_docs=4)
    calls = [0]

    def stop(*_: object) -> None:
        calls[0] += 1
        if calls[0] == 50:
            raise Stop

    with pytest.raises(Stop):
        build_source(tmp_path / "web", recipe, progress=stop)
    (inputs / "a.jsonl").write_text('{"text": "t1"}\n' + (inputs / "a.jsonl").read_text())
    with pytest.raises(ResumeError) as err:
        build_source(tmp_path / "web", resume=True)
    assert err.value.code is codes.RESUME_INPUT_MISMATCH


def test_missing_tokenizer(tmp_path: Path, inputs: Path) -> None:
    with pytest.raises(ConfigError) as err:
        build_source(tmp_path / "web", _recipe(inputs, tmp_path / "nope.json"))
    assert err.value.code is codes.TOKENIZER_FILE_INVALID


def test_cli(
    tmp_path: Path,
    inputs: Path,
    tokenizer_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "detect_mode", lambda: "full")
    argv = ["build", "c/web", "--from-jsonl", "data", "--tokenizer", str(tokenizer_path)]
    assert main([*argv, "--shard-size", "64"]) == 0
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[0] == "✔ built c/web [train]"
    assert lines[1] == "  60 docs · 488 tokens · 16 shards · 1 skipped"
    assert any("[TB-D206] Input record cannot be read: document 'a.jsonl:5'" in x for x in lines)
    assert lines[-1] == "! Status: complete · 2 warnings"
    assert "TB-D206" not in err  # shown once, in the summary, with its count

    assert main(argv) == 1  # the split exists
    assert "TB-C206" in capsys.readouterr().err
    assert main([*argv, "--overwrite", "--split", "valid", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["result"]["split"] == "valid"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["build", "w"], "pass --from-txt DIR or --from-jsonl PATH"),
        (["build", "w", "--resume", "--split", "valid"], "these options need an input: --split"),
        (["build", "w", "--from-txt", "d"], "--tokenizer PATH is required"),
        (["build", "w", "--from-txt", "d", "--tokenizer", "t", "--field", "x"], "--field"),
    ],
)
def test_cli_usage(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_cli_shard_size_units() -> None:
    from tokbin.cli.commands.build import parse_size

    assert parse_size("512M") == 512 * 1024**2
    assert parse_size("1g") == 1024**3
    assert parse_size("256MiB") == 256 * 1024**2
    assert parse_size("1.5K") == 1536
    assert parse_size("100") == 100
    with pytest.raises(Exception, match="not a size"):
        parse_size("lots")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery differs on Windows")
def test_ctrl_c_then_resume(tmp_path: Path, tokenizer_path: Path) -> None:
    """A real SIGINT: exit code 130, the resume command, then --resume completes."""
    data = tmp_path / "data.jsonl"
    with data.open("w") as f:
        for i in range(400_000):
            f.write(json.dumps({"text": f"t{i % 300} t{(i * 7) % 300} t5"}) + "\n")
    cmd = [sys.executable, "-m", "tokbin.cli.main", "build", "web"]
    args = ["--from-jsonl", str(data), "--tokenizer", str(tokenizer_path), "--shard-size", "4K"]
    proc = subprocess.Popen(
        [*cmd, *args],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    deadline = time.monotonic() + 60
    while not list(tmp_path.glob("web.partial/train-0000[2-9].bin")):
        assert time.monotonic() < deadline and proc.poll() is None, "the build ended too soon"
        time.sleep(0.05)
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 130
    assert "continue with: tokbin build web --resume" in err
    assert (tmp_path / "web.partial" / "checkpoint.json").is_file()

    done = subprocess.run([*cmd, "--resume"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert read_source(tmp_path / "web").n_docs == 400_000
