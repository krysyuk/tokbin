"""Interrupted writes and ``resume=True`` (spec 8.3, 19).

The central property: a write interrupted anywhere and then resumed produces a source
byte for byte equal to the one of an uninterrupted write.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tokbin import (
    ConfigError,
    DataQualityWarning,
    Dataset,
    ResumeError,
    ShardingWarning,
    StreamWriter,
    WriterConfig,
    codes,
    inspect_source,
)
from tokbin.format.checkpoint import read_checkpoint
from tokbin.write import partial as partial_mod
from tokbin.write import stream_writer

from support import make_docs, make_tokenizer

pytestmark = [
    pytest.mark.requires_tokenizers,
    pytest.mark.filterwarnings("ignore::tokbin.ShardingWarning"),
]

#: Tiny shards (24 uint16 items) so that a few dozen documents cross many boundaries.
CFG = WriterConfig(shard_bytes=48, prepend_bos=True, batch_docs=4)


class Interrupted(BaseException):  # like a kill: not wrapped at the API boundary
    pass


def _docs() -> list[tuple[str, bytes]]:
    docs: list[tuple[str, bytes]] = [(i, t.encode()) for i, t in make_docs(40, seed=3)]
    docs[5] = ("empty", b"")
    docs[17] = ("broken", b"t1 \xff")
    docs[30] = ("long", " ".join(f"t{i}" for i in range(70)).encode())  # spans shards
    return docs


def _until(docs: list[Any], stop: int, exc: BaseException | None = None) -> Iterator[Any]:
    """Yield ``stop`` documents, then fail like a crashing generator."""
    yield from docs[:stop]
    raise exc if exc is not None else Interrupted(f"stopped after {stop}")


def _write(root: Path, docs: Any, tok: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        warnings.simplefilter("ignore", ShardingWarning)
        return Dataset(root).write("web", docs, tok, config=kwargs.pop("config", CFG), **kwargs)


def _snapshot(source: Path) -> dict[str, bytes]:
    files = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    meta = json.loads(files.pop("meta.json"))
    for split in meta["splits"].values():
        split.pop("created_at")
    files["meta.json"] = json.dumps(meta, sort_keys=True).encode()
    return files


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    root = tmp_path_factory.mktemp("reference")
    _write(root, _docs(), make_tokenizer())
    return _snapshot(root / "web")


@pytest.fixture
def tok() -> Any:
    return make_tokenizer()


def test_reference_crosses_many_shards(reference: dict[str, bytes]) -> None:
    meta = json.loads(reference["meta.json"])["splits"]["train"]
    assert len(meta["shards"]) > 10
    assert meta["has_split_docs"]


@pytest.mark.parametrize("stop", range(len(_docs()) + 1))
def test_interrupt_anywhere_then_resume(
    tmp_path: Path, tok: Any, reference: dict[str, bytes], stop: int
) -> None:
    docs = _docs()
    with pytest.raises(Interrupted):
        _write(tmp_path, _until(docs, stop), tok)
    assert not (tmp_path / "web").exists()
    partial = tmp_path / "web.partial"
    assert (partial / "checkpoint.json").is_file()
    assert not (partial / ".lock").exists()
    assert not list(partial.glob("*.bin.open"))

    result = _write(tmp_path, docs, tok, resume=True)
    assert not partial.exists()
    assert _snapshot(tmp_path / "web") == reference
    assert result.stats.n_input == len(docs)
    assert result.stats.n_skipped == 2
    assert "TB-R207" in [i.code for i in result.issues]


def test_repeated_interruptions(tmp_path: Path, tok: Any, reference: dict[str, bytes]) -> None:
    docs = _docs()
    for stop in (7, 7, 19, 33, 34):
        with pytest.raises(Interrupted):
            _write(tmp_path, _until(docs, stop), tok, resume=True)
    _write(tmp_path, docs, tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


@pytest.mark.parametrize("batch_docs", [1, 1024])
def test_resume_with_another_batch_size(
    tmp_path: Path, tok: Any, reference: dict[str, bytes], batch_docs: int
) -> None:
    docs = _docs()
    with pytest.raises(Interrupted):
        _write(tmp_path, _until(docs, 25), tok)
    cfg = WriterConfig(shard_bytes=48, prepend_bos=True, batch_docs=batch_docs)
    _write(tmp_path, docs, tok, config=cfg, resume=True)
    assert _snapshot(tmp_path / "web") == reference


def test_keyboard_interrupt(tmp_path: Path, tok: Any, reference: dict[str, bytes]) -> None:
    docs = _docs()
    with pytest.raises(KeyboardInterrupt):
        _write(tmp_path, _until(docs, 21, KeyboardInterrupt()), tok)
    info = inspect_source(tmp_path / "web")
    assert info.state == "partial"
    assert info.partial is not None and info.partial.resumable
    _write(tmp_path, docs, tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


def test_checkpoint_follows_closed_shards(tmp_path: Path, tok: Any) -> None:
    docs = _docs()
    with pytest.raises(Interrupted):
        _write(tmp_path, _until(docs, 30), tok)
    partial = tmp_path / "web.partial"
    cp = read_checkpoint(partial)
    assert [s.name for s in cp.closed_shards] == sorted(p.name for p in partial.glob("*.bin"))
    assert cp.n_docs + cp.n_skipped == cp.n_input_consumed
    assert cp.last_input_id == docs[cp.n_input_consumed - 1][0]
    # Side files may hold entries written after the checkpoint; resume cuts them off.
    assert (partial / "train-ids.jsonl").stat().st_size >= cp.side_files.ids


def test_crash_between_shard_close_and_checkpoint(
    tmp_path: Path, tok: Any, reference: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = stream_writer.write_checkpoint
    calls = [0]

    def flaky(partial: Path, checkpoint: Any) -> None:
        calls[0] += 1
        if calls[0] == 6:
            raise Interrupted("crash after a shard was renamed")
        real(partial, checkpoint)

    monkeypatch.setattr(stream_writer, "write_checkpoint", flaky)
    with pytest.raises(Interrupted):
        _write(tmp_path, _docs(), tok)
    partial = tmp_path / "web.partial"
    # One closed shard more than the checkpoint knows about: resume must drop it.
    assert len(list(partial.glob("*.bin"))) == len(read_checkpoint(partial).closed_shards) + 1
    monkeypatch.setattr(stream_writer, "write_checkpoint", real)
    _write(tmp_path, _docs(), tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


def test_crash_during_publication(
    tmp_path: Path, tok: Any, reference: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = partial_mod.write_meta

    def fail(*args: Any) -> None:
        raise Interrupted("crash while publishing")

    monkeypatch.setattr(partial_mod, "write_meta", fail)
    with pytest.raises(Interrupted):
        _write(tmp_path, _docs(), tok)
    assert (tmp_path / "web.partial" / "train-offsets.npy").is_file()  # finish had run
    monkeypatch.setattr(partial_mod, "write_meta", real)
    _write(tmp_path, _docs(), tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


def test_crash_while_adding_a_split(
    tmp_path: Path, tok: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, _docs(), tok)
    valid = WriterConfig(split="valid", shard_bytes=48)
    real = partial_mod.write_meta

    def fail(*args: Any) -> None:
        raise Interrupted("crash after the other split was carried over")

    monkeypatch.setattr(partial_mod, "write_meta", fail)
    with pytest.raises(Interrupted):
        _write(tmp_path, make_docs(5), tok, config=valid)
    assert (tmp_path / "web.partial" / "train-00000.bin").is_file()
    monkeypatch.setattr(partial_mod, "write_meta", real)
    _write(tmp_path, make_docs(5), tok, config=valid, resume=True)
    meta = json.loads((tmp_path / "web" / "meta.json").read_text())
    assert list(meta["splits"]) == ["train", "valid"]


def test_low_disk_space_stops_before_a_shard(
    tmp_path: Path, tok: Any, reference: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokbin import _fs

    real = _fs.free_bytes
    calls = [0]

    def shrinking(path: Path) -> int:
        calls[0] += 1
        return real(path) if calls[0] < 5 else 95  # less than two 48-byte shards

    monkeypatch.setattr(_fs, "free_bytes", shrinking)
    with pytest.raises(ConfigError) as err:
        _write(tmp_path, _docs(), tok)
    assert err.value.code is codes.DISK_SPACE_LOW
    assert len(read_checkpoint(tmp_path / "web.partial").closed_shards) == 4
    monkeypatch.setattr(_fs, "free_bytes", real)
    _write(tmp_path, _docs(), tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


def test_resume_without_a_partial_is_a_fresh_write(
    tmp_path: Path, tok: Any, reference: dict[str, bytes]
) -> None:
    result = _write(tmp_path, _docs(), tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference
    assert "TB-R207" not in [i.code for i in result.issues]


def test_partial_without_checkpoint_starts_over(
    tmp_path: Path, tok: Any, reference: dict[str, bytes]
) -> None:
    partial = tmp_path / "web.partial"
    partial.mkdir()
    (partial / "train-00000.bin").write_bytes(b"junk")
    _write(tmp_path, _docs(), tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


def _interrupt(tmp_path: Path, tok: Any, stop: int = 20, docs: Any = None) -> Path:
    with pytest.raises(Interrupted):
        _write(tmp_path, _until(docs if docs is not None else _docs(), stop), tok)
    return tmp_path / "web.partial"


def test_existing_partial_needs_resume(tmp_path: Path, tok: Any) -> None:
    _interrupt(tmp_path, tok)
    with pytest.raises(ResumeError) as err:
        _write(tmp_path, _docs(), tok)
    assert err.value.code is codes.PARTIAL_EXISTS
    assert "resume=True" in err.value.fix
    assert "tokbin clean" in err.value.fix


def test_other_order_is_refused(tmp_path: Path, tok: Any) -> None:
    _interrupt(tmp_path, tok)
    docs = _docs()
    docs[3], docs[4] = docs[4], docs[3]
    with pytest.raises(ResumeError) as err:
        _write(tmp_path, docs, tok, resume=True)
    assert err.value.code is codes.RESUME_INPUT_MISMATCH
    assert (tmp_path / "web.partial" / "checkpoint.json").is_file()  # still resumable


def test_shorter_input_is_refused(tmp_path: Path, tok: Any) -> None:
    partial = _interrupt(tmp_path, tok, stop=30)
    consumed = read_checkpoint(partial).n_input_consumed
    with pytest.raises(ResumeError) as err:
        _write(tmp_path, _docs()[: consumed - 2], tok, resume=True)
    assert err.value.code is codes.RESUME_INPUT_MISMATCH
    assert "input ended" in err.value.why


def test_documents_without_ids(tmp_path: Path, tok: Any) -> None:
    texts = [t for _, t in _docs()]
    with pytest.raises(Interrupted):
        _write(tmp_path, _until(texts, 20), tok)
    with pytest.warns(DataQualityWarning, match="TB-R204"):
        result = Dataset(tmp_path).write("web", texts, tok, config=CFG, resume=True)
    assert "TB-R204" in [i.code for i in result.issues]


def test_changed_straddling_document_is_detected_without_ids(tmp_path: Path, tok: Any) -> None:
    texts = [t for _, t in _docs()]
    for stop in range(1, len(texts)):
        shutil.rmtree(tmp_path / "web.partial", ignore_errors=True)
        with pytest.raises(Interrupted):
            _write(tmp_path, _until(texts, stop), tok)
        cp = read_checkpoint(tmp_path / "web.partial")
        if cp.pending_doc_items >= 2:  # more than the BOS token is already written
            break
    else:
        pytest.fail("no interruption point with a straddling document")
    changed = list(texts)
    changed[cp.n_input_consumed] = b"t299 t298 t297 t296 t295 t294 t293 t292 t291"
    with pytest.raises(ResumeError) as err, warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        Dataset(tmp_path).write("web", changed, tok, config=CFG, resume=True)
    assert err.value.code is codes.RESUME_INPUT_MISMATCH


@pytest.mark.parametrize(
    "change",
    [
        {"config": WriterConfig(shard_bytes=64, prepend_bos=True, batch_docs=4)},
        {"config": WriterConfig(shard_bytes=48, prepend_bos=False, batch_docs=4)},
        {"tokenizer": "other"},
    ],
)
def test_other_settings_are_refused(tmp_path: Path, tok: Any, change: dict[str, Any]) -> None:
    _interrupt(tmp_path, tok)
    other_tok = make_tokenizer(specials=("<|endoftext|>", "<s>", "<pad>"))
    with pytest.raises(ResumeError) as err:
        _write(
            tmp_path,
            _docs(),
            other_tok if "tokenizer" in change else tok,
            config=change.get("config", CFG),
            resume=True,
        )
    assert err.value.code is codes.RESUME_SETTINGS_MISMATCH


def test_damaged_partial(tmp_path: Path, tok: Any) -> None:
    partial = _interrupt(tmp_path, tok)
    (partial / "train-00001.bin").unlink()
    with pytest.raises(ResumeError) as err:
        _write(tmp_path, _docs(), tok, resume=True)
    assert err.value.code is codes.PARTIAL_DAMAGED

    (partial / "checkpoint.json").write_text("{}")
    with pytest.raises(ResumeError) as err:
        _write(tmp_path, _docs(), tok, resume=True)
    assert err.value.code is codes.PARTIAL_DAMAGED
    assert not (partial / ".lock").exists()  # released after the failure


def test_truncated_side_file(tmp_path: Path, tok: Any) -> None:
    partial = _interrupt(tmp_path, tok)
    os.truncate(partial / "train-ids.jsonl", 3)
    with pytest.raises(ResumeError) as err:
        _write(tmp_path, _docs(), tok, resume=True)
    assert err.value.code is codes.PARTIAL_DAMAGED


# --- the lock -------------------------------------------------------------------------


def test_second_writer_is_refused(tmp_path: Path, tok: Any) -> None:
    with StreamWriter(tmp_path / "web", tok, config=CFG) as first:
        first.add("a", "t1 t2")
        for resume in (False, True):
            with pytest.raises(ResumeError) as err:
                StreamWriter(tmp_path / "web", tok, config=CFG, resume=resume).open()
            assert err.value.code is codes.WRITER_ACTIVE
            assert f"process {os.getpid()}" in err.value.why
    assert (tmp_path / "web" / "meta.json").is_file()
    assert not (tmp_path / "web" / ".lock").exists()


def _lock(partial: Path, **fields: Any) -> None:
    import socket

    data = {"pid": 1, "host": socket.gethostname(), "started_at": "2026-01-01T00:00:00Z"}
    data.update({"token": "x"}, **fields)
    (partial / ".lock").write_text(json.dumps(data))


def test_stale_lock_is_taken_over(tmp_path: Path, tok: Any, reference: dict[str, bytes]) -> None:
    partial = _interrupt(tmp_path, tok)
    _lock(partial, pid=2**22 + 12345)  # no such process
    _write(tmp_path, _docs(), tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


def test_lock_of_another_host_is_respected(tmp_path: Path, tok: Any) -> None:
    partial = _interrupt(tmp_path, tok)
    _lock(partial, pid=2**22 + 12345, host="elsewhere")
    with pytest.raises(ResumeError) as err:
        _write(tmp_path, _docs(), tok, resume=True)
    assert err.value.code is codes.WRITER_ACTIVE
    assert "elsewhere" in err.value.why


def test_unreadable_lock(tmp_path: Path, tok: Any, reference: dict[str, bytes]) -> None:
    partial = _interrupt(tmp_path, tok)
    (partial / ".lock").write_text("")
    with pytest.raises(ResumeError) as err:  # maybe being written right now
        _write(tmp_path, _docs(), tok, resume=True)
    assert err.value.code is codes.WRITER_ACTIVE
    os.utime(partial / ".lock", (0, 0))  # old: its writer is long gone
    _write(tmp_path, _docs(), tok, resume=True)
    assert _snapshot(tmp_path / "web") == reference


# The abandoned writer leaks its open files: that is the situation being modelled.
@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_lock_abandoned_in_this_process(tmp_path: Path, tok: Any) -> None:
    writer = StreamWriter(tmp_path / "web", tok, config=CFG)
    writer.add("a", "t1")  # opened, never closed or aborted
    with pytest.raises(ResumeError):
        StreamWriter(tmp_path / "web", tok, config=CFG, resume=True).open()
    del writer  # the owner is gone: its lock is stale now
    import gc

    gc.collect()
    result = Dataset(tmp_path).write("web", [("a", "t1")], tok, config=CFG, resume=True)
    assert result.stats.n_docs == 1


def test_finished_source_has_no_work_files(tmp_path: Path, tok: Any) -> None:
    _interrupt(tmp_path, tok)
    _write(tmp_path, _docs(), tok, resume=True)
    names = {p.name for p in (tmp_path / "web").iterdir()}
    assert not names & {"checkpoint.json", ".lock"}
    assert not [n for n in names if n.endswith((".i64", ".open"))]
