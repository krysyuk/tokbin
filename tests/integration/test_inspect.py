"""Inspection: states of sources and corpora (spec 12.1) without reading the data."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import pytest

from tokbin import ConfigError, Dataset, codes, inspect_corpus, inspect_source
from tokbin.format import schema
from tokbin.format.checkpoint import Checkpoint, SideFileLengths, write_checkpoint
from tokbin.format.meta import ShardInfo
from tokbin.ops import inspect as inspect_mod
from tokbin.ops.inspect import describe

from support import copy_fixture


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    copy_fixture(root / "web")
    return root


def _meta(source: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads((source / "meta.json").read_text())
    return data


def _codes(items: tuple[object, ...]) -> list[str]:
    return [getattr(i, "code") for i in items]  # noqa: B009


def test_complete_source(
    corpus: Path, no_side_effects: Callable[[], AbstractContextManager[None]]
) -> None:
    with no_side_effects():
        info = inspect_source(corpus / "web")
    assert info.state == "complete"
    assert info.problems == ()
    assert (info.dtype, info.vocab_size, info.eos_id, info.bos_id) == ("uint16", 303, 1, 2)
    assert info.tokenizer_hash == "5b6c9ce06a2c8850"
    assert [s.name for s in info.splits] == ["train", "valid"]
    assert (info.n_items, info.n_docs, info.n_shards, info.n_bytes) == (360, 45, 4, 720)
    # Split documents are a fact, not a warning.
    assert _codes(info.issues) == ["TB-S203"]
    assert info.n_warnings == 0
    assert info.partial is None


def test_missing_and_wrong_size_shards_make_it_corrupt(corpus: Path) -> None:
    (corpus / "web" / "train-00001.bin").unlink()
    with (corpus / "web" / "valid-00000.bin").open("ab") as f:
        f.write(b"\0\0")
    info = inspect_source(corpus / "web")
    assert info.state == "corrupt"
    assert _codes(info.problems) == ["TB-I301", "TB-I303"]
    assert info.splits  # counters from meta.json are still shown


def test_damaged_index_and_tokenizer(corpus: Path) -> None:
    (corpus / "web" / "valid-offsets.npy").write_bytes(b"garbage")
    (corpus / "web" / "tokenizer" / "tokenizer.json").unlink()
    info = inspect_source(corpus / "web")
    assert info.state == "corrupt"
    assert _codes(info.problems) == ["TB-I305", "TB-I306"]


def test_unreadable_meta(corpus: Path) -> None:
    (corpus / "web" / "meta.json").write_text("{not json")
    info = inspect_source(corpus / "web")
    assert info.state == "corrupt"
    assert _codes(info.problems) == ["TB-F102"]
    assert info.splits == ()
    assert info.dtype is None


def test_newer_schema_and_multistream_are_unsupported(corpus: Path, tmp_path: Path) -> None:
    meta = _meta(corpus / "web")
    meta["schema_version"] = 99
    (corpus / "web" / "meta.json").write_text(json.dumps(meta))
    info = inspect_source(corpus / "web")
    assert info.state == "unsupported"
    assert _codes(info.problems) == ["TB-F107"]

    multi = tmp_path / "multi"
    multi.mkdir()
    (multi / "dataset.json").write_text("{}")
    info = inspect_source(multi)
    assert info.state == "unsupported"
    assert _codes(info.problems) == ["TB-F109"]


def test_outdated_schema(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend the current schema is 2: the fixture becomes an old but readable source.
    monkeypatch.setattr(schema, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(schema, "SUPPORTED_SCHEMAS", frozenset({1, 2}))
    monkeypatch.setattr(inspect_mod, "SCHEMA_VERSION", 2)
    info = inspect_source(corpus / "web")  # no FutureWarning: it is reported as a state
    assert info.state == "outdated"
    assert "TB-F111" in _codes(info.issues)
    assert info.n_warnings == 1

    monkeypatch.setattr(schema, "SUPPORTED_SCHEMAS", frozenset({2}))
    info = inspect_source(corpus / "web")
    assert info.state == "outdated"
    assert _codes(info.problems) == ["TB-F108"]


def _checkpoint() -> Checkpoint:
    return Checkpoint(
        schema_version=1,
        split="train",
        dtype="uint16",
        config_hash="0" * 64,
        tokenizer_hash="5b6c9ce06a2c8850",
        n_input_consumed=12,
        pending_doc_items=0,
        last_input_id="doc-11",
        n_items=128,
        n_docs=11,
        n_skipped=1,
        n_split_docs=0,
        closed_shards=(
            ShardInfo(name="train-00000.bin", n_items=128, n_bytes=256, sha256="a" * 64),
        ),
        side_files=SideFileLengths(ids=0, ids_idx=0, offsets=0, skipped=0),
        updated_at="2026-09-26T10:00:00Z",
    )


def test_partial_only(corpus: Path) -> None:
    partial = corpus / "news.partial"
    partial.mkdir()
    (partial / "train-offsets.i64").touch()
    (partial / "train-00000.bin").touch()
    (partial / "train-00001.bin.open").touch()

    for path in (corpus / "news", partial):  # both names mean the same source
        info = inspect_source(path)
        assert info.state == "partial"
        assert info.path == corpus / "news"
        assert info.partial is not None
        assert info.partial.split == "train"
        assert info.partial.n_closed_shards == 1
        assert not info.partial.resumable
        assert info.partial.n_items is None

    write_checkpoint(partial, _checkpoint())
    info = inspect_source(corpus / "news")
    assert info.partial is not None
    assert info.partial.resumable
    assert (info.partial.n_items, info.partial.n_docs, info.partial.n_input_consumed) == (
        128,
        11,
        12,
    )
    assert info.partial.updated_at == "2026-09-26T10:00:00Z"

    (partial / "checkpoint.json").write_text("{}")
    info = inspect_source(corpus / "news")
    assert info.partial is not None
    assert not info.partial.resumable
    assert info.partial.problem is not None
    assert info.partial.problem.code == "TB-F103"


def test_partial_next_to_complete_source_is_a_warning(corpus: Path) -> None:
    (corpus / "web.partial").mkdir()
    info = inspect_source(corpus / "web")
    assert info.state == "complete"
    assert info.partial is not None
    assert "TB-R201" in _codes(info.issues)
    assert info.n_warnings == 1


def test_not_found_and_not_a_source(corpus: Path) -> None:
    with pytest.raises(ConfigError) as err:
        inspect_source(corpus / "missing")
    assert err.value.code is codes.SOURCE_NOT_FOUND
    assert err.value.where == "tokbin.ops.inspect.inspect_source"

    with pytest.raises(ConfigError) as err:
        inspect_source(corpus)
    assert err.value.code is codes.NOT_A_SOURCE
    assert "web" in err.value.why


def test_corpus(corpus: Path) -> None:
    copy_fixture(corpus / "code")
    (corpus / "news.partial").mkdir()
    (corpus / "tokenizers").mkdir()  # an unrelated directory is not a source
    (corpus / ".web.old-123").mkdir()  # nor is a leftover of an interrupted swap
    (corpus / "notes.txt").write_text("")
    (corpus / "mix.json").write_text('{"web": 3, "code": 1}')

    info = Dataset(corpus).info()
    assert [s.name for s in info.sources] == ["code", "news", "web"]
    assert [s.state for s in info.sources] == ["complete", "partial", "complete"]
    assert info.mix == (("web", 0.75), ("code", 0.25))
    assert info.tokenizer_hash == "5b6c9ce06a2c8850"
    assert info.issues == ()
    assert not info.ready  # an unfinished source
    assert info.n_items == 720

    (corpus / "news.partial").rmdir()
    assert Dataset(corpus).info().ready


def test_corpus_warnings(corpus: Path) -> None:
    copy_fixture(corpus / "code")
    meta = _meta(corpus / "code")
    meta["tokenizer_hash"] = "a" * 16
    (corpus / "code" / "meta.json").write_text(json.dumps(meta))
    (corpus / "mix.json").write_text('{"web": 1, "wiki": 1}')

    info = inspect_corpus(corpus)
    assert _codes(info.issues) == ["TB-M301", "TB-F110"]
    assert info.tokenizer_hash is None
    assert info.n_warnings == 2
    assert info.ready  # warnings do not make a corpus unusable


def test_broken_mix_is_a_problem(corpus: Path) -> None:
    (corpus / "mix.json").write_text('{"web": -1}')
    info = inspect_corpus(corpus)
    assert _codes(info.problems) == ["TB-F103"]
    assert not info.ready


def test_empty_and_wrong_corpus_paths(tmp_path: Path, corpus: Path) -> None:
    empty = inspect_corpus(tmp_path / "corpus" / "web" / "tokenizer")
    assert empty.sources == ()
    assert not empty.ready

    with pytest.raises(ConfigError) as err:
        inspect_corpus(corpus / "web")
    assert err.value.code is codes.NOT_A_CORPUS
    with pytest.raises(ConfigError) as err:
        inspect_corpus(tmp_path / "nothing")
    assert err.value.code is codes.SOURCE_NOT_FOUND


def test_describe_dispatch(corpus: Path) -> None:
    (corpus / "news.partial").mkdir()
    assert type(describe(corpus)).__name__ == "CorpusInfo"
    assert type(describe(corpus / "web")).__name__ == "SourceInfo"
    assert type(describe(corpus / "news")).__name__ == "SourceInfo"
    assert type(describe(corpus / "news.partial")).__name__ == "SourceInfo"
    with pytest.raises(ConfigError) as err:
        describe(corpus / "missing")
    assert err.value.code is codes.SOURCE_NOT_FOUND


def test_dataset_status_checks_the_name(corpus: Path) -> None:
    assert Dataset(corpus).status("web").state == "complete"
    with pytest.raises(ConfigError) as err:
        Dataset(corpus).status("../web")
    assert err.value.code is codes.SOURCE_NAME_INVALID


def test_to_dict_is_json_serializable(corpus: Path) -> None:
    (corpus / "web.partial").mkdir()
    data = Dataset(corpus).info().to_dict()
    text = json.dumps(data)
    assert json.loads(text)["sources"][0]["partial"]["path"].endswith("web.partial")


def test_inspection_does_not_read_shards(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_memmap, real_open = np.memmap, builtins.open

    def no_shards(path: object) -> None:
        if str(path).endswith(".bin"):
            raise AssertionError(f"inspection must not read shards: {path}")

    def memmap(filename: object, *args: object, **kwargs: object) -> object:
        no_shards(filename)
        return real_memmap(filename, *args, **kwargs)  # type: ignore[call-overload]

    def open_(file: object, *args: object, **kwargs: object) -> object:
        no_shards(file)
        return real_open(file, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(np, "memmap", memmap)
    monkeypatch.setattr(builtins, "open", open_)
    assert inspect_source(corpus / "web").state == "complete"
