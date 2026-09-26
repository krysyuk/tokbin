"""Batched encoding must be invisible: same bytes for any batch size."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tokbin import (
    DataQualityWarning,
    Dataset,
    ShardingWarning,
    StreamWriter,
    WriterConfig,
    codes,
)
from tokbin.tokenizer.hf import HFTokenizer
from tokbin.write import stream_writer

from support import EOS, encode, make_docs, read_ids, read_skipped, read_stream

pytestmark = pytest.mark.requires_tokenizers


def _mixed_docs() -> list[tuple[str, bytes]]:
    docs: list[tuple[str, bytes]] = [(i, t.encode()) for i, t in make_docs(60, seed=5, max_len=30)]
    docs[3] = ("empty", b"")
    docs[10] = ("blank", b"   ")
    docs[11] = ("broken", b"t1 \xff")
    docs[40] = ("empty2", b"")
    return docs


def _snapshot(source: Path) -> dict[str, bytes]:
    files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    meta = json.loads(files.pop("meta.json"))
    for split in meta["splits"].values():
        split.pop("created_at")
    files["meta.json"] = json.dumps(meta, sort_keys=True).encode()
    return files


@pytest.mark.parametrize("batch_docs", [1, 7, 1024])
def test_output_does_not_depend_on_batch_size(
    tmp_path: Path, tokenizer_path: Path, batch_docs: int
) -> None:
    def write(name: str, batch: int) -> Path:
        cfg = WriterConfig(shard_bytes=128, prepend_bos=True, batch_docs=batch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            warnings.simplefilter("ignore", ShardingWarning)
            Dataset(tmp_path).write(name, _mixed_docs(), tokenizer_path, config=cfg)
        return tmp_path / name

    reference = _snapshot(write("one_by_one", 1))
    assert _snapshot(write("batched", batch_docs)) == reference
    skipped = read_skipped(tmp_path / "batched")
    assert [s["input_index"] for s in skipped] == [3, 10, 11, 40]


def test_out_of_range_id_skips_only_its_document(
    tmp_path: Path, tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_one = HFTokenizer.encode

    def one(self: HFTokenizer, text: str) -> Any:
        return [70000] if text == "t7" else real_one(self, text)

    def batch(self: HFTokenizer, texts: Any) -> Any:
        lists = [list(one(self, t)) for t in texts]
        lengths = np.array([len(ids) for ids in lists], dtype=np.int64)
        return np.array([i for ids in lists for i in ids], dtype=np.int64), lengths

    monkeypatch.setattr(HFTokenizer, "encode_batch", batch)
    monkeypatch.setattr(HFTokenizer, "encode", one)
    docs = [("a", "t1"), ("b", "t7"), ("c", "t2")]
    with pytest.warns(DataQualityWarning, match="TB-D202"):
        result = Dataset(tmp_path).write("code", docs, tokenizer_path)
    assert result.stats.n_skipped == 1
    assert read_ids(tmp_path / "code") == ["a", "c"]
    assert read_stream(tmp_path / "code").tolist() == [*encode("t1"), EOS, *encode("t2"), EOS]


def test_stats_are_exact_after_every_add(tmp_path: Path, tokenizer_path: Path) -> None:
    with StreamWriter(tmp_path / "code", tokenizer_path) as w:
        for i in range(5):
            w.add(f"d{i}", "t1 t2")
            assert w.stats.n_docs == i + 1
            assert w.stats.n_items == 3 * (i + 1)


def test_batches_are_bounded_by_text_size(
    tmp_path: Path, tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stream_writer, "_BATCH_CHARS", 20)
    calls: list[int] = []
    real = HFTokenizer.encode_batch

    def batch(self: HFTokenizer, texts: Any) -> Any:
        calls.append(len(texts))
        return real(self, texts)

    monkeypatch.setattr(HFTokenizer, "encode_batch", batch)
    docs = [(f"d{i}", "t1 t2 t3 t4") for i in range(10)]  # 11 characters each
    Dataset(tmp_path).write("code", docs, tokenizer_path)
    assert max(calls) == 2
    assert sum(calls) == 10


def test_batch_docs_validation() -> None:
    from tokbin import ConfigError

    for bad in (0, -1, 1.5, True):
        with pytest.raises(ConfigError) as info:
            WriterConfig(batch_docs=bad)  # type: ignore[arg-type]
        assert info.value.code is codes.CONFIG_VALUE_INVALID


@pytest.mark.parametrize("bad", [(0,), (2,), (0, 2), (0, 1, 2)])
def test_offenders_at_batch_edges(
    tmp_path: Path, tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch, bad: tuple[int, ...]
) -> None:
    real_one = HFTokenizer.encode
    texts = ["t1 t2", "t3", "t4 t5 t6"]
    poisoned = {texts[k] for k in bad}

    def one(self: HFTokenizer, text: str) -> Any:
        ids = list(real_one(self, text))
        return [*ids[:-1], 70000] if text in poisoned else ids

    def batch(self: HFTokenizer, batch_texts: Any) -> Any:
        lists = [one(self, t) for t in batch_texts]
        lengths = np.array([len(ids) for ids in lists], dtype=np.int64)
        return np.array([i for ids in lists for i in ids], dtype=np.int64), lengths

    monkeypatch.setattr(HFTokenizer, "encode_batch", batch)
    monkeypatch.setattr(HFTokenizer, "encode", one)
    docs = [(f"d{k}", t) for k, t in enumerate(texts)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        Dataset(tmp_path).write("code", docs, tokenizer_path)
    assert read_ids(tmp_path / "code") == [f"d{k}" for k in range(3) if k not in bad]
    assert [s["input_index"] for s in read_skipped(tmp_path / "code")] == list(bad)


def test_hf_encode_batch_matches_encode(tokenizer_path: Path) -> None:
    tok = HFTokenizer.from_file(tokenizer_path)
    texts = ["t1 t2 t3", "t299", "", "   ", "t0 t0"]
    flat, lengths = tok.encode_batch(texts)
    assert flat.dtype == np.int64
    assert lengths.tolist() == [len(tok.encode(t)) for t in texts]
    assert flat.tolist() == [i for t in texts for i in tok.encode(t)]
