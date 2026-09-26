from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tokbin import (
    CompatibilityError,
    ConfigError,
    ContractError,
    DataError,
    DataQualityWarning,
    Dataset,
    ErrorPolicy,
    ResumeError,
    ShardingWarning,
    StreamWriter,
    UnsupportedFeatureError,
    WriterConfig,
    codes,
)
from tokbin.tokenizer.hf import HFTokenizer

from support import (
    BOS,
    EOS,
    encode,
    expected_stream,
    make_docs,
    make_tokenizer,
    read_ids,
    read_meta_json,
    read_skipped,
    read_stream,
)

pytestmark = pytest.mark.requires_tokenizers


def _files(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir())


# --- the happy path ---------------------------------------------------------------


def test_write_produces_the_expected_stream(
    tmp_path: Path,
    tokenizer_path: Path,
    no_side_effects: Callable[[], AbstractContextManager[None]],
) -> None:
    docs = make_docs(50)
    with no_side_effects():
        result = Dataset(tmp_path / "corpus").write("code", iter(docs), tokenizer_path)

    source = tmp_path / "corpus" / "code"
    assert result.path == source
    assert result.status == "complete"
    assert result.stats.n_docs == 50
    assert result.stats.n_input == 50
    assert result.stats.n_skipped == 0

    expected = expected_stream(docs)
    assert np.array_equal(read_stream(source), expected)
    assert result.stats.n_items == expected.size

    # Nothing but the finished source: no partial directory, no service files.
    assert _files(tmp_path / "corpus") == ["code"]
    assert _files(source) == [
        "meta.json",
        "tokenizer",
        "train-00000.bin",
        "train-ids.idx.npy",
        "train-ids.jsonl",
        "train-offsets.npy",
        "train-skipped.jsonl",
    ]


def test_meta_describes_the_data(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = make_docs(10)
    Dataset(tmp_path).write("code", docs, tokenizer_path)
    meta = read_meta_json(tmp_path / "code")
    assert meta["schema_version"] == 1
    assert meta["dtype"] == "uint16"
    assert meta["vocab_size"] == 303
    assert meta["tokenizer_id"] == "tiny"
    assert meta["eos_id"] == EOS
    assert meta["bos_id"] == BOS
    train = meta["splits"]["train"]
    assert train["n_docs"] == 10
    assert train["n_items"] == expected_stream(docs).size
    shard = train["shards"][0]
    data = (tmp_path / "code" / shard["name"]).read_bytes()
    assert shard["n_bytes"] == len(data)
    assert shard["sha256"] == hashlib.sha256(data).hexdigest()


def test_offsets_and_ids_mark_documents(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = make_docs(30, seed=3)
    Dataset(tmp_path).write("code", docs, tokenizer_path)
    source = tmp_path / "code"

    offsets = np.load(source / "train-offsets.npy", allow_pickle=False)
    assert offsets.dtype == np.int64
    assert offsets.shape == (31,)
    stream = read_stream(source)
    for i, (_, text) in enumerate(docs):
        assert stream[offsets[i] : offsets[i + 1]].tolist() == [*encode(text), EOS]
    assert offsets[-1] == stream.size

    assert read_ids(source) == [doc_id for doc_id, _ in docs]
    idx = np.load(source / "train-ids.idx.npy", allow_pickle=False)
    raw = (source / "train-ids.jsonl").read_bytes()
    assert idx.dtype == np.int64
    assert idx.shape == (31,)
    assert idx[-1] == len(raw)
    for i, (doc_id, _) in enumerate(docs):
        assert json.loads(raw[idx[i] : idx[i + 1]]) == {"id": doc_id}


def test_plain_text_documents_have_null_ids(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = make_docs(5)
    Dataset(tmp_path).write("code", (text for _, text in docs), tokenizer_path)
    assert read_ids(tmp_path / "code") == [None] * 5
    assert np.array_equal(read_stream(tmp_path / "code"), expected_stream(docs))


def test_bytes_documents(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = make_docs(5)
    Dataset(tmp_path).write("code", ((i, t.encode()) for i, t in docs), tokenizer_path)
    assert np.array_equal(read_stream(tmp_path / "code"), expected_stream(docs))


def test_tokenizer_object_and_stored_copy(tmp_path: Path, tokenizer: Any) -> None:
    Dataset(tmp_path).write("code", make_docs(3), tokenizer)
    stored = tmp_path / "code" / "tokenizer" / "tokenizer.json"
    copy = HFTokenizer.from_file(stored)
    assert copy.fingerprint() == HFTokenizer(tokenizer).fingerprint()
    assert read_meta_json(tmp_path / "code")["tokenizer_hash"] == copy.fingerprint()
    assert read_meta_json(tmp_path / "code")["tokenizer_id"] is None


def test_fingerprint_is_pinned(tokenizer: Any, tokenizer_path: Path) -> None:
    # If this changes, hashes stored in existing datasets stop matching.
    assert HFTokenizer(tokenizer).fingerprint() == "5b6c9ce06a2c8850"
    assert HFTokenizer.from_file(tokenizer_path).fingerprint() == "5b6c9ce06a2c8850"


# --- shards -------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::tokbin.errors.ShardingWarning")  # tiny shards
def test_small_shards_split_the_stream(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = make_docs(40, seed=1)
    cfg = WriterConfig(shard_bytes=64)  # 32 uint16 items per shard
    result = Dataset(tmp_path).write("code", docs, tokenizer_path, config=cfg)
    source = tmp_path / "code"
    meta = read_meta_json(source)
    shards = meta["splits"]["train"]["shards"]

    assert len(shards) == result.stats.n_shards > 3
    assert all(s["n_items"] == 32 for s in shards[:-1])
    assert 0 < shards[-1]["n_items"] <= 32
    assert [s["name"] for s in shards] == [f"train-{i:05d}.bin" for i in range(len(shards))]
    assert np.array_equal(read_stream(source), expected_stream(docs))

    assert meta["splits"]["train"]["has_split_docs"] is True
    assert result.stats.n_split_docs > 0
    info = {i.code: i for i in result.issues}
    assert info["TB-S203"].level == "info"
    assert info["TB-S203"].count == result.stats.n_split_docs


@pytest.mark.filterwarnings("ignore::tokbin.errors.ShardingWarning")  # tiny shards
def test_stream_that_fills_shards_exactly(tmp_path: Path, tokenizer_path: Path) -> None:
    # 4 documents of 7 words + EOS = 32 items = exactly one shard of 64 bytes.
    docs = [(f"d{i}", " ".join(["t1"] * 7)) for i in range(4)]
    result = Dataset(tmp_path).write(
        "code", docs, tokenizer_path, config=WriterConfig(shard_bytes=64)
    )
    assert result.stats.n_shards == 1
    assert result.stats.n_split_docs == 0
    assert _files(tmp_path / "code").count("train-00001.bin") == 0


def test_document_larger_than_a_shard(tmp_path: Path, tokenizer_path: Path) -> None:
    # 64 items per shard: "big" (101 items) exceeds it, "small" (2) is under 5%.
    docs = [("big", " ".join(["t2"] * 100)), ("small", "t3")]
    with pytest.warns(ShardingWarning, match="TB-S201"):
        result = Dataset(tmp_path).write(
            "code", docs, tokenizer_path, config=WriterConfig(shard_bytes=128)
        )
    assert result.status == "complete_with_issues"
    assert np.array_equal(read_stream(tmp_path / "code"), expected_stream(docs))


def test_large_document_warning(tmp_path: Path, tokenizer_path: Path) -> None:
    # 60 items in shards of 1000 items: above 5%.
    docs = [("d", " ".join(["t2"] * 59))]
    with pytest.warns(ShardingWarning, match="TB-S202"):
        Dataset(tmp_path).write("code", docs, tokenizer_path, config=WriterConfig(shard_bytes=2000))


def test_info_issues_keep_the_status_complete(tmp_path: Path, tokenizer_path: Path) -> None:
    # 1000 items per shard, documents of at most 13 items (< 5%): only split info.
    docs = make_docs(300, seed=4)
    result = Dataset(tmp_path).write(
        "code", docs, tokenizer_path, config=WriterConfig(shard_bytes=2000)
    )
    assert {i.level for i in result.issues} == {"info"}
    assert result.status == "complete"


def test_explicit_wider_dtype(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = make_docs(5)
    Dataset(tmp_path).write("code", docs, tokenizer_path, config=WriterConfig(dtype="uint32"))
    assert read_meta_json(tmp_path / "code")["dtype"] == "uint32"
    assert np.array_equal(read_stream(tmp_path / "code"), expected_stream(docs))


# --- special tokens -----------------------------------------------------------------


def test_prepend_bos_and_no_eos(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = make_docs(5)
    cfg = WriterConfig(prepend_bos=True, append_eos=False)
    Dataset(tmp_path).write("code", docs, tokenizer_path, config=cfg)
    assert np.array_equal(
        read_stream(tmp_path / "code"), expected_stream(docs, eos=False, bos=True)
    )


def test_missing_eos_is_an_error(tmp_path: Path) -> None:
    tok = make_tokenizer(specials=("<pad>",))
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("code", make_docs(3), tok)
    assert info.value.code is codes.NO_EOS_TOKEN
    assert not (tmp_path / "code").exists()
    assert not (tmp_path / "code.partial").exists()


def test_explicit_eos_token(tmp_path: Path) -> None:
    tok = make_tokenizer(specials=("<pad>",))
    cfg = WriterConfig(eos_token="<pad>")
    Dataset(tmp_path).write("code", [("a", "t0")], tok, config=cfg)
    assert read_stream(tmp_path / "code").tolist() == [2 + 0, 1]


def test_unknown_explicit_eos_token(tmp_path: Path, tokenizer: Any) -> None:
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("code", [], tokenizer, config=WriterConfig(eos_token="<nope>"))
    assert info.value.code is codes.SPECIAL_TOKEN_NOT_FOUND


# DeepSeek-style special tokens: fullwidth bars and a lower one-eighth block.
DS_EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"
DS_BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"


def _deepseek_like(tmp_path: Path, config: object) -> Path:
    tok = make_tokenizer(specials=(DS_BOS, DS_EOS))
    path = tmp_path / "ds" / "tokenizer.json"
    path.parent.mkdir()
    tok.save(str(path))
    if config is not None:
        text = config if isinstance(config, str) else json.dumps(config)
        (path.parent / "tokenizer_config.json").write_text(text, encoding="utf-8")
    return path


def test_special_tokens_from_tokenizer_config(tmp_path: Path) -> None:
    added = {"__type": "AddedToken", "content": DS_EOS, "lstrip": False}
    path = _deepseek_like(tmp_path, {"eos_token": added, "bos_token": DS_BOS})
    result = Dataset(tmp_path).write("code", [("a", "t0")], path)
    assert read_stream(tmp_path / "code").tolist() == [3, 2]  # t0, then EOS (id 2)
    meta = read_meta_json(tmp_path / "code")
    assert (meta["eos_id"], meta["bos_id"]) == (2, 1)
    note = next(i for i in result.issues if i.code == "TB-C215")
    assert note.level == "info"
    assert "tokenizer_config.json" in note.message


def test_without_config_deepseek_names_are_not_guessed(tmp_path: Path) -> None:
    path = _deepseek_like(tmp_path, None)
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("code", [("a", "t0")], path)
    assert info.value.code is codes.NO_EOS_TOKEN


def test_explicit_token_wins_over_config(tmp_path: Path) -> None:
    path = _deepseek_like(tmp_path, {"eos_token": DS_EOS})
    cfg = WriterConfig(eos_token=DS_BOS)
    result = Dataset(tmp_path).write("code", [("a", "t0")], path, config=cfg)
    assert read_stream(tmp_path / "code").tolist() == [3, 1]
    assert "TB-C215" not in {i.code for i in result.issues}


def test_config_naming_an_unknown_token(tmp_path: Path) -> None:
    path = _deepseek_like(tmp_path, {"eos_token": "<nope>"})
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("code", [("a", "t0")], path)
    assert info.value.code is codes.SPECIAL_TOKEN_NOT_FOUND
    assert "tokenizer_config.json" in str(info.value)


@pytest.mark.parametrize("content", ["{broken", '{"eos_token": 5}'])
def test_invalid_tokenizer_config(tmp_path: Path, content: str) -> None:
    path = _deepseek_like(tmp_path, content)
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("code", [("a", "t0")], path)
    assert info.value.code is codes.TOKENIZER_FILE_INVALID


def test_stored_copy_is_enough_for_another_split(tmp_path: Path) -> None:
    path = _deepseek_like(tmp_path, {"eos_token": DS_EOS})
    ds = Dataset(tmp_path / "corpus")
    ds.write("code", [("a", "t0")], path)
    stored = tmp_path / "corpus" / "code" / "tokenizer" / "tokenizer.json"
    assert json.loads(stored.with_name("tokenizer_config.json").read_text()) == {
        "eos_token": DS_EOS,
        "bos_token": None,
    }
    ds.write("code", [("b", "t1")], stored, config=WriterConfig(split="valid"))
    assert read_stream(tmp_path / "corpus" / "code", "valid").tolist() == [4, 2]


def test_truncation_is_disabled_without_touching_the_user_object(
    tmp_path: Path, tokenizer: Any
) -> None:
    tokenizer.enable_truncation(max_length=2)
    docs = [("long", "t1 t2 t3 t4 t5")]
    result = Dataset(tmp_path).write("code", docs, tokenizer)
    assert read_stream(tmp_path / "code").tolist() == [*encode(docs[0][1]), EOS]
    assert {i.code for i in result.issues} == {"TB-C213", "TB-C215"}
    assert tokenizer.truncation is not None  # the user's object is unchanged
    stored = json.loads((tmp_path / "code" / "tokenizer" / "tokenizer.json").read_text())
    assert stored["truncation"] is None


# --- skipped documents --------------------------------------------------------------


def test_undecodable_and_empty_documents_are_skipped(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = [
        ("ok1", b"t1 t2"),
        ("bad", b"t1 \xff t2"),
        ("empty", b""),
        ("blank", b"   "),
        ("ok2", b"t3"),
    ]
    with pytest.warns(DataQualityWarning):
        result = Dataset(tmp_path).write("code", docs, tokenizer_path)
    source = tmp_path / "code"
    assert result.status == "complete_with_issues"
    assert result.stats.n_docs == 2
    assert result.stats.n_skipped == 3
    assert result.stats.n_input == 5
    assert read_ids(source) == ["ok1", "ok2"]
    assert read_stream(source).tolist() == [*encode("t1 t2"), EOS, *encode("t3"), EOS]

    skipped = read_skipped(source)
    assert [(s["input_index"], s["id"], s["code"]) for s in skipped] == [
        (1, "bad", "TB-D201"),
        (2, "empty", "TB-D203"),
        (3, "blank", "TB-D203"),
    ]
    assert "0xff at position 3" in skipped[0]["reason"]
    assert "t1" not in json.dumps(skipped)  # content is never stored
    assert read_meta_json(source)["splits"]["train"]["n_skipped"] == 3
    counts = {i.code: i.count for i in result.issues if i.level == "warning"}
    assert counts == {"TB-D201": 1, "TB-D203": 2}


def test_tokenizer_failure_skips_the_document(
    tmp_path: Path, tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Like a real tokenizer: the batch call fails as a whole, the single call only on
    # the offending text. The writer must isolate that one document.
    real_one, real_batch = HFTokenizer.encode, HFTokenizer.encode_batch

    def one(self: HFTokenizer, text: str) -> Any:
        if "t13" in text.split():
            raise RuntimeError("unlucky")
        return real_one(self, text)

    def batch(self: HFTokenizer, texts: Any) -> Any:
        if any("t13" in t.split() for t in texts):
            raise RuntimeError("unlucky batch")
        return real_batch(self, texts)

    monkeypatch.setattr(HFTokenizer, "encode", one)
    monkeypatch.setattr(HFTokenizer, "encode_batch", batch)
    docs = [("a", "t1"), ("b", "t13"), ("c", "t2")]
    with pytest.warns(DataQualityWarning, match="TB-D204"):
        result = Dataset(tmp_path).write("code", docs, tokenizer_path)
    assert result.stats.n_skipped == 1
    assert read_skipped(tmp_path / "code")[0]["reason"] == "RuntimeError: unlucky"
    assert read_ids(tmp_path / "code") == ["a", "c"]
    assert read_stream(tmp_path / "code").tolist() == [*encode("t1"), EOS, *encode("t2"), EOS]


def test_policy_raise_stops_the_write(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = [("a", "t1"), ("b", b"\xff")]
    with pytest.raises(DataError) as info:
        Dataset(tmp_path).write(
            "code", docs, tokenizer_path, policy=ErrorPolicy(on_data_error="raise")
        )
    assert info.value.code is codes.DOCUMENT_NOT_UTF8
    assert not (tmp_path / "code").exists()
    assert (tmp_path / "code.partial").is_dir()
    assert not list((tmp_path / "code.partial").glob("*.open"))


def test_too_many_skips_stop_the_write(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = [("ok", "t1")] * 5 + [("bad", b"\xff")] * 5
    policy = ErrorPolicy(max_skip_ratio=0.2, min_docs_for_ratio=8)
    with pytest.warns(DataQualityWarning), pytest.raises(ContractError) as info:
        Dataset(tmp_path).write("code", docs, tokenizer_path, policy=policy)
    assert info.value.code is codes.TOO_MANY_SKIPPED
    assert not (tmp_path / "code").exists()


def test_duplicate_ids_warn_once_at_the_end(tmp_path: Path, tokenizer_path: Path) -> None:
    docs = [("same", "t1"), ("other", "t2"), ("same", "t3"), ("other", "t4"), ("same", "t5")]
    with pytest.warns(DataQualityWarning, match="TB-D205") as caught:
        result = Dataset(tmp_path).write("code", docs, tokenizer_path)
    assert len(caught) == 1
    assert result.stats.n_docs == 5  # duplicates are written, only reported
    dup = next(i for i in result.issues if i.code == "TB-D205")
    assert dup.count == 3
    assert "3 documents repeat an earlier id" in dup.message
    assert "'same'" in dup.message or "'other'" in dup.message


def test_plain_text_documents_are_never_duplicates(tmp_path: Path, tokenizer_path: Path) -> None:
    result = Dataset(tmp_path).write("code", ["t1", "t1", "t1"], tokenizer_path)
    assert result.status == "complete"


# --- contract violations and interruptions ------------------------------------------


@pytest.mark.parametrize(
    ("docs", "code"),
    [
        (["t1", ("id", "t2")], "TB-K202"),
        ([("id", "t1"), "t2"], "TB-K202"),
        ([42], "TB-K203"),
        ([("id", "t1", "extra")], "TB-K203"),
        ([(1, "t1")], "TB-K203"),
    ],
)
def test_contract_violations(
    tmp_path: Path, tokenizer_path: Path, docs: list[object], code: str
) -> None:
    with pytest.raises(ContractError) as info:
        Dataset(tmp_path).write("code", docs, tokenizer_path)
    assert info.value.code.id == code
    assert not (tmp_path / "code").exists()


def test_segments_are_recognized_but_unsupported(tmp_path: Path, tokenizer_path: Path) -> None:
    with pytest.raises(UnsupportedFeatureError):
        Dataset(tmp_path).write("code", [("id", ["t1"])], tokenizer_path)
    with pytest.raises(UnsupportedFeatureError):
        Dataset(tmp_path).write("code", [], tokenizer_path, mode="multi")


def test_generator_errors_propagate_unchanged(tmp_path: Path, tokenizer_path: Path) -> None:
    class BoomError(Exception):
        pass

    def docs() -> Iterator[tuple[str, str]]:
        yield "a", "t1"
        raise BoomError("from the generator")

    with pytest.raises(BoomError, match="from the generator"):
        Dataset(tmp_path).write("code", docs(), tokenizer_path)
    assert not (tmp_path / "code").exists()
    assert not list((tmp_path / "code.partial").glob("*.open"))


@pytest.mark.filterwarnings("ignore::tokbin.errors.ShardingWarning")  # tiny shards
def test_keyboard_interrupt_leaves_no_open_shard(tmp_path: Path, tokenizer_path: Path) -> None:
    def docs() -> Iterator[tuple[str, str]]:
        yield from make_docs(20)
        raise KeyboardInterrupt

    # Small batches: buffered documents are only written when their batch is processed.
    with pytest.raises(KeyboardInterrupt):
        Dataset(tmp_path).write(
            "code", docs(), tokenizer_path, config=WriterConfig(shard_bytes=64, batch_docs=4)
        )
    partial = tmp_path / "code.partial"
    assert not (tmp_path / "code").exists()
    assert list(partial.glob("train-*.bin"))  # closed shards are kept for resume
    assert not list(partial.glob("*.open"))


def test_unfinished_write_blocks_a_new_one(tmp_path: Path, tokenizer_path: Path) -> None:
    (tmp_path / "code.partial").mkdir()
    with pytest.raises(ResumeError) as info:
        Dataset(tmp_path).write("code", make_docs(2), tokenizer_path)
    assert info.value.code is codes.PARTIAL_EXISTS


def test_unsupported_tokenizer_object(tmp_path: Path) -> None:
    with pytest.raises(ContractError) as info:
        Dataset(tmp_path).write("code", [], object())
    assert info.value.code is codes.TOKENIZER_UNSUPPORTED


def test_missing_tokenizer_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("code", [], tmp_path / "nope.json")
    assert info.value.code is codes.TOKENIZER_FILE_INVALID


def test_invalid_source_name(tmp_path: Path, tokenizer_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("../escape", [], tokenizer_path)
    assert info.value.code is codes.SOURCE_NAME_INVALID


# --- splits and existing sources ----------------------------------------------------


@pytest.mark.filterwarnings("ignore::tokbin.errors.ShardingWarning")  # tiny shards
def test_add_a_split_to_an_existing_source(tmp_path: Path, tokenizer_path: Path) -> None:
    ds = Dataset(tmp_path)
    train, valid = make_docs(20, seed=1), make_docs(5, seed=2)
    ds.write("code", train, tokenizer_path, config=WriterConfig(shard_bytes=64))
    ds.write("code", valid, tokenizer_path, config=WriterConfig(split="valid"))

    source = tmp_path / "code"
    meta = read_meta_json(source)
    assert list(meta["splits"]) == ["train", "valid"]
    assert np.array_equal(read_stream(source, "train"), expected_stream(train))
    assert np.array_equal(read_stream(source, "valid"), expected_stream(valid))
    # No partial directory and no swapped-out old copy are left behind.
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".code.old-*"))


@pytest.mark.filterwarnings("ignore::tokbin.errors.ShardingWarning")  # tiny shards
def test_existing_split_needs_overwrite(tmp_path: Path, tokenizer_path: Path) -> None:
    ds = Dataset(tmp_path)
    ds.write("code", make_docs(20, seed=1), tokenizer_path, config=WriterConfig(shard_bytes=64))
    ds.write("code", make_docs(3), tokenizer_path, config=WriterConfig(split="valid"))
    with pytest.raises(ConfigError) as info:
        ds.write("code", make_docs(3), tokenizer_path)
    assert info.value.code is codes.SPLIT_EXISTS

    new_train = make_docs(2, seed=9)
    ds.write("code", new_train, tokenizer_path, overwrite=True)
    source = tmp_path / "code"
    assert np.array_equal(read_stream(source, "train"), expected_stream(new_train))
    assert np.array_equal(read_stream(source, "valid"), expected_stream(make_docs(3)))
    # Old train shards beyond the new ones are gone.
    assert not (source / "train-00001.bin").exists()


def test_other_tokenizer_is_incompatible(tmp_path: Path, tokenizer_path: Path) -> None:
    ds = Dataset(tmp_path)
    ds.write("code", make_docs(3), tokenizer_path)
    other = make_tokenizer(specials=("<|endoftext|>", "<s>", "<extra>"))
    with pytest.raises(CompatibilityError) as info:
        ds.write("code", make_docs(3), other, config=WriterConfig(split="valid"))
    assert info.value.code is codes.TOKENIZER_MISMATCH


def test_other_dtype_is_incompatible(tmp_path: Path, tokenizer_path: Path) -> None:
    ds = Dataset(tmp_path)
    ds.write("code", make_docs(3), tokenizer_path)
    with pytest.raises(CompatibilityError) as info:
        ds.write(
            "code", make_docs(3), tokenizer_path, config=WriterConfig(split="valid", dtype="uint32")
        )
    assert info.value.code is codes.DTYPE_MISMATCH


def test_target_that_is_not_a_source(tmp_path: Path, tokenizer_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "notes.txt").write_text("mine")
    with pytest.raises(ConfigError) as info:
        Dataset(tmp_path).write("code", make_docs(2), tokenizer_path, overwrite=True)
    assert info.value.code is codes.TARGET_NOT_A_SOURCE
    assert (tmp_path / "code" / "notes.txt").read_text() == "mine"


def test_empty_target_directory_is_replaced(tmp_path: Path, tokenizer_path: Path) -> None:
    (tmp_path / "code").mkdir()
    Dataset(tmp_path).write("code", make_docs(2), tokenizer_path)
    assert (tmp_path / "code" / "meta.json").is_file()


# --- the low-level writer -----------------------------------------------------------


def test_stream_writer_add(tmp_path: Path, tokenizer_path: Path) -> None:
    with StreamWriter(tmp_path / "code", tokenizer_path) as w:
        w.add("a", "t1 t2")
        w.add("b", "t3")
        assert w.stats.n_docs == 2
        with pytest.raises(ConfigError) as info:
            _ = w.result
        assert info.value.code is codes.WRITER_NOT_FINISHED
    assert w.result.stats.n_items == 5
    assert read_ids(tmp_path / "code") == ["a", "b"]


def test_empty_write_produces_an_empty_split(tmp_path: Path, tokenizer_path: Path) -> None:
    result = Dataset(tmp_path).write("code", [], tokenizer_path)
    assert result.stats.n_shards == 0
    meta = read_meta_json(tmp_path / "code")
    assert meta["splits"]["train"]["shards"] == []
