from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import pytest

from tokbin import (
    ConfigError,
    Dataset,
    IntegrityError,
    OutOfRangeError,
    ShardingWarning,
    Source,
    UnsupportedFeatureError,
    WriterConfig,
    codes,
    read_source,
)

from support import expected_stream, make_docs, make_tokenizer

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "schema_v1" / "tiny"

# The frozen fixture was written from these documents (see tests/fixtures/make_fixtures.py).
TRAIN_DOCS = make_docs(40, seed=1)
VALID_DOCS = make_docs(5, seed=2)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A private copy of the frozen fixture: tests may damage it."""
    dst = tmp_path / "tiny"
    shutil.copytree(FIXTURE, dst)
    return dst


# --- the frozen fixture: works in core mode too ---------------------------------------


def test_fixture_reads_back_the_documents() -> None:
    src = read_source(FIXTURE)
    expected = expected_stream(TRAIN_DOCS)
    assert len(src) == expected.size
    assert src.n_docs == 40
    assert src.n_shards == 3
    assert src.dtype == np.dtype("<u2")
    assert np.array_equal(src[:], expected)

    valid = read_source(FIXTURE, "valid")
    assert np.array_equal(valid[:], expected_stream(VALID_DOCS))


def test_every_window_matches_the_stream() -> None:
    src = read_source(FIXTURE)
    expected = expected_stream(TRAIN_DOCS)
    n = len(src)
    # Every start, lengths up to 80 items: every seam of the 128-item shards is crossed.
    for start in range(n):
        for length in range(min(80, n - start) + 1):
            got = src.window(start, length)
            assert np.array_equal(got, expected[start : start + length]), (start, length)


def test_window_inside_a_shard_is_a_read_only_view() -> None:
    src = read_source(FIXTURE)
    inside = src.window(2, 10)  # shard 0 holds items 0..127
    assert not inside.flags.writeable
    across = src.window(120, 10)
    assert across.dtype == src.dtype
    assert np.array_equal(across, np.concatenate([src.window(120, 8), src.window(128, 2)]))


def test_indexing_and_slicing() -> None:
    src = read_source(FIXTURE)
    expected = expected_stream(TRAIN_DOCS)
    n = len(src)
    assert src[0] == expected[0]
    assert src[-1] == expected[-1]
    assert isinstance(src[5], int)
    for key in [
        slice(None),
        slice(10, 70),
        slice(-40, None),
        slice(5, 100, 3),
        slice(None, None, -1),
        slice(90, 10, -7),
        slice(50, 10),
        slice(n + 5, n + 10),
    ]:
        assert np.array_equal(src[key], expected[key]), key
    with pytest.raises(IndexError):
        src[n]
    with pytest.raises(OutOfRangeError):
        src[-n - 1]


def test_documents_and_ids() -> None:
    src = read_source(FIXTURE)
    for i, (doc_id, text) in enumerate(TRAIN_DOCS):
        expected = expected_stream([(doc_id, text)])
        assert np.array_equal(src.doc(i), expected)
        assert src.id_of_doc(i) == doc_id
    assert src.id_of_doc(-1) == TRAIN_DOCS[-1][0]
    with pytest.raises(OutOfRangeError) as info:
        src.doc(40)
    assert info.value.code is codes.OUT_OF_RANGE
    with pytest.raises(IndexError):
        src.id_of_doc(-41)


def test_window_bounds() -> None:
    src = read_source(FIXTURE)
    n = len(src)
    assert src.window(n, 0).size == 0
    for start, length in [(-1, 1), (0, n + 1), (n, 1), (5, -1)]:
        with pytest.raises(OutOfRangeError):
            src.window(start, length)
    with pytest.raises(ConfigError) as info:
        src.window(1.5, 2)  # type: ignore[arg-type]
    assert info.value.code is codes.CONFIG_VALUE_INVALID


def test_sample_windows_is_reproducible_and_correct() -> None:
    src = read_source(FIXTURE)
    expected = expected_stream(TRAIN_DOCS)
    batch = src.sample_windows(16, 24, np.random.default_rng(7))
    assert batch.shape == (16, 24)
    assert batch.dtype == src.dtype
    starts = np.random.default_rng(7).integers(0, len(src) - 24 + 1, size=16)
    for row, start in zip(batch, starts, strict=True):
        assert np.array_equal(row, expected[start : start + 24])
    again = src.sample_windows(16, 24, np.random.default_rng(7))
    assert np.array_equal(batch, again)


def test_sample_windows_requires_an_explicit_generator() -> None:
    src = read_source(FIXTURE)
    for bad in (None, 7, np.random.RandomState(0)):
        with pytest.raises(ConfigError) as info:
            src.sample_windows(2, 4, bad)  # type: ignore[arg-type]
        assert info.value.code is codes.RNG_REQUIRED
    with pytest.raises(ConfigError) as info:
        src.sample_windows(2, len(src) + 1, np.random.default_rng(0))
    assert info.value.code is codes.WINDOW_TOO_LARGE


def test_pickle_reopens_the_files() -> None:
    src = read_source(FIXTURE)
    src.window(0, 40)  # map some shards first
    data = pickle.dumps(src)
    assert len(data) < 1024  # only the path and the split travel
    clone = pickle.loads(data)  # noqa: S301 (our own object)
    assert isinstance(clone, Source)
    assert np.array_equal(clone[:], src[:])
    assert clone.id_of_doc(3) == src.id_of_doc(3)


def test_reading_has_no_side_effects(
    no_side_effects: Callable[[], AbstractContextManager[None]],
) -> None:
    with no_side_effects():
        src = read_source(FIXTURE)
        src.sample_windows(4, 8, np.random.default_rng(0))
        src.doc(0)
        src.id_of_doc(0)


def test_reading_never_imports_tokenizers() -> None:
    script = textwrap.dedent(
        f"""
        import sys
        import numpy as np
        from tokbin import read_source
        src = read_source({str(FIXTURE)!r})
        src.sample_windows(2, 8, np.random.default_rng(0)); src.doc(0); src.id_of_doc(0)
        print("tokenizers" in sys.modules)
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- opening checks -------------------------------------------------------------------


def test_unknown_split(source: Path) -> None:
    with pytest.raises(ConfigError) as info:
        read_source(source, "test")
    assert info.value.code is codes.SPLIT_NOT_FOUND
    assert "train, valid" in str(info.value)


def test_missing_shard(source: Path) -> None:
    (source / "train-00001.bin").unlink()
    with pytest.raises(IntegrityError) as info:
        read_source(source)
    assert info.value.code is codes.SHARD_MISSING
    assert "train-00001.bin" in str(info.value)
    assert "tokbin verify" in str(info.value)
    read_source(source, "valid")  # other splits stay readable


def test_truncated_shard(source: Path) -> None:
    path = source / "train-00002.bin"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(IntegrityError) as info:
        read_source(source)
    assert info.value.code is codes.SHARD_WRONG_SIZE


def test_missing_offsets(source: Path) -> None:
    (source / "train-offsets.npy").unlink()
    with pytest.raises(IntegrityError) as info:
        read_source(source)
    assert info.value.code is codes.INDEX_MISSING


@pytest.mark.parametrize(
    "array",
    [
        np.arange(5, dtype=np.int64),  # wrong length
        np.zeros(41, dtype=np.int32),  # wrong dtype
        np.zeros(41, dtype=np.int64),  # wrong end point
    ],
)
def test_corrupt_offsets(source: Path, array: np.ndarray) -> None:
    np.save(source / "train-offsets.npy", array)
    with pytest.raises(IntegrityError) as info:
        read_source(source)
    assert info.value.code is codes.INDEX_CORRUPT


def test_pickled_index_is_refused_without_loading(source: Path) -> None:
    np.save(source / "train-offsets.npy", np.array([object()] * 41), allow_pickle=True)
    with pytest.raises(IntegrityError) as info:
        read_source(source)
    assert info.value.code is codes.INDEX_CORRUPT


def test_ids_file_that_does_not_match_its_index(source: Path) -> None:
    with (source / "train-ids.jsonl").open("a") as f:
        f.write('{"id": "extra"}\n')
    with pytest.raises(IntegrityError) as info:
        read_source(source)
    assert info.value.code is codes.INDEX_CORRUPT


def test_multimodal_source_is_recognized(source: Path) -> None:
    (source / "dataset.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(UnsupportedFeatureError):
        read_source(source)


def test_not_a_source(tmp_path: Path) -> None:
    from tokbin import FormatError

    with pytest.raises(FormatError) as info:
        read_source(tmp_path)
    assert info.value.code is codes.METADATA_MISSING


# --- round trip with the writer -------------------------------------------------------


@pytest.mark.requires_tokenizers
def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    docs = make_docs(200, seed=11, max_len=40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShardingWarning)
        Dataset(tmp_path).write("web", docs, make_tokenizer(), config=WriterConfig(shard_bytes=100))
    src = read_source(tmp_path / "web")
    expected = expected_stream(docs)
    assert src.n_shards > 50
    assert np.array_equal(src[:], expected)
    for i in range(src.n_docs):
        assert src.id_of_doc(i) == docs[i][0]
        assert np.array_equal(src.doc(i), expected_stream([docs[i]]))
    # Windows of up to three shards (50 items each) cross several seams at once.
    n = len(src)
    for start in range(0, n, 7):
        for length in (1, 49, 50, 51, 99, 100, 101, 150):
            if start + length <= n:
                assert np.array_equal(src.window(start, length), expected[start : start + length])


@pytest.mark.requires_tokenizers
def test_plain_text_source_has_no_ids(tmp_path: Path) -> None:
    Dataset(tmp_path).write("web", ["t1 t2", "t3"], make_tokenizer())
    src = read_source(tmp_path / "web")
    assert src.id_of_doc(0) is None
    assert src.id_of_doc(1) is None


@pytest.mark.requires_tokenizers
def test_empty_split(tmp_path: Path) -> None:
    Dataset(tmp_path).write("web", [], make_tokenizer())
    src = read_source(tmp_path / "web")
    assert len(src) == 0
    assert src.n_docs == 0
    assert src.window(0, 0).size == 0
    assert src[:].size == 0
    with pytest.raises(ConfigError):
        src.sample_windows(1, 1, np.random.default_rng(0))
