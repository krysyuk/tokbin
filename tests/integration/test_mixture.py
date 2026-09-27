"""Mixtures: weights, tokenizer compatibility, determinism by seed (spec 6.6, 17, 19)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tokbin import CompatibilityError, ConfigError, FormatError, Mixture, codes, read_source

from support import copy_fixture


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    copy_fixture(tmp_path / "web")
    copy_fixture(tmp_path / "code")
    (tmp_path / "mix.json").write_text('{"web": 3, "code": 1}')
    return tmp_path


def test_from_mix_json(corpus: Path) -> None:
    mix = Mixture.from_config(corpus)
    assert mix.names == ("web", "code")
    assert mix.weights == (0.75, 0.25)
    assert mix.dtype == np.dtype("uint16")
    batch = mix.batch(8, 16, np.random.default_rng(0))
    assert batch.shape == (8, 16)
    assert batch.dtype == np.uint16


def test_explicit_weights_override_the_file(corpus: Path) -> None:
    mix = Mixture.from_config(corpus, {"code": 1})
    assert mix.names == ("code",)
    assert Mixture(corpus, {"web": 1, "code": 0}).names == ("web",)


def test_same_seed_same_batch(corpus: Path) -> None:
    mix = Mixture.from_config(corpus)
    a = mix.batch(32, 20, np.random.default_rng(7))
    b = mix.batch(32, 20, np.random.default_rng(7))
    c = mix.batch(32, 20, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_single_source_matches_the_source(corpus: Path) -> None:
    mix = Mixture(corpus, {"web": 1})
    got = mix.batch(10, 30, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    rng.choice(1, size=10, p=[1.0])  # the mixture draws sources first
    expected = read_source(corpus / "web").sample_windows(10, 30, rng)
    assert np.array_equal(got, expected)


def test_distribution_follows_weights(corpus: Path) -> None:
    mix = Mixture(corpus, {"web": 0.7, "code": 0.3})
    which = mix.choose(100_000, np.random.default_rng(1))
    share = np.bincount(which, minlength=2) / which.size
    assert abs(share[0] - 0.7) < 0.01
    assert abs(share[1] - 0.3) < 0.01


def test_rows_come_from_the_chosen_source(corpus: Path) -> None:
    # Make the sources distinguishable: "code" gets a different valid split size.
    mix = Mixture(corpus, {"web": 1, "code": 1}, split="valid")
    batch = mix.batch(50, 57, np.random.default_rng(0))  # the whole valid split
    whole = read_source(corpus / "web", "valid")[:]
    assert all(np.array_equal(row, whole) for row in batch)


def test_different_tokenizers_are_refused(corpus: Path) -> None:
    meta = json.loads((corpus / "code" / "meta.json").read_text())
    meta["tokenizer_hash"] = "a" * 16
    (corpus / "code" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(CompatibilityError) as err:
        Mixture.from_config(corpus)
    assert err.value.code is codes.TOKENIZERS_DIFFER
    assert "code=aaaaaaaaaaaaaaaa" in err.value.why
    assert Mixture.from_config(corpus, {"web": 1, "code": 0}).names == ("web",)


def test_missing_sources(corpus: Path) -> None:
    (corpus / "mix.json").write_text('{"web": 1, "wiki": 1}')
    with pytest.raises(FormatError) as err:
        Mixture.from_config(corpus)
    assert err.value.code is codes.MIX_SOURCE_MISSING
    with pytest.raises(ConfigError) as err2:
        Mixture(corpus, {"wiki": 1})
    assert err2.value.code is codes.SOURCE_NOT_FOUND
    assert Mixture(corpus, {"web": 1, "wiki": 0}).names == ("web",)


@pytest.mark.parametrize(
    "weights",
    [{}, {"web": -1}, {"web": 0}, {"web": float("nan")}, {"web": True}, {"../x": 1}, {"web": "1"}],
)
def test_invalid_weights(corpus: Path, weights: dict[str, object]) -> None:
    with pytest.raises(ConfigError) as err:
        Mixture(corpus, weights)  # type: ignore[arg-type]
    assert err.value.code is codes.MIX_WEIGHTS_INVALID


def test_sampling_arguments(corpus: Path) -> None:
    mix = Mixture.from_config(corpus)
    with pytest.raises(ConfigError) as err:
        mix.batch(2, 4, np.random.RandomState(0))  # type: ignore[arg-type]
    assert err.value.code is codes.RNG_REQUIRED
    with pytest.raises(ConfigError):
        mix.batch(2, 0, np.random.default_rng())
    with pytest.raises(ConfigError):
        mix.batch(-1, 4, np.random.default_rng())
    with pytest.raises(ConfigError) as err:
        mix.batch(2, 10_000, np.random.default_rng())
    assert err.value.code is codes.WINDOW_TOO_LARGE
    assert mix.batch(0, 4, np.random.default_rng()).shape == (0, 4)


def test_pickle(corpus: Path) -> None:
    mix = Mixture.from_config(corpus)
    clone = pickle.loads(pickle.dumps(mix))  # noqa: S301 (our own object)
    assert (clone.names, clone.weights, clone.split) == (mix.names, mix.weights, mix.split)
    assert np.array_equal(
        clone.batch(4, 8, np.random.default_rng(2)), mix.batch(4, 8, np.random.default_rng(2))
    )
    assert "web=0.75" in repr(mix)
