"""The torch adapter, with a stand-in torch module (and with the real one if installed)."""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tokbin import ConfigError, DependencyError, Mixture, OutOfRangeError, codes, read_source

from support import copy_fixture


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake = types.ModuleType("torch")
    fake.from_numpy = lambda array: ("tensor", array)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    copy_fixture(tmp_path / "web")
    copy_fixture(tmp_path / "code")
    return tmp_path


def test_window_dataset(corpus: Path, fake_torch: Any) -> None:
    from tokbin.adapters.torch import WindowDataset

    src = read_source(corpus / "web")  # 303 items
    ds = WindowDataset(src, block_size=100)
    assert len(ds) == 3
    kind, first = ds[0]
    assert kind == "tensor"
    assert first.dtype == np.int64
    assert np.array_equal(first, src.window(0, 100))
    assert np.array_equal(ds[-1][1], src.window(200, 100))
    with pytest.raises(OutOfRangeError):
        ds[3]

    overlapping = WindowDataset(src, block_size=100, stride=1)
    assert len(overlapping) == 204
    assert np.array_equal(overlapping[203][1], src.window(203, 100))


@pytest.mark.parametrize("kwargs", [{"block_size": 0}, {"block_size": 10, "stride": 0}])
def test_window_dataset_arguments(corpus: Path, fake_torch: Any, kwargs: dict[str, int]) -> None:
    from tokbin.adapters.torch import WindowDataset

    with pytest.raises(ConfigError):
        WindowDataset(read_source(corpus / "web"), **kwargs)
    with pytest.raises(ConfigError) as err:
        WindowDataset(read_source(corpus / "web"), block_size=10_000)
    assert err.value.code is codes.WINDOW_TOO_LARGE


def test_mixture_dataset_is_deterministic_per_index(corpus: Path, fake_torch: Any) -> None:
    from tokbin.adapters.torch import MixtureDataset

    mix = Mixture(corpus, {"web": 1, "code": 1})
    ds = MixtureDataset(mix, block_size=32, length=100, seed=5)
    assert len(ds) == 100
    assert np.array_equal(ds[7][1], ds[7][1])
    assert not all(np.array_equal(ds[i][1], ds[0][1]) for i in range(1, 20))
    other = MixtureDataset(mix, block_size=32, length=100, seed=6)
    assert not np.array_equal(ds[7][1], other[7][1])
    # What DataLoader workers receive.
    clone = pickle.loads(pickle.dumps(ds))  # noqa: S301 (our own object)
    assert np.array_equal(clone[42][1], ds[42][1])


def test_torch_is_required(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tokbin.adapters.torch import WindowDataset

    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    with pytest.raises(DependencyError) as err:
        WindowDataset(read_source(corpus / "web"), block_size=10)
    assert "tokbin-core[torch]" in err.value.fix


def test_importing_the_adapter_does_not_import_torch() -> None:
    import subprocess

    code = "import sys, tokbin.adapters.torch; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_with_real_torch(corpus: Path) -> None:
    torch = pytest.importorskip("torch")
    from tokbin.adapters.torch import MixtureDataset, WindowDataset

    loader = torch.utils.data.DataLoader(
        WindowDataset(read_source(corpus / "web"), block_size=16), batch_size=4, shuffle=True
    )
    batch = next(iter(loader))
    assert batch.shape == (4, 16)
    assert batch.dtype == torch.int64
    mixed = MixtureDataset(Mixture(corpus, {"web": 1}), block_size=16, length=8, seed=0)
    assert next(iter(torch.utils.data.DataLoader(mixed, batch_size=8))).shape == (8, 16)
