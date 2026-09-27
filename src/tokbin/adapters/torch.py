"""Datasets for ``torch.utils.data.DataLoader`` (the ``torch`` extra).

::

    from tokbin.adapters.torch import WindowDataset, MixtureDataset

    loader = DataLoader(WindowDataset(read_source("corpus/web"), block_size=1024),
                        batch_size=32, shuffle=True, num_workers=4)
    loader = DataLoader(MixtureDataset(Mixture.from_config("corpus"), block_size=1024,
                                       length=100_000, seed=0), batch_size=32)

Both are map-style datasets (``__len__`` and ``__getitem__``): they work with any
sampler, shuffling and worker count. Items are ``torch.int64`` tensors of
``block_size`` tokens; shifting into inputs and targets is up to the training code.

The classes do not subclass torch types, so importing this module does not import
torch; torch is required (``DependencyError`` otherwise) when a dataset is created.
They pickle cheaply: sources reopen their files in each worker.
"""

from __future__ import annotations

import operator
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from tokbin import _deps, codes
from tokbin.errors import ConfigError, OutOfRangeError
from tokbin.read.mixture import Mixture
from tokbin.read.source import Source

if TYPE_CHECKING:
    import torch

__all__ = ["MixtureDataset", "WindowDataset"]


def _torch() -> ModuleType:
    return _deps.require("torch")


def _positive(name: str, value: object) -> int:
    try:
        number = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        number = 0
    if isinstance(value, bool) or number < 1:
        raise ConfigError(
            codes.CONFIG_VALUE_INVALID,
            f"{name}={value!r}",
            why=f"{name} must be a positive integer",
            fix=f"pass a positive {name}",
        )
    return number


def _index(i: object, n: int) -> int:
    index = operator.index(i)  # type: ignore[arg-type]
    if index < 0:
        index += n
    if not 0 <= index < n:
        raise OutOfRangeError(
            codes.OUT_OF_RANGE,
            f"item {i}",
            why=f"the dataset has {n} items",
            fix="index within len(dataset)",
        )
    return index


class WindowDataset:
    """Consecutive windows of one source: item ``i`` starts at ``i * stride``.

    ``stride`` defaults to ``block_size`` (non-overlapping windows). The tail shorter
    than a window is not used.
    """

    def __init__(self, source: Source, block_size: int, *, stride: int | None = None) -> None:
        _torch()
        self.source = source
        self.block_size = _positive("block_size", block_size)
        self.stride = _positive("stride", stride if stride is not None else block_size)
        if self.block_size > len(source):
            raise ConfigError(
                codes.WINDOW_TOO_LARGE,
                f"block_size={block_size}",
                why=f"the source has only {len(source)} items",
                fix="use a smaller block_size",
            )

    def __len__(self) -> int:
        return (len(self.source) - self.block_size) // self.stride + 1

    def __getitem__(self, i: int) -> torch.Tensor:
        start = _index(i, len(self)) * self.stride
        window = self.source.window(start, self.block_size)
        return _torch().from_numpy(window.astype(np.int64))


class MixtureDataset:
    """``length`` windows drawn from a mixture by weight.

    Item ``i`` is fully determined by ``(seed, i)``: the same index always gives the
    same window, in any worker and any epoch; reshuffle across epochs by changing the
    seed.
    """

    def __init__(self, mixture: Mixture, block_size: int, *, length: int, seed: int) -> None:
        _torch()
        self.mixture = mixture
        self.block_size = _positive("block_size", block_size)
        self.length = _positive("length", length)
        self.seed = operator.index(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> torch.Tensor:
        index = _index(i, self.length)
        rng = np.random.default_rng(np.random.SeedSequence(entropy=self.seed, spawn_key=(index,)))
        window = self.mixture.batch(1, self.block_size, rng)[0]
        return _torch().from_numpy(window.astype(np.int64))
