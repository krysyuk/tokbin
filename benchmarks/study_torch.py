"""torch study: the DataLoader adapters against a hand-written memmap dataset.

Needs torch (not a dependency of the benchmarks)::

    python benchmarks/study_torch.py --source benchmarks/.work/full/deepseek/src

Writes ``results/torch-loader.json``: batches per second and time to the first batch
for ``WindowDataset``, ``MixtureDataset`` and a plain ``np.memmap`` dataset, with 0 to 8
worker processes (``spawn``, the default on macOS and Windows).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import tokbin
from tokbin.adapters.torch import MixtureDataset, WindowDataset

from bench_common import RESULTS, environment, save_json


class RawWindows(Dataset):  # type: ignore[type-arg]
    """nanoGPT-style: one flat file, non-overlapping windows, opened lazily per worker."""

    def __init__(self, path: Path, dtype: str, block: int) -> None:
        self.path, self.dtype, self.block = path, dtype, block
        self.n = (path.stat().st_size // np.dtype(dtype).itemsize - block) // block + 1
        self.mm: np.memmap | None = None

    def __getstate__(self) -> dict[str, Any]:
        # A mapped np.memmap would be pickled with all its data: send only the path.
        return {**self.__dict__, "mm": None}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        if self.mm is None:
            self.mm = np.memmap(self.path, dtype=self.dtype, mode="r")
        s = i * self.block
        return torch.from_numpy(self.mm[s : s + self.block].astype(np.int64))


def run(ds: Any, workers: int, batches: int, batch: int) -> dict[str, float]:
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(0),
    )
    t0 = time.perf_counter()
    it = iter(loader)
    next(it)
    first = time.perf_counter() - t0
    for _ in range(max(50, 10 * workers)):  # let the workers' prefetch fill up
        next(it)
    t1 = time.perf_counter()
    for _ in range(batches):
        next(it)
    steady = time.perf_counter() - t1
    del it
    return {"first_batch_s": first, "batches_per_s": batches / steady}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="the shards as one flat file")
    parser.add_argument("--block", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batches", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    # Warm the page cache for both layouts, so neither pays for the disk.
    for path in [args.raw, *sorted(args.source.glob("*.bin"))]:
        with path.open("rb") as f:
            while f.read(64 << 20):
                pass
    src = tokbin.read_source(args.source)
    mix = tokbin.Mixture(args.source.parent, {args.source.name: 1.0})
    datasets = {
        "raw memmap": RawWindows(args.raw, str(src.dtype), args.block),
        "WindowDataset": WindowDataset(src, args.block),
        "MixtureDataset": MixtureDataset(mix, args.block, length=10**7, seed=0),
    }
    rows = []
    for workers in (0, 2, 4, 8):
        for name, ds in datasets.items():
            res = run(ds, workers, args.batches, args.batch)
            rows.append({"dataset": name, "workers": workers, **res})
            print(json.dumps(rows[-1]))
    save_json(
        args.out / "torch-loader.json",
        {
            "experiment": "loader",
            "env": {**environment(), "torch": torch.__version__},
            "block": args.block,
            "batch": args.batch,
            "data": rows,
        },
    )


if __name__ == "__main__":
    main()
