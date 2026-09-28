"""Read-path study: opening, random access, sampling, memory and cold storage.

Uses a real source (``--source``, e.g. one written by ``study_write.py full``) and
synthetic sources it writes itself for the scaling curves::

    uv run --group bench python benchmarks/study_read.py --source benchmarks/.work/full/deepseek/src

Every experiment writes ``results/read-<name>.json``. Experiments:

``calls``     latency of single calls against raw numpy on a memmap
``windows``   window latency by length, inside a shard and across a boundary
``sampling``  batches of random windows: tokbin against hand-written numpy
``shards``    open time and window latency as the number of shards grows
``docs``      ``doc`` / ``id_of_doc`` latency as the number of documents grows
``cold``      first reads from disk (the page cache bypassed on macOS)
``memory``    resident memory while sampling random windows
``ops``       ``inspect_source`` and ``verify_source``
``mixture``   ``Mixture.batch`` with several sources
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import tokbin
from tokbin import Mixture, WriterConfig, read_source

from bench_common import (
    RESULTS,
    ROOT,
    cold_copy,
    dir_size,
    environment,
    fresh_dir,
    measure,
    replay_tokenizer,
    rss_bytes,
    save_json,
    synthetic_docs,
    time_once,
    word_tokenizer,
)

WORK = ROOT / "benchmarks" / ".work" / "read"


def raw_concat(source: Path) -> Path:
    """All shards of the train split as one flat file: the hand-written baseline."""
    out = WORK / "concat.bin"
    meta = json.loads((source / "meta.json").read_text())
    shards = [source / s["name"] for s in meta["splits"]["train"]["shards"]]
    expected = sum(p.stat().st_size for p in shards)
    if out.is_file() and out.stat().st_size == expected:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        for path in shards:
            with path.open("rb") as shard:
                shutil.copyfileobj(shard, f, 16 << 20)
    return out


def us(timing: Any) -> dict[str, float]:
    """A Timing in microseconds."""
    return {k: v * 1e6 if k != "n" else v for k, v in timing.to_dict().items()}


# --- synthetic sources -------------------------------------------------------------------

_TABLES: dict[tuple[int, int], tuple[list[tuple[str, str]], Any]] = {}


def synthetic_source(
    root: Path, *, n_docs: int, mean_words: int, shard_bytes: int, seed: int = 0
) -> Path:
    """A uint16 source of synthetic documents, written once and reused."""
    target = root / f"d{n_docs}-w{mean_words}-s{shard_bytes}"
    if (target / "meta.json").is_file():
        return target
    key = (n_docs, mean_words)
    if key not in _TABLES:
        from tokbin.tokenizer.hf import HFTokenizer

        docs = synthetic_docs(n_docs, mean_words=mean_words, seed=seed)
        real = word_tokenizer()
        table = {}
        texts = [t for _, t in docs]
        for i in range(0, len(texts), 4096):
            chunk = texts[i : i + 4096]
            for text, enc in zip(chunk, real.encode_batch(chunk), strict=True):
                table[text] = np.asarray(enc.ids, dtype=np.int64)
        hf = HFTokenizer(real, eos_token="<|endoftext|>")
        _TABLES[key] = (docs, replay_tokenizer(hf, table))
    docs, tok = _TABLES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokbin.Dataset(root).write(
            target.name, docs, tok, config=WriterConfig(shard_bytes=shard_bytes)
        )
    return target


# --- experiments -------------------------------------------------------------------------


def exp_calls(args: argparse.Namespace) -> dict[str, Any]:
    src = read_source(args.source)
    raw = np.memmap(raw_concat(args.source), dtype=src.dtype, mode="r")
    offsets = np.load(args.source / "train-offsets.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    n = len(src)
    pos = rng.integers(0, n - 4096, size=4096)
    docs = rng.integers(0, src.n_docs, size=4096)
    it = iter(range(1 << 62))

    def pick() -> int:
        return int(pos[next(it) & 4095])

    def pick_doc() -> int:
        return int(docs[next(it) & 4095])

    rows = {
        "raw_item": measure(lambda: int(raw[pick()])),
        "src_item": measure(lambda: src[pick()]),
        "raw_slice_256": measure(lambda: raw[(p := pick()) : p + 256]),
        "src_window_256": measure(lambda: src.window(pick(), 256)),
        "raw_doc": measure(lambda: raw[offsets[(d := pick_doc())] : offsets[d + 1]]),
        "src_doc": measure(lambda: src.doc(pick_doc())),
        "src_id_of_doc": measure(lambda: src.id_of_doc(pick_doc())),
        "pick_only": measure(pick),
        "open": measure(lambda: read_source(args.source), repeat=7),
        "pickle_roundtrip": measure(lambda: pickle.loads(pickle.dumps(src)), repeat=7),
    }
    out = {k: us(v) for k, v in rows.items()}
    for k, v in out.items():
        print(f"calls {k}: {v['median']:.2f} us")
    return {
        "n_items": n,
        "n_docs": src.n_docs,
        "n_shards": src.n_shards,
        "dtype": str(src.dtype),
        "us": out,
    }


def exp_windows(args: argparse.Namespace) -> dict[str, Any]:
    src = read_source(args.source)
    raw = np.memmap(raw_concat(args.source), dtype=src.dtype, mode="r")
    meta = json.loads((args.source / "meta.json").read_text())
    boundary = meta["splits"]["train"]["shards"][0]["n_items"]
    rows = []
    for length in (1, 16, 256, 2048, 16_384, 131_072, 1_048_576):
        rng = np.random.default_rng(length)
        starts = rng.integers(0, boundary - length, size=256)
        it = iter(range(1 << 62))

        def start(starts: npt.NDArray[np.int64] = starts) -> int:
            return int(starts[next(it) & 255])

        row = {
            "length": length,
            # A view: the call returns before any data is read.
            "src_view": us(measure(lambda n=length: src.window(start(), n))),
            "raw_view": us(measure(lambda n=length: raw[(s := start()) : s + n])),
            # What training does next: materialize the window as int64 for torch.
            "src_int64": us(measure(lambda n=length: src.window(start(), n).astype(np.int64))),
            "raw_int64": us(measure(lambda n=length: raw[(s := start()) : s + n].astype(np.int64))),
            # Across the boundary of shards 0 and 1: always a copy.
            "src_cross": us(measure(lambda n=length: src.window(boundary - n // 2 - 1, n)))
            if length > 1
            else None,
        }
        rows.append(row)
        print(
            f"windows {length}: view {row['src_view']['median']:.2f} us, "
            f"int64 {row['src_int64']['median']:.1f} us (raw {row['raw_int64']['median']:.1f})"
        )
    return {"dtype": str(src.dtype), "rows": rows}


def _raw_loop(raw: np.memmap, starts: npt.NDArray[np.int64], block: int) -> npt.NDArray[Any]:
    return np.stack([raw[s : s + block] for s in starts.tolist()])


def _raw_fancy(raw: np.memmap, starts: npt.NDArray[np.int64], block: int) -> npt.NDArray[Any]:
    return raw[starts[:, None] + np.arange(block)]


def exp_sampling(args: argparse.Namespace) -> dict[str, Any]:
    src = read_source(args.source)
    raw = np.memmap(raw_concat(args.source), dtype=src.dtype, mode="r")
    mix = Mixture(args.source.parent, {args.source.name: 1.0})
    n = len(src)
    rows = []
    for batch, block in (
        (1, 2048),
        (8, 2048),
        (32, 128),
        (32, 512),
        (32, 2048),
        (32, 8192),
        (128, 2048),
        (512, 2048),
        (512, 128),
    ):
        rng = np.random.default_rng(0)
        starts_rng = np.random.default_rng(0)

        def starts(b: int = batch, k: int = block) -> npt.NDArray[np.int64]:
            return starts_rng.integers(0, n - k + 1, size=b)

        row = {
            "batch": batch,
            "block": block,
            "tokbin": us(measure(lambda b=batch, k=block: src.sample_windows(b, k, rng))),
            "mixture": us(measure(lambda b=batch, k=block: mix.batch(b, k, rng))),
            "raw_loop": us(measure(lambda b=batch, k=block: _raw_loop(raw, starts(b, k), k))),
            "raw_fancy": us(measure(lambda b=batch, k=block: _raw_fancy(raw, starts(b, k), k))),
        }
        rows.append(row)
        print(
            f"sampling {batch}x{block}: tokbin {row['tokbin']['median']:.0f} us, "
            f"mixture {row['mixture']['median']:.0f}, loop {row['raw_loop']['median']:.0f}, "
            f"fancy {row['raw_fancy']['median']:.0f}"
        )
    return {"dtype": str(src.dtype), "n_items": n, "rows": rows}


def exp_shards(args: argparse.Namespace) -> dict[str, Any]:
    root = WORK / "shards"
    root.mkdir(parents=True, exist_ok=True)
    total_bytes = 200_000 * 100 * 2  # about 20M uint16 tokens
    rows = []
    for n_shards in (1, 10, 100, 1000, 10_000):
        shard_bytes = max(16, total_bytes // n_shards)
        t0 = time.perf_counter()
        path = synthetic_source(root, n_docs=200_000, mean_words=100, shard_bytes=shard_bytes)
        written = time.perf_counter() - t0
        src = read_source(path)
        rng = np.random.default_rng(0)
        starts = rng.integers(0, len(src) - 2048, size=4096)
        # First pass: shards are mapped lazily, so early windows pay for np.memmap.
        first = []
        for start in starts.tolist():
            t1 = time.perf_counter()
            src.window(start, 2048)
            first.append((time.perf_counter() - t1) * 1e6)
        it = iter(range(1 << 62))
        row = {
            "n_shards": src.n_shards,
            "write_s": written,
            "meta_bytes": (path / "meta.json").stat().st_size,
            "open": us(measure(lambda p=path: read_source(p), repeat=5)),
            "window_first_pass_us": {
                "median": float(np.median(first)),
                "total_ms": sum(first) / 1e3,
            },
            "window_2048": us(
                measure(lambda s=src, st=starts: s.window(int(st[next(it) & 4095]), 2048))
            ),
            "sample_32x2048": us(measure(lambda s=src: s.sample_windows(32, 2048, rng))),
            "inspect": us(measure(lambda p=path: tokbin.inspect_source(p), repeat=5)),
        }
        rows.append(row)
        print(
            f"shards {row['n_shards']}: open {row['open']['median']:.0f} us, window "
            f"{row['window_2048']['median']:.2f} us, meta {row['meta_bytes']} B"
        )
    return {"rows": rows}


def exp_docs(args: argparse.Namespace) -> dict[str, Any]:
    root = WORK / "docs"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_docs in (10_000, 100_000, 1_000_000, 10_000_000):
        path = synthetic_source(root, n_docs=n_docs, mean_words=8, shard_bytes=512 * 1024**2)
        src = read_source(path)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n_docs, size=4096)
        it = iter(range(1 << 62))
        row = {
            "n_docs": n_docs,
            "doc": us(measure(lambda s=src, i=idx: s.doc(int(i[next(it) & 4095])))),
            "id_of_doc": us(measure(lambda s=src, i=idx: s.id_of_doc(int(i[next(it) & 4095])))),
            "open": us(measure(lambda p=path: read_source(p), repeat=5)),
            "index_bytes": sum(
                (path / f).stat().st_size
                for f in ("train-offsets.npy", "train-ids.idx.npy", "train-ids.jsonl")
            ),
        }
        rows.append(row)
        print(
            f"docs {n_docs}: doc {row['doc']['median']:.2f} us, id_of_doc "
            f"{row['id_of_doc']['median']:.2f} us, open {row['open']['median']:.0f} us"
        )
    return {"rows": rows}


def _drop_cache_copy(source: Path, name: str) -> Path:
    return cold_copy(source, WORK / "cold" / name)


def exp_cold(args: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {"skipped": "cold copies need F_NOCACHE (macOS)"}
    out: dict[str, Any] = {}
    # Open.
    path = _drop_cache_copy(args.source, "open")
    out["open_cold_s"] = time_once(lambda: read_source(path))
    out["open_warm_s"] = measure(lambda: read_source(path), repeat=5).median
    # Random windows: every first touch comes from the disk.
    path = _drop_cache_copy(args.source, "windows")
    src = read_source(path)
    rng = np.random.default_rng(0)
    for block in (256, 2048, 16_384):
        starts = rng.integers(0, len(src) - block, size=500)
        cold, warm = [], []
        for s in starts.tolist():
            t0 = time.perf_counter()
            src.window(s, block).sum()  # touch every page
            cold.append(time.perf_counter() - t0)
        for s in starts.tolist():
            t0 = time.perf_counter()
            src.window(s, block).sum()
            warm.append(time.perf_counter() - t0)
        out[f"window_{block}"] = {
            "cold_us": [x * 1e6 for x in cold],
            "warm_us": [x * 1e6 for x in warm],
        }
        print(
            f"cold window {block}: median cold {np.median(cold) * 1e6:.0f} us, "
            f"warm {np.median(warm) * 1e6:.1f} us, p99 cold {np.percentile(cold, 99) * 1e6:.0f}"
        )
    # A batch of 32 random windows, cold.
    path = _drop_cache_copy(args.source, "batch")
    src = read_source(path)
    batches = [time_once(lambda: src.sample_windows(32, 2048, rng)) for _ in range(100)]
    out["batch_32x2048_cold_ms"] = [x * 1e3 for x in batches]
    # Sequential: verify reads and hashes every byte.
    path = _drop_cache_copy(args.source, "verify")
    size = dir_size(path)
    out["verify_cold_s"] = time_once(lambda: tokbin.verify_source(path))
    out["verify_warm_s"] = time_once(lambda: tokbin.verify_source(path))
    out["verify_bytes"] = size
    print(
        f"cold verify: {size / out['verify_cold_s'] / 1e9:.2f} GB/s, "
        f"warm {size / out['verify_warm_s'] / 1e9:.2f} GB/s"
    )
    shutil.rmtree(WORK / "cold", ignore_errors=True)
    return out


def exp_memory(args: argparse.Namespace) -> dict[str, Any]:
    """Resident memory while sampling: runs in a subprocess, so the baseline is clean."""
    cmd = [sys.executable, __file__, "_memory_child", "--source", str(args.source)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout.strip().splitlines()[-1])
    print(f"memory: rss {data['rss'][0][1] / 2**20:.0f} -> {data['rss'][-1][1] / 2**20:.0f} MiB")
    return data


def memory_child(args: argparse.Namespace) -> None:
    src = read_source(args.source)
    rng = np.random.default_rng(0)
    base = rss_bytes()
    size = len(src) * src.dtype.itemsize
    rss = [(0, base)]
    sampled = 0
    step = 32 * 2048
    while sampled < 4 * len(src):
        for _ in range(64):
            src.sample_windows(32, 2048, rng)
            sampled += step
        rss.append((sampled, rss_bytes()))
    print(json.dumps({"data_bytes": size, "n_items": len(src), "rss": rss}))


def exp_ops(args: argparse.Namespace) -> dict[str, Any]:
    size = dir_size(args.source)
    out = {
        "bytes": size,
        "inspect_s": measure(lambda: tokbin.inspect_source(args.source), repeat=5).median,
        "verify_warm_s": min(
            time_once(lambda: tokbin.verify_source(args.source)) for _ in range(3)
        ),
    }
    print(
        f"ops: inspect {out['inspect_s'] * 1e3:.1f} ms, verify {out['verify_warm_s']:.2f} s "
        f"({size / out['verify_warm_s'] / 1e9:.2f} GB/s)"
    )
    return out


def exp_mixture(args: argparse.Namespace) -> dict[str, Any]:
    root = fresh_dir(WORK / "mixture")
    rows = []
    for k in (1, 2, 4, 8, 16):
        for i in range(k):
            (root / f"k{k}-{i}").symlink_to(args.source.resolve(), target_is_directory=True)
        weights = {f"k{k}-{i}": 1.0 for i in range(k)}
        mix = Mixture(root, weights)
        rng = np.random.default_rng(0)
        rows.append(
            {
                "sources": k,
                "batch_32x2048": us(measure(lambda m=mix: m.batch(32, 2048, rng))),
                "batch_512x128": us(measure(lambda m=mix: m.batch(512, 128, rng))),
            }
        )
        print(f"mixture {k}: {rows[-1]['batch_32x2048']['median']:.0f} us")
    src = read_source(args.source)
    rng = np.random.default_rng(0)
    return {
        "rows": rows,
        "single_32x2048": us(measure(lambda: src.sample_windows(32, 2048, rng))),
        "single_512x128": us(measure(lambda: src.sample_windows(512, 128, rng))),
    }


EXPERIMENTS = {
    "calls": exp_calls,
    "windows": exp_windows,
    "sampling": exp_sampling,
    "shards": exp_shards,
    "docs": exp_docs,
    "cold": exp_cold,
    "memory": exp_memory,
    "ops": exp_ops,
    "mixture": exp_mixture,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("only", nargs="*", help=f"experiments: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.only == ["_memory_child"]:
        memory_child(args)
        return
    for name in args.only or list(EXPERIMENTS):
        started = time.perf_counter()
        data = EXPERIMENTS[name](args)
        save_json(
            args.out / f"read-{name}.json",
            {
                "experiment": name,
                "env": environment(),
                "source": str(args.source),
                "wall_s": time.perf_counter() - started,
                "data": data,
            },
        )


if __name__ == "__main__":
    main()
