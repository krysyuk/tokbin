"""Regression benchmarks: a fast, synthetic suite compared against a saved baseline.

Needs no downloads (a word-level tokenizer and documents are generated) and runs in
about a minute::

    uv run --group bench python benchmarks/regress.py                  # compare
    uv run --group bench python benchmarks/regress.py --save-baseline  # accept

Every metric is a time (lower is better). A metric more than ``--threshold`` (10%)
slower than the baseline is measured again; if it is still slower the run fails with
exit code 1. Timings only compare on the same machine: the baseline records where it
was made, and a comparison on another machine prints a warning first.

The writer is measured with tokenization replayed from a table, so the number is the
cost of tokbin itself and does not move with the tokenizers package.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import tokbin
from tokbin import Mixture, WriterConfig, read_source
from tokbin.tokenizer.hf import HFTokenizer

from bench_common import (
    ROOT,
    environment,
    measure,
    replay_tokenizer,
    save_json,
    synthetic_docs,
    word_tokenizer,
)

BASELINE = ROOT / "benchmarks" / "baseline.json"
WORK = ROOT / "benchmarks" / ".work" / "regress"

N_DOCS = 50_000
MEAN_WORDS = 200


class Suite:
    """Builds the data once; each metric is a callable returning seconds."""

    def __init__(self) -> None:
        self.docs = synthetic_docs(N_DOCS, mean_words=MEAN_WORDS)
        real = word_tokenizer()
        texts = [t for _, t in self.docs]
        table = {}
        for i in range(0, len(texts), 4096):
            chunk = texts[i : i + 4096]
            for text, enc in zip(chunk, real.encode_batch(chunk), strict=True):
                table[text] = np.asarray(enc.ids, dtype=np.int64)
        self.n_tokens = sum(a.size + 1 for a in table.values())
        self.tok = replay_tokenizer(HFTokenizer(real, eos_token="<|endoftext|>"), table)
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        # 100 shards: reads cross boundaries and the shard search is exercised.
        self.config = WriterConfig(shard_bytes=self.n_tokens * 2 // 100 + 2)
        self._write("src")
        self.src = read_source(WORK / "src")
        self.mix = Mixture(WORK, {"src": 1.0})
        self.rng = np.random.default_rng(0)
        self.starts = self.rng.integers(0, len(self.src) - 2048, size=4096)
        self.doc_idx = self.rng.integers(0, self.src.n_docs, size=4096)
        self.k = 0

    def _write(self, name: str) -> None:
        shutil.rmtree(WORK / name, ignore_errors=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokbin.Dataset(WORK).write(name, self.docs, self.tok, config=self.config)

    def _next(self) -> int:
        self.k = (self.k + 1) & 4095
        return self.k

    def metrics(self) -> dict[str, Callable[[], float]]:
        src, rng = self.src, self.rng
        return {
            # Per token of the writer's own work, tokenization excluded.
            "write_ns_per_token": lambda: (
                measure(lambda: self._write("w"), repeat=5, number=1, warmup=1).best
                / self.n_tokens
                * 1e9
            ),
            "open_100_shards_us": lambda: measure(lambda: read_source(WORK / "src")).median * 1e6,
            "item_us": lambda: measure(lambda: src[int(self.starts[self._next()])]).median * 1e6,
            "window_2048_us": lambda: (
                measure(lambda: src.window(int(self.starts[self._next()]), 2048)).median * 1e6
            ),
            "doc_us": lambda: (
                measure(lambda: src.doc(int(self.doc_idx[self._next()]))).median * 1e6
            ),
            "id_of_doc_us": lambda: (
                measure(lambda: src.id_of_doc(int(self.doc_idx[self._next()]))).median * 1e6
            ),
            "sample_32x2048_us": lambda: (
                measure(lambda: src.sample_windows(32, 2048, rng)).median * 1e6
            ),
            "sample_512x128_us": lambda: (
                measure(lambda: src.sample_windows(512, 128, rng)).median * 1e6
            ),
            "mixture_32x2048_us": lambda: (
                measure(lambda: self.mix.batch(32, 2048, rng)).median * 1e6
            ),
            "inspect_ms": lambda: measure(lambda: tokbin.inspect_source(WORK / "src")).median * 1e3,
            "verify_ms": lambda: (
                measure(lambda: tokbin.verify_source(WORK / "src"), repeat=5).median * 1e3
            ),
        }


def compare(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold: float,
    remeasure: Callable[[str], float],
) -> list[str]:
    failed = []
    print(f"\n{'metric':24} {'baseline':>12} {'current':>12} {'change':>8}")
    for name, value in current.items():
        base = baseline.get(name)
        if base is None:
            print(f"{name:24} {'-':>12} {value:12.3f}      new")
            continue
        change = value / base - 1
        if change > threshold:
            value = min(value, remeasure(name))
            current[name] = value
            change = value / base - 1
        mark = "  FAIL" if change > threshold else ""
        print(f"{name:24} {base:12.3f} {value:12.3f} {change:+8.1%}{mark}")
        if change > threshold:
            failed.append(name)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--save-baseline", action="store_true", help="store this run")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args()

    suite = Suite()
    metrics = suite.metrics()
    current = {}
    for name, fn in metrics.items():
        current[name] = fn()
        print(f"{name:24} {current[name]:12.3f}", flush=True)
    env = environment()
    record = {"env": env, "n_docs": N_DOCS, "n_tokens": suite.n_tokens, "metrics": current}
    if args.json is not None:
        save_json(args.json, record)
    if args.save_baseline:
        save_json(args.baseline, record)
        print(f"\nbaseline saved: {args.baseline}")
        return 0
    if not args.baseline.is_file():
        print(f"\nno baseline at {args.baseline}; run with --save-baseline first")
        return 0
    saved: dict[str, Any] = json.loads(args.baseline.read_text())
    if saved["env"].get("cpu") != env["cpu"] or saved["env"].get("python") != env["python"]:
        print(
            f"\nwarning: the baseline was made on {saved['env'].get('cpu')} / Python "
            f"{saved['env'].get('python')}; timings from another machine do not compare"
        )
    failed = compare(current, saved["metrics"], args.threshold, lambda n: metrics[n]())
    if failed:
        print(f"\n{len(failed)} regression(s) above {args.threshold:.0%}: {', '.join(failed)}")
        return 1
    print(f"\nno regression above {args.threshold:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
