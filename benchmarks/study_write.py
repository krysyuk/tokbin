"""Write-path study on real text: where the time and the memory of a write go.

Needs a TinyStories ``.txt`` file and one or more ``tokenizer.json`` files::

    uv run --group bench python benchmarks/study_write.py \\
        --data TinyStoriesV2-GPT4-train.txt \\
        --tokenizer deepseek=path/to/tokenizer.json --tokenizer bpe12k=path/to/tokenizer.json

Every experiment writes ``results/write-<name>.json``. Experiments:

``breakdown``  tokbin write against its parts and against a hand-written writer
``batch``      the ``batch_docs`` setting
``threads``    tokenizer threads (``RAYON_NUM_THREADS``), each in a subprocess
``doclen``     short and long documents with the same text
``shards``     the ``shard_bytes`` setting (tokenization replayed, I/O isolated)
``resume``     the cost of continuing an interrupted write
``inputs``     ``tokbin build`` from JSON Lines and from a directory of text files
``idhashes``   memory and time of duplicate-id detection per document
``full``       the whole file with a memory timeline, and the hand-written writer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Iterable, Iterator
from itertools import chain, islice
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import tokbin
from tokbin import WriterConfig
from tokbin.tokenizer.hf import HFTokenizer
from tokbin.write.duplicates import IdHashes

from bench_common import (
    RESULTS,
    ROOT,
    RssSampler,
    dir_size,
    environment,
    fresh_dir,
    peak_rss_bytes,
    replay_tokenizer,
    save_json,
    time_once,
    tinystories,
)

Docs = list[tuple[str, str]]

WORK = ROOT / "benchmarks" / ".work"


class Interrupted(BaseException):
    """Stops a write the way Ctrl+C would, without touching the signal handlers."""


# --- baselines ---------------------------------------------------------------------------


def generate(docs: Iterable[tuple[str, str]]) -> None:
    """Only iterate the input: what reading the documents costs by itself."""
    for _ in docs:
        pass


def batches(docs: Iterable[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    it = iter(docs)
    while batch := list(islice(it, size)):
        yield batch


def tokenize_only(tok: Any, docs: Iterable[tuple[str, str]], batch: int) -> int:
    """``encode_batch`` plus ids to numpy: the least any writer has to pay.

    The conversion is the one tokbin's adapter does, so the difference to a tokbin
    write is exactly the writer's own work.
    """
    total = 0
    for chunk in batches(docs, batch):
        lists = [e.ids for e in tok.encode_batch([t for _, t in chunk], add_special_tokens=False)]
        lengths = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
        flat = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=int(lengths.sum()))
        total += flat.size
    return total


def encode_only(tok: Any, docs: Iterable[tuple[str, str]], batch: int) -> None:
    """Only the Rust ``encode_batch`` call, results thrown away."""
    for chunk in batches(docs, batch):
        tok.encode_batch([t for _, t in chunk], add_special_tokens=False)


def hand_written(
    tok: Any, docs: Iterable[tuple[str, str]], out: Path, dtype: str, eos: int, batch: int
) -> int:
    """What people write by hand (nanoGPT ``prepare.py`` style), made reasonably fast.

    One ``.bin`` file, EOS after each document, nothing else: no index of documents,
    no ids, no checksums, no fsync, no atomic rename, no resume.
    """
    total = 0
    with out.open("wb") as f:
        for chunk in batches(docs, batch):
            encs = tok.encode_batch([t for _, t in chunk], add_special_tokens=False)
            flat = np.fromiter(chain.from_iterable([*e.ids, eos] for e in encs), dtype=np.int64)
            f.write(flat.astype(dtype).tobytes())
            total += flat.size
    return total


def tokbin_write(
    root: Path, docs: Iterable[object], tokenizer: object, config: WriterConfig | None = None
) -> tokbin.WriteResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tokbin.Dataset(root).write("src", docs, tokenizer, config=config)


def clear(root: Path) -> None:
    """Remove the previous result outside the timed region."""
    shutil.rmtree(root / "src", ignore_errors=True)


def load_tokenizer(path: Path) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def adapter(path: Path) -> HFTokenizer:
    return HFTokenizer.from_file(path)


def token_table(tok: Any, docs: Docs, batch: int = 1024) -> dict[str, npt.NDArray[np.int64]]:
    table: dict[str, npt.NDArray[np.int64]] = {}
    for chunk in batches(docs, batch):
        texts = [t for _, t in chunk]
        for text, enc in zip(texts, tok.encode_batch(texts, add_special_tokens=False), strict=True):
            table[text] = np.asarray(enc.ids, dtype=np.int64)
    return table


def interleaved(
    repeat: int, variants: dict[str, tuple[Any, Path | None]]
) -> dict[str, dict[str, Any]]:
    """Like ``best_of`` for several variants, run in rounds (A B C, A B C, ...).

    Rounds spread slow drifts (heat, background load) over all variants equally.
    """
    samples: dict[str, list[float]] = {k: [] for k in variants}
    for _ in range(repeat):
        for name, (fn, root) in variants.items():
            if root is not None:
                clear(root)
            samples[name].append(time_once(fn))
    return {
        k: {"best": min(v), "median": float(np.median(v)), "samples": v} for k, v in samples.items()
    }


def best_of(repeat: int, fn: Any, root: Path | None = None) -> dict[str, Any]:
    """Run ``fn`` ``repeat`` times; ``root/src`` is removed before each run, untimed."""
    samples = []
    for _ in range(repeat):
        if root is not None:
            clear(root)
        samples.append(time_once(fn))
    return {"best": min(samples), "median": float(np.median(samples)), "samples": samples}


# --- experiments -------------------------------------------------------------------------


def exp_breakdown(args: argparse.Namespace) -> dict[str, Any]:
    docs = list(tinystories(args.data, limit=args.n))
    n_chars = sum(len(t) for _, t in docs)
    out: dict[str, Any] = {"n_docs": len(docs), "n_chars": n_chars, "tokenizers": {}}
    for name, path in args.tokenizer.items():
        tok = load_tokenizer(path)
        hf = adapter(path)
        table = token_table(tok, docs)
        replay = replay_tokenizer(hf, table)
        n_tokens = sum(a.size for a in table.values())
        dtype = _dtype_for(hf.vocab_size)
        eos = hf.eos_id if hf.eos_id is not None else 0
        root = fresh_dir(WORK / "breakdown" / name)
        diy = root / "diy.bin"
        variants = {
            "generate": (lambda: generate(docs), None),
            "encode_batch": (lambda tok=tok: encode_only(tok, docs, 1024), None),
            "tokenize": (lambda tok=tok: tokenize_only(tok, docs, 1024), None),
            "hand_written": (
                lambda tok=tok, eos=eos, dtype=dtype: hand_written(
                    tok, docs, diy, dtype, eos, 1024
                ),
                None,
            ),
            "tokbin": (lambda hf=hf: tokbin_write(root, docs, hf), root),
            "tokbin_replay": (lambda replay=replay: tokbin_write(root, docs, replay), root),
            "replay_lookup": (lambda replay=replay: _replay_only(replay, docs), None),
        }
        row: dict[str, Any] = {"vocab_size": hf.vocab_size, "dtype": dtype, "n_tokens": n_tokens}
        row.update(interleaved(args.repeat, variants))
        row["disk_bytes"] = dir_size(root / "src")
        row["diy_bytes"] = diy.stat().st_size
        out["tokenizers"][name] = row
        print(
            f"breakdown {name}: "
            + json.dumps({k: v["best"] for k, v in row.items() if isinstance(v, dict)})
        )
    return out


def _replay_only(replay: Any, docs: Docs) -> None:
    for chunk in batches(docs, 1024):
        replay.encode_batch([t for _, t in chunk])


def _dtype_for(vocab: int) -> str:
    return "uint8" if vocab <= 256 else "uint16" if vocab <= 65536 else "uint32"


def exp_batch(args: argparse.Namespace) -> dict[str, Any]:
    docs = list(tinystories(args.data, limit=args.n))
    out: dict[str, Any] = {"n_docs": len(docs), "tokenizers": {}}
    sizes = [1, 4, 16, 64, 256, 1024, 4096, 16384]
    for name, path in args.tokenizer.items():
        hf = adapter(path)
        tok = load_tokenizer(path)
        root = WORK / "batch" / name
        rows = []
        for size in sizes:
            # One document at a time is slow: fewer documents keep the run short.
            subset = docs if size >= 64 else docs[: len(docs) // 8]
            config = WriterConfig(batch_docs=size)
            timings = interleaved(
                args.repeat,
                {
                    "write": (lambda c=config, s=subset: tokbin_write(root, s, hf, c), root),
                    "tokenize": (lambda s=subset, b=size: tokenize_only(tok, s, b), None),
                },
            )
            write, tokenize = timings["write"], timings["tokenize"]
            rows.append(
                {"batch_docs": size, "n_docs": len(subset), "write": write, "tokenize": tokenize}
            )
            print(
                f"batch {name} {size}: write {write['best']:.2f}s "
                f"tokenize {tokenize['best']:.2f}s ({len(subset)} docs)"
            )
        out["tokenizers"][name] = rows
    return out


def exp_threads(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"tokenizers": {}}
    counts = sorted({1, 2, 4, 8, os.cpu_count() or 1})
    for name, path in args.tokenizer.items():
        rows = []
        for threads in counts:
            env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
            cmd = [
                sys.executable,
                __file__,
                "_threads_child",
                "--data",
                str(args.data),
                "--n",
                str(args.n),
                "--repeat",
                str(args.repeat),
                "--tokenizer",
                f"{name}={path}",
            ]
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            row = json.loads(res.stdout.strip().splitlines()[-1])
            row["threads"] = threads
            rows.append(row)
            print(
                f"threads {name} {threads}: write {row['write']['best']:.2f}s "
                f"tokenize {row['tokenize']['best']:.2f}s"
            )
        out["tokenizers"][name] = rows
    return out


def threads_child(args: argparse.Namespace) -> None:
    ((name, path),) = args.tokenizer.items()
    docs = list(tinystories(args.data, limit=args.n))
    hf = adapter(path)
    tok = load_tokenizer(path)
    root = WORK / "threads" / name
    print(
        json.dumps(
            interleaved(
                args.repeat,
                {
                    "write": (lambda: tokbin_write(root, docs, hf), root),
                    "tokenize": (lambda: tokenize_only(tok, docs, 1024), None),
                },
            )
        )
    )


def _sentences(docs: Docs) -> Docs:
    out = []
    for doc_id, text in docs:
        parts = [p.strip() for p in text.replace("\n", " ").split(". ") if p.strip()]
        out.extend((f"{doc_id}/{k}", p) for k, p in enumerate(parts))
    return out


def _joined(docs: Docs, k: int) -> Docs:
    return [
        (docs[i][0] + f"+{k}", "\n\n".join(t for _, t in docs[i : i + k]))
        for i in range(0, len(docs), k)
    ]


def exp_doclen(args: argparse.Namespace) -> dict[str, Any]:
    base = list(tinystories(args.data, limit=args.n))
    variants = {
        "sentence": _sentences(base),
        "story": base,
        "8 stories": _joined(base, 8),
        "64 stories": _joined(base, 64),
        "1024 stories": _joined(base, 1024),
    }
    out: dict[str, Any] = {"tokenizers": {}}
    for name, path in args.tokenizer.items():
        tok = load_tokenizer(path)
        hf = adapter(path)
        rows = []
        for label, docs in variants.items():
            table = token_table(tok, docs, batch=1024 if len(docs) > 10_000 else 16)
            replay = replay_tokenizer(hf, table)
            root = WORK / "doclen" / name
            n_tokens = sum(table[t].size for _, t in docs)
            timings = interleaved(
                args.repeat,
                {
                    "write": (lambda d=docs: tokbin_write(root, d, hf), root),
                    "tokenize": (lambda d=docs: tokenize_only(tok, d, 1024), None),
                    "writer_only": (lambda d=docs, r=replay: tokbin_write(root, d, r), root),
                },
            )
            write, tokenize, replayed = (
                timings["write"],
                timings["tokenize"],
                timings["writer_only"],
            )
            files = {p.name: p.stat().st_size for p in (root / "src").iterdir() if p.is_file()}
            shard_bytes = sum(v for k, v in files.items() if k.endswith(".bin"))
            rows.append(
                {
                    "variant": label,
                    "n_docs": len(docs),
                    "n_tokens": n_tokens,
                    "write": write,
                    "tokenize": tokenize,
                    "writer_only": replayed,
                    "shard_bytes": shard_bytes,
                    "files": files,
                }
            )
            print(
                f"doclen {name} {label}: {len(docs)} docs, write {write['best']:.2f}s, "
                f"tokenize {tokenize['best']:.2f}s, writer-only {replayed['best']:.2f}s"
            )
        out["tokenizers"][name] = rows
    return out


def exp_shards(args: argparse.Namespace) -> dict[str, Any]:
    ((name, path), *_) = args.tokenizer.items()
    docs = list(tinystories(args.data, limit=args.n))
    tok, hf = load_tokenizer(path), adapter(path)
    replay = replay_tokenizer(hf, token_table(tok, docs))
    root = WORK / "shards"
    rows = []
    for mib in (1, 4, 16, 64, 256, 1024):
        config = WriterConfig(shard_bytes=mib * 1024**2)
        timing = best_of(args.repeat, lambda c=config: tokbin_write(root, docs, replay, c), root)
        clear(root)
        result = tokbin_write(root, docs, replay, config)
        rows.append(
            {
                "shard_mib": mib,
                "n_shards": result.stats.n_shards,
                "write": timing,
                "n_tokens": result.stats.n_items,
            }
        )
        print(f"shards {mib} MiB: {result.stats.n_shards} shards, {timing['best']:.2f}s")
    return {"tokenizer": name, "n_docs": len(docs), "rows": rows}


def _interrupt_after(docs: Docs, n: int) -> Iterator[tuple[str, str]]:
    for i, doc in enumerate(docs):
        if i == n:
            raise Interrupted
        yield doc


def exp_resume(args: argparse.Namespace) -> dict[str, Any]:
    ((name, path), *_) = args.tokenizer.items()
    docs = list(tinystories(args.data, limit=args.n))
    hf = adapter(path)
    config = WriterConfig(shard_bytes=16 * 1024**2)
    root = fresh_dir(WORK / "resume")
    full = best_of(args.repeat, lambda: tokbin_write(root, docs, hf, config), root)
    rows = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
        cut = int(len(docs) * fraction)
        samples, first_part = [], []
        for _ in range(args.repeat):
            fresh_dir(root)
            t0 = time.perf_counter()
            try:
                tokbin_write(root, _interrupt_after(docs, cut), hf, config)
            except Interrupted:
                pass
            first_part.append(time.perf_counter() - t0)
            cp = json.loads((root / "src.partial" / "checkpoint.json").read_text())
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tokbin.Dataset(root).write("src", docs, hf, config=config, resume=True)
            samples.append(time.perf_counter() - t0)
        # The skip part alone: consumed inputs are iterated and their ids checked.
        consumed = cp["n_input_consumed"]
        rows.append(
            {
                "fraction": fraction,
                "cut_docs": cut,
                "consumed": consumed,
                "resume": {"best": min(samples), "samples": samples},
                "first_part": {"best": min(first_part), "samples": first_part},
                "lost_docs": cut - consumed,
            }
        )
        print(
            f"resume at {fraction:.0%}: consumed {consumed} of {cut}, "
            f"first {min(first_part):.2f}s + resume {min(samples):.2f}s "
            f"(full {full['best']:.2f}s)"
        )
    return {"tokenizer": name, "n_docs": len(docs), "shard_mib": 16, "full": full, "rows": rows}


def exp_inputs(args: argparse.Namespace) -> dict[str, Any]:
    ((name, path), *_) = args.tokenizer.items()
    docs = list(tinystories(args.data, limit=args.n))
    hf = adapter(path)
    base = fresh_dir(WORK / "inputs")
    jsonl_dir = fresh_dir(base / "jsonl")
    with (jsonl_dir / "data.jsonl").open("w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(json.dumps({"id": doc_id, "text": text}) + "\n")
    txt_dir = fresh_dir(base / "txt")
    t0 = time.perf_counter()
    for i, (_, text) in enumerate(docs):
        sub = txt_dir / f"{i // 1000:04d}"
        if i % 1000 == 0:
            sub.mkdir()
        (sub / f"{i:07d}.txt").write_text(text, encoding="utf-8")
    txt_create = time.perf_counter() - t0
    out_root = base / "out"

    def build(kind: str, src: Path) -> None:
        recipe = tokbin.BuildRecipe(input_kind=kind, input_path=src, tokenizer=path)  # type: ignore[arg-type]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokbin.build_source(out_root / "src", recipe)

    rows = {
        "in_memory": best_of(args.repeat, lambda: tokbin_write(out_root, docs, hf), out_root),
        "jsonl": best_of(args.repeat, lambda: build("jsonl", jsonl_dir), out_root),
        "txt": best_of(args.repeat, lambda: build("txt", txt_dir), out_root),
        "read_jsonl": best_of(args.repeat, lambda: generate(tokbin.jsonl(jsonl_dir))),
        "read_txt": best_of(args.repeat, lambda: generate(tokbin.txt_dir(txt_dir))),
    }
    for k, v in rows.items():
        print(f"inputs {k}: {v['best']:.2f}s")
    return {
        "tokenizer": name,
        "n_docs": len(docs),
        "txt_files_created_s": txt_create,
        "jsonl_bytes": (jsonl_dir / "data.jsonl").stat().st_size,
        "rows": rows,
    }


def exp_idhashes(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for n in (1_000_000, 4_000_000, 16_000_000):
        cmd = [sys.executable, __file__, "_idhashes_child", "--n", str(n)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        row = json.loads(res.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"idhashes {n}: {row}")
    return {"rows": rows}


def idhashes_child(args: argparse.Namespace) -> None:
    import psutil

    proc = psutil.Process()
    ids = [f"TinyStoriesV2-GPT4-train.txt:{i}" for i in range(args.n)]
    before = proc.memory_info().rss
    hashes = IdHashes()
    t0 = time.perf_counter()
    for doc_id in ids:
        hashes.add(doc_id)
    add = time.perf_counter() - t0
    held = proc.memory_info().rss - before
    t0 = time.perf_counter()
    peak_before = peak_rss_bytes()
    report = hashes.report(lambda i: ids[i])
    check = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "n": args.n,
                "add_s": add,
                "add_ns_per_id": add / args.n * 1e9,
                "held_bytes": held,
                "report_s": check,
                "duplicates": report.count,
                "peak_growth_in_report": max(0, peak_rss_bytes() - peak_before),
            }
        )
    )


def exp_full(args: argparse.Namespace) -> dict[str, Any]:
    """Each run is a fresh subprocess: memory numbers do not carry over between runs.

    Variants run in rounds (tokbin, hand-written, tokenize only; again), so a slow
    drift of the machine affects them alike.
    """
    out: dict[str, Any] = {"tokenizers": {}}
    for name, path in args.tokenizer.items():
        runs: dict[str, list[dict[str, Any]]] = {"tokbin": [], "hand_written": [], "tokenize": []}
        for _ in range(args.repeat):
            for variant in runs:
                cmd = [
                    sys.executable,
                    __file__,
                    "_full_child",
                    "--data",
                    str(args.data),
                    "--tokenizer",
                    f"{name}={path}",
                    "--variant",
                    variant,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                run = json.loads(res.stdout.strip().splitlines()[-1])
                runs[variant].append(run)
                print(
                    f"full {name} {variant}: {run['seconds']:.1f}s, "
                    f"peak {run['peak_rss'] / 2**20:.0f} MiB",
                    flush=True,
                )
        out["tokenizers"][name] = runs
    return out


def full_child(args: argparse.Namespace) -> None:
    ((name, path),) = args.tokenizer.items()
    hf = adapter(path)
    tok = load_tokenizer(path)
    root = WORK / "full" / name
    progress: list[tuple[float, int]] = []
    t0 = time.perf_counter()

    def tracked() -> Iterator[tuple[str, str]]:
        for i, doc in enumerate(tinystories(args.data)):
            if i % 20_000 == 0:
                progress.append((time.perf_counter() - t0, i))
            yield doc

    if args.variant == "tokbin":
        clear(root)
    result: dict[str, Any] = {}
    with RssSampler(interval=0.5) as rss:
        t0 = time.perf_counter()
        if args.variant == "tokbin":
            written = tokbin_write(root, tracked(), hf)
            result["stats"] = {
                "n_docs": written.stats.n_docs,
                "n_items": written.stats.n_items,
                "n_shards": written.stats.n_shards,
                "n_split_docs": written.stats.n_split_docs,
            }
        elif args.variant == "hand_written":
            diy = root / "hand_written.bin"
            diy.parent.mkdir(parents=True, exist_ok=True)
            eos = hf.eos_id if hf.eos_id is not None else 0
            hand_written(tok, tracked(), diy, _dtype_for(hf.vocab_size), eos, 1024)
            result["bytes"] = diy.stat().st_size
            diy.unlink()
        else:
            tokenize_only(tok, tracked(), 1024)
        seconds = time.perf_counter() - t0
    if args.variant == "tokbin":
        result["files"] = {
            p.name: p.stat().st_size for p in (root / "src").iterdir() if p.is_file()
        }
    result.update(
        {
            "seconds": seconds,
            "progress": progress,
            "rss": rss.samples,
            "peak_rss": peak_rss_bytes(),
        }
    )
    print(json.dumps(result))


EXPERIMENTS = {
    "breakdown": exp_breakdown,
    "batch": exp_batch,
    "threads": exp_threads,
    "doclen": exp_doclen,
    "shards": exp_shards,
    "resume": exp_resume,
    "inputs": exp_inputs,
    "idhashes": exp_idhashes,
    "full": exp_full,
}


def _tokenizers(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        name, _, path = value.partition("=")
        out[name] = Path(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("only", nargs="*", help=f"experiments: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tokenizer", action="append", default=[], help="name=path")
    parser.add_argument("--n", type=int, default=200_000, help="documents per experiment")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.tokenizer = _tokenizers(args.tokenizer)
    if args.only == ["_threads_child"]:
        threads_child(args)
        return
    if args.only == ["_full_child"]:
        full_child(args)
        return
    if args.only == ["_idhashes_child"]:
        idhashes_child(args)
        return
    for name in args.only or list(EXPERIMENTS):
        started = time.perf_counter()
        data = EXPERIMENTS[name](args)
        save_json(
            args.out / f"write-{name}.json",
            {
                "experiment": name,
                "env": environment(),
                "n": args.n,
                "repeat": args.repeat,
                "tokenizers": {k: str(v) for k, v in args.tokenizer.items()},
                "wall_s": time.perf_counter() - started,
                "data": data,
            },
        )


if __name__ == "__main__":
    main()
