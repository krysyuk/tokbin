"""Storage study: bytes on disk, index overhead, and pack/unpack.

::

    uv run --group bench python benchmarks/study_storage.py \\
        --source deepseek=benchmarks/.work/storage/deepseek --source bpe12k=... \\
        --text TinyStoriesV2-GPT4-train.txt

Experiments (``results/storage-<name>.json``):

``layout``  every file of each source and its share of the total
``pack``    pack/unpack time and ratio per method and level, against compressing the text
"""

from __future__ import annotations

import argparse
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import tokbin
from tokbin.ops.compression import compress_stream

from bench_common import RESULTS, ROOT, dir_size, environment, save_json, time_once

WORK = ROOT / "benchmarks" / ".work" / "storage"


def exp_layout(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for name, path in args.source.items():
        info = tokbin.read_source(path)
        files = {
            str(p.relative_to(path)): p.stat().st_size
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
        out[name] = {
            "dtype": str(info.dtype),
            "n_items": len(info),
            "n_docs": info.n_docs,
            "files": files,
            "total": sum(files.values()),
        }
        print(f"layout {name}: {out[name]['total'] / 2**20:.1f} MiB")
    return out


def _compress_file(src: Path, method: str, level: int) -> tuple[int, float]:
    out = WORK / "text.cmp"
    t0 = time.perf_counter()
    with src.open("rb") as fin, out.open("wb") as fout:
        compress_stream(fin, fout, method, level=level)  # type: ignore[arg-type]
    seconds = time.perf_counter() - t0
    size = out.stat().st_size
    out.unlink()
    return size, seconds


def exp_pack(args: argparse.Namespace) -> dict[str, Any]:
    settings = [("zstd", 1), ("zstd", 3), ("zstd", 9), ("zstd", 19), ("lzma", 1), ("lzma", 3)]
    out: dict[str, Any] = {"sources": {}, "text": {}}
    for name, path in args.source.items():
        size = dir_size(path)
        rows = []
        for method, level in settings:
            if (method == "lzma" or level == 19) and size > args.slow_limit:
                rows.append({"method": method, "level": level, "skipped": True})
                continue
            pack_dir = WORK / "packs" / name
            shutil.rmtree(pack_dir, ignore_errors=True)
            pack_dir.mkdir(parents=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pack_s = time_once(
                    lambda m=method, lv=level, d=pack_dir: tokbin.pack_source(
                        path, d, method=m, level=lv
                    )
                )
            packed = next(pack_dir.iterdir())
            packed_bytes = dir_size(packed)
            unpack_to = WORK / "unpacked" / name
            shutil.rmtree(unpack_to.parent, ignore_errors=True)
            unpack_to.parent.mkdir(parents=True)
            unpack_s = time_once(lambda p=packed, u=unpack_to: tokbin.unpack_pack(p, u))
            rows.append(
                {
                    "method": method,
                    "level": level,
                    "bytes": size,
                    "packed": packed_bytes,
                    "ratio": packed_bytes / size,
                    "pack_s": pack_s,
                    "unpack_s": unpack_s,
                }
            )
            print(
                f"pack {name} {method}-{level}: {packed_bytes / size:.3f}, "
                f"{pack_s:.1f}s / {unpack_s:.1f}s"
            )
            shutil.rmtree(pack_dir)
            shutil.rmtree(unpack_to.parent)
        out["sources"][name] = rows
    if args.text is not None:
        size = args.text.stat().st_size
        for method, level in (("zstd", 3), ("zstd", 19)):
            if level == 19 and size > args.slow_limit:
                continue
            packed, seconds = _compress_file(args.text, method, level)
            out["text"][f"{method}-{level}"] = {"bytes": size, "packed": packed, "seconds": seconds}
            print(f"text {method}-{level}: {packed / size:.3f}")
    return out


EXPERIMENTS = {"layout": exp_layout, "pack": exp_pack}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("only", nargs="*", help=f"experiments: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--source", action="append", default=[], help="name=path")
    parser.add_argument("--text", type=Path, help="the raw text the sources were made from")
    parser.add_argument(
        "--slow-limit",
        type=int,
        default=600 * 2**20,
        help="skip lzma and zstd-19 for inputs larger than this (bytes)",
    )
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.source = {k: Path(v) for k, _, v in (s.partition("=") for s in args.source)}
    WORK.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(EXPERIMENTS):
        started = time.perf_counter()
        data = EXPERIMENTS[name](args)
        save_json(
            args.out / f"storage-{name}.json",
            {
                "experiment": name,
                "env": environment(),
                "sources": {k: str(v) for k, v in args.source.items()},
                "wall_s": time.perf_counter() - started,
                "data": data,
            },
        )


if __name__ == "__main__":
    main()
