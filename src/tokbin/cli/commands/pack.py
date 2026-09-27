"""``tokbin pack <path>``: compress a source for transfer (spec 15)."""

from __future__ import annotations

import argparse

from tokbin.cli.context import Context, Outcome
from tokbin.cli.render import fmt_bytes
from tokbin.ops.compression import METHODS
from tokbin.ops.pack import pack_source

NAME = "pack"
HELP = "compress every file of a source into <name>.tbpack (a directory, or one .tar)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="a complete source directory")
    parser.add_argument("--out", help="output directory (default: next to the source)")
    parser.add_argument("--tar", action="store_true", help="write one <name>.tbpack.tar file")
    parser.add_argument(
        "--method", choices=METHODS, help="compression (default: zstd if available, else lzma)"
    )
    parser.add_argument("--level", type=int, help="compression level (default: zstd 3, lzma 3)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing pack")


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    progress = ctx.progress()

    def report(done: int, total: int, name: str) -> None:
        if progress is not None:
            progress.update(done, total, name)

    try:
        result = pack_source(
            args.path,
            args.out,
            method=args.method,
            level=args.level,
            tar=args.tar,
            overwrite=args.overwrite,
            progress=report,
        )
    finally:
        if progress is not None:
            progress.close()
    s = ctx.style
    m = result.manifest
    ratio = m.stored_bytes / m.n_bytes if m.n_bytes else 1.0
    ctx.line(f"{s.mark('ok')} packed {args.path} {s.arrow} {result.path}")
    ctx.line()
    ctx.line(
        f"{s.mark('ok')} {s.bold('Status:')} "
        + s.join(
            f"{len(m.files)} files",
            f"{fmt_bytes(m.n_bytes)} {s.arrow} {fmt_bytes(m.stored_bytes)} ({ratio:.0%})",
            f"{m.method}-{m.level}",
        )
    )
    n_warnings = sum(1 for i in result.issues if i.level == "warning")
    return Outcome(exit_code=0, data=result.to_dict(), n_warnings=n_warnings)
