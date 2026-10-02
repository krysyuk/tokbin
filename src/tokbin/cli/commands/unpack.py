"""``tokbin unpack <pack>``: decompress and verify a pack (spec 15)."""

from __future__ import annotations

import argparse

from tokbin.cli.context import Context, Outcome
from tokbin.cli.render import fmt_bytes
from tokbin.ops.pack import unpack_pack

NAME = "unpack"
HELP = "Unpack a .tbpack, check every sha256, publish the source"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pack", help="A <name>.tbpack directory or <name>.tbpack.tar file")
    parser.add_argument(
        "--into", metavar="DIR", help="Corpus directory to unpack into [default: next to the pack]"
    )
    parser.add_argument("--name", help="Source name [default: the name stored in the pack]")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing source")


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    progress = ctx.progress()

    def report(done: int, total: int, name: str) -> None:
        if progress is not None:
            progress.update(done, total, name)

    try:
        result = unpack_pack(
            args.pack, args.into, name=args.name, overwrite=args.overwrite, progress=report
        )
    finally:
        if progress is not None:
            progress.close()
    s = ctx.style
    m = result.manifest
    ctx.line(f"{s.mark('ok')} unpacked {args.pack} {s.arrow} {result.path}")
    ctx.line()
    ctx.line(
        f"{s.mark('ok')} {s.bold('Status:')} "
        + s.join(f"{len(m.files)} files", fmt_bytes(m.n_bytes), "sha256 ok")
    )
    return Outcome(exit_code=0, data=result.to_dict())
