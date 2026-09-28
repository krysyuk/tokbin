"""``tokbin migrate <path>``: convert a source to the current format schema (spec 13.3)."""

from __future__ import annotations

import argparse

from tokbin.cli.context import Context, Outcome
from tokbin.ops.migrate import migrate_source

NAME = "migrate"
HELP = "convert a source to the current format schema (a new copy unless --in-place)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="a source directory")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", help="where to write the converted copy (default: <name>-v<N>)")
    target.add_argument(
        "--in-place", action="store_true", help="replace the source itself instead of copying"
    )


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    result = migrate_source(args.path, args.out, in_place=args.in_place)
    s = ctx.style
    if result.migrated:
        ctx.line(
            f"{s.mark('ok')} {args.path}: schema {result.from_version} {s.arrow} "
            f"{result.to_version}, written to {result.path}"
        )
    else:
        ctx.line(
            f"{s.mark('ok')} {args.path} already uses schema {result.to_version}; nothing to do"
        )
    return Outcome(exit_code=0, data=result.to_dict())
