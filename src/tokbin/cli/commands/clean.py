"""``tokbin clean <path>``: remove the unfinished write of a source (spec 14.2)."""

from __future__ import annotations

import argparse

from tokbin.cli.context import Context, Outcome
from tokbin.cli.render import fmt_bytes
from tokbin.ops.clean import clean_source

NAME = "clean"
HELP = "remove the unfinished write (<name>.partial) of a source; the source itself stays"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="a source directory, or its .partial directory")


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    result = clean_source(args.path)
    s = ctx.style
    if not result.removed:
        ctx.line(f"{s.mark('skip')} nothing to clean: {result.path} has no unfinished write")
    for path in result.removed:
        ctx.line(f"{s.mark('ok')} removed {path}")
    if result.removed:
        ctx.line()
        ctx.line(f"{s.mark('ok')} {s.bold('Status:')} {fmt_bytes(result.n_bytes)} freed")
    return Outcome(exit_code=0, data=result.to_dict())
