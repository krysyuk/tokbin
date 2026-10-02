"""``tokbin rm <path> --yes``: delete a source and its unfinished write (spec 14.2)."""

from __future__ import annotations

import argparse

from tokbin.cli.context import Context, Outcome
from tokbin.cli.render import fmt_bytes
from tokbin.ops.remove import remove_source, source_size

NAME = "rm"
HELP = "Delete a source and its unfinished write (requires --yes)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="A source directory")
    parser.add_argument("--yes", action="store_true", help="Confirm the deletion")


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    s = ctx.style
    if not args.yes:
        size = fmt_bytes(source_size(args.path))
        ctx.err.write(
            f"{ctx.err_style.mark('error')} not removed: {args.path} ({size}) would be deleted "
            "for good; repeat with --yes to confirm\n"
        )
        return Outcome(exit_code=2, data={"path": str(args.path), "removed": []})
    result = remove_source(args.path)
    for path in result.removed:
        ctx.line(f"{s.mark('ok')} removed {path}")
    ctx.line()
    ctx.line(f"{s.mark('ok')} {s.bold('Status:')} {fmt_bytes(result.n_bytes)} freed")
    return Outcome(exit_code=0, data=result.to_dict())
