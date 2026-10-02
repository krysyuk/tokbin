"""``tokbin ls [root]``: sources of a corpus with states and sizes."""

from __future__ import annotations

import argparse

from tokbin.cli.context import Context, Outcome, plural
from tokbin.cli.render import fmt_bytes, fmt_int
from tokbin.cli.views import dir_label, source_rows
from tokbin.ops.inspect import CorpusInfo, inspect_corpus

NAME = "ls"
HELP = "List the sources of a corpus with states and sizes"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("root", nargs="?", default=".", help="Corpus directory [default: .]")


def exit_code(corpus: CorpusInfo) -> int:
    """3 when a source is damaged: an integrity check did not pass (spec 14.4)."""
    return 3 if any(s.state == "corrupt" for s in corpus.sources) else 0


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    corpus = inspect_corpus(args.root)
    s = ctx.style
    ctx.line(
        f"{s.bold(dir_label(args.root))}   "
        + s.join(
            plural(len(corpus.sources), "source"),
            f"{fmt_int(corpus.n_items)} tokens",
            fmt_bytes(corpus.n_bytes),
        )
    )
    ctx.line()
    if corpus.sources:
        ctx.lines(source_rows(corpus.sources, s, with_state=True, details=False))
    else:
        ctx.line(s.dim("  no sources"))
    return Outcome(exit_code=exit_code(corpus), data=corpus.to_dict(), n_warnings=corpus.n_warnings)
