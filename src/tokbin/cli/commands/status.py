"""``tokbin status <path>``: the write state of a source, unfinished writes included."""

from __future__ import annotations

import argparse

from tokbin import codes
from tokbin.cli.commands.ls import exit_code as corpus_exit_code
from tokbin.cli.context import Context, Outcome
from tokbin.cli.render import Mark
from tokbin.cli.views import (
    dir_label,
    partial_summary,
    problem_block,
    source_mark,
    source_rows,
    status_line,
)
from tokbin.ops.inspect import CorpusInfo, SourceInfo, describe

NAME = "status"
HELP = "the write state of a source or of every source of a corpus"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="a source directory or a corpus directory")


def _render_source(info: SourceInfo, ctx: Context, label: str) -> None:
    s = ctx.style
    splits = ", ".join(sp.name for sp in info.splits)
    ctx.line(f"{s.bold(dir_label(label))}   " + s.join(info.state, splits))
    partial = info.partial
    if partial is not None:
        ctx.line()
        ctx.line(f"  {s.mark('warn')} unfinished write   {partial.path}")
        pad = " " * 23
        ctx.line(pad + s.dim(partial_summary(partial, s)))
        if partial.writer is not None:
            pid, host = partial.writer
            ctx.line(pad + s.dim(f"being written right now by process {pid} on {host}"))
        elif partial.problem is not None:
            ctx.line(pad + s.red(f"[{partial.problem.code}] {partial.problem.what}"))
            ctx.line(pad + s.dim(f"cannot be resumed; remove it with `tokbin clean {label}`"))
        elif partial.resumable:
            ctx.line(pad + s.dim("resume: repeat the write with resume=True and the same input"))
            ctx.line(pad + s.dim(f"discard: `tokbin clean {label}`"))
        else:
            ctx.line(pad + s.dim(f"no checkpoint; remove it with `tokbin clean {label}`"))
    warnings = [
        i for i in info.issues if i.level == "warning" and i.code != codes.PARTIAL_EXISTS.id
    ]
    if warnings:
        ctx.line()
        ctx.lines([f"  {s.mark('warn')} {s.yellow(f'[{i.code}] {i.message}')}" for i in warnings])
    for problem in info.problems:
        ctx.line()
        ctx.lines(problem_block(problem, s))
    ctx.line()
    ctx.line(status_line(s, source_mark(info), info.state, info.n_warnings))


def _render_corpus(corpus: CorpusInfo, ctx: Context, label: str) -> None:
    s = ctx.style
    ctx.line(s.bold(dir_label(label)))
    ctx.line()
    if corpus.sources:
        ctx.lines(source_rows(corpus.sources, s, with_state=True))
    else:
        ctx.line(s.dim("  no sources"))
    ctx.line()
    states = [src.state for src in corpus.sources]
    summary = s.join(*(f"{states.count(st)} {st}" for st in dict.fromkeys(states)))
    mark: Mark = "warn"
    if "corrupt" in states or "unsupported" in states:
        mark = "error"
    elif states and all(st == "complete" for st in states):
        mark = "ok"
    ctx.line(status_line(s, mark, summary or "no sources", corpus.n_warnings))


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    found = describe(args.path)
    if isinstance(found, SourceInfo):
        _render_source(found, ctx, args.path)
        code = 3 if found.state == "corrupt" else 0
        return Outcome(exit_code=code, data=found.to_dict(), n_warnings=found.n_warnings)
    _render_corpus(found, ctx, args.path)
    return Outcome(
        exit_code=corpus_exit_code(found), data=found.to_dict(), n_warnings=found.n_warnings
    )
