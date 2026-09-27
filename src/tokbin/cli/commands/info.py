"""``tokbin info <path>``: dtype, shards, tokenizer, mixture and remarks (spec 14.3)."""

from __future__ import annotations

import argparse
from typing import Literal

from tokbin.cli.commands.ls import exit_code as corpus_exit_code
from tokbin.cli.context import Context, Outcome, plural
from tokbin.cli.render import Cell, Mark, Style, fmt_bytes, fmt_int, table
from tokbin.cli.views import (
    detail_lines,
    dir_label,
    partial_summary,
    problem_block,
    source_rows,
    status_line,
)
from tokbin.ops.inspect import CorpusInfo, SourceInfo, SourceState, describe

NAME = "info"
HELP = "describe a corpus or a source: tokens, shards, tokenizer, mixture, remarks"

_STATE_TEXT: dict[SourceState, tuple[Mark, str]] = {
    "complete": ("ok", "ready for training"),
    "partial": ("warn", "unfinished write"),
    "corrupt": ("error", "damaged"),
    "outdated": ("warn", "outdated format; convert it with `tokbin migrate`"),
    "unsupported": ("error", "not readable by this tokbin"),
}
_SPLIT_HEADER: tuple[tuple[str, Literal["<", ">"]], ...] = (
    ("split", "<"),
    ("tokens", ">"),
    ("docs", ">"),
    ("shards", ">"),
    ("size", ">"),
    ("skipped", ">"),
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="a corpus directory or a source directory")


def _tokenizer(style: Style, tokenizer_id: str | None, tokenizer_hash: str | None) -> str:
    return style.join(tokenizer_id or "unnamed", tokenizer_hash[:8] if tokenizer_hash else "")


def _field(style: Style, label: str, value: str) -> str:
    return f"  {style.dim(label.ljust(11))} {value}"


def _render_source(info: SourceInfo, ctx: Context, label: str) -> None:
    s = ctx.style
    head = [f"{fmt_int(info.n_items)} tokens"] if info.splits else []
    ctx.line(f"{s.bold(dir_label(label))}   " + s.join(*head, info.state))
    ctx.line()
    if info.dtype is not None:
        vocab = f"vocab {fmt_int(info.vocab_size or 0)}"
        eos = f"eos {info.eos_id}" if info.eos_id is not None else "no eos"
        bos = f"bos {info.bos_id}" if info.bos_id is not None else "no bos"
        written = f"written by tokbin {info.tokbin_version}"
        ctx.line(_field(s, "dtype", s.join(info.dtype, vocab)))
        tok = _tokenizer(s, info.tokenizer_id, info.tokenizer_hash)
        ctx.line(_field(s, "tokenizer", s.join(tok, eos, bos)))
        ctx.line(_field(s, "format", s.join(f"schema {info.schema_version}", written)))
    if info.partial is not None:
        summary = partial_summary(info.partial, s)
        ctx.line(_field(s, "unfinished", s.join(str(info.partial.path), summary)))
    if info.splits:
        ctx.line()
        rows = [[Cell(text, align, s.dim) for text, align in _SPLIT_HEADER]]
        rows += [
            [
                Cell(sp.name),
                Cell(fmt_int(sp.n_items), ">"),
                Cell(fmt_int(sp.n_docs), ">"),
                Cell(fmt_int(sp.n_shards), ">"),
                Cell(fmt_bytes(sp.n_bytes), ">"),
                Cell(fmt_int(sp.n_skipped), ">"),
            ]
            for sp in info.splits
        ]
        ctx.lines(table(rows))
    notes = detail_lines(info, s, "  ")
    if notes:
        ctx.line()
        ctx.lines(notes)
    for problem in info.problems:
        ctx.line()
        ctx.lines(problem_block(problem, s))
    ctx.line()
    mark, text = _STATE_TEXT[info.state]
    ctx.line(status_line(s, mark, text, info.n_warnings))


def _corpus_status(corpus: CorpusInfo, style: Style) -> str:
    if corpus.ready:
        return status_line(style, "ok", "ready for training", corpus.n_warnings)
    states = [src.state for src in corpus.sources if src.state != "complete"]
    counts = [f"{states.count(st)} {st}" for st in dict.fromkeys(states)]
    if corpus.problems:
        counts.append(plural(len(corpus.problems), "problem"))
    if not corpus.sources:
        counts.append("no sources")
    bad = bool(corpus.problems) or any(st in ("corrupt", "unsupported") for st in states)
    text = style.join("not ready", *counts)
    return status_line(style, "error" if bad else "warn", text, corpus.n_warnings)


def _render_corpus(corpus: CorpusInfo, ctx: Context, label: str) -> None:
    s = ctx.style
    totals = s.join(f"{fmt_int(corpus.n_items)} tokens", plural(len(corpus.sources), "source"))
    ctx.line(f"{s.bold(dir_label(label))}   {totals}")
    ctx.line()
    if corpus.sources:
        ctx.lines(source_rows(corpus.sources, s, with_state=False))
    else:
        ctx.line(s.dim("  no sources"))
    ctx.line()
    if corpus.tokenizer_hash is not None:
        tok = _tokenizer(s, corpus.tokenizer_id, corpus.tokenizer_hash)
        if len(corpus.sources) > 1:
            tok = s.join(tok, "same in all sources")
        ctx.line(_field(s, "tokenizer", tok))
    elif any(src.tokenizer_hash for src in corpus.sources):
        ctx.line(_field(s, "tokenizer", s.yellow("differs between sources")))
    if corpus.mix is not None:
        ctx.line(_field(s, "mix", s.join(*(f"{name} {w:.2f}" for name, w in corpus.mix))))
    else:
        ctx.line(_field(s, "mix", s.dim("none (no mix.json)")))
    if corpus.issues:
        ctx.line()
        for issue in corpus.issues:
            paint = s.yellow if issue.level == "warning" else s.dim
            ctx.line(f"  {s.dim(s.branch)} {paint(f'[{issue.code}] {issue.message}')}")
    for problem in corpus.problems:
        ctx.line()
        ctx.lines(problem_block(problem, s))
    ctx.line()
    ctx.line(_corpus_status(corpus, s))


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
