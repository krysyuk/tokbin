"""``tokbin verify <path>``: sha256 of every shard, completeness and sizes (spec 14.3)."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokbin.cli.context import Context, Outcome, plural
from tokbin.cli.render import Cell, fmt_bytes, table
from tokbin.cli.views import dir_label, problem_block, status_line
from tokbin.ops.inspect import SourceInfo, describe
from tokbin.ops.verify import ShardCheck, VerifyProgress, VerifyReport, verify_source

NAME = "verify"
HELP = "Hash every shard and check all files of a source or a corpus"

_SHARD_TEXT = {
    "ok": "sha256 ok",
    "missing": "missing",
    "wrong_size": "wrong size",
    "corrupt": "sha256 mismatch",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="A source directory or a corpus directory")


def _verify(path: Path, ctx: Context) -> VerifyReport:
    progress = ctx.progress()

    def report(p: VerifyProgress) -> None:
        if progress is not None:
            progress.update(p.done_bytes, p.total_bytes, p.file)

    try:
        return verify_source(path, progress=report)
    finally:
        if progress is not None:
            progress.close()


def _shard_rows(shards: tuple[ShardCheck, ...], ctx: Context) -> list[str]:
    s = ctx.style
    rows = [
        [
            s.mark_cell("ok" if sh.status == "ok" else "error"),
            Cell(sh.name),
            Cell(fmt_bytes(sh.n_bytes), ">"),
            Cell(_SHARD_TEXT[sh.status], paint=s.dim if sh.status == "ok" else s.red),
        ]
        for sh in shards
    ]
    return table(rows, marked=True)


def _render_report(report: VerifyReport, ctx: Context) -> None:
    ctx.lines(_shard_rows(report.shards, ctx))
    for problem in report.problems:
        ctx.line()
        ctx.lines(problem_block(problem, ctx.style))


def _summary(reports: list[VerifyReport], ctx: Context) -> str:
    s = ctx.style
    shards = [sh for r in reports for sh in r.shards]
    bad = sum(1 for sh in shards if sh.status != "ok")
    problems = sum(len(r.problems) for r in reports)
    if not problems:
        checked = fmt_bytes(sum(r.n_bytes_checked for r in reports))
        return status_line(
            s, "ok", s.join("intact", plural(len(shards), "shard"), f"{checked} checked")
        )
    parts = ["damaged"]
    if bad:
        parts.append(f"{bad} of {plural(len(shards), 'shard')}")
    if problems > bad:
        parts.append(plural(problems - bad, "other problem"))
    return status_line(s, "error", s.join(*parts))


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    s = ctx.style
    found = describe(args.path)
    reports: list[VerifyReport] = []
    if isinstance(found, SourceInfo):
        report = _verify(Path(args.path), ctx)
        ctx.line(s.bold(dir_label(args.path)))
        ctx.line()
        _render_report(report, ctx)
        reports.append(report)
        data: dict[str, object] = report.to_dict()
    else:
        ctx.line(s.bold(dir_label(args.path)))
        skipped: list[str] = []
        for source in found.sources:
            ctx.line()
            if not source.splits and source.state == "partial":
                note = s.dim("unfinished write, nothing to verify")
                ctx.line(f"  {s.mark('skip')} {s.bold(source.name)}   {note}")
                skipped.append(source.name)
                continue
            report = _verify(source.path, ctx)
            ctx.line(f"  {s.bold(dir_label(source.name))}")
            ctx.lines(["  " + line for line in _shard_rows(report.shards, ctx)])
            for problem in report.problems:
                ctx.line()
                ctx.lines(["  " + line for line in problem_block(problem, s)])
            reports.append(report)
        data = {
            "path": str(found.path),
            "ok": all(r.ok for r in reports),
            "sources": [r.to_dict() for r in reports],
            "skipped": skipped,
        }
    ctx.line()
    ctx.line(_summary(reports, ctx))
    return Outcome(exit_code=0 if all(r.ok for r in reports) else 3, data=data)
