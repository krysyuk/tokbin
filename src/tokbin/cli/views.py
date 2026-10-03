"""Text views shared by commands: source rows, detail lines, problem blocks."""

from __future__ import annotations

import os

from tokbin.cli.context import plural
from tokbin.cli.render import Cell, Mark, Style, fmt_bytes, fmt_count, table
from tokbin.ops.inspect import PartialInfo, Problem, SourceInfo
from tokbin.write.result import Issue

__all__ = [
    "dir_label",
    "partial_summary",
    "problem_block",
    "source_mark",
    "source_rows",
    "status_line",
]

_GAP = "   "


def dir_label(path: object) -> str:
    """A directory as the user wrote it, with a trailing separator of this OS."""
    text = str(path)
    return text if text.endswith(("/", "\\")) else text + os.sep


def source_mark(info: SourceInfo) -> Mark:
    if info.state in ("corrupt", "unsupported"):
        return "error"
    if info.state != "complete" or info.n_warnings:
        return "warn"
    return "ok"


def partial_summary(partial: PartialInfo, style: Style) -> str:
    """``train · 3 shards closed · 12.3M tokens · updated 2026-...``."""
    parts = [
        f"split {partial.split}" if partial.split else "",
        plural(partial.n_closed_shards, "shard") + " closed",
        f"{fmt_count(partial.n_items)} tokens" if partial.n_items is not None else "",
        f"updated {partial.updated_at}",
    ]
    return style.join(*parts)


def _issue_line(issue: Issue, style: Style) -> str:
    text = f"[{issue.code}] {issue.message}"
    if issue.count > 1:
        text += f" (x{issue.count:,})"
    return style.yellow(text) if issue.level == "warning" else style.dim(text)


def detail_lines(info: SourceInfo, style: Style, indent: str) -> list[str]:
    """Problems first, then warnings, then facts; one line each under the row."""
    lines = [
        f"{indent}{style.dim(style.branch)} {style.red(f'[{p.code}] {p.what}')}"
        for p in info.problems
    ]
    ordered = sorted(info.issues, key=lambda i: i.level != "warning")
    lines += [f"{indent}{style.dim(style.branch)} {_issue_line(i, style)}" for i in ordered]
    return lines


def source_rows(
    sources: tuple[SourceInfo, ...], style: Style, *, with_state: bool, details: bool = True
) -> list[str]:
    """One aligned row per source, each followed by its detail lines.

    A source that exists only as an unfinished write has no counters; its summary is a
    free cell, so it does not widen the columns of other rows.
    """
    rows: list[list[Cell]] = []
    for info in sources:
        row = [style.mark_cell(source_mark(info)), Cell(info.name, paint=style.bold)]
        if with_state:
            row.append(Cell(info.state))
        if info.splits:
            row += [
                Cell(f"{fmt_count(info.n_items)} tokens", ">"),
                Cell(f"{fmt_count(info.n_docs)} docs", ">"),
                Cell(plural(info.n_shards, "shard"), ">"),
                Cell(fmt_bytes(info.n_bytes), ">"),
            ]
        elif info.partial is not None and not info.problems:
            row.append(Cell(partial_summary(info.partial, style), paint=style.dim, free=True))
        rows.append(row)
    lines = table(rows, gap=_GAP, marked=True)
    if not details:
        return lines
    mark_width = max(len(r[0].text) for r in rows) if rows else 1
    indent = "  " + " " * mark_width + " "
    out: list[str] = []
    for line, info in zip(lines, sources, strict=True):
        out.append(line)
        out += detail_lines(info, style, indent)
    return out


def problem_block(problem: Problem, style: Style) -> list[str]:
    """The full three-part message of a problem, as an exception would print it."""
    return [
        style.red(f"[{problem.code}] {problem.what}"),
        f"  cause: {problem.why}",
        f"  fix:   {problem.fix}",
    ]


def status_line(style: Style, mark: Mark, text: str, n_warnings: int = 0) -> str:
    parts = [text]
    if n_warnings:
        parts.append(plural(n_warnings, "warning"))
    return f"{style.mark(mark)} {style.bold('Status:')} {style.join(*parts)}"
