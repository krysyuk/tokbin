"""``tokbin doctor``: version, mode, dependencies and features (spec 2.5)."""

from __future__ import annotations

import argparse

from tokbin.cli.context import Context, Outcome
from tokbin.cli.render import Cell, Mark, table
from tokbin.ops.doctor import doctor

NAME = "doctor"
HELP = "Show version, mode, dependencies and features"

_MARKS: dict[str, Mark] = {"ok": "ok", "missing": "error", "optional": "skip"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    del args
    report = doctor()
    s = ctx.style
    ctx.line(
        s.join(
            s.bold(f"tokbin {report.version}"),
            f"mode {report.mode}",
            f"Python {report.python}",
            report.platform,
        )
    )
    rows = []
    for pkg in report.packages:
        row = [s.mark_cell(_MARKS[pkg.state]), Cell(pkg.name)]
        if pkg.state == "ok":
            row.append(Cell(pkg.version or "installed"))
        else:
            row += [
                Cell("not installed", paint=s.dim),
                Cell(f"{s.arrow} needed for {pkg.unavailable}: {pkg.install}", paint=s.dim),
            ]
        rows.append(row)
    ctx.lines(table(rows, marked=True))
    if report.mode == "core":
        ctx.line()
        ctx.line(f"To write datasets: {s.bold('pip install tokbin')}")
    return Outcome(exit_code=0, data=report.to_dict())
