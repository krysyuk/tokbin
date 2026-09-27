"""Entry point of the ``tokbin`` CLI (spec 14).

The CLI is a thin layer: every command calls public library functions and renders
the dataclasses they return. Exit codes (spec 14.4):

===  ========================================================
 0   success, warnings included
 1   runtime error; also warnings under ``--strict``
 2   wrong usage of the command
 3   an integrity check did not pass
 4   a required package is missing
130  interrupted by the user
===  ========================================================

With ``--json`` stdout carries exactly one JSON document of a stable shape::

    {"tokbin": "0.2.0", "command": "info", "mode": "full", "exit_code": 0,
     "result": {...}, "warnings": [...], "error": null}
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence

from tokbin._deps import detect_mode
from tokbin._version import get_version
from tokbin.cli.commands import COMMANDS
from tokbin.cli.context import Context, Outcome
from tokbin.errors import DependencyError, IntegrityError, InternalError, TokbinError

__all__ = ["EXIT_DEPENDENCY", "EXIT_ERROR", "EXIT_INTEGRITY", "EXIT_INTERRUPTED", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTEGRITY = 3
EXIT_DEPENDENCY = 4
EXIT_INTERRUPTED = 130

#: Version of the ``--json`` envelope; bumped only on incompatible changes.
JSON_FORMAT = 1


def _common_options(default: object) -> argparse.ArgumentParser:
    """Options accepted both before and after the command name.

    The main parser and every subparser get their own copies: argparse shares action
    objects between a parent and its children, and a default set on one would leak
    into the others. Subparsers use ``SUPPRESS``, so a flag given before the command
    is not reset by the subcommand's default.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=default, help="machine-readable output"
    )
    common.add_argument("--no-color", action="store_true", default=default, help="disable colors")
    common.add_argument(
        "--strict",
        action="store_true",
        default=default,
        help="exit with code 1 when there are warnings (for CI)",
    )
    return common


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokbin",
        description="Tokenized corpora as binary shards.",
        parents=[_common_options(False)],
    )
    parser.set_defaults(command=None)
    parser.add_argument("--version", action="version", version=f"tokbin {get_version()}")
    sub = parser.add_subparsers(title="commands", metavar="<command>")
    for module in COMMANDS:
        cmd = sub.add_parser(
            module.NAME,
            help=module.HELP,
            description=module.HELP,
            parents=[_common_options(argparse.SUPPRESS)],
        )
        module.add_arguments(cmd)
        cmd.set_defaults(command=module)
    return parser


def exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    if isinstance(exc, IntegrityError):
        return EXIT_INTEGRITY
    if isinstance(exc, DependencyError):
        return EXIT_DEPENDENCY
    return EXIT_ERROR


def _error_dict(exc: BaseException) -> dict[str, object]:
    if isinstance(exc, TokbinError):
        return {
            "type": type(exc).__name__,
            "code": exc.code.id,
            "what": exc.what,
            "why": exc.why,
            "fix": exc.fix,
            "where": exc.where,
        }
    return {"type": type(exc).__name__, "code": None, "what": str(exc), "why": None, "fix": None}


def _render_error(ctx: Context, exc: BaseException) -> None:
    s = ctx.err_style
    if isinstance(exc, KeyboardInterrupt):
        ctx.err.write(f"{s.mark('error')} interrupted\n")
        return
    first, *rest = str(exc).splitlines() or [type(exc).__name__]
    if not isinstance(exc, TokbinError):
        first = f"{type(exc).__name__}: {first}"
    ctx.err.write(f"{s.mark('error')} {s.red(first)}\n")
    for line in rest:
        ctx.err.write(line + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    if args.command is None:
        parser.print_usage(sys.stderr)
        sys.stderr.write("tokbin: error: a command is required (see `tokbin --help`)\n")
        return EXIT_USAGE

    ctx = Context.create(json=args.json, strict=args.strict, no_color=args.no_color)
    name = args.command.NAME
    mode = detect_mode()
    if mode == "core" and name != "doctor":
        ctx.line(
            ctx.style.dim(
                f"tokbin {get_version()} {ctx.style.dot} mode core: reading and checking only; "
                "`pip install tokbin` adds writing"
            )
        )
        ctx.line()

    outcome: Outcome | None = None
    error: BaseException | None = None
    # The CLI is the application, so it may decide how warnings are shown: they are
    # collected and printed after the output, and counted for --strict.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            outcome = args.command.run(args, ctx)
        except KeyboardInterrupt as exc:
            error = exc
        except (TokbinError, OSError) as exc:
            error = exc
        except Exception as exc:  # a bug in the CLI itself
            error = InternalError.wrap(exc, where=f"tokbin.cli.commands.{name}")
            error.__cause__ = exc

    shown = [str(w.message) for w in caught]
    code = exit_code_for(error) if error is not None else outcome.exit_code if outcome else 1
    n_warnings = len(shown) + (outcome.n_warnings if outcome else 0)
    if ctx.strict and n_warnings and code == EXIT_OK:
        code = EXIT_ERROR

    if ctx.json:
        document = {
            "format": JSON_FORMAT,
            "tokbin": get_version(),
            "command": name,
            "mode": mode,
            "exit_code": code,
            "result": outcome.data if outcome else None,
            "warnings": shown,
            "error": _error_dict(error) if error is not None else None,
        }
        ctx.out.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        return code

    # Warnings and errors go after the output: flush it first, the streams interleave.
    ctx.out.flush()
    s = ctx.err_style
    for text in shown:
        ctx.err.write(f"{s.mark('warn')} {s.yellow(text)}\n")
    if error is not None:
        _render_error(ctx, error)
    if ctx.strict and n_warnings and error is None and outcome and outcome.exit_code == EXIT_OK:
        ctx.err.write(f"{s.mark('error')} --strict: warnings are treated as errors\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
