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
from tokbin.cli.commands import COMMANDS, GROUPS
from tokbin.cli.context import Context, Outcome
from tokbin.cli.help import GLOBAL_OPTIONS, Parser
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


TAGLINE = "tokenized corpora as memory-mappable binary shards"

EXAMPLES = (
    ("tokbin build", "corpus/web --from-jsonl data/ --tokenizer tokenizer.json"),
    ("tokbin ls", "corpus"),
    ("tokbin verify", "corpus/web"),
)


def _common_options(default: object, *, version: bool = False) -> Parser:
    """Options accepted both before and after the command name.

    The main parser and every subparser get their own copies: argparse shares action
    objects between a parent and its children, and a default set on one would leak
    into the others. Subparsers use ``SUPPRESS``, so a flag given before the command
    is not reset by the subcommand's default.
    """
    common = Parser(add_help=False)
    group = common.add_argument_group(GLOBAL_OPTIONS)
    group.add_argument(
        "--json", action="store_true", default=default, help="Print one JSON document to stdout"
    )
    group.add_argument(
        "--no-color",
        action="store_true",
        default=default,
        help="Disable colors (also: NO_COLOR=1)",
    )
    group.add_argument(
        "--strict",
        action="store_true",
        default=default,
        help="Exit with code 1 on warnings (for CI)",
    )
    group.add_argument("-h", "--help", action="help", help="Print help")
    if version:
        group.add_argument(
            "-V",
            "--version",
            action="version",
            version=f"tokbin {get_version()}",
            help="Print version",
        )
    return common


def _parser(*, no_color: bool = False) -> Parser:
    parser = Parser(
        prog="tokbin",
        description=TAGLINE,
        add_help=False,
        parents=[_common_options(False, version=True)],
    )
    parser.set_defaults(command=None, topic=None)
    parser.no_color = no_color
    parser.version = get_version()
    parser.command_groups = [
        (title, [(m.NAME, m.HELP) for m in modules]) for title, modules in GROUPS
    ]
    parser.examples = EXAMPLES
    parser.footer = lambda s: (
        f"Use {s.accent('tokbin help <command>')} for more information on a command."
    )
    sub = parser.add_subparsers(metavar="<command>", parser_class=Parser)
    for module in COMMANDS:
        cmd = sub.add_parser(
            module.NAME,
            prog=f"tokbin {module.NAME}",
            description=module.HELP,
            add_help=False,
            parents=[_common_options(argparse.SUPPRESS)],
        )
        cmd.no_color = no_color
        module.add_arguments(cmd)
        cmd.set_defaults(command=module)
        parser.subcommands[module.NAME] = cmd
    helper = sub.add_parser(
        "help",
        prog="tokbin help",
        description="Show help for tokbin or one of its commands",
        add_help=False,
        parents=[_common_options(argparse.SUPPRESS)],
    )
    helper.no_color = no_color
    helper.add_argument("topic", nargs="?", metavar="COMMAND", help="Command to describe")
    helper.set_defaults(command=None)
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
    argv = list(sys.argv[1:] if argv is None else argv)
    # Parsed before argparse runs: --help and usage errors are printed while parsing.
    parser = _parser(no_color="--no-color" in argv)
    try:
        args, extras = parser.parse_known_args(argv)
        # Report extra arguments against the command they were given to.
        if extras:
            target = parser.subcommands[args.command.NAME] if args.command else parser
            target.error(f"unrecognized arguments: {' '.join(extras)}")
        if args.command is None:
            # `tokbin`, `tokbin help` and `tokbin help <command>` print help.
            if args.topic is None:
                parser.print_help()
                return EXIT_OK
            if args.topic not in parser.subcommands:
                parser.unknown_command(args.topic)
            parser.subcommands[args.topic].print_help()
            return EXIT_OK
    except SystemExit as exc:  # --help, --version and usage errors
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE

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

    handled = outcome.shown_codes if outcome else frozenset()
    shown = [
        text
        for text in (str(w.message) for w in caught)
        if not any(text.startswith(f"[{code}]") for code in handled)
    ]
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
