"""Help screens and usage errors in the style of uv and ruff (spec 14.1).

argparse still parses the command line; only its text output is replaced:

- sections with capitalized headers, commands grouped by purpose, examples;
- ``<VALUE>`` placeholders, ``[default: ...]`` and ``[possible values: ...]`` dimmed;
- commands and flags in the accent color;
- usage errors as ``error:`` with the offending value, a tip and the usage line.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, TextIO, cast

from tokbin.cli.render import Style, detect_style

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

__all__ = ["GLOBAL_OPTIONS", "Parser", "error_lines"]

#: Title of the argument group with the options every command accepts; shown last.
GLOBAL_OPTIONS = "Global options"

_TITLES = {"positional arguments": "Arguments", "options": "Options"}
#: A wider left column puts the description on the next line.
_MAX_LEFT = 30
_MAX_WIDTH = 100
_BRACKETS = re.compile(r"\[[^\]]*\]")
_QUOTED = re.compile(r"'[^']*'")

Painter = Callable[[str], str]
#: One line of a two-column list: plain left text, painted left text, description.
Row = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _Problem:
    message: str
    details: tuple[str, ...] = ()
    tip: str | None = None


class Parser(argparse.ArgumentParser):
    """argparse that prints tokbin's help screens and usage errors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - argparse's signature
        super().__init__(*args, **kwargs)
        self.no_color = False
        #: Shown in the headline of the main help.
        self.version: str | None = None
        #: The main help lists commands by group: ``(title, [(name, help), ...])``.
        self.command_groups: Sequence[tuple[str, Sequence[tuple[str, str]]]] = ()
        #: ``(command, arguments)``: the command is painted with the accent.
        self.examples: Sequence[tuple[str, str]] = ()
        #: The last line, painted with the style of the stream.
        self.footer: Callable[[Style], str] | None = None
        self.subcommands: dict[str, Parser] = {}

    # --- help -------------------------------------------------------------------------

    def format_help(self) -> str:
        return "\n".join(self.help_lines(detect_style(sys.stdout, no_color=True))) + "\n"

    def print_help(self, file: SupportsWrite[str] | None = None) -> None:
        stream = cast("TextIO", file if file is not None else sys.stdout)
        style = detect_style(stream, no_color=self.no_color)
        stream.write("\n".join(self.help_lines(style)) + "\n")

    def format_usage(self) -> str:
        return self.usage_line(detect_style(sys.stdout, no_color=True)) + "\n"

    def print_usage(self, file: SupportsWrite[str] | None = None) -> None:
        stream = cast("TextIO", file if file is not None else sys.stdout)
        stream.write(self.usage_line(detect_style(stream, no_color=self.no_color)) + "\n")

    def usage_line(self, s: Style) -> str:
        parts = [s.accent(self.prog)]
        if any(a.option_strings and a.help != argparse.SUPPRESS for a in self._actions):
            parts.append(s.dim("[OPTIONS]"))
        parts += [s.dim(_placeholder(a)) for a in self._actions if not a.option_strings]
        return f"{s.bold('Usage:')} {' '.join(parts)}"

    def help_lines(self, s: Style) -> list[str]:
        width = max(60, min(shutil.get_terminal_size().columns, _MAX_WIDTH))
        lines: list[str] = []
        if self.version is not None:
            lines.append(f"{s.accent(self.prog)} {self.version} {s.dot} {self.description}")
        elif self.description:
            lines.append(self.description)
        lines += ["", self.usage_line(s)]
        if self.command_groups:
            lines += _sections(
                [
                    (title, [(name, s.accent(name), text) for name, text in commands])
                    for title, commands in self.command_groups
                ],
                s,
                width,
            )
        lines += _sections(self._argument_sections(s), s, width)
        if self.examples:
            lines += ["", s.bold("Examples:")]
            lines += [f"  {s.dim('$')} {s.accent(cmd)} {rest}" for cmd, rest in self.examples]
        if self.footer:
            lines += ["", self.footer(s)]
        return lines

    def _argument_sections(self, s: Style) -> list[tuple[str, list[Row]]]:
        groups = sorted(self._action_groups, key=lambda g: g.title == GLOBAL_OPTIONS)
        sections = []
        for group in groups:
            rows = [
                _argument_row(a, s)
                for a in group._group_actions
                if a.help != argparse.SUPPRESS and not isinstance(a, argparse._SubParsersAction)
            ]
            if rows:
                title = group.title or "Options"
                sections.append((_TITLES.get(title, title), rows))
        return sections

    # --- errors -----------------------------------------------------------------------

    def error(self, message: str) -> NoReturn:
        self._fail(self._explain(message))

    def unknown_command(self, value: str) -> NoReturn:
        close = difflib.get_close_matches(value, list(self.subcommands), n=1)
        tip = f"a similar command exists: {close[0]!r}" if close else None
        self._fail(_Problem(f"unrecognized command {value!r}", tip=tip))

    def _fail(self, problem: _Problem) -> NoReturn:
        s = detect_style(sys.stderr, no_color=self.no_color)
        lines = error_lines(
            s,
            problem.message,
            details=problem.details,
            tip=problem.tip,
            usage=self.usage_line(s),
            prog=self.prog,
        )
        sys.stderr.write("\n".join(lines) + "\n")
        self.exit(2)

    def _explain(self, message: str) -> _Problem:
        """Turn an argparse message into a problem with the values quoted."""
        if m := re.fullmatch(r"the following arguments are required: (.+)", message):
            names = [self._shown(n, bare=True) for n in m.group(1).split(", ")]
            return _Problem("the following required arguments were not provided:", tuple(names))
        if m := re.fullmatch(r"unrecognized arguments: (.+)", message):
            extras = m.group(1).split()
            if len(extras) == 1:
                return _Problem(f"unexpected argument {extras[0]!r} found")
            return _Problem(f"unexpected arguments: {', '.join(map(repr, extras))}")
        m = re.fullmatch(r"argument (\S+): (.*)", message, re.DOTALL)
        if m is None:
            return _Problem(message)
        name, rest = m.groups()
        action = self._action(name)
        choice = re.match(r"invalid choice: ('[^']*')", rest)
        if isinstance(action, argparse._SubParsersAction) and choice:
            self.unknown_command(choice.group(1)[1:-1])
        shown = self._shown(name)
        if choice:
            values = ", ".join(map(str, action.choices or ())) if action else ""
            details = (f"[possible values: {values}]",) if values else ()
            return _Problem(f"invalid value {choice.group(1)} for {shown}", details)
        if rest.startswith("expected "):
            return _Problem(f"a value is required for {shown} but none was supplied")
        if other := re.match(r"not allowed with argument (\S+)", rest):
            return _Problem(f"the argument {shown} cannot be used with {self._shown(other[1])}")
        if value := re.match(r"invalid \w+ value: ('.*')", rest):
            return _Problem(f"invalid value {value.group(1)} for {shown}")
        return _Problem(f"invalid value for {shown}: {rest}")

    def _action(self, name: str) -> argparse.Action | None:
        for action in self._actions:
            if name in ("/".join(action.option_strings), action.metavar, action.dest):
                return action
        return None

    def _shown(self, name: str, *, bare: bool = False) -> str:
        """``'--split <SPLIT>'``; ``<TARGET>`` for a positional with ``bare``."""
        action = self._action(name)
        if action is None:
            text = name
        elif action.option_strings:
            flag = max(action.option_strings, key=len)
            text = flag if action.nargs == 0 else f"{flag} <{_metavar(action)}>"
        else:
            text = _placeholder(action)
        return text if bare else f"'{text}'"


def error_lines(
    s: Style,
    message: str,
    *,
    details: Sequence[str] = (),
    tip: str | None = None,
    usage: str | None = None,
    prog: str,
) -> list[str]:
    """A usage error: values in quotes are highlighted, a tip and the usage follow."""
    lines = [f"{s.red('error:')} {_paint_quoted(message, s.yellow)}"]
    lines += [f"  {s.green(d)}" for d in details]
    if tip:
        lines += ["", f"  {s.green('tip:')} {_paint_quoted(tip, s.green)}"]
    if usage:
        lines += ["", usage]
    hint = s.accent(f"'{prog} --help'")
    lines += ["", f"For more information, try {hint}."]
    return lines


# --- layout ---------------------------------------------------------------------------


def _metavar(action: argparse.Action) -> str:
    return action.metavar if isinstance(action.metavar, str) else action.dest.upper()


def _placeholder(action: argparse.Action) -> str:
    """``<TARGET>``, ``[ROOT]`` for an optional positional, ``<COMMAND>``."""
    if isinstance(action, argparse._SubParsersAction):
        return "<COMMAND>"
    name = _metavar(action).strip("<>").upper()
    return f"[{name}]" if action.nargs == "?" else f"<{name}>"


def _argument_row(action: argparse.Action, s: Style) -> Row:
    text = action.help or ""
    if action.choices and not isinstance(action, argparse._SubParsersAction):
        text += f" [possible values: {', '.join(map(str, action.choices))}]"
    if not action.option_strings:
        name = _placeholder(action)
        return name, s.dim(name), text
    shorts = [o for o in action.option_strings if not o.startswith("--")]
    flags = shorts + [o for o in action.option_strings if o.startswith("--")]
    # Long-only flags line up with the long flags after "-h, ".
    indent = "" if shorts else "    "
    plain = indent + ", ".join(flags)
    painted = indent + ", ".join(s.accent(f) for f in flags)
    if action.nargs != 0:
        value = f"<{_metavar(action)}>"
        plain, painted = f"{plain} {value}", f"{painted} {s.dim(value)}"
    return plain, painted, text


def _sections(sections: list[tuple[str, list[Row]]], s: Style, width: int) -> list[str]:
    """Titled two-column lists sharing one column width."""
    rows = [row for _, section in sections for row in section]
    if not rows:
        return []
    column = min(max(len(plain) for plain, _, _ in rows), _MAX_LEFT)
    indent = " " * (2 + column + 2)
    lines = []
    for title, section in sections:
        lines += ["", s.bold(f"{title}:")]
        for plain, painted, text in section:
            wrapped = _wrap(_spans(text, s), width - len(indent)) or [""]
            if len(plain) > column:
                lines.append(f"  {painted}")
                lines += [indent + w for w in wrapped if w]
                continue
            lines.append(f"  {painted}{' ' * (column - len(plain))}  {wrapped[0]}".rstrip())
            lines += [indent + w for w in wrapped[1:]]
    return lines


def _spans(text: str, s: Style) -> list[tuple[str, Painter | None]]:
    """Words of a description; words inside ``[...]`` are dimmed."""
    spans: list[tuple[str, Painter | None]] = []
    pos = 0
    for m in _BRACKETS.finditer(text):
        spans += [(w, None) for w in text[pos : m.start()].split()]
        spans += [(w, s.dim) for w in m.group().split()]
        pos = m.end()
    spans += [(w, None) for w in text[pos:].split()]
    return spans


def _wrap(spans: list[tuple[str, Painter | None]], width: int) -> list[str]:
    """Greedy word wrap by the plain length; words are painted one by one."""
    lines: list[str] = []
    words: list[str] = []
    length = 0
    for word, paint in spans:
        if words and length + 1 + len(word) > width:
            lines.append(" ".join(words))
            words, length = [], 0
        length += len(word) + (1 if words else 0)
        words.append(paint(word) if paint else word)
    if words:
        lines.append(" ".join(words))
    return lines


def _paint_quoted(text: str, paint: Painter) -> str:
    return _QUOTED.sub(lambda m: paint(m.group()), text)
