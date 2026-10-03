from __future__ import annotations

import io

import pytest

from tokbin.cli import render
from tokbin.cli.progress import Progress
from tokbin.cli.render import Cell, Style, detect_style, fmt_bytes, fmt_count, fmt_int, table


class _Stream(io.StringIO):
    def __init__(self, *, tty: bool, encoding: str = "utf-8") -> None:
        super().__init__()
        self._tty = tty
        self._encoding = encoding

    def isatty(self) -> bool:
        return self._tty

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (0, "0"),
        (999, "999"),
        (1000, "1.00K"),
        (12_000, "12.0K"),
        (890_400, "890.4K"),
        (999_950, "1.00M"),
        (1_204_881, "1.20M"),
        (412_000_000, "412.0M"),
        (3_000_000_000, "3.00B"),
        (5 * 10**15, "5000.0T"),
    ],
)
def test_fmt_count(n: int, text: str) -> None:
    assert fmt_count(n) == text


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KiB"),
        (512 * 1024**2, "512.0 MiB"),
        (1024**3 - 1, "1.0 GiB"),
        (int(1.5 * 1024**3), "1.5 GiB"),
    ],
)
def test_fmt_bytes(n: int, text: str) -> None:
    assert fmt_bytes(n) == text


def test_fmt_int() -> None:
    assert fmt_int(847392014) == "847,392,014"


def test_style_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake stream has no Windows console to switch into VT mode.
    monkeypatch.setattr(render, "_enable_windows_vt", lambda stream: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    assert detect_style(_Stream(tty=True), no_color=False) == Style(color=True, unicode=True)
    assert not detect_style(_Stream(tty=False), no_color=False).color
    assert not detect_style(_Stream(tty=True), no_color=True).color
    monkeypatch.setenv("TERM", "dumb")
    assert not detect_style(_Stream(tty=True), no_color=False).color
    monkeypatch.delenv("TERM")
    monkeypatch.setenv("NO_COLOR", "1")
    assert not detect_style(_Stream(tty=True), no_color=False).color
    # An empty NO_COLOR does not count (no-color.org).
    monkeypatch.setenv("NO_COLOR", "")
    assert detect_style(_Stream(tty=True), no_color=False).color
    assert not detect_style(_Stream(tty=False, encoding="cp1252"), no_color=False).unicode
    assert detect_style(_Stream(tty=False, encoding="UTF8"), no_color=False).unicode


def test_truecolor_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake stream has no Windows console to switch into VT mode.
    monkeypatch.setattr(render, "_enable_windows_vt", lambda stream: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert detect_style(_Stream(tty=True), no_color=False).truecolor
    # Without color there is nothing to render in 24 bits.
    assert not detect_style(_Stream(tty=False), no_color=False).truecolor
    monkeypatch.setenv("COLORTERM", "")
    assert not detect_style(_Stream(tty=True), no_color=False).truecolor


def test_accent() -> None:
    assert Style(color=False, unicode=True).accent("build") == "build"
    assert Style(color=True, unicode=True).accent("build") == "\x1b[1;38;5;223mbuild\x1b[0m"
    truecolor = Style(color=True, unicode=True, truecolor=True)
    assert truecolor.accent("build") == "\x1b[1;38;2;242;196;170mbuild\x1b[0m"


def test_marks() -> None:
    plain = Style(color=False, unicode=True)
    assert [plain.mark(k) for k in ("ok", "warn", "error")] == ["✔", "!", "✖"]
    ascii_ = Style(color=False, unicode=False)
    assert [ascii_.mark(k) for k in ("ok", "warn", "error")] == ["[ok]", "[!]", "[x]"]
    assert ascii_.join("a", "", "b") == "a | b"
    colored = Style(color=True, unicode=True)
    assert colored.mark("error") == "\x1b[31m✖\x1b[0m"
    assert colored.green("") == ""


def test_table_aligns_plain_text_and_paints_after() -> None:
    style = Style(color=True, unicode=False)
    rows = [
        [style.mark_cell("ok"), Cell("web", paint=style.bold), Cell("12", ">")],
        [style.mark_cell("warn"), Cell("books"), Cell("3", ">"), Cell("free text", free=True)],
    ]
    lines = table(rows, marked=True)
    plain = [line.replace("\x1b[0m", "").replace("\x1b[1m", "") for line in lines]
    plain = [line.replace("\x1b[32m", "").replace("\x1b[33m", "") for line in plain]
    assert plain == ["  [ok] web     12", "  [!]  books    3   free text"]


def test_progress_on_a_terminal() -> None:
    now = [0.0]
    stream = _Stream(tty=True)
    progress = Progress(stream, tty=True, clock=lambda: now[0])
    progress.update(10, 100, "a.bin")  # too soon after start: skipped
    now[0] = 0.5
    progress.update(50, 100, "a.bin")
    progress.update(60, 100, "a.bin")  # throttled
    progress.update(100, 100, "b")  # the final state is always drawn
    progress.close()
    out = stream.getvalue()
    assert out.split("\r")[1:] == [
        "   50%  50 / 100  a.bin",
        "  100%  100 / 100  b   ",
        " " * 23,
        "",
    ]


def test_progress_in_a_log() -> None:
    now = [0.0]
    stream = _Stream(tty=False)
    progress = Progress(stream, tty=False, interval=10, clock=lambda: now[0])
    for t in (1, 5, 11, 12, 25):
        now[0] = t
        progress.update(t, 25, "x")
    progress.close()
    assert stream.getvalue() == "   44%  11 / 25  x\n  100%  25 / 25  x\n"
