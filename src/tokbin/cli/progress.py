"""Progress of long operations (spec 14.1).

On a terminal: one line, redrawn in place at most ten times a second. Elsewhere (a
pipe, a CI log): a plain line every ``interval`` seconds, so logs are not flooded and
short operations print nothing at all.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TextIO

__all__ = ["Progress"]

_REDRAW_SECONDS = 0.1


class Progress:
    """Reports ``done`` of ``total`` units with a label."""

    __slots__ = ("_clock", "_drawn", "_interval", "_last", "_start", "_stream", "_tty", "_unit")

    def __init__(
        self,
        stream: TextIO,
        *,
        tty: bool,
        unit: Callable[[int], str] = str,
        interval: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream
        self._tty = tty
        self._unit = unit
        self._interval = interval
        self._clock = clock
        self._start = clock()
        self._last = self._start
        #: Width of the line currently on screen (terminal mode).
        self._drawn = 0

    def _text(self, done: int, total: int, label: str) -> str:
        percent = 100 * done // total if total else 100
        return f"  {percent:3d}%  {self._unit(done)} / {self._unit(total)}  {label}"

    def update(self, done: int, total: int, label: str) -> None:
        now = self._clock()
        if self._tty:
            if now - self._last < _REDRAW_SECONDS and done < total:
                return
            self._last = now
            text = self._text(done, total, label)
            pad = max(self._drawn - len(text), 0)
            self._stream.write("\r" + text + " " * pad)
            self._stream.flush()
            self._drawn = len(text) + pad
        elif now - self._last >= self._interval:
            self._last = now
            self._stream.write(self._text(done, total, label) + "\n")
            self._stream.flush()

    def close(self) -> None:
        """Erase the terminal line, so that the final output starts clean."""
        if self._tty and self._drawn:
            self._stream.write("\r" + " " * self._drawn + "\r")
            self._stream.flush()
            self._drawn = 0
