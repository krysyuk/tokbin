"""Entry point of the ``tokbin`` CLI.

Only ``--version`` is available for now; commands arrive in 0.2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tokbin._version import get_version

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokbin",
        description="Tokenized corpora as binary shards.",
    )
    parser.add_argument("--version", action="version", version=f"tokbin {get_version()}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    parser = _parser()
    parser.parse_args(argv)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
