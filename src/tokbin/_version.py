"""Package version. The single source of truth is ``pyproject.toml``, read via metadata."""

from __future__ import annotations

import functools
from importlib import metadata

__all__ = ["DISTRIBUTION", "get_version"]

DISTRIBUTION = "tokbin-core"


@functools.cache
def get_version() -> str:
    """Version of the installed ``tokbin-core``; ``0+unknown`` if metadata is missing."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return "0+unknown"
