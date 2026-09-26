"""UTC timestamps in the fixed form used by metadata: ``2026-09-26T10:22:31Z``."""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["is_timestamp", "utc_timestamp"]

_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_timestamp() -> str:
    """The current time, UTC, to the second."""
    return datetime.now(timezone.utc).strftime(_FORMAT)


def is_timestamp(value: str) -> bool:
    try:
        datetime.strptime(value, _FORMAT)
    except ValueError:
        return False
    return True
