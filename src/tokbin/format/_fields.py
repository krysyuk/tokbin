"""Strict, typed access to fields of a parsed JSON object.

Every accessor either returns a value of the declared type or raises
:class:`FormatError` naming the file and the field. ``bool`` is never accepted where
an integer is expected, even though it is an ``int`` subclass in Python.

Unknown keys are ignored: they do not affect how data is read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from tokbin import codes
from tokbin.errors import FormatError

__all__ = ["Fields"]

_FIX = "the file is corrupted or was not written by tokbin; restore it from the original"


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    return {
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }.get(type(value), type(value).__name__)


class Fields:
    """A JSON object together with its location, for error messages."""

    __slots__ = ("_data", "where")

    def __init__(self, data: object, where: str) -> None:
        if not isinstance(data, dict):
            raise FormatError(
                codes.METADATA_FIELD_INVALID,
                where,
                why=f"expected an object, got {_type_name(data)}",
                fix=_FIX,
            )
        self._data: Mapping[str, object] = data
        self.where = where

    def error(self, key: str, why: str) -> FormatError:
        """Build an error about ``key`` of this object."""
        return FormatError(codes.METADATA_FIELD_INVALID, f"{self.where}: {key}", why=why, fix=_FIX)

    def _get(self, key: str) -> object:
        if key not in self._data:
            raise self.error(key, "the field is missing")
        return self._data[key]

    def _wrong(self, key: str, expected: str, value: object) -> FormatError:
        return self.error(key, f"expected {expected}, got {_type_name(value)} {value!r}")

    def get_int(self, key: str, *, minimum: int | None = None) -> int:
        value = self._get(key)
        if type(value) is not int:
            raise self._wrong(key, "an integer", value)
        if minimum is not None and value < minimum:
            raise self.error(key, f"expected an integer >= {minimum}, got {value}")
        return value

    def get_optional_int(self, key: str, *, minimum: int | None = None) -> int | None:
        if self._get(key) is None:
            return None
        return self.get_int(key, minimum=minimum)

    def get_number(self, key: str) -> float:
        value = self._get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._wrong(key, "a number", value)
        result = float(value)
        if not math.isfinite(result):
            raise self.error(key, f"expected a finite number, got {value!r}")
        return result

    def get_bool(self, key: str) -> bool:
        value = self._get(key)
        if type(value) is not bool:
            raise self._wrong(key, "a boolean", value)
        return value

    def get_str(self, key: str) -> str:
        value = self._get(key)
        if type(value) is not str:
            raise self._wrong(key, "a string", value)
        return value

    def get_optional_str(self, key: str) -> str | None:
        if self._get(key) is None:
            return None
        return self.get_str(key)

    def get_list(self, key: str) -> Sequence[object]:
        value = self._get(key)
        if type(value) is not list:
            raise self._wrong(key, "an array", value)
        return value

    def get_object(self, key: str) -> Fields:
        return Fields(self._get(key), f"{self.where}: {key}")

    def get_any(self, key: str) -> object:
        """The raw value; the caller validates it (for example by a constructor)."""
        return self._get(key)

    def field_names(self) -> Sequence[str]:
        return list(self._data)
