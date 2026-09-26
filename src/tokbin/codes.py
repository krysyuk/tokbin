"""Registry of error and warning codes.

A code is stable: once released, it never changes meaning and is never reused, even
after the error itself is removed (such codes move to ``RETIRED``). The code table in
the documentation is generated from this module.

Format: ``TB-<category><three digits>``. Categories:

====  ==================  =========================================================
 C    configuration       ``ConfigError``
 K    contract            ``ContractError``
 D    data                ``DataError``, ``DataQualityWarning``
 F    format              ``FormatError``, ``SchemaVersionError``,
                          ``UnsupportedFeatureError``
 I    integrity           ``IntegrityError``
 M    compatibility       ``CompatibilityError``, ``CompatibilityWarning``
 R    resume              ``ResumeError``
 P    dependencies        ``DependencyError``
 S    sharding            ``ShardingWarning``
 X    internal            ``InternalError``
====  ==================  =========================================================
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["CODES", "RETIRED", "Code"]

_CODE_RE = re.compile(r"^TB-[CKDFIMRPSX]\d{3}$")


@dataclass(frozen=True, slots=True)
class Code:
    """A stable diagnostic code and its short title ("what happened")."""

    id: str
    title: str

    @property
    def category(self) -> str:
        """Category letter: ``"I"`` for ``TB-I301``."""
        return self.id[3]

    def __str__(self) -> str:
        return self.id


_registry: dict[str, Code] = {}

#: Codes that are no longer issued. They must never be registered again.
RETIRED: frozenset[str] = frozenset()


def _register(code_id: str, title: str) -> Code:
    if not _CODE_RE.match(code_id):
        raise ValueError(f"malformed code: {code_id!r}")
    if code_id in _registry:
        raise ValueError(f"code registered twice: {code_id}")
    if code_id in RETIRED:
        raise ValueError(f"code is retired and cannot be reused: {code_id}")
    code = Code(code_id, title)
    _registry[code_id] = code
    return code


# --- P: dependencies ----------------------------------------------------------------

DEPENDENCY_MISSING = _register("TB-P001", "Required package is not installed")

# --- X: internal --------------------------------------------------------------------

INTERNAL = _register("TB-X001", "Internal tokbin error")


#: All active codes, keyed by id.
CODES: Mapping[str, Code] = MappingProxyType(_registry)
