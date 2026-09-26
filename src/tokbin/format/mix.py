"""``mix.json``: default mixture weights of a corpus (spec 6.6).

``{"web": 0.6, "code": 0.25, "books": 0.15}``. Keys are source directory names.
Weights are normalized when read; the mixture is a run parameter, not a property of
the data, so it can be overridden without touching the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokbin import codes
from tokbin.errors import FormatError
from tokbin.format import naming
from tokbin.format._fields import Fields
from tokbin.format._json import read_json_bounded, write_json_atomic

__all__ = ["Mix", "read_mix", "write_mix"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Mix:
    """Source weights as given, in file order. Weights are finite and non-negative."""

    weights: tuple[tuple[str, float], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.weights)

    def normalized(self) -> tuple[tuple[str, float], ...]:
        """Weights scaled to sum to 1."""
        total = sum(w for _, w in self.weights)
        return tuple((name, w / total) for name, w in self.weights)

    def to_dict(self) -> dict[str, object]:
        return dict(self.weights)

    @classmethod
    def from_dict(cls, data: object, where: str) -> Mix:
        f = Fields(data, where)
        weights: list[tuple[str, float]] = []
        for name in f.field_names():
            if not naming.is_valid_source_name(name):
                raise f.error(name, "not a valid source directory name")
            weight = f.get_number(name)
            if weight < 0:
                raise f.error(name, f"expected a non-negative weight, got {weight}")
            weights.append((name, weight))
        if not weights or sum(w for _, w in weights) <= 0:
            raise FormatError(
                codes.METADATA_INCONSISTENT,
                where,
                why="the mixture has no source with a positive weight",
                fix="give at least one source a positive weight",
            )
        return cls(weights=tuple(weights))


def read_mix(corpus_root: Path) -> Mix:
    path = corpus_root / naming.MIX_JSON
    return Mix.from_dict(read_json_bounded(path), where=str(path))


def write_mix(corpus_root: Path, mix: Mix) -> None:
    write_json_atomic(corpus_root / naming.MIX_JSON, mix.to_dict())
