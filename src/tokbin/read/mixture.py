"""``Mixture``: sampling training windows from several sources by weight (spec 6.6, 17).

::

    mix = Mixture.from_config("corpus")                 # weights from corpus/mix.json
    mix = Mixture.from_config("corpus", {"web": 3, "code": 1})
    mix.batch(batch_size=32, block_size=1024, rng=rng)  # (32, 1024)

A weight is the probability that a window comes from that source; weights are
normalized. Sources with weight 0 are not opened. All sources must share the tokenizer
(spec 23, invariant 5), otherwise ``CompatibilityError``.

Randomness comes only from the ``rng`` passed in. A ``Mixture`` survives pickling the
same way a ``Source`` does: only the root, weights and split travel.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import CompatibilityError, ConfigError, FormatError
from tokbin.format import naming
from tokbin.format.mix import read_mix
from tokbin.read.source import Source

__all__ = ["Mixture"]


def _weights_error(detail: str, why: str) -> ConfigError:
    return ConfigError(
        codes.MIX_WEIGHTS_INVALID,
        detail,
        why=why,
        fix='pass non-negative numbers with a positive sum, e.g. {"web": 0.7, "code": 0.3}',
    )


def _check_weights(weights: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(weights, Mapping) or not weights:
        raise _weights_error(repr(weights), "weights must be a non-empty mapping")
    checked: dict[str, float] = {}
    for name, weight in weights.items():
        if not isinstance(name, str) or not naming.is_valid_source_name(name):
            raise _weights_error(repr(name), "keys must be source directory names")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise _weights_error(f"{name}={weight!r}", "weights must be numbers")
        if not math.isfinite(weight) or weight < 0:
            raise _weights_error(f"{name}={weight!r}", "weights must be finite and >= 0")
        checked[name] = float(weight)
    if sum(checked.values()) <= 0:
        raise _weights_error(repr(dict(weights)), "at least one weight must be positive")
    return checked


class Mixture:
    """Several sources of one corpus, sampled by weight."""

    __slots__ = ("_p", "names", "root", "sources", "split", "weights")

    def __init__(
        self, root: str | Path, weights: Mapping[str, float], *, split: str = "train"
    ) -> None:
        self._open(Path(root), dict(weights), split, from_file=False)

    def _open(self, root: Path, weights: dict[str, float], split: str, *, from_file: bool) -> None:
        checked = _check_weights(weights)
        naming.check_split(split)
        self.root = root
        self.split = split
        active = {name: w for name, w in checked.items() if w > 0}
        total = sum(active.values())
        self.names: tuple[str, ...] = tuple(active)
        self.weights: tuple[float, ...] = tuple(w / total for w in active.values())
        self._p = np.array(self.weights, dtype=np.float64)
        for name in self.names:
            path = root / name
            if not ((path / naming.META_JSON).is_file() or (path / naming.DATASET_JSON).is_file()):
                where = str(root / naming.MIX_JSON) if from_file else f"weights[{name!r}]"
                cls = FormatError if from_file else ConfigError
                code = codes.MIX_SOURCE_MISSING if from_file else codes.SOURCE_NOT_FOUND
                raise cls(
                    code,
                    f"{where}: {name}",
                    why=f"{path} is not a finished source",
                    fix="write the source, fix the name, or give it weight 0",
                )
        self.sources: tuple[Source, ...] = tuple(Source(root / n, split) for n in self.names)
        hashes = {s.meta.tokenizer_hash for s in self.sources}
        if len(hashes) > 1:
            listing = ", ".join(f"{n}={s.meta.tokenizer_hash}" for n, s in self._pairs())
            raise CompatibilityError(
                codes.TOKENIZERS_DIFFER,
                str(root),
                why=f"sources were written with different tokenizers ({listing}); their "
                "token ids mean different things",
                fix="mix only sources written with the same tokenizer, or give the others weight 0",
            )

    def _pairs(self) -> list[tuple[str, Source]]:
        return list(zip(self.names, self.sources, strict=True))

    @classmethod
    @public_api
    def from_config(
        cls,
        root: str | Path,
        weights: Mapping[str, float] | None = None,
        *,
        split: str = "train",
    ) -> Mixture:
        """Weights from ``weights``, or from ``root/mix.json`` when not given."""
        root = Path(root)
        mix = object.__new__(cls)
        if weights is None:
            mix._open(root, dict(read_mix(root).weights), split, from_file=True)
        else:
            mix._open(root, dict(weights), split, from_file=False)
        return mix

    # --- pickling ---------------------------------------------------------------------

    def __getstate__(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "weights": dict(zip(self.names, self.weights, strict=True)),
            "split": self.split,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._open(Path(state["root"]), state["weights"], state["split"], from_file=False)

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}={w:.3g}" for n, w in zip(self.names, self.weights, strict=True))
        return f"Mixture({str(self.root)!r}, {parts}, split={self.split!r})"

    # --- sampling ---------------------------------------------------------------------

    @property
    def dtype(self) -> np.dtype[Any]:
        """The widest item dtype among the sources; batches are returned in it."""
        return np.result_type(*(s.dtype for s in self.sources))

    @public_api
    def choose(self, n: int, rng: np.random.Generator) -> npt.NDArray[np.intp]:
        """Source index for each of ``n`` samples, drawn by weight."""
        if not isinstance(rng, np.random.Generator):
            raise ConfigError(
                codes.RNG_REQUIRED,
                type(rng).__qualname__,
                why="sampling takes its randomness from an explicit generator",
                fix="pass rng=np.random.default_rng(seed)",
            )
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ConfigError(
                codes.CONFIG_VALUE_INVALID,
                f"n={n!r}",
                why="n must be a non-negative integer",
                fix="pass a count",
            )
        return rng.choice(len(self.sources), size=n, p=self._p)

    @public_api
    def batch(
        self, batch_size: int, block_size: int, rng: np.random.Generator
    ) -> npt.NDArray[np.unsignedinteger[Any]]:
        """``batch_size`` windows of ``block_size`` items: ``(batch_size, block_size)``.

        Each row comes from a source drawn by weight, at a uniform random position of
        that source. The result depends only on the state of ``rng``.
        """
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ConfigError(
                codes.CONFIG_VALUE_INVALID,
                f"block_size={block_size!r}",
                why="block_size must be a positive integer",
                fix="pass the window length in items",
            )
        which = self.choose(batch_size, rng)
        out = np.empty((batch_size, block_size), dtype=self.dtype)
        for k, source in enumerate(self.sources):
            rows = np.flatnonzero(which == k)
            if rows.size:
                out[rows] = source.sample_windows(int(rows.size), block_size, rng)
        return out
