"""The tokenizer protocol the writer works with.

Adding sentencepiece or custom vocabularies later means a new implementation of this
protocol, not a breaking API change.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from tokbin.codes import Code

__all__ = ["TokenizerProtocol"]


class TokenizerProtocol(Protocol):
    @property
    def vocab_size(self) -> int:
        """Number of token ids, special tokens included."""
        ...

    @property
    def eos_id(self) -> int | None:
        """End-of-sequence token id, or ``None`` if the tokenizer has none."""
        ...

    @property
    def bos_id(self) -> int | None:
        """Beginning-of-sequence token id, or ``None`` if the tokenizer has none."""
        ...

    @property
    def identifier(self) -> str | None:
        """A human-readable name (``gpt2``), for display only; not used for matching."""
        ...

    @property
    def notes(self) -> tuple[tuple[Code, str], ...]:
        """Facts about how the tokenizer was prepared, for the result summary."""
        ...

    def encode(self, text: str) -> Sequence[int]:
        """Token ids of ``text``, without any special tokens added."""
        ...

    def encode_batch(
        self, texts: Sequence[str]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """``encode`` for many texts at once, possibly on several threads.

        Returns the ids of all texts concatenated, as int64, and the number of ids of
        each text. Implementations must produce genuine integers: the writer checks
        the range, not the type.
        """
        ...

    def fingerprint(self) -> str:
        """16 hex digits identifying the tokenizer; equal fingerprints mean equal ids."""
        ...

    def save(self, directory: Path) -> None:
        """Store the tokenizer in ``directory`` so the dataset stays self-describing."""
        ...
