"""Shared helpers for tests: a tiny deterministic tokenizer and raw readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

N_WORDS = 300
#: Ids of the fixture vocabulary: specials first, then "t0".."t299".
UNK, EOS, BOS = 0, 1, 2
FIRST_WORD = 3


def make_tokenizer(*, specials: tuple[str, ...] = ("<|endoftext|>", "<s>")) -> Any:
    """A deterministic WordLevel tokenizer: "t<N>" -> N + 3, whitespace-separated."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": UNK, **{token: i for i, token in enumerate(specials, start=1)}}
    base = len(vocab)
    vocab.update({f"t{i}": base + i for i in range(N_WORDS)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def make_docs(n: int, *, seed: int = 0, max_len: int = 12) -> list[tuple[str, str]]:
    """``n`` named documents of 1..max_len random words."""
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        words = rng.integers(0, N_WORDS, size=int(rng.integers(1, max_len + 1)))
        docs.append((f"doc-{i}", " ".join(f"t{w}" for w in words)))
    return docs


def encode(text: str) -> list[int]:
    return [FIRST_WORD + int(word[1:]) for word in text.split()]


def expected_stream(
    docs: list[tuple[str, str]], *, eos: bool = True, bos: bool = False
) -> npt.NDArray[np.int64]:
    out: list[int] = []
    for _, text in docs:
        out += ([BOS] if bos else []) + encode(text) + ([EOS] if eos else [])
    return np.array(out, dtype=np.int64)


def read_meta_json(source: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    return result


def read_stream(source: Path, split: str = "train") -> npt.NDArray[np.int64]:
    """Concatenate the shards of a split without the reader (it arrives in 0.1 later)."""
    meta = read_meta_json(source)
    dtype = np.dtype(meta["dtype"]).newbyteorder("<")
    parts = [np.fromfile(source / s["name"], dtype=dtype) for s in meta["splits"][split]["shards"]]
    return np.concatenate(parts).astype(np.int64) if parts else np.empty(0, np.int64)


def read_ids(source: Path, split: str = "train") -> list[str | None]:
    lines = (source / f"{split}-ids.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["id"] for line in lines]


def read_skipped(source: Path, split: str = "train") -> list[dict[str, Any]]:
    path = source / f"{split}-skipped.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
