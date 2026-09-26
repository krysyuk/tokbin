"""Turn what the user passed as a tokenizer into a :class:`TokenizerProtocol`."""

from __future__ import annotations

from pathlib import Path

from tokbin import _deps, codes
from tokbin.errors import ContractError, DependencyError
from tokbin.tokenizer.hf import HFTokenizer
from tokbin.tokenizer.protocol import TokenizerProtocol

__all__ = ["TokenizerLike", "resolve_tokenizer"]

#: A ``tokenizers.Tokenizer`` object, or a path to ``tokenizer.json``.
TokenizerLike = object


def _require_tokenizers() -> None:
    try:
        _deps.require("tokenizers")
    except DependencyError as exc:
        err = DependencyError(
            codes.WRITE_UNAVAILABLE,
            why="this is the lightweight tokbin-core install: it reads, verifies and "
            "unpacks datasets. Writing needs the `tokenizers` package, which the full "
            "`tokbin` package installs",
            fix="pip install tokbin",
        )
        err.name = "tokenizers"
        raise err from exc


def resolve_tokenizer(
    tokenizer: TokenizerLike,
    *,
    eos_token: str | None = None,
    bos_token: str | None = None,
) -> TokenizerProtocol:
    """Accept a ``tokenizers.Tokenizer`` or a path to ``tokenizer.json``."""
    _require_tokenizers()
    if isinstance(tokenizer, HFTokenizer):
        return tokenizer
    if isinstance(tokenizer, (str, Path)):
        return HFTokenizer.from_file(Path(tokenizer), eos_token=eos_token, bos_token=bos_token)

    tokenizers = _deps.require("tokenizers")
    if isinstance(tokenizer, tokenizers.Tokenizer):
        return HFTokenizer(tokenizer, eos_token=eos_token, bos_token=bos_token)
    raise ContractError(
        codes.TOKENIZER_UNSUPPORTED,
        type(tokenizer).__qualname__,
        why="this version accepts a `tokenizers.Tokenizer` object or a path to "
        "tokenizer.json; other tokenizers (sentencepiece, custom vocabularies) are planned",
        fix="convert the tokenizer to the Hugging Face tokenizers format, or pass the "
        "path to its tokenizer.json",
    )
