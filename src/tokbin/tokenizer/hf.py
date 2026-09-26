"""Adapter over ``tokenizers.Tokenizer`` (Hugging Face).

The adapter works on its own copy of the tokenizer with truncation and padding turned
off: a ``tokenizer.json`` may carry ``"truncation": {"max_length": 512}``, and encoding
with it would silently cut every document. The user's object is never modified.
The same normalized copy is saved next to the data and fingerprinted.

Special tokens are resolved in this order:

1. an explicit ``eos_token`` / ``bos_token``;
2. ``tokenizer_config.json`` next to ``tokenizer.json`` (the Hugging Face convention
   that names them, e.g. the DeepSeek end-of-sentence token);
3. a single match among well-known names (``<|endoftext|>``, ``</s>``...).

Anything chosen automatically is reported, because it changes the written data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from tokbin import _deps, _fs, codes
from tokbin.codes import Code
from tokbin.errors import ConfigError, FormatError
from tokbin.format import naming
from tokbin.format._json import canonical_dumps, read_json_bounded

__all__ = ["BOS_CANDIDATES", "EOS_CANDIDATES", "HFTokenizer", "SpecialToken"]

#: Special tokens recognized as end-of-sequence, in no particular order.
EOS_CANDIDATES: Final = ("<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>")
#: Special tokens recognized as beginning-of-sequence.
BOS_CANDIDATES: Final = ("<s>", "<bos>", "<|begin_of_text|>", "<|startoftext|>")

_Origin = Literal["explicit", "tokenizer_config.json", "well-known name"]
_PARAMS: Final = ("eos_token", "bos_token")


@dataclass(frozen=True, slots=True)
class SpecialToken:
    text: str
    id: int
    origin: _Origin


def _config_hints(path: Path) -> dict[str, str | None]:
    """``eos_token`` / ``bos_token`` named by a ``tokenizer_config.json``, if there is one."""
    if not path.is_file():
        return {}
    try:
        data = read_json_bounded(path)
    except FormatError as exc:
        raise ConfigError(
            codes.TOKENIZER_FILE_INVALID,
            str(path),
            why=exc.why,
            fix="fix or remove the file, or pass eos_token / bos_token explicitly",
        ) from exc
    hints: dict[str, str | None] = {}
    if not isinstance(data, dict):
        return hints
    for key in _PARAMS:
        value = data.get(key)
        # Hugging Face stores either a plain string or an AddedToken object.
        if isinstance(value, dict):
            value = value.get("content")
        if value is not None and not isinstance(value, str):
            raise ConfigError(
                codes.TOKENIZER_FILE_INVALID,
                f"{path}: {key}",
                why=f"expected a string or an object with 'content', got {value!r}",
                fix=f"fix the file, or pass {key} explicitly",
            )
        hints[key] = value or None
    return hints


class HFTokenizer:
    """:class:`~tokbin.tokenizer.protocol.TokenizerProtocol` over ``tokenizers.Tokenizer``."""

    __slots__ = ("_bos", "_eos", "_fingerprint", "_identifier", "_notes", "_tok")

    def __init__(
        self,
        tokenizer: object,
        *,
        identifier: str | None = None,
        eos_token: str | None = None,
        bos_token: str | None = None,
        hints: Mapping[str, str | None] | None = None,
        _owned: bool = False,
    ) -> None:
        tokenizers = _deps.require("tokenizers")
        source: Any = tokenizer  # a tokenizers.Tokenizer; the package ships no usable types
        # A tokenizer passed by the user is copied, so disabling its limits below never
        # changes the user's object. One loaded by tokbin itself needs no copy.
        tok = source if _owned else tokenizers.Tokenizer.from_str(source.to_str())
        notes: list[tuple[Code, str]] = []
        if tok.truncation is not None or tok.padding is not None:
            notes.append(
                (
                    codes.TOKENIZER_LIMITS_DISABLED,
                    "the tokenizer truncated or padded its output; tokbin encodes documents "
                    "in full",
                )
            )
        tok.no_truncation()
        tok.no_padding()
        self._tok: Any = tok
        self._identifier = identifier
        self._fingerprint: str | None = None

        hints = hints or {}
        self._eos = self._special("eos_token", eos_token, hints.get("eos_token"), EOS_CANDIDATES)
        self._bos = self._special("bos_token", bos_token, hints.get("bos_token"), BOS_CANDIDATES)
        detected = [
            f"{name} {s.text!r} (id {s.id}) from {s.origin}"
            for name, s in (("EOS", self._eos), ("BOS", self._bos))
            if s is not None and s.origin != "explicit"
        ]
        if detected:
            notes.append((codes.SPECIAL_TOKENS_DETECTED, "; ".join(detected)))
        self._notes = tuple(notes)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        eos_token: str | None = None,
        bos_token: str | None = None,
    ) -> HFTokenizer:
        tokenizers = _deps.require("tokenizers")
        if not path.is_file():
            raise ConfigError(
                codes.TOKENIZER_FILE_INVALID,
                str(path),
                why="the file does not exist",
                fix="pass the path to a tokenizer.json file",
            )
        try:
            tok = tokenizers.Tokenizer.from_file(str(path))
        except Exception as exc:
            raise ConfigError(
                codes.TOKENIZER_FILE_INVALID,
                str(path),
                why=f"tokenizers could not load it: {exc}",
                fix="pass a tokenizer.json saved by the Hugging Face tokenizers library",
            ) from exc
        identifier = path.parent.name if path.name == naming.TOKENIZER_JSON else path.stem
        return cls(
            tok,
            identifier=identifier or None,
            eos_token=eos_token,
            bos_token=bos_token,
            hints=_config_hints(path.parent / naming.TOKENIZER_CONFIG_JSON),
            _owned=True,
        )

    def _special(
        self,
        param: str,
        explicit: str | None,
        hint: str | None,
        candidates: tuple[str, ...],
    ) -> SpecialToken | None:
        if explicit is not None:
            return SpecialToken(explicit, self._lookup(param, explicit, "explicit"), "explicit")
        if hint is not None:
            return SpecialToken(
                hint,
                self._lookup(param, hint, "tokenizer_config.json"),
                "tokenizer_config.json",
            )
        matches = {c: int(i) for c in candidates if (i := self._tok.token_to_id(c)) is not None}
        # Ambiguous detection yields None: guessing wrong would corrupt every document.
        if len(set(matches.values())) != 1:
            return None
        text = next(iter(matches))
        return SpecialToken(text, matches[text], "well-known name")

    def _lookup(self, param: str, text: str, origin: _Origin) -> int:
        token_id = self._tok.token_to_id(text)
        if token_id is None:
            where = (
                f"{param}={text!r}" if origin == "explicit" else f"{origin} names {param} {text!r}"
            )
            raise ConfigError(
                codes.SPECIAL_TOKEN_NOT_FOUND,
                repr(text),
                why=f"{where}, but it is not in the tokenizer vocabulary",
                fix=f"check the spelling, or pass {param} explicitly",
            )
        return int(token_id)

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size(with_added_tokens=True))

    @property
    def eos_id(self) -> int | None:
        return self._eos.id if self._eos is not None else None

    @property
    def bos_id(self) -> int | None:
        return self._bos.id if self._bos is not None else None

    @property
    def identifier(self) -> str | None:
        return self._identifier

    @property
    def notes(self) -> tuple[tuple[Code, str], ...]:
        return self._notes

    def encode(self, text: str) -> Sequence[int]:
        # Special tokens are handled by the writer (append_eos / prepend_bos); the
        # tokenizer's post-processor must not add its own.
        ids: Sequence[int] = self._tok.encode(text, add_special_tokens=False).ids
        return ids

    def encode_batch(
        self, texts: Sequence[str]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        # Runs on the tokenizers thread pool (size: RAYON_NUM_THREADS, default all cores).
        lists = [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]
        lengths = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
        # np.fromiter would silently truncate floats, but tokenizers ids are Rust u32:
        # always non-negative integers below 2**32. It is twice as fast as np.asarray.
        flat = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=int(lengths.sum()))
        return flat, lengths

    def to_json(self) -> str:
        """The normalized tokenizer as ``tokenizer.json`` text."""
        text: str = self._tok.to_str(pretty=True)
        return text

    def fingerprint(self) -> str:
        # Computed once: serializing a large tokenizer (DeepSeek: 6 MB) is not free.
        if self._fingerprint is None:
            canonical = canonical_dumps(json.loads(self._tok.to_str()))
            self._fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return self._fingerprint

    def save(self, directory: Path) -> None:
        """Save ``tokenizer.json`` and, if special tokens are known, ``tokenizer_config.json``.

        The config makes the stored copy usable on its own: writing another split with
        it finds the same special tokens without extra arguments.
        """
        directory.mkdir(parents=True, exist_ok=True)
        _fs.atomic_write_bytes(directory / naming.TOKENIZER_JSON, self.to_json().encode("utf-8"))
        if self._eos is not None or self._bos is not None:
            config = {
                "eos_token": self._eos.text if self._eos is not None else None,
                "bos_token": self._bos.text if self._bos is not None else None,
            }
            text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
            _fs.atomic_write_bytes(directory / naming.TOKENIZER_CONFIG_JSON, text.encode("utf-8"))
