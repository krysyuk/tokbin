"""Forms of input documents (spec 10.2).

::

    "text"                         plain text (str or UTF-8 bytes)
    ("doc_id", "text")             text with a name (recommended: enables resume checks)
    ("doc_id", [Segment, ...])     segments (multimodal; not supported before 2.x)
    SkipDocument("doc_id", "why")  an input record the generator could not read

All documents of one write must have the same form. Text may be given as ``bytes``:
a decoding error then becomes a skipped document instead of a crashed generator.
``SkipDocument`` does the same for records the generator itself cannot parse (a broken
JSON line): the document is recorded in ``skipped.jsonl`` with the reason, and it counts
towards ``ErrorPolicy.max_skip_ratio``. It fits either form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from tokbin import codes
from tokbin.errors import ConfigError, ContractError, DataError, UnsupportedFeatureError

__all__ = ["Document", "DocumentNormalizer", "Mode", "Segment", "SkipDocument", "decode_text"]

Mode = Literal["single", "multi"]
_Form = Literal["text", "named"]
_FORM_NAMES = {"text": "plain text", "named": "(doc_id, text) tuples"}


class Segment(NamedTuple):
    """A piece of a multimodal document. Reserved; not supported before 2.x."""

    kind: Literal["text", "image", "audio"]
    data: object


class SkipDocument(NamedTuple):
    """An input record that cannot be read; the writer records it as skipped."""

    id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class Document:
    """A normalized input document."""

    id: str | None
    text: str | bytes
    #: Why the generator could not read it (a ``SkipDocument``); ``None`` normally.
    unreadable: str | None = None


def _describe(value: object) -> str:
    return type(value).__qualname__


class DocumentNormalizer:
    """Validates document forms and pins the form of the first document."""

    __slots__ = ("_form",)

    def __init__(self, mode: Mode | None = None) -> None:
        if mode == "multi":
            raise UnsupportedFeatureError(
                codes.FEATURE_UNSUPPORTED,
                "mode='multi'",
                why="multimodal documents are not supported by this version",
                fix="write text documents with mode='single' (the default)",
            )
        if mode not in (None, "single"):
            raise ConfigError(
                codes.CONFIG_VALUE_INVALID,
                f"mode={mode!r}",
                why="expected 'single', 'multi' or None",
                fix="omit mode for text documents",
            )
        self._form: _Form | None = None

    def normalize(self, doc: object) -> Document:
        if isinstance(doc, SkipDocument):
            if doc.id is not None and not isinstance(doc.id, str):
                raise ContractError(
                    codes.DOCUMENT_FORM_INVALID,
                    f"SkipDocument id {_describe(doc.id)}",
                    why="the id of a SkipDocument must be a string or None",
                    fix="pass SkipDocument(doc_id: str | None, reason: str)",
                )
            return Document(doc.id, b"", unreadable=str(doc.reason))
        form, result = self._classify(doc)
        if self._form is None:
            self._form = form
        elif form != self._form:
            raise ContractError(
                codes.DOCUMENT_FORMS_MIXED,
                why=f"the first document was {_FORM_NAMES[self._form]}, "
                f"this one is {_FORM_NAMES[form]}",
                fix="make the generator yield every document in the same form",
            )
        return result

    @staticmethod
    def _classify(doc: object) -> tuple[_Form, Document]:
        if isinstance(doc, (str, bytes)):
            return "text", Document(None, doc)
        if isinstance(doc, tuple) and len(doc) == 2 and isinstance(doc[0], str):
            doc_id, body = doc
            if isinstance(body, (str, bytes)):
                return "named", Document(doc_id, body)
            if isinstance(body, (list, tuple)):
                raise UnsupportedFeatureError(
                    codes.FEATURE_UNSUPPORTED,
                    f"document {doc_id!r}",
                    why="segmented (multimodal) documents are not supported by this version",
                    fix="yield (doc_id, text) tuples",
                )
        raise ContractError(
            codes.DOCUMENT_FORM_INVALID,
            _describe(doc),
            why="a document must be text (str or bytes) or a (doc_id: str, text) tuple",
            fix="make the generator yield `text` or `(doc_id, text)`",
        )


def decode_text(text: str | bytes) -> str:
    """Decode bytes as strict UTF-8; ``DataError`` on invalid input."""
    if isinstance(text, str):
        return text
    try:
        return text.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataError(
            codes.DOCUMENT_NOT_UTF8,
            why=f"byte 0x{text[exc.start]:02x} at position {exc.start}: {exc.reason}",
            fix="fix the encoding of the source file, or decode it yourself in the generator",
        ) from exc
