"""``Dataset``: the entry point for a corpus directory (spec 12.1).

::

    ds = Dataset("corpus")
    result = ds.write("code", txt_dir("data/code"), tokenizer="gpt2/tokenizer.json")
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokbin._boundary import public_api
from tokbin.format import naming
from tokbin.tokenizer.resolve import TokenizerLike
from tokbin.write.config import ErrorPolicy, WriterConfig
from tokbin.write.documents import Mode
from tokbin.write.result import WriteResult
from tokbin.write.stream_writer import StreamWriter

__all__ = ["Dataset"]


class Dataset:
    """A corpus: a directory of sources, each in its own subdirectory."""

    __slots__ = ("root",)

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"Dataset({str(self.root)!r})"

    @public_api
    def write(
        self,
        name: str,
        docs: Iterable[object],
        tokenizer: TokenizerLike,
        *,
        config: WriterConfig | None = None,
        policy: ErrorPolicy | None = None,
        mode: Mode | None = None,
        overwrite: bool = False,
    ) -> WriteResult:
        """Write one split of the source ``name`` from ``docs``.

        ``docs`` yields ``text`` or ``(doc_id, text)``; text may be ``str`` or UTF-8
        ``bytes``. ``tokenizer`` is a ``tokenizers.Tokenizer`` or a path to
        ``tokenizer.json``. With ``overwrite=True`` an existing split of the same name
        is replaced; other splits of the source are kept.
        """
        naming.check_source_name(name)
        with StreamWriter(
            self.root / name,
            tokenizer,
            config=config,
            policy=policy,
            mode=mode,
            overwrite=overwrite,
        ) as writer:
            writer.write(docs)
        return writer.result
