"""``Dataset``: the entry point for a corpus directory (spec 12.1).

::

    ds = Dataset("corpus")
    result = ds.write("code", txt_dir("data/code"), tokenizer="gpt2/tokenizer.json")
    ds.info()            # all sources, tokenizer, mixture
    ds.status("code")    # complete / partial / corrupt / outdated / unsupported
    ds.verify("code")    # sha256 of every shard
    ds.clean("code")     # remove an unfinished write
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokbin._boundary import public_api
from tokbin.format import naming
from tokbin.ops.clean import CleanResult, clean_source
from tokbin.ops.inspect import CorpusInfo, SourceInfo, inspect_corpus, inspect_source
from tokbin.ops.verify import ProgressCallback, VerifyReport, verify_source
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
        resume: bool = False,
    ) -> WriteResult:
        """Write one split of the source ``name`` from ``docs``.

        ``docs`` yields ``text`` or ``(doc_id, text)``; text may be ``str`` or UTF-8
        ``bytes``. ``tokenizer`` is a ``tokenizers.Tokenizer`` or a path to
        ``tokenizer.json``. With ``overwrite=True`` an existing split of the same name
        is replaced; other splits of the source are kept.

        With ``resume=True`` an interrupted write of the source continues from its last
        checkpoint (if there is none, the write simply starts). ``docs`` must then
        yield the same documents in the same order as before: those already consumed
        are skipped, and their ids are checked.
        """
        naming.check_source_name(name)
        with StreamWriter(
            self.root / name,
            tokenizer,
            config=config,
            policy=policy,
            mode=mode,
            overwrite=overwrite,
            resume=resume,
        ) as writer:
            writer.write(docs)
        return writer.result

    @public_api
    def info(self) -> CorpusInfo:
        """All sources of the corpus with their states, the tokenizer and the mixture."""
        return inspect_corpus(self.root)

    @public_api
    def status(self, name: str) -> SourceInfo:
        """The state of the source ``name``, including an unfinished write of it."""
        naming.check_source_name(name)
        return inspect_source(self.root / name)

    @public_api
    def verify(self, name: str, *, progress: ProgressCallback | None = None) -> VerifyReport:
        """Hash every shard of the source ``name`` and check its index files.

        Reads the whole source. ``progress`` is called with the bytes hashed so far.
        """
        naming.check_source_name(name)
        return verify_source(self.root / name, progress=progress)

    @public_api
    def clean(self, name: str) -> CleanResult:
        """Remove the unfinished write of the source ``name``; the source itself stays.

        ``ResumeError`` if another process is writing it right now.
        """
        naming.check_source_name(name)
        return clean_source(self.root / name)
