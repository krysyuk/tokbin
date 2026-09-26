"""``StreamWriter``: documents in, a finished source out (spec 12.2, 16).

::

    with StreamWriter("corpus/code", "gpt2/tokenizer.json", config=cfg) as w:
        for doc_id, text in docs:
            w.add(doc_id, text)
    result = w.result

Packing is ``stream``: items of consecutive documents are concatenated and cut into
shards wherever a shard fills up. Document boundaries live in ``offsets`` in global
item coordinates and know nothing about shards.

On success the context manager publishes the source atomically. On any exception
(``KeyboardInterrupt`` included) the open shard is dropped and the partial directory
is left in place; no finished source appears.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from tokbin import codes
from tokbin._boundary import mark_user_error, public_api
from tokbin._version import get_version
from tokbin.errors import (
    CompatibilityError,
    ConfigError,
    ContractError,
    DataError,
    DataQualityWarning,
    InternalError,
    ShardingWarning,
)
from tokbin.format import naming
from tokbin.format._time import utc_timestamp
from tokbin.format.dtypes import TokenArray, dtype_for_vocab, resolve_dtype, to_dtype_checked
from tokbin.format.meta import Meta, ShardInfo, SplitMeta, read_meta
from tokbin.format.schema import SCHEMA_VERSION, source_kind
from tokbin.tokenizer.protocol import TokenizerProtocol
from tokbin.tokenizer.resolve import TokenizerLike, resolve_tokenizer
from tokbin.write.config import ErrorPolicy, WriterConfig
from tokbin.write.documents import Document, DocumentNormalizer, Mode, decode_text
from tokbin.write.duplicates import IdHashes
from tokbin.write.partial import SideFiles, create_partial, discard_open_shards, publish
from tokbin.write.result import IssueCollector, WriteResult, WriteStats
from tokbin.write.shard import ShardFile

__all__ = ["StreamWriter"]

#: A document above this share of a shard is reported as large (spec 7.4).
LARGE_DOC_SHARE: Final = 0.05

#: Upper bound of buffered text per batch, so a few huge documents stay bounded.
_BATCH_CHARS: Final = 64 * 1024**2

_MISSING: Final = object()


class _Pending(NamedTuple):
    doc: Document
    #: Decoded text, or None when decoding failed (then ``error`` is set).
    text: str | None
    error: DataError | None


def _empty(why: str) -> DataError:
    return DataError(
        codes.DOCUMENT_EMPTY,
        why=why,
        fix="filter out empty and whitespace-only documents in the generator",
    )


#: Placeholder result for empty texts, which are never sent to the tokenizer.
_EMPTY: Final[TokenArray] = np.empty(0, dtype=np.uint8)


def _where(doc: Document, index: int) -> str:
    return f"document {doc.id!r}" if doc.id is not None else f"input #{index}"


class StreamWriter:
    """Writes one split of one source."""

    @public_api
    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerLike,
        *,
        config: WriterConfig | None = None,
        policy: ErrorPolicy | None = None,
        mode: Mode | None = None,
        overwrite: bool = False,
    ) -> None:
        self._target = Path(path)
        naming.check_source_name(self._target.name)
        self._config = config if config is not None else WriterConfig()
        self._policy = policy if policy is not None else ErrorPolicy()
        self._overwrite = overwrite
        self._normalizer = DocumentNormalizer(mode)
        self._tok: TokenizerProtocol = resolve_tokenizer(
            tokenizer, eos_token=self._config.eos_token, bos_token=self._config.bos_token
        )
        self._issues = IssueCollector()

        self._partial: Path | None = None
        self._side: SideFiles | None = None
        self._shard: ShardFile | None = None
        self._closed_shards: list[ShardInfo] = []
        self._existing: Meta | None = None
        self._result: WriteResult | None = None
        self._dtype = np.dtype("uint8")
        self._shard_items = 0
        self._bos: int | None = None
        self._eos: int | None = None
        self._pending: list[_Pending] = []
        self._pending_chars = 0
        self._id_hashes = IdHashes()

        self._n_input = 0
        self._n_docs = 0
        self._n_skipped = 0
        self._n_items = 0
        self._n_split_docs = 0

    # --- lifecycle --------------------------------------------------------------------

    def __enter__(self) -> StreamWriter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    @public_api
    def open(self) -> None:
        """Validate everything that can be validated up front and start the write."""
        if self._partial is not None:
            return
        cfg = self._config
        fingerprint = self._tok.fingerprint()
        self._existing = self._inspect_target(fingerprint)
        self._dtype = self._choose_dtype()

        if cfg.append_eos and self._tok.eos_id is None:
            raise ConfigError(
                codes.NO_EOS_TOKEN,
                why="append_eos=True, but no end-of-sequence token was found in the tokenizer",
                fix="set WriterConfig(eos_token='...') to the right token, "
                "or append_eos=False to write documents without separators",
            )
        if cfg.prepend_bos and self._tok.bos_id is None:
            raise ConfigError(
                codes.NO_BOS_TOKEN,
                why="prepend_bos=True, but no beginning-of-sequence token was found",
                fix="set WriterConfig(bos_token='...') to the right token, or prepend_bos=False",
            )
        self._shard_items = cfg.shard_bytes // self._dtype.itemsize
        if self._shard_items < 1:
            raise ConfigError(
                codes.CONFIG_VALUE_INVALID,
                f"shard_bytes={cfg.shard_bytes}",
                why=f"a shard must hold at least one {self._dtype.name} item",
                fix="increase shard_bytes",
            )
        self._bos = self._tok.bos_id if cfg.prepend_bos else None
        self._eos = self._tok.eos_id if cfg.append_eos else None

        for code, message in self._tok.notes:
            self._issues.add(code, message, level="info")

        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._partial = create_partial(self._target)
        self._tok.save(self._partial / naming.TOKENIZER_DIR)
        self._side = SideFiles(self._partial, cfg.split)

    def _inspect_target(self, fingerprint: str) -> Meta | None:
        target = self._target
        if not target.exists():
            return None
        if not target.is_dir() or not (
            (target / naming.META_JSON).exists() or (target / naming.DATASET_JSON).exists()
        ):
            if target.is_dir() and not any(target.iterdir()):
                return None  # an empty directory is simply replaced
            raise ConfigError(
                codes.TARGET_NOT_A_SOURCE,
                str(target),
                why="the path exists, but it is not a tokbin source",
                fix="choose another name, or remove the directory yourself",
            )
        source_kind(target)
        existing = read_meta(target)
        split = self._config.split
        if split in existing.split_names and not self._overwrite:
            raise ConfigError(
                codes.SPLIT_EXISTS,
                f"{target} [{split}]",
                why=f"the source already has a {split!r} split",
                fix="pass overwrite=True to replace this split, or write another split",
            )
        if existing.tokenizer_hash != fingerprint:
            raise CompatibilityError(
                codes.TOKENIZER_MISMATCH,
                str(target),
                why=f"the source was written with tokenizer {existing.tokenizer_hash}, "
                f"this one is {fingerprint}; splits of one source must share a tokenizer",
                fix="use the tokenizer stored in the source's tokenizer/ directory, "
                "or write to a new source",
            )
        return existing

    def _choose_dtype(self) -> np.dtype[np.unsignedinteger]:
        vocab = self._tok.vocab_size
        existing = self._existing
        if existing is None:
            return resolve_dtype(vocab, self._config.dtype)
        wanted = (
            resolve_dtype(vocab, self._config.dtype)
            if self._config.dtype is not None
            else existing.np_dtype
        )
        if wanted.itemsize < dtype_for_vocab(vocab).itemsize or wanted.name != existing.dtype:
            raise CompatibilityError(
                codes.DTYPE_MISMATCH,
                str(self._target),
                why=f"the source stores {existing.dtype}, this write would use {wanted.name}",
                fix="omit dtype, or set it to the dtype of the existing source",
            )
        return np.dtype(existing.dtype)

    # --- adding documents -------------------------------------------------------------
    #
    # Documents are buffered and processed in batches: one encode_batch call per batch
    # (multi-threaded in tokenizers), and vectorized bookkeeping. The result is exactly
    # the one of processing documents one by one, in order.

    @public_api
    def add(self, first: object, second: object = _MISSING) -> None:
        """Add one document: ``add(text)`` or ``add(doc_id, text)``.

        Documents are buffered and tokenized in batches of ``WriterConfig.batch_docs``.
        """
        self._add(first if second is _MISSING else (first, second))

    @public_api
    def write(self, docs: Iterable[object]) -> None:
        """Add every document of ``docs``.

        Exceptions raised by the iterable itself (the user's generator) propagate
        unchanged: tokbin cannot resume a generator that failed.
        """
        try:
            iterator = iter(docs)
        except TypeError as exc:
            raise ContractError(
                codes.DOCUMENT_FORM_INVALID,
                type(docs).__qualname__,
                why="documents must be given as an iterable (list, generator...)",
                fix="pass a generator that yields documents",
            ) from exc
        while True:
            try:
                raw = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                mark_user_error(exc)
                raise
            self._add(raw)
        self._flush()

    def _add(self, raw: object) -> None:
        if self._partial is None:
            self.open()
        # Form and encoding are checked right away, so errors point at this document.
        doc = self._normalizer.normalize(raw)
        try:
            text: str | None = decode_text(doc.text)
            error: DataError | None = None
        except DataError as err:
            text, error = None, err
        self._pending.append(_Pending(doc, text, error))
        self._pending_chars += len(text) if text is not None else 0
        if len(self._pending) >= self._config.batch_docs or self._pending_chars >= _BATCH_CHARS:
            self._flush()

    def _flush(self) -> None:
        """Process the buffered documents, in order."""
        if not self._pending:
            return
        entries, self._pending, self._pending_chars = self._pending, [], 0
        encoded = iter(self._encode_many([e.text for e in entries if e.text is not None]))

        kept_docs: list[Document] = []
        kept_items: list[TokenArray] = []
        kept_inputs: list[int] = []
        for entry in entries:
            index = self._n_input
            self._n_input += 1
            error = entry.error
            if error is None:
                # One result per decoded text, empty ones included: stay aligned.
                result = next(encoded)
                if not entry.text:
                    error = _empty("the text is empty")
                elif isinstance(result, DataError):
                    error = result
                elif result.size == 0:
                    error = _empty("the text produced no tokens")
                else:
                    kept_docs.append(entry.doc)
                    kept_items.append(result)
                    kept_inputs.append(index)
            if error is not None:
                self._skip(entry.doc, error, index)
            self._check_skip_ratio()
        self._write_batch(kept_docs, kept_items, kept_inputs)

    def _encode_many(self, texts: list[str]) -> list[TokenArray | DataError]:
        """Token arrays of ``texts`` (empty strings included), or a DataError per text."""
        work = [t for t in texts if t]
        try:
            flat, lengths = self._tok.encode_batch(work)
        except Exception:
            # Some text breaks the tokenizer: find which, one by one.
            arrays = [self._encode_one(t) for t in work]
        else:
            arrays = self._split_checked(flat, lengths)
        it = iter(arrays)
        return [next(it) if t else _EMPTY for t in texts]

    def _encode_one(self, text: str) -> TokenArray | DataError:
        try:
            ids = self._tok.encode(text)
        except Exception as exc:
            return DataError(
                codes.TOKENIZER_FAILED,
                why=f"{type(exc).__name__}: {exc}",
                fix="inspect the document; if many documents fail, check the tokenizer",
            )
        try:
            return to_dtype_checked(ids, self._dtype)
        except DataError as err:
            return err

    def _split_checked(
        self, flat: npt.NDArray[np.int64], lengths: npt.NDArray[np.int64]
    ) -> list[TokenArray | DataError]:
        """Range-check the whole batch at once; only offending documents become errors."""
        bounds = np.cumsum(lengths)[:-1]
        top = int(np.iinfo(self._dtype).max)
        bad = np.flatnonzero((flat < 0) | (flat > top))
        result: list[TokenArray | DataError] = []
        if not bad.size:
            # Common case: one conversion for the batch, pieces are views.
            result.extend(np.split(flat.astype(self._dtype), bounds))
            return result
        offenders = set(np.searchsorted(bounds, bad, side="right").tolist())
        for k, piece in enumerate(np.split(flat, bounds)):
            result.append(self._range_error(piece) if k in offenders else piece.astype(self._dtype))
        return result

    def _range_error(self, ids: npt.NDArray[np.int64]) -> DataError:
        try:
            to_dtype_checked(ids, self._dtype)
        except DataError as err:
            return err
        raise InternalError.wrap(RuntimeError("range check disagreement"))  # pragma: no cover

    def _write_batch(
        self, docs: list[Document], items: list[TokenArray], inputs: list[int]
    ) -> None:
        if not docs:
            return
        body_lengths = np.fromiter((a.size for a in items), dtype=np.int64, count=len(items))
        extra = int(self._bos is not None) + int(self._eos is not None)
        lengths = body_lengths + extra
        ends = np.cumsum(lengths)
        starts = ends - lengths
        body = np.concatenate(items)
        if extra:
            stream = np.empty(int(ends[-1]), dtype=self._dtype)
            is_body = np.ones(stream.size, dtype=bool)
            if self._bos is not None:
                stream[starts] = self._bos
                is_body[starts] = False
            if self._eos is not None:
                stream[ends - 1] = self._eos
                is_body[ends - 1] = False
            stream[is_body] = body
        else:
            stream = body.astype(self._dtype, copy=False)

        self._check_sizes(docs, lengths, inputs)
        first = self._n_items
        # A document is split when its first and last items fall into different shards;
        # every shard but the last holds exactly shard_items items.
        size = self._shard_items
        self._n_split_docs += int(
            np.count_nonzero((first + starts) // size != (first + ends - 1) // size)
        )
        self._emit(stream)
        self._n_docs += len(docs)
        self._open_side().add_docs([d.id for d in docs], first + ends)
        for doc in docs:
            if doc.id is not None:
                self._id_hashes.add(doc.id)

    def _skip(self, doc: Document, err: DataError, index: int) -> None:
        if self._policy.on_data_error == "raise":
            raise err
        self._n_skipped += 1
        reason = f"{err.detail}: {err.why}" if err.detail else err.why
        self._open_side().add_skipped(index, doc.id, err.code.id, reason)
        self._issues.add(
            err.code,
            f"{_where(doc, index)} ({reason}); skipped, "
            f"see {naming.skipped_name(self._config.split)}",
            category=DataQualityWarning,
        )

    def _check_skip_ratio(self) -> None:
        policy = self._policy
        if self._n_input < policy.min_docs_for_ratio or self._n_skipped == 0:
            return
        ratio = self._n_skipped / self._n_input
        if ratio > policy.max_skip_ratio:
            raise ContractError(
                codes.TOO_MANY_SKIPPED,
                f"{self._n_skipped} of {self._n_input}",
                why=f"{ratio:.1%} of documents were skipped, above the limit of "
                f"{policy.max_skip_ratio:.1%}; a systematically broken input must not "
                "produce a dataset made of fragments",
                fix=f"see {naming.skipped_name(self._config.split)} in the partial "
                "directory for reasons; fix the input, or raise ErrorPolicy.max_skip_ratio",
            )

    def _check_sizes(
        self, docs: list[Document], lengths: npt.NDArray[np.int64], inputs: list[int]
    ) -> None:
        size = self._shard_items
        huge = np.flatnonzero(lengths > size)
        large = np.flatnonzero((lengths > LARGE_DOC_SHARE * size) & (lengths <= size))
        for code, found in (
            (codes.DOC_LARGER_THAN_SHARD, huge),
            (codes.DOC_LARGE_FOR_SHARD, large),
        ):
            if found.size:
                k = int(found[0])
                self._issues.add(
                    code,
                    f"{_where(docs[k], inputs[k])} has {int(lengths[k])} items, "
                    f"a shard holds {size}",
                    category=ShardingWarning,
                    count=int(found.size),
                )

    def _report_duplicates(self, partial: Path) -> None:
        split = self._config.split
        idx = np.load(partial / naming.ids_idx_name(split), mmap_mode="r", allow_pickle=False)

        def id_of(i: int) -> str | None:
            with (partial / naming.ids_name(split)).open("rb") as f:
                f.seek(int(idx[i]))
                doc_id: str | None = json.loads(f.read(int(idx[i + 1]) - int(idx[i])))["id"]
            return doc_id

        report = self._id_hashes.report(id_of)
        if report.count:
            example = f", e.g. {report.example!r}" if report.example is not None else ""
            self._issues.add(
                codes.DUPLICATE_DOC_ID,
                f"{report.count} documents repeat an earlier id{example}",
                category=DataQualityWarning,
                count=report.count,
            )

    def _emit(self, items: TokenArray) -> None:
        pos, total = 0, items.size
        while pos < total:
            if self._shard is None:
                self._shard = ShardFile(
                    self._open_partial(),
                    self._config.split,
                    len(self._closed_shards),
                    self._dtype.itemsize,
                )
            take = min(self._shard_items - self._shard.n_items, total - pos)
            self._shard.write(items[pos : pos + take])
            pos += take
            self._n_items += take
            if self._shard.n_items == self._shard_items:
                self._close_shard()

    def _close_shard(self) -> None:
        if self._shard is not None:
            self._closed_shards.append(self._shard.close())
            self._shard = None

    def _open_partial(self) -> Path:
        if self._partial is None:
            raise InternalError.wrap(RuntimeError("the writer is not open"))
        return self._partial

    def _open_side(self) -> SideFiles:
        if self._side is None:
            raise InternalError.wrap(RuntimeError("the writer is not open"))
        return self._side

    # --- finishing --------------------------------------------------------------------

    @property
    def stats(self) -> WriteStats:
        """Exact counters. Buffered documents are processed first, so reading stats
        after every ``add`` disables batching; read them periodically instead."""
        if self._partial is not None:
            self._flush()
        return WriteStats(
            n_input=self._n_input,
            n_docs=self._n_docs,
            n_skipped=self._n_skipped,
            n_items=self._n_items,
            n_shards=len(self._closed_shards),
            n_split_docs=self._n_split_docs,
        )

    @property
    def result(self) -> WriteResult:
        """The result of a finished write."""
        if self._result is None:
            raise ConfigError(
                codes.WRITER_NOT_FINISHED,
                str(self._target),
                why="the result exists only after the writer is closed",
                fix="read `result` after the `with` block, or call close() first",
            )
        return self._result

    @public_api
    def close(self) -> WriteResult:
        """Finish the split and publish the source."""
        if self._result is not None:
            return self._result
        if self._partial is None:
            self.open()
        self._flush()
        partial, side = self._open_partial(), self._open_side()
        if self._shard is not None:
            if self._shard.n_items:
                self._close_shard()
            else:
                self._shard.discard()
                self._shard = None
        side.finish()
        self._side = None
        self._report_duplicates(partial)

        split = self._config.split
        split_meta = SplitMeta(
            name=split,
            created_at=utc_timestamp(),
            n_items=self._n_items,
            n_docs=self._n_docs,
            n_skipped=self._n_skipped,
            has_split_docs=self._n_split_docs > 0,
            shards=tuple(self._closed_shards),
        )
        if self._existing is not None:
            meta = replace(self._existing.with_split(split_meta), tokbin_version=get_version())
        else:
            meta = Meta(
                schema_version=SCHEMA_VERSION,
                tokbin_version=get_version(),
                dtype=self._dtype.name,
                vocab_size=self._tok.vocab_size,
                tokenizer_id=self._tok.identifier,
                tokenizer_hash=self._tok.fingerprint(),
                eos_id=self._tok.eos_id,
                bos_id=self._tok.bos_id,
                splits=(split_meta,),
            )
        publish(
            partial,
            self._target,
            meta,
            replace_split=split if self._existing is not None else None,
        )
        self._partial = None

        self._issues.set_count(
            codes.DOCS_SPLIT_ACROSS_SHARDS,
            "their items continue in the next shard; readers handle this transparently",
            self._n_split_docs,
            level="info",
        )
        self._result = WriteResult(
            path=self._target,
            split=split,
            status="complete_with_issues" if self._issues.has_warnings else "complete",
            stats=self.stats,
            issues=self._issues.snapshot(),
        )
        return self._result

    def abort(self) -> None:
        """Drop the open shard and leave the partial directory as it is."""
        self._pending, self._pending_chars = [], 0
        if self._shard is not None:
            self._shard.discard()
            self._shard = None
        if self._side is not None:
            self._side.close()
            self._side = None
        if self._partial is not None:
            discard_open_shards(self._partial)
