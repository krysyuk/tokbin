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
is left in place with a valid checkpoint; no finished source appears.

Reliability (spec 8.2, 8.3). One process writes a partial directory at a time
(``.lock``). A checkpoint is written when the write starts and after every closed
shard; it records the counters, the closed shards and the lengths of the side files.
``resume=True`` continues an interrupted write from its last checkpoint: side files are
cut back to the recorded lengths, the documents already consumed are skipped (their
ids are compared with the checkpoint), and the document that straddled the last shard
boundary is written without the items already in closed shards. The result is byte
for byte the one of an uninterrupted write, provided the generator yields the same
documents in the same order.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from tokbin import _fs, codes
from tokbin._boundary import mark_user_error, public_api
from tokbin._version import get_version
from tokbin.codes import Code
from tokbin.errors import (
    CompatibilityError,
    ConfigError,
    ContractError,
    DataError,
    DataQualityWarning,
    InternalError,
    ResumeError,
    ShardingWarning,
    TokbinError,
)
from tokbin.format import naming
from tokbin.format._time import utc_timestamp
from tokbin.format.checkpoint import Checkpoint, read_checkpoint, write_checkpoint
from tokbin.format.dtypes import TokenArray, dtype_for_vocab, resolve_dtype, to_dtype_checked
from tokbin.format.meta import Meta, ShardInfo, SplitMeta, read_meta
from tokbin.format.schema import SCHEMA_VERSION, source_kind
from tokbin.tokenizer.protocol import TokenizerProtocol
from tokbin.tokenizer.resolve import TokenizerLike, resolve_tokenizer
from tokbin.write.config import ErrorPolicy, WriterConfig, config_hash
from tokbin.write.documents import Document, DocumentNormalizer, Mode, decode_text
from tokbin.write.duplicates import IdHashes
from tokbin.write.lock import WriteLock
from tokbin.write.partial import (
    SideFiles,
    create_partial,
    discard_open_shards,
    discard_stale_outputs,
    partial_exists_error,
    partial_path,
    publish,
)
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


class _Skip(NamedTuple):
    input_index: int
    doc: Document
    error: DataError


def _resume_error(code: Code, partial: Path, why: str, fix: str) -> ResumeError:
    return ResumeError(code, str(partial), why=why, fix=fix)


def _reason(err: DataError) -> str:
    return f"{err.detail}: {err.why}" if err.detail else err.why


def read_checkpoint_for_resume(partial: Path) -> Checkpoint:
    try:
        return read_checkpoint(partial)
    except TokbinError as exc:
        raise _resume_error(
            codes.PARTIAL_DAMAGED,
            partial,
            f"its checkpoint cannot be read: {exc.what} ({exc.why})",
            f"delete the unfinished write (`tokbin clean {partial}`) and start again",
        ) from exc


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
        resume: bool = False,
    ) -> None:
        self._target = Path(path)
        naming.check_source_name(self._target.name)
        self._config = config if config is not None else WriterConfig()
        self._policy = policy if policy is not None else ErrorPolicy()
        self._overwrite = overwrite
        self._resume = resume
        self._normalizer = DocumentNormalizer(mode)
        self._tok: TokenizerProtocol = resolve_tokenizer(
            tokenizer, eos_token=self._config.eos_token, bos_token=self._config.bos_token
        )
        self._issues = IssueCollector()

        self._partial: Path | None = None
        self._lock: WriteLock | None = None
        self._side: SideFiles | None = None
        self._shard: ShardFile | None = None
        self._closed_shards: list[ShardInfo] = []
        self._existing: Meta | None = None
        self._result: WriteResult | None = None
        self._dtype = np.dtype("uint8")
        self._config_hash = ""
        self._shard_items = 0
        self._bos: int | None = None
        self._eos: int | None = None
        self._pending: list[_Pending] = []
        self._pending_chars = 0
        self._id_hashes = IdHashes()

        #: Input documents seen, skipped ones included (the input index of the next one).
        self._n_input = 0
        #: Counters of what is recorded in the side files.
        self._n_docs = 0
        self._n_skipped = 0
        self._n_items = 0
        self._n_split_docs = 0
        #: Id of the last input document of the previous batch (for checkpoints).
        self._prev_last_id: str | None = None

        # Resume state: inputs still to skip, the id the last skipped one must have,
        # and the items of the next document that are already in closed shards.
        self._skip_inputs = 0
        self._expected_last_id: str | None = None
        self._drop_items = 0
        #: Input index of the first document after the resumed position.
        self._resume_input = 0
        #: Per-document check of consumed inputs (see _check_consumed).
        self._check_every_id = False
        self._skipped_ids: dict[int, str | None] = {}
        self._consumed_kept = 0

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
        """Validate everything that can be validated up front and start the write.

        With ``resume=True`` an unfinished write of the same source continues from its
        last checkpoint; without one, the write starts from the beginning.
        """
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
        self._config_hash = config_hash(
            cfg, dtype=self._dtype, eos_id=self._tok.eos_id, bos_id=self._tok.bos_id
        )

        for code, message in self._tok.notes:
            self._issues.add(code, message, level="info")

        self._target.parent.mkdir(parents=True, exist_ok=True)
        partial = partial_path(self._target)
        if partial.exists():
            self._continue(partial)
        else:
            self._start(create_partial(self._target))

    def _start(self, partial: Path) -> None:
        """A fresh write: lock, tokenizer copy, empty side files, first checkpoint."""
        lock = WriteLock(partial)
        lock.acquire()
        try:
            self._tok.save(partial / naming.TOKENIZER_DIR)
            self._side = SideFiles(partial, self._config.split)
            self._partial, self._lock = partial, lock
            self._write_checkpoint(consumed=0, pending=0, last_id=None)
        except BaseException:
            self._partial = self._lock = None
            if self._side is not None:
                self._side.close()
                self._side = None
            lock.release()
            shutil.rmtree(partial, ignore_errors=True)
            raise

    def _continue(self, partial: Path) -> None:
        """An unfinished write exists: resume it, or refuse without ``resume=True``."""
        lock = WriteLock(partial)
        lock.acquire()  # ResumeError if another process is writing it
        if not self._resume:
            lock.release()
            raise partial_exists_error(partial)
        try:
            if not (partial / naming.CHECKPOINT).is_file():
                # The previous run stopped before its first checkpoint: nothing was
                # saved, so starting over is exactly resuming from zero.
                self._clear(partial)
                self._start_in(partial, lock)
                return
            self._restore(partial, read_checkpoint_for_resume(partial))
            self._partial, self._lock = partial, lock
        except BaseException:
            self._partial = self._lock = None
            if self._side is not None:
                self._side.close()
                self._side = None
            lock.release()
            raise

    def _clear(self, partial: Path) -> None:
        for entry in partial.iterdir():
            if entry.name == naming.LOCK:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def _start_in(self, partial: Path, lock: WriteLock) -> None:
        self._tok.save(partial / naming.TOKENIZER_DIR)
        self._side = SideFiles(partial, self._config.split)
        self._partial, self._lock = partial, lock
        self._write_checkpoint(consumed=0, pending=0, last_id=None)

    def _restore(self, partial: Path, cp: Checkpoint) -> None:
        """Bring the writer to the state of the checkpoint."""
        fix_settings = (
            "resume with exactly the settings and tokenizer of the interrupted write, "
            f"or delete it (`tokbin clean {partial}`) and start again"
        )
        mismatches = []
        if cp.split != self._config.split:
            mismatches.append(f"split {cp.split!r} vs {self._config.split!r}")
        if cp.tokenizer_hash != self._tok.fingerprint():
            mismatches.append(f"tokenizer {cp.tokenizer_hash} vs {self._tok.fingerprint()}")
        if cp.dtype != self._dtype.name:
            mismatches.append(f"dtype {cp.dtype} vs {self._dtype.name}")
        if cp.config_hash != self._config_hash:
            mismatches.append("WriterConfig (shard_bytes, append_eos, prepend_bos...)")
        if mismatches:
            raise _resume_error(
                codes.RESUME_SETTINGS_MISMATCH,
                partial,
                "the interrupted write used different settings: " + "; ".join(mismatches),
                fix_settings,
            )

        split = cp.split
        discard_open_shards(partial)
        discard_stale_outputs(partial, split)
        kept = {s.name for s in cp.closed_shards}
        for path in partial.glob(f"{split}-*.bin"):
            if path.name not in kept:  # closed after the last checkpoint
                path.unlink()
        for shard in cp.closed_shards:
            path = partial / shard.name
            size = path.stat().st_size if path.is_file() else -1
            if size != shard.n_bytes:
                raise _resume_error(
                    codes.PARTIAL_DAMAGED,
                    partial,
                    f"{shard.name} should have {shard.n_bytes} bytes, found "
                    + (f"{size}" if size >= 0 else "no file"),
                    f"delete the unfinished write (`tokbin clean {partial}`) and start again",
                )
        if not (partial / naming.TOKENIZER_DIR / naming.TOKENIZER_JSON).is_file():
            self._tok.save(partial / naming.TOKENIZER_DIR)

        self._side = SideFiles(partial, split, resume=cp.side_files)
        self._closed_shards = list(cp.closed_shards)
        self._n_input = cp.n_input_consumed
        self._n_docs = cp.n_docs
        self._n_skipped = cp.n_skipped
        self._n_items = cp.n_items
        self._n_split_docs = cp.n_split_docs
        self._prev_last_id = cp.last_input_id
        self._skip_inputs = cp.n_input_consumed
        self._expected_last_id = cp.last_input_id
        self._drop_items = cp.pending_doc_items
        self._resume_input = cp.n_input_consumed
        self._restore_ids(partial, cp.n_docs)
        self._restore_skipped(partial, cp.n_skipped)
        self._check_every_id = cp.n_docs > 0 and len(self._id_hashes) == cp.n_docs
        self._issues.add(
            codes.RESUMED,
            f"{len(cp.closed_shards)} closed shards kept, "
            f"{cp.n_input_consumed} input documents skipped",
            level="info",
        )

    def _restore_ids(self, partial: Path, n_docs: int) -> None:
        """Rebuild the duplicate detector from the ids already written."""
        with (partial / naming.ids_name(self._config.split)).open("rb") as f:
            for _, line in zip(range(n_docs), f, strict=False):
                doc_id = json.loads(line)["id"]
                if doc_id is not None:
                    self._id_hashes.add(doc_id)

    def _restore_skipped(self, partial: Path, n_skipped: int) -> None:
        """Count skips recorded before the interruption (their warnings were shown)."""
        counts: dict[str, int] = {}
        with (partial / naming.skipped_name(self._config.split)).open("rb") as f:
            for _, line in zip(range(n_skipped), f, strict=False):
                record = json.loads(line)
                counts[record["code"]] = counts.get(record["code"], 0) + 1
                self._skipped_ids[record["input_index"]] = record["id"]
        for code_id, count in counts.items():
            code = codes.CODES.get(code_id)
            if code is not None:
                self._issues.add(
                    code,
                    f"before the interruption; see {naming.skipped_name(self._config.split)}",
                    count=count,
                )

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
        if self._skip_inputs:
            # Resuming: this document was consumed before the interruption.
            self._check_consumed(doc)
            self._skip_inputs -= 1
            if self._skip_inputs == 0:
                self._check_resume_position(doc)
            return
        try:
            text: str | None = decode_text(doc.text)
            error: DataError | None = None
        except DataError as err:
            text, error = None, err
        self._pending.append(_Pending(doc, text, error))
        self._pending_chars += len(text) if text is not None else 0
        if len(self._pending) >= self._config.batch_docs or self._pending_chars >= _BATCH_CHARS:
            self._flush()

    def _check_consumed(self, doc: Document) -> None:
        """Compare a skipped-over document with what the interrupted write recorded.

        When every written document had an id, each consumed input is checked: a
        skipped one against ``skipped.jsonl``, a written one against the id hashes
        kept for duplicate detection. Otherwise only the last position is checked.
        """
        if not self._check_every_id:
            return
        index = self._resume_input - self._skip_inputs
        if index in self._skipped_ids:
            same = doc.id == self._skipped_ids[index]
        else:
            same = doc.id is not None and (
                IdHashes.digest(doc.id) == self._id_hashes.at(self._consumed_kept)
            )
            self._consumed_kept += 1
        if not same:
            raise self._order_error(
                f"input #{index} is {doc.id!r}, the interrupted write had another "
                "document at this position"
            )

    def _check_resume_position(self, doc: Document) -> None:
        """The last skipped document must be the last one consumed before."""
        expected = self._expected_last_id
        if doc.id is None and expected is None:
            self._issues.add(
                codes.RESUME_ORDER_UNCHECKED,
                "documents have no ids; make sure the generator yields them in the same "
                "order as before",
                category=DataQualityWarning,
            )
            return
        if doc.id != expected:
            raise self._order_error(
                f"input #{self._n_input - 1} is {doc.id!r}, the interrupted write "
                f"consumed {expected!r} at that position"
            )

    def _order_error(self, why: str) -> ResumeError:
        return ResumeError(
            codes.RESUME_INPUT_MISMATCH,
            str(self._partial),
            why=why,
            fix="resume with a generator that yields the same documents in the same "
            f"order, or delete the unfinished write (`tokbin clean {self._partial}`) "
            "and start again",
        )

    def _flush(self) -> None:
        """Process the buffered documents, in order."""
        if not self._pending:
            return
        entries, self._pending, self._pending_chars = self._pending, [], 0
        encoded = iter(self._encode_many([e.text for e in entries if e.text is not None]))

        base = self._n_input
        kept_docs: list[Document] = []
        kept_items: list[TokenArray] = []
        kept_inputs: list[int] = []
        skips: list[_Skip] = []
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
                self._note_skip(entry.doc, error, index)
                skips.append(_Skip(index, entry.doc, error))
            self._check_skip_ratio(len(skips))
        batch_ids = [e.doc.id for e in entries]
        self._write_batch(kept_docs, kept_items, kept_inputs, skips, base, batch_ids)

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
        self,
        docs: list[Document],
        items: list[TokenArray],
        inputs: list[int],
        skips: list[_Skip],
        base: int,
        batch_ids: list[str | None],
    ) -> None:
        """Emit a batch and record it in the side files, checkpointing at shard ends.

        Side file entries are written in input order, and only for inputs that are
        complete when a shard closes, so that every checkpoint describes a prefix of
        the input exactly.
        """
        side = self._open_side()
        n = len(docs)
        size = self._shard_items
        drop = 0
        if n:
            body_lengths = np.fromiter((a.size for a in items), dtype=np.int64, count=n)
            extra = int(self._bos is not None) + int(self._eos is not None)
            lengths = body_lengths + extra
            ends = np.cumsum(lengths)
            starts = ends - lengths
            stream = self._assemble(items, starts, ends)
            if self._drop_items:
                drop = self._take_resumed_prefix(stream, lengths, inputs)
        elif self._drop_items:
            self._take_resumed_prefix(np.empty(0, self._dtype), np.empty(0, np.int64), inputs)
        if n:
            self._check_sizes(docs, lengths, inputs)
            first = self._n_items - drop
            g_ends = first + ends
            # A document is split when its first and last items fall into different
            # shards; every shard but the last holds exactly shard_items items.
            split = (first + starts) // size != (first + ends - 1) // size
            doc_ids = [d.id for d in docs]
        else:
            stream = np.empty(0, dtype=self._dtype)
            first, g_ends, starts = self._n_items, np.empty(0, np.int64), np.empty(0, np.int64)
            split = np.empty(0, dtype=bool)
            doc_ids = []

        recorded = 0
        next_skip = 0

        def record(k: int, consumed: int) -> None:
            """Write side entries of kept docs [recorded, k) and skips before ``consumed``."""
            nonlocal recorded, next_skip
            if k > recorded:
                side.add_docs(doc_ids[recorded:k], g_ends[recorded:k])
                self._n_docs += k - recorded
                self._n_split_docs += int(np.count_nonzero(split[recorded:k]))
                for doc_id in doc_ids[recorded:k]:
                    if doc_id is not None:
                        self._id_hashes.add(doc_id)
                recorded = k
            while next_skip < len(skips) and skips[next_skip].input_index < consumed:
                skip = skips[next_skip]
                side.add_skipped(
                    skip.input_index, skip.doc.id, skip.error.code.id, _reason(skip.error)
                )
                self._n_skipped += 1
                next_skip += 1

        def on_full(position: int) -> tuple[int, int, str | None]:
            """A shard is full at global ``position``: record what is complete."""
            k = int(np.searchsorted(g_ends, position, side="right"))
            if k < n:
                consumed, pending = inputs[k], position - int(first + starts[k])
            else:
                consumed, pending = inputs[-1] + 1, 0
            record(k, consumed)
            last = consumed - 1
            last_id = batch_ids[last - base] if last >= base else self._prev_last_id
            return consumed, pending, last_id

        self._emit(stream[drop:], on_full)
        record(n, base + len(batch_ids))
        self._prev_last_id = batch_ids[-1]

    def _assemble(
        self,
        items: list[TokenArray],
        starts: npt.NDArray[np.int64],
        ends: npt.NDArray[np.int64],
    ) -> TokenArray:
        """Concatenate document bodies, inserting BOS and EOS where configured."""
        body = np.concatenate(items)
        if self._bos is None and self._eos is None:
            return body.astype(self._dtype, copy=False)
        stream = np.empty(int(ends[-1]), dtype=self._dtype)
        is_body = np.ones(stream.size, dtype=bool)
        if self._bos is not None:
            stream[starts] = self._bos
            is_body[starts] = False
        if self._eos is not None:
            stream[ends - 1] = self._eos
            is_body[ends - 1] = False
        stream[is_body] = body
        return stream

    def _take_resumed_prefix(
        self, stream: TokenArray, lengths: npt.NDArray[np.int64], inputs: list[int]
    ) -> int:
        """The first document after resume straddled the last shard boundary.

        Its first items are already in closed shards: check they are the same items
        (this also verifies the document order), and do not write them again.
        """
        drop, self._drop_items = self._drop_items, 0
        if not inputs or inputs[0] != self._resume_input or int(lengths[0]) <= drop:
            raise self._order_error(
                f"input #{self._resume_input} continued into the next shard before the "
                "interruption, but now it is skipped or shorter"
            )
        if not np.array_equal(stream[:drop], self._tail(drop)):
            raise self._order_error(
                f"the start of input #{self._resume_input} differs from the items already "
                "written for it"
            )
        return drop

    def _tail(self, n: int) -> TokenArray:
        """The last ``n`` items of the closed shards."""
        partial = self._open_partial()
        parts: list[TokenArray] = []
        for info in reversed(self._closed_shards):
            take = min(n, info.n_items)
            parts.append(
                np.fromfile(
                    partial / info.name,
                    dtype=self._dtype,
                    count=take,
                    offset=(info.n_items - take) * self._dtype.itemsize,
                )
            )
            n -= take
            if n == 0:
                break
        return np.concatenate(parts[::-1])

    def _note_skip(self, doc: Document, err: DataError, index: int) -> None:
        if self._policy.on_data_error == "raise":
            raise err
        self._issues.add(
            err.code,
            f"{_where(doc, index)} ({_reason(err)}); skipped, "
            f"see {naming.skipped_name(self._config.split)}",
            category=DataQualityWarning,
        )

    def _check_skip_ratio(self, unrecorded: int) -> None:
        policy = self._policy
        skipped = self._n_skipped + unrecorded
        if self._n_input < policy.min_docs_for_ratio or skipped == 0:
            return
        ratio = skipped / self._n_input
        if ratio > policy.max_skip_ratio:
            raise ContractError(
                codes.TOO_MANY_SKIPPED,
                f"{skipped} of {self._n_input}",
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

    def _emit(
        self, items: TokenArray, on_full: Callable[[int], tuple[int, int, str | None]]
    ) -> None:
        pos, total = 0, items.size
        while pos < total:
            if self._shard is None:
                self._check_space()
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
                consumed, pending, last_id = on_full(self._n_items)
                self._close_shard()
                self._write_checkpoint(consumed=consumed, pending=pending, last_id=last_id)

    def _check_space(self) -> None:
        """At least two shards of free space before a new shard starts (spec 7.4)."""
        partial = self._open_partial()
        need = 2 * self._config.shard_bytes
        free = _fs.free_bytes(partial)
        if free < need:
            raise ConfigError(
                codes.DISK_SPACE_LOW,
                str(partial),
                why=f"{free:,} bytes are free, a new shard needs at least two shards "
                f"({need:,} bytes) of free space",
                fix="free disk space and continue the write with resume=True, or write "
                "to another disk",
            )

    def _write_checkpoint(self, *, consumed: int, pending: int, last_id: str | None) -> None:
        """Record the state after a closed shard (the shard is already durable)."""
        lengths = self._open_side().lengths()
        checkpoint = Checkpoint(
            schema_version=SCHEMA_VERSION,
            split=self._config.split,
            dtype=self._dtype.name,
            config_hash=self._config_hash,
            tokenizer_hash=self._tok.fingerprint(),
            n_input_consumed=consumed,
            pending_doc_items=pending,
            last_input_id=last_id,
            n_items=self._n_items,
            n_docs=self._n_docs,
            n_skipped=self._n_skipped,
            n_split_docs=self._n_split_docs,
            closed_shards=tuple(self._closed_shards),
            side_files=lengths,
            updated_at=utc_timestamp(),
        )
        write_checkpoint(self._open_partial(), checkpoint)

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
        try:
            return self._finish()
        except BaseException:
            self.abort()
            raise

    def _finish(self) -> WriteResult:
        self._flush()
        if self._skip_inputs:
            raise self._order_error(
                f"the input ended after {self._n_input - self._skip_inputs} documents, "
                f"but the interrupted write had already consumed {self._n_input}"
            )
        if self._drop_items:
            raise self._order_error(
                f"the input ended before input #{self._resume_input}, which the "
                "interrupted write had started"
            )
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
        if self._lock is not None:
            self._lock.release()  # the file went away with the publication
        self._partial = self._lock = None

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
        """Drop the open shard and unwritten batch, release the lock.

        The partial directory stays with its last checkpoint, ready for ``resume=True``.
        """
        self._pending, self._pending_chars = [], 0
        if self._shard is not None:
            self._shard.discard()
            self._shard = None
        if self._side is not None:
            self._side.close()
            self._side = None
        if self._partial is not None:
            discard_open_shards(self._partial)
        if self._lock is not None:
            self._lock.release()
            self._lock = None
        self._partial = None
