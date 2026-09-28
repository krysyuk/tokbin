# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/); versions follow
SemVer (PEP 440). A **Breaking** section comes first in an entry whenever it is not empty.

## [Unreleased]

schema_version: 1

### Added

- Project skeleton: the `tokbin-core` package and the `tokbin` meta-package (uv workspace).
- Exception and warning hierarchy, registry of stable error codes.
- Lazy import of optional dependencies, detection of the `full` / `core` mode.
- Public API boundary: foreign exceptions are wrapped in `InternalError`.
- On-disk format, schema 1: `meta.json` with per-split sections (`splits.train`,
  `splits.valid`, ...), `checkpoint.json`, `mix.json`, file naming, strict bounded
  metadata parsing, path confinement to the dataset root.
- Storage dtype derivation from the vocabulary size and an explicit range check before
  narrowing token ids (no silent wraparound).
- Writing: `Dataset.write`, `StreamWriter`, `WriterConfig`, `ErrorPolicy`. Stream
  packing into shards, document offsets, ids with an O(1) line index, skipped documents
  with reasons. Writes go to `<name>.partial/` and are published atomically; a split can
  be added to an existing source or replaced with `overwrite=True`.
- Hugging Face `tokenizers` adapter: EOS/BOS detection or explicit tokens, normalized
  fingerprint, truncation and padding disabled on a private copy, tokenizer stored next
  to the data. In `tokbin-core` writing fails with an error explaining the two packages.
- Reading: `read_source(path, split)` returns a `Source` with `len`, indexing and
  slicing, `window` across shard boundaries (zero-copy inside a shard),
  `sample_windows` with an explicit `numpy.random.Generator`, `doc(i)` and O(1)
  `id_of_doc(i)`. Opening checks shards and index files without reading them in full;
  a `Source` survives pickling by reopening its files. Reading never imports
  `tokenizers` and works in `tokbin-core`.
- `OutOfRangeError`, a `ConfigError` that is also an `IndexError`.
- Special tokens are taken from `tokenizer_config.json` next to `tokenizer.json` when
  not given explicitly (DeepSeek, Llama and other Hugging Face tokenizers work without
  settings); automatic choices are reported as `TB-C215`. The tokenizer copy stored in
  a source includes a `tokenizer_config.json` with the special tokens it used.
- Duplicate document ids are detected with 8 bytes of memory per document and
  reported once, at the end of a write, with an exact count and an example.
- Batched tokenization: documents are encoded with `encode_batch` in batches of
  `WriterConfig.batch_docs` (default 1024, bounded by 64 M characters of text) with
  vectorized bookkeeping. The output is byte-identical for any batch size; a failing
  batch falls back to one-by-one encoding so only the offending document is skipped.
- A frozen schema 1 dataset fixture in `tests/fixtures/schema_v1/`.
- Inspection: `inspect_source(path)` / `Dataset.status(name)` return a `SourceInfo`
  with the state (`complete`, `partial`, `corrupt`, `outdated`, `unsupported`),
  counters per split, an unfinished write with its checkpoint, warnings and problems;
  `inspect_corpus(root)` / `Dataset.info()` return a `CorpusInfo` with all sources,
  the shared tokenizer and the mixture. Inspection never reads shard data.
- Verification: `verify_source(path)` / `Dataset.verify(name)` hash every shard and
  check index files value by value, listing every problem in one `VerifyReport`, with
  a progress callback.
- `doctor()`: version, operating mode, optional packages and what they enable.
- CLI commands `ls`, `info`, `status`, `verify` and `doctor`, with `--json` (a stable
  envelope with the result, warnings and error), `--no-color` (and `NO_COLOR`),
  `--strict` for CI, ASCII symbols when the output is not UTF-8, a one-line progress
  indicator on terminals and periodic lines in logs. Exit codes: 0 success, 1 error,
  2 usage, 3 integrity check failed, 4 missing dependency, 130 interrupted.
- `docs/errors.md`, the table of error codes, generated from `tokbin.codes`.
- Reliable writing:
  - one writer per source: a `.lock` file with the pid, host and start time; a lock
    left by a crashed process is taken over, a live or remote writer is refused;
  - a checkpoint when a write starts and after every closed shard, with the lengths of
    the side files;
  - `resume=True` for `Dataset.write` and `StreamWriter` continues an interrupted
    write from its last checkpoint, byte for byte equal to an uninterrupted write.
    The generator must yield the same documents in the same order; when documents
    have ids, every consumed input is checked (not only the last one), otherwise a
    `TB-R204` warning is issued. Different settings or tokenizer are refused;
  - `KeyboardInterrupt` and other exceptions drop only the open shard; the checkpoint
    stays valid;
  - free disk space is checked before every new shard (at least two shards).
- `clean_source(path)`, `Dataset.clean(name)` and `tokbin clean <path>` remove an
  unfinished write (never one a live process is writing). `status` shows who is
  writing a source and how to resume or discard an unfinished write.
- Packing for transfer: `pack_source(path)` / `tokbin pack` compress every file of a
  complete source separately into `<name>.tbpack/` (or one reproducible
  `<name>.tbpack.tar` with `--tar`) with a manifest of sizes and sha256 of the original
  and compressed data. zstd through `compression.zstd` (Python 3.14+) or `zstandard`
  (the `zstd` extra), otherwise lzma with a `TB-P003` warning. A shard that does not
  match its sha256 is never packed.
- `unpack_pack(pack)` / `tokbin unpack` decompress into `<name>.partial/`, check the
  size and sha256 of every compressed and decompressed file, check the result against
  its `meta.json` and only then publish it. Decompression never writes more than the
  declared size (decompression bombs); archive entries with absolute paths, `..`,
  links or devices are refused on every Python version.
- `Mixture.from_config(root, weights=None)`: weighted sampling of windows from several
  sources (`mix.json` or explicit weights), `batch(batch_size, block_size, rng)`;
  sources with different tokenizers are refused (`TB-M301`).
- `tokbin.adapters.torch`: `WindowDataset` and `MixtureDataset`, map-style datasets
  for `DataLoader` returning `int64` tensors; importing the module does not import
  torch.
- Standard inputs `txt_dir(root)` and `jsonl(root, field="text")`: files in a
  portable deterministic order (resume depends on it), progress in input bytes. A
  record the generator cannot read is yielded as `SkipDocument(doc_id, reason)`, a new
  document form: it is recorded in `skipped.jsonl` (`TB-D206`) instead of crashing
  the write.
- `build_source(target, recipe)` and `tokbin build <target> --from-txt DIR |
  --from-jsonl PATH --tokenizer PATH [--shard-size 512M ...]`. The settings are saved
  in the partial directory, so `tokbin build <target> --resume` alone continues an
  interrupted build; `Ctrl+C` prints that command and exits with code 130.
- `remove_source(path)` and `tokbin rm <path> --yes`: only tokbin sources, never
  through a symlink or under a live writer; the source disappears in one rename.
- `migrate_source(path, out=None, in_place=False)` and `tokbin migrate`: a registry
  of schema steps applied on a hard-linked copy, checked, then published; a copy
  `<name>-v<schema>` by default, the source itself only with `--in-place`. Schema 1
  is current, so there are no steps yet.

### Fixed

- Calling a public function with wrong arguments raises the usual `TypeError`
  instead of `InternalError`.
