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
