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
