# tokbin

Tokenized corpora as binary shards: write from a stream of documents, read random
windows through `numpy.memmap` without loading the corpus into memory, mix sources
by weight, pack datasets for transfer and verify their integrity.

> Status: pre-alpha, version 0.1 is in development.

## Installation

```
pip install tokbin          # full: write and read
pip install tokbin-core     # lightweight: read, verify, unpack
```

`tokbin-core` depends only on `numpy`. The `tokbin` meta-package adds `tokenizers`,
which is required for writing datasets.

## Development

```
uv sync --all-packages
git config core.hooksPath scripts/hooks   # enable the pre-commit quality gate
```

The pre-commit hook runs ruff, mypy, the test suite and a guard against non-English
text in staged files. The full OS x Python matrix runs in GitHub Actions on demand and
on release tags.

## License

MIT
