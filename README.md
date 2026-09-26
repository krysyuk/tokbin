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
uv run ruff check && uv run ruff format --check
uv run mypy
uv run pytest
```

## License

MIT
