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

## Interrupted writes

A write goes to `<name>.partial/` and becomes visible only when it is complete. It
saves a checkpoint after every closed shard; after a crash or `Ctrl+C` it continues
from there:

```python
ds.write("web", docs(), tokenizer, resume=True)
```

`docs()` must yield the same documents in the same order as before. At most one shard
of work is lost; the result is identical to an uninterrupted write.

## Mixtures

```python
from tokbin import Mixture

mix = Mixture.from_config("corpus")                 # weights from corpus/mix.json
x = mix.batch(batch_size=32, block_size=1024, rng=np.random.default_rng(0))
```

For torch, `tokbin.adapters.torch` has `WindowDataset` and `MixtureDataset`
(`pip install 'tokbin-core[torch]'`).

## Command line

```
tokbin ls corpus              # sources, states and sizes
tokbin info corpus            # tokens, tokenizer, mixture, remarks
tokbin info corpus/web        # dtype, splits, special tokens of one source
tokbin status corpus/web      # write state, unfinished writes included
tokbin verify corpus/web      # sha256 of every shard and all index files
tokbin clean corpus/web       # remove an unfinished write (corpus/web.partial)
tokbin pack corpus/web --tar  # corpus/web.tbpack.tar: compressed, with sha256 manifest
tokbin unpack web.tbpack.tar --into corpus   # unpack, verify, publish
tokbin doctor                 # version, mode, optional packages
```

Every command accepts `--json` (machine-readable output), `--no-color` and `--strict`
(warnings fail with exit code 1). Exit codes: 0 success, 1 error, 2 wrong usage,
3 integrity check failed, 4 missing dependency, 130 interrupted. Error codes are
listed in [docs/errors.md](docs/errors.md).

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
