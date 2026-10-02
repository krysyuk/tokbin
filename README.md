# tokbin

Pretraining data for language models. tokbin tokenizes a text corpus once into binary
shards, reads random windows through `numpy.memmap` without loading the corpus into memory,
mixes sources by weight, packs datasets for transfer and verifies their integrity.

tokbin supports **pretraining only**: plain text packed into a continuous token stream. It
is not a tool for fine-tuning, SFT or chat data.

> Status: alpha. The API and the on-disk format may still change before 1.0.

## Is tokbin for you?

tokbin is a narrow tool: it prepares **pretraining data for language models**. It was built
for one workflow: tokenize a large text corpus ahead of time on a cheap machine, move it to
a rented GPU server, and start training at once, without paying for idle GPUs.

**Use it if** you pretrain or continue pretraining on plain text that is too large for
memory, prepare data on one machine and train on another, need writes that survive crashes,
or mix several sources by weight.

**Skip it if** you fine-tune on chats or instructions (no chat templates, loss masks or
padding), need attention masks at document borders, use a tokenizer that is not a Hugging
Face `tokenizer.json`, work with non-text data, need distributed writing or streaming from
object storage, or your data fits in memory. A 30-line `numpy.memmap` script may be all you
need.

The details, a comparison with a hand-written script and the measured costs are in
[Is tokbin for you?](https://github.com/krysyuk/tokbin/blob/main/docs/when-to-use.md)

## Documentation

The full documentation, with runnable examples and their output, is in [docs/](https://github.com/krysyuk/tokbin/blob/main/docs/README.md):

- [Getting started](https://github.com/krysyuk/tokbin/blob/main/docs/getting-started.md) and [core concepts](https://github.com/krysyuk/tokbin/blob/main/docs/concepts.md)
- [Writing data](https://github.com/krysyuk/tokbin/blob/main/docs/writing.md), [data generators](https://github.com/krysyuk/tokbin/blob/main/docs/generators.md),
  [resuming interrupted writes](https://github.com/krysyuk/tokbin/blob/main/docs/resume.md)
- [Reading data](https://github.com/krysyuk/tokbin/blob/main/docs/reading.md), [mixing sources](https://github.com/krysyuk/tokbin/blob/main/docs/mixtures.md),
  [PyTorch integration](https://github.com/krysyuk/tokbin/blob/main/docs/pytorch.md)
- [Statistics and verification](https://github.com/krysyuk/tokbin/blob/main/docs/statistics.md), [archives](https://github.com/krysyuk/tokbin/blob/main/docs/archives.md)
- [Command line](https://github.com/krysyuk/tokbin/blob/main/docs/cli.md), [errors and troubleshooting](https://github.com/krysyuk/tokbin/blob/main/docs/troubleshooting.md)

## Installation

```bash
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

mix = Mixture.from_config("corpus")    # weights from corpus/mix.json
x = mix.batch(batch_size=32, block_size=1024, rng=np.random.default_rng(0))
```

For torch, `tokbin.adapters.torch` has `WindowDataset` and `MixtureDataset`
(`pip install 'tokbin-core[torch]'`).

## Command line

```bash
tokbin build corpus/web --from-jsonl data/web --tokenizer gpt2/tokenizer.json
tokbin build corpus/web --resume            # after Ctrl+C or a crash: same settings
tokbin ls corpus                            # sources, states and sizes
tokbin info corpus                          # tokens, tokenizer, mixture, remarks
tokbin info corpus/web                      # dtype, splits, special tokens of one source
tokbin status corpus/web                    # write state, unfinished writes included
tokbin verify corpus/web                    # sha256 of every shard and all index files
tokbin clean corpus/web                     # remove an unfinished write (corpus/web.partial)
tokbin pack corpus/web --tar                # corpus/web.tbpack.tar: compressed, with sha256 manifest
tokbin unpack web.tbpack.tar --into corpus  # unpack, verify, publish
tokbin migrate corpus/web                   # convert to the current format schema (a copy)
tokbin rm corpus/web --yes                  # delete a source
tokbin doctor                               # version, mode, optional packages
```

Every command accepts `--json` (machine-readable output), `--no-color` and `--strict`
(warnings fail with exit code 1). Exit codes: 0 success, 1 error, 2 wrong usage,
3 integrity check failed, 4 missing dependency, 130 interrupted. Error codes are
listed in [docs/errors.md](https://github.com/krysyuk/tokbin/blob/main/docs/errors.md).

## Development

```bash
uv sync --all-packages
git config core.hooksPath scripts/hooks   # enable the pre-commit quality gate
```

The pre-commit hook runs ruff, mypy, the test suite and a guard against non-English
text in staged files. The full OS x Python matrix runs in GitHub Actions on demand and
on release tags.

## License

MIT
