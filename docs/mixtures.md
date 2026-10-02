# Mixing sources

**In short:** real training data comes from several places: web pages, books, code. A
**mixture** reads from several sources at once with weights you choose, for example
"75% web, 25% books". You write the weights in a small `mix.json` file or pass them in code.

## The data used on this page

Two sources in one corpus: `web` (60 short web pages) and `books` (3 book openings):

```python
# prepare_corpus.py
import warnings
from tokbin import Dataset, DataQualityWarning, jsonl, txt_dir

warnings.simplefilter("ignore", DataQualityWarning)

ds = Dataset("corpus")
ds.write("web", jsonl("data/web"), "gpt2/tokenizer.json")
ds.write("books", txt_dir("data/books"), "gpt2/tokenizer.json")
```

## Step 1: write `mix.json`

Put a file `mix.json` in the corpus folder. Keys are source names, values are weights:

```text
corpus/
├── mix.json          ← {"web": 3, "books": 1}
├── books/
└── web/
```

```bash
$ cat corpus/mix.json
```

```text
{"web": 3, "books": 1}
```

Weights do not need to add up to 1: `{"web": 3, "books": 1}` means "3 parts web, 1 part
books", which is the same as `{"web": 0.75, "books": 0.25}`.

`tokbin info` shows the normalized weights:

```bash
$ tokbin info corpus
```

```text
corpus/   1,879 tokens · 2 sources

  ✔ books     133 tokens    3 docs   1 shard     266 B
  ✔ web     1.75K tokens   60 docs   1 shard   3.4 KiB
    └ [TB-D301] Documents were skipped when the source was written: the reasons are in train-skipped.jsonl

  tokenizer   gpt2 · 31dba887 · same in all sources
  mix         web 0.75 · books 0.25

✔ Status: ready for training
```

## Step 2: sample batches

```python
# mix_batches.py
import numpy as np
from tokbin import Mixture

mix = Mixture.from_config("corpus")     # reads corpus/mix.json
print(mix)
print(mix.names, mix.weights)

rng = np.random.default_rng(0)
batch = mix.batch(batch_size=4, block_size=8, rng=rng)
print(batch)
print(batch.shape, batch.dtype)
```

```text
Mixture('corpus', web=0.75, books=0.25, split='train')
('web', 'books') (0.75, 0.25)
[[ 1909    13 50256    49  1191   389  4697   621]
 [50256    33 32124   389  4697   621   749   661]
 [  749   661   892    13  4650  3835   423   587]
 [ 4697   621   749   661   892    13  4650  3835]]
(4, 8) uint16
```

For every row, the mixture first picks a source using the weights, then takes a random
window from that source. As always in tokbin, all randomness comes from the `rng` you pass:
the same seed gives the same batch.

## Do the weights really work?

`mix.choose(n, rng)` shows which source each of `n` samples would come from. Count them:

```python
# check_weights.py
import numpy as np
from tokbin import Mixture

mix = Mixture.from_config("corpus")
rng = np.random.default_rng(0)

picks = mix.choose(10_000, rng)          # which source each of 10,000 samples comes from
counts = np.bincount(picks)
for name, count in zip(mix.names, counts):
    print(f"{name:6} {count:5}  ({count / 10_000:.1%})")
```

```text
web     7495  (75.0%)
books   2505  (25.1%)
```

> **Note** Weights are about **samples**, not documents or files. `books` has 13 times fewer
> tokens than `web`, yet a quarter of all windows come from it. A small source with a big
> weight is repeated more often; that is how you up-sample high-quality data.

## Weights in code

`mix.json` is the default for the corpus. In an experiment you can override it without
touching the file:

```python
# custom_weights.py
from tokbin import Mixture

only_books = Mixture.from_config("corpus", {"books": 1})          # ignore mix.json
print(only_books)

half_half = Mixture("corpus", {"web": 0.5, "books": 0.5})
print(half_half)

no_books = Mixture("corpus", {"web": 1, "books": 0})              # weight 0: not even opened
print(no_books.names)
```

```text
Mixture('corpus', books=1, split='train')
Mixture('corpus', web=0.5, books=0.5, split='train')
('web',)
```

| You write | Weights come from |
|---|---|
| `Mixture.from_config("corpus")` | `corpus/mix.json` |
| `Mixture.from_config("corpus", {...})` | the dict; `mix.json` is ignored |
| `Mixture("corpus", {...})` | the dict (always required) |

A source with weight `0` is not opened at all, so it may even be unfinished or broken.

## Validation mixtures

Pass `split=` to mix the `valid` (or `test`) splits instead of `train`:

```python
valid_mix = Mixture.from_config("corpus", split="valid")
```

Every source with a positive weight must have that split.

## Things that can go wrong

A source that does not exist:

```python
Mixture("corpus", {"web": 1, "code": 1})
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ConfigError: [TB-C305] Dataset path not found: weights['code']: code
  cause: corpus/code is not a finished source
  fix:   write the source, fix the name, or give it weight 0
```

A window longer than a source (`books` has only 133 tokens):

```python
Mixture("corpus", {"books": 1}).batch(2, 500, np.random.default_rng(0))
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ConfigError: [TB-C304] Window is larger than the split: block_size=500
  where: tokbin.read.source.Source.sample_windows
  cause: the split 'train' has only 133 items
  fix:   use a smaller block_size
```

Sources written with **different tokenizers** cannot be mixed: token `464` would mean
different things. Opening such a mixture fails with `CompatibilityError` `TB-M301`, and
`tokbin info corpus` warns about it.

## Next

- [PyTorch integration](pytorch.md#mixing-sources-mixturedataset): `MixtureDataset` for a
  `DataLoader`.
- [Statistics and verification](statistics.md)
