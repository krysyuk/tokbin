# Reading data

**In short:** `read_source("corpus/web")` opens a source and gives you an object that behaves
like one long NumPy array of tokens. You can index it, slice it, take a whole document, or
sample random windows for training. Nothing is loaded into memory until you ask for it.

> **Note** Reading needs only `numpy`. It works with the light `tokbin-core` package too,
> for example on a training server.

## The data used on this page

The examples read a source `corpus/web` with a `train` split (the 60 JSON Lines documents
from [Data generators](generators.md#ready-made-reader-json-lines-jsonl)) and a small
`valid` split. To follow along, create it with this script:

```python
# prepare_web.py
import warnings
from tokbin import Dataset, DataQualityWarning, WriterConfig, jsonl

warnings.simplefilter("ignore", DataQualityWarning)  # one broken line in data/web, we know

ds = Dataset("corpus")
ds.write("web", jsonl("data/web"), "gpt2/tokenizer.json", config=WriterConfig(shard_bytes=2048))

valid = [("valid-0", "Owls can turn their heads almost all the way around."),
         ("valid-1", "Honey never spoils if it is kept sealed.")]
ds.write("web", valid, "gpt2/tokenizer.json", config=WriterConfig(split="valid"))
```

`shard_bytes=2048` makes 2 shards of 1024 tokens, so we can see what happens at the border.

## Open a source

```python
# open_source.py
from tokbin import read_source

src = read_source("corpus/web")            # the train split by default

print(src)
print("tokens:   ", len(src))
print("documents:", src.n_docs)
print("shards:   ", src.n_shards)
print("dtype:    ", src.dtype)
print("splits:   ", src.meta.split_names)
```

```text
Source('corpus/web', split='train', n_items=1746)
tokens:    1746
documents: 60
shards:    2
dtype:     uint16
splits:    ('train', 'valid')
```

Opening is instant even for a terabyte source: tokbin checks that every shard is there and
has the right size, but does not read the tokens.

## Tokens: index and slice

Think of `src` as one long array:

```python
# tokens.py
from tokbin import read_source

src = read_source("corpus/web")

print(src[0])          # the first token
print(src[-1])         # the last token
print(src[0:10])       # the first 10 tokens
```

```text
34
50256
[  34 1381  389 4697  621  749  661  892   13 4650]
```

> **Warning** `src[a:b]` reads the whole range into memory. `src[:]` on a 100 GB source
> needs 100 GB of RAM. For training, use windows (below).

## Documents

`src.doc(i)` returns document number `i` with its EOS token, and `src.id_of_doc(i)` its id.
To see the text, decode the tokens with the same tokenizer:

```python
# documents.py
from tokenizers import Tokenizer
from tokbin import read_source

src = read_source("corpus/web")
tok = Tokenizer.from_file("gpt2/tokenizer.json")

tokens = src.doc(1)
print(src.id_of_doc(1))
print(tokens)
print(tok.decode(tokens.tolist()))
```

```text
part-00.jsonl:1
[   49  1191   389  4697   621   749   661   892    13  4650  3835   423
   587  3194   546 18180    13  8990  1690  1265  2683   546 18180    13
 50256]
Rivers are older than most people think. Many books have been written about rivers. Children often ask questions about rivers.
```

> **Tip** A copy of the tokenizer is saved inside every source:
> `Tokenizer.from_file("corpus/web/tokenizer/tokenizer.json")`. You can decode a dataset
> even if the original tokenizer file is gone.

`id_of_doc` is instant for any `i`, so you can walk over documents:

```python
# walk_documents.py
from tokbin import read_source

src = read_source("corpus/web")

lengths = [len(src.doc(i)) for i in range(src.n_docs)]
print("shortest:", min(lengths), "longest:", max(lengths), "average:", sum(lengths) / len(lengths))

wanted = "part-01.jsonl:5"
index = next(i for i in range(src.n_docs) if src.id_of_doc(i) == wanted)
print(wanted, "is document", index)
```

```text
shortest: 17 longest: 41 average: 29.1
part-01.jsonl:5 is document 35
```

If the documents were written as plain text without ids, `id_of_doc` returns `None`.

## Windows

A **window** is a run of tokens starting anywhere, ignoring document borders. This is what
a language model is trained on.

```python
# windows.py
from tokbin import read_source

src = read_source("corpus/web")      # 2 shards of 1024 tokens each

inside = src.window(100, 8)          # tokens 100..107, all in shard 0
across = src.window(1020, 8)         # tokens 1020..1027: 4 from shard 0, 4 from shard 1

print(inside, "writeable:", inside.flags.writeable)
print(across, "writeable:", across.flags.writeable)

mine = inside.copy()                 # copy before changing
mine[0] = 0
print(mine)
```

```text
[13404  1909    13  1318   318   257  1402 13257] writeable: False
[ 1690  1265  2683   546 25476    13 23782   991] writeable: True
[    0  1909    13  1318   318   257  1402 13257]
```

- A window inside one shard is a **view** straight into the file: no copy, very fast, and
  read-only, so you cannot damage the dataset by accident.
- A window that crosses a shard border is glued into a new array.
- Either way you get the same tokens; you never need to think about shards.

Changing a read-only window raises `ValueError: assignment destination is read-only`. Call
`.copy()` first, as above.

## Random windows for training

`sample_windows(n, block_size, rng)` returns `n` random windows as one 2-D array, which is
exactly a training batch:

```python
# sample.py
import numpy as np
from tokbin import read_source

src = read_source("corpus/web")
rng = np.random.default_rng(seed=42)

batch = src.sample_windows(3, 8, rng)   # 3 windows of 8 tokens
print(batch)
print(batch.shape)
```

```text
[[50256    42  2737   389  4697   621   749   661]
 [   13 23782   991  2050   479  2737  1909    13]
 [  546 19432    13  8990  1690  1265  2683   546]]
(3, 8)
```

The randomness always comes from the `rng` you pass, never from a global seed. The same seed
gives the same batch on any machine, which makes experiments reproducible.

To train with PyTorch, see [PyTorch integration](pytorch.md): it wraps this into a
`Dataset` for a `DataLoader`.

## Other splits

```python
# valid_split.py
from tokbin import read_source

valid = read_source("corpus/web", split="valid")
print(valid)
print(valid.id_of_doc(0), valid.doc(0))
```

```text
Source('corpus/web', split='valid', n_items=26)
valid-0 [   46    86  7278   460  1210   511  6665  2048   477   262   835  1088
    13 50256]
```

Asking for a split that does not exist tells you which ones do:

```python
read_source("corpus/web", split="test")
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ConfigError: [TB-C301] Split is not in the source: corpus/web [test]
  where: tokbin.read.source.read_source
  cause: the source has the splits: train, valid
  fix:   pass one of the available splits
```

## Out of range

Positions are checked, so a wrong index is a clear error and not garbage data:

```python
read_source("corpus/web").doc(60)
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.OutOfRangeError: [TB-C302] Position is out of range: document 60
  where: tokbin.read.source.Source.doc
  cause: the split 'train' has 60 documents
  fix:   check the position against len() / n_docs
```

`OutOfRangeError` is also an `IndexError`, so `except IndexError` catches it too.

## Cheat sheet

| You want | Write |
|---|---|
| open the train split | `src = read_source("corpus/web")` |
| open another split | `read_source("corpus/web", split="valid")` |
| number of tokens / documents / shards | `len(src)`, `src.n_docs`, `src.n_shards` |
| one token | `src[i]` |
| a range of tokens (loads it all) | `src[a:b]` |
| a window of tokens | `src.window(start, length)` |
| document `i` and its id | `src.doc(i)`, `src.id_of_doc(i)` |
| a random batch | `src.sample_windows(n, block_size, np.random.default_rng(seed))` |
| metadata: tokenizer, EOS id, vocab | `src.meta.tokenizer_id`, `src.meta.eos_id`, `src.meta.vocab_size` |

A `Source` can be pickled and sent to another process: it carries only its path and reopens
the files there. That is why it works with multiprocess `DataLoader` workers.

## Next

- [Mixing sources](mixtures.md): read from several sources with weights.
- [PyTorch integration](pytorch.md)
