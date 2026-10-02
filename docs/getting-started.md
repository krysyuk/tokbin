# Getting started

**In short:** in five minutes you will install tokbin, turn three sentences into a tokenized
dataset, look at it from Python and from the command line, and pull random training windows
out of it.

## 1. Install

```bash
$ pip install tokbin
```

This installs the library, the `tokbin` command and the Hugging Face
[`tokenizers`](https://github.com/huggingface/tokenizers) package that tokbin uses to turn
text into tokens.

Check that it works:

```bash
$ tokbin doctor
```

```text
tokbin 0.1.0 · mode full · Python 3.12.9 · Darwin arm64
  ✔ numpy             2.5.1
  ✔ tokenizers        0.23.1
  ✔ zstd              0.25.0
  – torch             not installed   → needed for torch DataLoader adapter: pip install 'tokbin-core[torch]'
  ✔ huggingface-hub   1.27.0
```

`mode full` means you can both write and read. A line with `–` is an optional extra that is
not installed; you only need `torch` for the [PyTorch integration](pytorch.md).

## 2. Get a tokenizer

tokbin needs a `tokenizer.json` file. All examples in these docs use the GPT-2 tokenizer.
Download it into a folder called `gpt2`:

```bash
$ mkdir gpt2
$ curl -L -o gpt2/tokenizer.json https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
```

Your project folder now looks like this:

```text
my-project/
└── gpt2/
    └── tokenizer.json
```

> **Tip** Any Hugging Face model with a `tokenizer.json` works the same way: Llama, Qwen,
> Mistral, DeepSeek and so on.

## 3. Write your first dataset

Create a file `make_dataset.py` in `my-project/`:

```python
# make_dataset.py
from tokbin import Dataset

docs = [
    "The cat sat on the mat.",
    "Dogs like to play in the park.",
    "The sun is a star.",
]

ds = Dataset("corpus")                                   # 1. a folder for your data
result = ds.write("hello", docs, tokenizer="gpt2/tokenizer.json")  # 2. tokenize and save

print(result.status)
print(result.stats)
```

Run it:

```bash
$ python make_dataset.py
```

```text
complete
WriteStats(n_input=3, n_docs=3, n_skipped=0, n_items=25, n_shards=1, n_split_docs=0)
```

What happened:

- `Dataset("corpus")` points at a folder. It is created if it does not exist.
- `ds.write("hello", ...)` tokenized the three sentences, added an end-of-text token after
  each one, and saved everything into `corpus/hello/`.
- 3 documents became **25 tokens** (`n_items`), stored in **1 shard**.

Here is what appeared on disk:

```text
my-project/
├── gpt2/
│   └── tokenizer.json
├── make_dataset.py
└── corpus/
    └── hello/                     ← your first source
        ├── meta.json
        ├── tokenizer/
        ├── train-00000.bin        ← the 25 tokens
        ├── train-offsets.npy
        ├── train-ids.jsonl
        ├── train-ids.idx.npy
        └── train-skipped.jsonl
```

Do not worry about the extra files; [Core concepts](concepts.md#what-a-corpus-looks-like-on-disk)
explains each of them.

## 4. Read it back

Create `read_dataset.py`:

```python
# read_dataset.py
import numpy as np
from tokenizers import Tokenizer
from tokbin import read_source

src = read_source("corpus/hello")      # opens instantly, reads nothing yet

print(src)
print("tokens:   ", len(src))
print("documents:", src.n_docs)
print("dtype:    ", src.dtype)

print("document 0:", src.doc(0))

tok = Tokenizer.from_file("gpt2/tokenizer.json")
print("as text:   ", tok.decode(src.doc(0).tolist()))

rng = np.random.default_rng(0)
batch = src.sample_windows(4, 6, rng)  # 4 random windows of 6 tokens
print(batch)
```

```bash
$ python read_dataset.py
```

```text
Source('corpus/hello', split='train', n_items=25)
tokens:    25
documents: 3
dtype:     uint16
document 0: [  464  3797  3332   319   262  2603    13 50256]
as text:    The cat sat on the mat.
[[50256   464  4252   318   257  3491]
 [  711   287   262  3952    13 50256]
 [  588   284   711   287   262  3952]
 [ 2603    13 50256    35 18463   588]]
```

- `src.doc(0)` is the first document: 7 tokens of text plus `50256`, the end-of-text token.
- `sample_windows` is what you use for training: random slices of the stream, ready to be
  turned into a batch. The same `rng` seed always gives the same windows.

## 5. Look at it from the command line

```bash
$ tokbin info corpus/hello
```

```text
corpus/hello/   25 tokens · complete

  dtype       uint16 · vocab 50,257
  tokenizer   gpt2 · 31dba887 · eos 50256 · no bos
  format      schema 1 · written by tokbin 0.1.0

  split   tokens   docs   shards   size   skipped
  train       25      3        1   50 B         0

✔ Status: ready for training
```

```bash
$ tokbin ls corpus
```

```text
corpus/   1 source · 25 tokens · 50 B

  ✔ hello   complete   25 tokens   3 docs   1 shard   50 B
```

## Where to go next

| I want to... | Read |
|---|---|
| write real data from files, generators, JSON Lines | [Writing data](writing.md), [Data generators](generators.md) |
| survive Ctrl+C and crashes during a long write | [Resuming interrupted writes](resume.md) |
| read documents and windows | [Reading data](reading.md) |
| train a PyTorch model | [PyTorch integration](pytorch.md) |
| see statistics of my dataset | [Statistics and verification](statistics.md) |
| send a dataset to another machine | [Archives](archives.md) |
| use the command line only | [Command line](cli.md) |
