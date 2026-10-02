# Writing data

**In short:** you give tokbin documents and a tokenizer, and it saves the tokens to disk.
One call, `Dataset.write`, does it all. This page shows what a document may look like, how
to add a validation split, and which settings you can change.

All examples run in a project folder with the GPT-2 tokenizer in `gpt2/tokenizer.json`
(see [Getting started](getting-started.md#2-get-a-tokenizer)).

## The one call you need

```python
# write_animals.py
from tokbin import Dataset

docs = [
    ("animals/cat", "The cat sat on the mat."),
    ("animals/dog", "Dogs like to play in the park."),
    ("space/sun", "The sun is a star."),
]

ds = Dataset("corpus")
result = ds.write("animals", docs, tokenizer="gpt2/tokenizer.json")
print(result)
```

```text
WriteResult(path=PosixPath('corpus/animals'), split='train', status='complete', stats=WriteStats(n_input=3, n_docs=3, n_skipped=0, n_items=25, n_shards=1, n_split_docs=0), issues=(Issue(code='TB-C215', level='info', message="Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name", count=1),))
```

`ds.write(name, docs, tokenizer)` takes three things:

| Argument | What to pass |
|---|---|
| `name` | The name of the source. It becomes the folder `corpus/<name>/`. Letters, digits, `.`, `_` and `-`. |
| `docs` | Anything you can loop over that gives documents: a list, a generator, a file reader. See [Data generators](generators.md). |
| `tokenizer` | A path to `tokenizer.json`, or a `tokenizers.Tokenizer` object you already loaded. |

The `issues` in the result are notes about the write. `TB-C215` just tells you that tokbin
found the end-of-text token `<|endoftext|>` by itself. More on issues
[below](#reading-the-result).

## What a document can be

A document is one of these:

| Form | Example | When to use |
|---|---|---|
| text | `"The cat sat."` | quick experiments |
| `(id, text)` | `("animals/cat", "The cat sat.")` | **recommended** for real data |
| bytes | `b"The cat sat."` or `("animals/cat", b"...")` | text read from files as UTF-8 |
| `SkipDocument(id, reason)` | `SkipDocument("row-17", "empty")` | your reader found a broken record, see [Data generators](generators.md#skipping-broken-records) |

### Why ids are worth it

An id is just a name for the document. tokbin saves it next to the tokens, so later you can
ask "which document is this?":

```bash
$ cat corpus/animals/train-ids.jsonl
```

```text
{"id": "animals/cat"}
{"id": "animals/dog"}
{"id": "space/sun"}
```

Ids also let tokbin check that you feed the same documents when you
[resume an interrupted write](resume.md), and warn you about duplicates.

### Bytes work too

```python
# write_bytes.py
from tokbin import Dataset

docs = ["Hello world.".encode("utf-8"), "Bytes are fine too.".encode("utf-8")]
result = Dataset("corpus").write("raw", docs, tokenizer="gpt2/tokenizer.json")
print(result.stats.n_docs, "documents,", result.stats.n_items, "tokens")
```

```text
2 documents, 10 tokens
```

Bytes that are not valid UTF-8 do not crash the write: such a document is skipped and
recorded in `train-skipped.jsonl`.

### Do not mix forms

The first document decides the form for the whole write. Mixing plain text and
`(id, text)` is an error:

```python
# mixed_forms.py
from tokbin import Dataset

docs = ["just text", ("an-id", "text with an id")]
Dataset("corpus").write("mixed", docs, tokenizer="gpt2/tokenizer.json")
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ContractError: [TB-K202] Documents mix different forms
  where: tokbin.write.stream_writer.StreamWriter.write
  cause: the first document was plain text, this one is (doc_id, text) tuples
  fix:   make the generator yield every document in the same form
```

Every tokbin error looks like this: **what** happened, **cause** and **fix**. See
[Errors and troubleshooting](troubleshooting.md).

## Train and valid splits

A source can hold up to three splits: `train`, `valid` and `test`. Each `write` call writes
**one** split. Choose it with `WriterConfig(split=...)`; the default is `train`.

```python
# write_splits.py
from tokbin import Dataset, WriterConfig

train_docs = [(f"train-{i}", f"This is training document number {i}.") for i in range(100)]
valid_docs = [(f"valid-{i}", f"This is validation document number {i}.") for i in range(10)]

ds = Dataset("corpus")
config = WriterConfig(shard_bytes=350)          # tiny shards, just to see several of them

train = ds.write("numbers", train_docs, "gpt2/tokenizer.json", config=config)
valid = ds.write("numbers", valid_docs, "gpt2/tokenizer.json",
                 config=WriterConfig(split="valid"))

print("train:", train.stats)
print("valid:", valid.stats)
for issue in train.issues:
    print(issue.level, issue.code, issue.message)
```

```text
train: WriteStats(n_input=100, n_docs=100, n_skipped=0, n_items=800, n_shards=5, n_split_docs=4)
valid: WriteStats(n_input=10, n_docs=10, n_skipped=0, n_items=80, n_shards=1, n_split_docs=0)
info TB-C215 Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name
info TB-S203 Documents are split across shards: their items continue in the next shard; readers handle this transparently
```

Both splits now live in the same folder:

```text
corpus/numbers/
├── meta.json
├── tokenizer/
├── train-00000.bin   ┐
├── train-00001.bin   │
├── train-00002.bin   │ 800 train tokens in 5 shards of 175 tokens (350 bytes)
├── train-00003.bin   │
├── train-00004.bin   ┘
├── train-ids.idx.npy
├── train-ids.jsonl
├── train-offsets.npy
├── train-skipped.jsonl
├── valid-00000.bin     80 valid tokens in 1 shard
├── valid-ids.idx.npy
├── valid-ids.jsonl
├── valid-offsets.npy
└── valid-skipped.jsonl
```

`n_split_docs=4` and the `TB-S203` note say that 4 documents start in one shard and end in
the next one. This is normal and costs nothing: readers stitch shards together.

> **Note** `shard_bytes=350` is only for the demo. Keep the default (512 MiB) for real
> data. With very small shards tokbin also warns `TB-S202` when one document takes more
> than 5% of a shard.

### Replacing a split

Writing a split that already exists is refused, so you never lose data by accident:

```python
# write_again.py
from tokbin import Dataset

docs = [("new-0", "A brand new training set.")]
Dataset("corpus").write("numbers", docs, "gpt2/tokenizer.json")
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ConfigError: [TB-C206] Split already exists in the source: corpus/numbers [train]
  where: tokbin.write.stream_writer.StreamWriter.open
  cause: the source already has a 'train' split
  fix:   pass overwrite=True to replace this split, or write another split
```

Add `overwrite=True` to replace it. The other splits stay untouched:

```python
# write_again.py
from tokbin import Dataset

docs = [("new-0", "A brand new training set.")]
result = Dataset("corpus").write("numbers", docs, "gpt2/tokenizer.json", overwrite=True)
print(result.stats)
```

```text
WriteStats(n_input=1, n_docs=1, n_skipped=0, n_items=7, n_shards=1, n_split_docs=0)
```

```bash
$ tokbin info corpus/numbers
```

```text
corpus/numbers/   87 tokens · complete

  dtype       uint16 · vocab 50,257
  tokenizer   gpt2 · 31dba887 · eos 50256 · no bos
  format      schema 1 · written by tokbin 0.1.0.dev0

  split   tokens   docs   shards    size   skipped
  train        7      1        1    14 B         0
  valid       80     10        1   160 B         0

✔ Status: ready for training
```

The new `train` has 7 tokens; `valid` still has its 80.

> **Tip** A write is all-or-nothing. Until it finishes, the data goes into
> `corpus/<name>.partial/`, and readers keep seeing the old version. If the write fails,
> the old split is still there.

## Settings: `WriterConfig`

All settings are optional. Pass them by name:

```python
from tokbin import WriterConfig

config = WriterConfig(split="valid", shard_bytes=256 * 1024**2)
```

| Setting | Default | What it does |
|---|---|---|
| `split` | `"train"` | Which split to write: `"train"`, `"valid"` or `"test"`. |
| `shard_bytes` | `512 * 1024**2` (512 MiB) | Size of one shard file. The last shard holds the rest. |
| `append_eos` | `True` | Put the EOS token after every document. |
| `prepend_bos` | `False` | Put the BOS token before every document. |
| `eos_token` | found automatically | The EOS token text, e.g. `"</s>"`. |
| `bos_token` | found automatically | The BOS token text, e.g. `"<s>"`. |
| `dtype` | smallest that fits | `"uint8"`, `"uint16"` or `"uint32"`. GPT-2 (50,257 tokens) fits `uint16`, so 2 bytes per token. |
| `batch_docs` | `1024` | How many documents are tokenized together. Changes speed only, never the result. |

### Special tokens (EOS and BOS)

tokbin looks for the special tokens in this order:

1. what you pass in `eos_token` / `bos_token`;
2. `tokenizer_config.json` next to `tokenizer.json` (this is how Llama, Qwen, DeepSeek and
   most Hugging Face models describe them);
3. well-known names: `<|endoftext|>`, `</s>`, `<eos>`, `<|end_of_text|>` for EOS and
   `<s>`, `<bos>`, `<|begin_of_text|>`, `<|startoftext|>` for BOS.

GPT-2 has no BOS token, so asking for one fails with a clear message:

```python
# bos_missing.py
from tokbin import Dataset, WriterConfig

config = WriterConfig(prepend_bos=True)
Dataset("corpus").write("bos-missing", ["Hello there."], "gpt2/tokenizer.json", config=config)
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ConfigError: [TB-C209] Tokenizer has no beginning-of-sequence token
  where: tokbin.write.stream_writer.StreamWriter.open
  cause: prepend_bos=True, but no beginning-of-sequence token was found
  fix:   set WriterConfig(bos_token='...') to the right token, or prepend_bos=False
```

Tell it which token to use:

```python
# bos_explicit.py
from tokbin import Dataset, WriterConfig, read_source

config = WriterConfig(
    prepend_bos=True,                  # put a BOS token before every document
    bos_token="<|endoftext|>",         # GPT-2 has no separate BOS, so reuse its EOS token
)
Dataset("corpus").write("with-bos", ["Hello there."], "gpt2/tokenizer.json", config=config)

print(read_source("corpus/with-bos").doc(0))
```

```text
[50256 15496   612    13 50256]
```

`50256` now appears on both sides of `Hello there.`

## Reading the result

`write` returns a `WriteResult`. If something went really wrong, you get an exception
instead, so a returned result always means the data is on disk.

| Field | Meaning |
|---|---|
| `result.path` | Where the source is, e.g. `corpus/numbers`. |
| `result.split` | Which split was written. |
| `result.status` | `"complete"`, or `"complete_with_issues"` when there is at least one warning. |
| `result.stats.n_input` | Documents you gave, including skipped ones. |
| `result.stats.n_docs` | Documents written. |
| `result.stats.n_skipped` | Documents skipped (empty, broken UTF-8, ...). |
| `result.stats.n_items` | Tokens written, including EOS/BOS. |
| `result.stats.n_shards` | Shard files. |
| `result.stats.n_split_docs` | Documents that continue into the next shard. |
| `result.issues` | Notes and warnings: each has `code`, `level` (`"info"` or `"warning"`), `message`, `count`. |

## Writing one document at a time: `StreamWriter`

`Dataset.write` wants all documents as one iterable. If your code **pushes** documents one by
one instead (a web crawler, a callback, a chat log), use `StreamWriter` and its `add`
method:

```python
# write_chat.py
from tokbin import StreamWriter

with StreamWriter("corpus/chat", "gpt2/tokenizer.json") as writer:
    writer.add("msg-1", "Hi! How are you?")
    writer.add("msg-2", "Fine, thanks. And you?")
    writer.add("msg-3", "Great!")

print(writer.result.stats)
```

```text
WriteStats(n_input=3, n_docs=3, n_skipped=0, n_items=18, n_shards=1, n_split_docs=0)
```

- The first argument is the **source folder itself** (`corpus/chat`), not the corpus.
- `writer.add(text)` and `writer.add(doc_id, text)` both work.
- When the `with` block ends normally, the source is published. If an exception escapes the
  block, nothing is published and the unfinished write is kept for
  [resuming](resume.md).
- `StreamWriter` accepts the same `config=`, `policy=`, `overwrite=` and `resume=` as
  `Dataset.write`.

## Next

- [Data generators](generators.md): feed files, JSON Lines and your own readers.
- [Resuming interrupted writes](resume.md): what happens on Ctrl+C.
