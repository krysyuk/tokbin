# Statistics and verification

**In short:** tokbin answers three questions about your data. *What is in it?* (`info`)
*Is it finished?* (`status`) *Is it intact?* (`verify`). The first two are instant because
they read only small metadata files. `verify` reads every byte and checks sha256.

| Question | Python | Command line | Speed |
|---|---|---|---|
| what is in the corpus | `Dataset("corpus").info()` | `tokbin info corpus` | instant |
| what is in one source | `Dataset("corpus").status("web")` | `tokbin info corpus/web` | instant |
| is a write unfinished | `Dataset("corpus").status("web").partial` | `tokbin status corpus/web` | instant |
| are the files intact | `Dataset("corpus").verify("web")` | `tokbin verify corpus/web` | reads everything |

## The data used on this page

```python
# prepare_corpus.py
import warnings
from tokbin import Dataset, DataQualityWarning, WriterConfig, jsonl, txt_dir

warnings.simplefilter("ignore", DataQualityWarning)

ds = Dataset("corpus")
ds.write("web", jsonl("data/web"), "gpt2/tokenizer.json", config=WriterConfig(shard_bytes=2048))
ds.write("web", [("v-0", "Owls can turn their heads.")], "gpt2/tokenizer.json",
         config=WriterConfig(split="valid"))
ds.write("books", txt_dir("data/books"), "gpt2/tokenizer.json")
```

and a `corpus/mix.json` with `{"web": 3, "books": 1}`.

## The whole corpus

```python
# corpus_info.py
from tokbin import Dataset

info = Dataset("corpus").info()

print("ready:  ", info.ready)
print("tokens: ", info.n_items)
print("bytes:  ", info.n_bytes)
print("mix:    ", info.mix)
print("tokenizer:", info.tokenizer_id, info.tokenizer_hash)
for source in info.sources:
    print(f"  {source.name:6} {source.state:9} {source.n_items:6} tokens {source.n_docs:4} docs")
```

```text
ready:   True
tokens:  1888
bytes:   3776
mix:     (('web', 0.75), ('books', 0.25))
tokenizer: gpt2 31dba887f217f446
  books  complete     133 tokens    3 docs
  web    complete    1755 tokens   61 docs
```

- `ready` is `True` when every source is complete and nothing is wrong. Check it at the start
  of a training script.
- `tokenizer_hash` is a fingerprint of the tokenizer. It is the same in all sources here, so
  they can be mixed.

### `CorpusInfo` fields

| Field | Meaning |
|---|---|
| `path` | the corpus folder |
| `sources` | one `SourceInfo` per source, sorted by name |
| `ready` | every source complete, no problems |
| `n_items`, `n_bytes` | tokens and bytes in all sources |
| `mix` | weights from `mix.json`, normalized, or `None` |
| `tokenizer_id`, `tokenizer_hash` | the shared tokenizer, or `None` when sources differ |
| `issues`, `problems` | notes and errors about the corpus as a whole |

## One source

```python
# source_info.py
from tokbin import Dataset

web = Dataset("corpus").status("web")

print(web.state, web.dtype, "vocab", web.vocab_size, "eos", web.eos_id)
for split in web.splits:
    print(split)
```

```text
complete uint16 vocab 50257 eos 50256
SplitInfo(name='train', n_items=1746, n_docs=60, n_skipped=1, n_shards=2, n_bytes=3492, has_split_docs=True, created_at='2026-10-02T11:33:53Z')
SplitInfo(name='valid', n_items=9, n_docs=1, n_skipped=0, n_shards=1, n_bytes=18, has_split_docs=False, created_at='2026-10-02T11:33:53Z')
```

`Dataset("corpus").status("web")` is the same as `inspect_source("corpus/web")`, and
`Dataset("corpus").info()` is the same as `inspect_corpus("corpus")`. Use whichever reads
better in your code.

### The state of a source

| `state` | Meaning | What to do |
|---|---|---|
| `complete` | finished and readable | train on it |
| `partial` | only an unfinished write exists | [resume](resume.md) or `clean` |
| `corrupt` | files are missing, have the wrong size, or metadata is broken | see `problems`, restore the files |
| `outdated` | written by an older tokbin format | `tokbin migrate` |
| `unsupported` | written by a newer tokbin | upgrade tokbin |

### `SourceInfo` fields

| Field | Meaning |
|---|---|
| `name`, `path`, `state` | which source and how it is |
| `n_items`, `n_docs`, `n_shards`, `n_bytes` | totals over all splits |
| `splits` | one `SplitInfo` per split (fields below) |
| `dtype`, `vocab_size` | how tokens are stored |
| `tokenizer_id`, `tokenizer_hash`, `eos_id`, `bos_id` | the tokenizer used |
| `schema_version`, `tokbin_version` | format version and the tokbin that wrote it |
| `partial` | the unfinished write, if any: `n_closed_shards`, `n_items`, `n_input_consumed`, `resumable`, ... |
| `issues` | notes, e.g. `TB-D301` "documents were skipped" |
| `problems` | what is broken, each with `code`, `what`, `why`, `fix` |

| `SplitInfo` field | Meaning |
|---|---|
| `name` | `train`, `valid` or `test` |
| `n_items` | tokens, including EOS/BOS |
| `n_docs` | documents written |
| `n_skipped` | documents skipped while writing |
| `n_shards`, `n_bytes` | shard files and their total size |
| `has_split_docs` | some documents continue into the next shard |
| `created_at` | when the split was written (UTC) |

### As JSON

Every result has `to_dict()`, ready for `json.dumps`, logging or a web dashboard:

```python
# source_json.py
import json
from tokbin import Dataset

web = Dataset("corpus").status("web")
print(json.dumps(web.to_dict(), indent=2))
```

```text
{
  "path": "corpus/web",
  "name": "web",
  "state": "complete",
  "schema_version": 1,
  "tokbin_version": "0.1.0.dev0",
  "dtype": "uint16",
  "vocab_size": 50257,
  "tokenizer_id": "gpt2",
  "tokenizer_hash": "31dba887f217f446",
  "eos_id": 50256,
  "bos_id": null,
  "n_items": 1755,
  "n_docs": 61,
  "n_shards": 3,
  "n_bytes": 3510,
  "splits": [
    {
      "name": "train",
      "n_items": 1746,
      "n_docs": 60,
      "n_skipped": 1,
      "n_shards": 2,
      "n_bytes": 3492,
      "has_split_docs": true,
      "created_at": "2026-10-02T11:33:53Z"
    },
    {
      "name": "valid",
      "n_items": 9,
      "n_docs": 1,
      "n_skipped": 0,
      "n_shards": 1,
      "n_bytes": 18,
      "has_split_docs": false,
      "created_at": "2026-10-02T11:33:53Z"
    }
  ],
  "partial": null,
  "issues": [
    {
      "code": "TB-D301",
      "level": "info",
      "message": "Documents were skipped when the source was written: the reasons are in train-skipped.jsonl",
      "count": 1
    },
    {
      "code": "TB-S203",
      "level": "info",
      "message": "Documents are split across shards: in train; readers handle this transparently",
      "count": 1
    }
  ],
  "problems": []
}
```

The command line gives the same document with `tokbin info corpus/web --json`.

### As a pandas table

`to_dict()` also makes it easy to build a table with
[pandas](https://pandas.pydata.org/) (`pip install pandas`): one row per split of every
source.

```python
# corpus_table.py
import pandas as pd
from tokbin import Dataset

info = Dataset("corpus").info()

rows = [
    {"source": source.name, **split.to_dict()}
    for source in info.sources
    for split in source.splits
]
table = pd.DataFrame(rows).set_index(["source", "name"])
print(table[["n_items", "n_docs", "n_skipped", "n_shards", "n_bytes"]])
print()
print("tokens per source:")
print(table.groupby("source")["n_items"].sum())
```

```text
              n_items  n_docs  n_skipped  n_shards  n_bytes
source name                                                
books  train      133       3          0         1      266
web    train     1746      60          1         2     3492
       valid        9       1          0         1       18

tokens per source:
source
books     133
web      1755
Name: n_items, dtype: int64
```

## Statistics of the content

`info` and `status` count tokens and documents. For anything deeper, read the data: it is
just NumPy.

### Document lengths

```python
# doc_lengths.py
import numpy as np
from tokbin import read_source

src = read_source("corpus/web")
lengths = np.array([len(src.doc(i)) for i in range(src.n_docs)])

print("documents:      ", len(lengths))
print("tokens:         ", lengths.sum())
print("mean length:    ", round(lengths.mean(), 1))
print("median length:  ", np.median(lengths))
print("90th percentile:", np.percentile(lengths, 90))
print("longest:        ", lengths.max(), "->", src.id_of_doc(int(lengths.argmax())))
```

```text
documents:       60
tokens:          1746
mean length:     29.1
median length:   28.0
90th percentile: 40.0
longest:         41 -> part-00.jsonl:15
```

### The most frequent tokens

Reading the stream in chunks keeps memory small, whatever the size of the source:

```python
# top_tokens.py
import numpy as np
from tokenizers import Tokenizer
from tokbin import read_source

src = read_source("corpus/web")
tok = Tokenizer.from_file("corpus/web/tokenizer/tokenizer.json")

counts = np.zeros(src.meta.vocab_size, dtype=np.int64)
step = 1_000_000                                   # count in chunks: works for any size
for start in range(0, len(src), step):
    chunk = src.window(start, min(step, len(src) - start))
    counts += np.bincount(chunk, minlength=src.meta.vocab_size)

for token_id in counts.argsort()[::-1][:5]:
    print(f"{token_id:6} {tok.id_to_token(int(token_id))!r:18} {counts[token_id]}")
```

```text
    13 '.'                210
   546 'Ġabout'           105
 50256 '<|endoftext|>'    60
   892 'Ġthink'           60
  3194 'Ġwritten'         60
```

(`Ġ` is how GPT-2 shows a leading space.)

## Verification: are the files intact?

When tokbin writes a shard, it saves the shard's sha256 in `meta.json`. `verify` reads every
shard again, computes the sha256 and compares. It also checks the index files value by value.
Do it after copying a dataset to another machine, or when a training run behaves strangely.

```python
# verify_web.py
from tokbin import Dataset

report = Dataset("corpus").verify("web")

print("ok:", report.ok)
print("checked:", report.n_bytes_checked, "bytes")
for shard in report.shards:
    print(f"  {shard.name}  {shard.status}  {shard.actual_sha256[:16]}")
```

```text
ok: True
checked: 3510 bytes
  train-00000.bin  ok  9ce56e9b9ca5480c
  train-00001.bin  ok  74808de8d3e41d32
  valid-00000.bin  ok  b18e75226c61e700
```

### Showing progress

Verifying a terabyte takes a while. Pass a `progress` function; it is called with how many
bytes are done:

```python
# verify_progress.py
from tokbin import verify_source

def show(progress):
    print(f"{progress.done_bytes:>5} / {progress.total_bytes} bytes  {progress.file}")

verify_source("corpus/web", progress=show)
```

```text
 2048 / 3510 bytes  train-00000.bin
 3492 / 3510 bytes  train-00001.bin
 3510 / 3510 bytes  valid-00000.bin
```

### When something is broken

Let's damage a shard on purpose by overwriting two bytes:

```python
# break_a_shard.py  (do not do this to real data!)
with open("corpus/web/train-00001.bin", "r+b") as f:
    f.seek(10)
    f.write(b"\x00\x00")
```

`verify` finds it and tells you what to do:

```python
# verify_broken.py
from tokbin import Dataset

report = Dataset("corpus").verify("web")

print("ok:", report.ok, "| bad shards:", report.n_bad_shards)
for shard in report.shards:
    print(f"  {shard.name}  {shard.status}")
for problem in report.problems:
    print(problem.code, problem.what)
    print("  fix:", problem.fix)
```

```text
ok: False | bad shards: 1
  train-00000.bin  ok
  train-00001.bin  corrupt
  valid-00000.bin  ok
TB-I302 Shard is corrupted: corpus/web/train-00001.bin
  fix: download the shard again and repeat `tokbin verify corpus/web`
```

`verify` never raises for damaged data: it collects every problem into one report, so you see
them all at once. A shard's `status` is one of `ok`, `missing`, `wrong_size` or `corrupt`.

The same from the command line (exit code `3` means "integrity check failed"):

```bash
$ tokbin verify corpus/web
```

```text
corpus/web/

  ✔ train-00000.bin   2.0 KiB   sha256 ok
  ✖ train-00001.bin   1.4 KiB   sha256 mismatch
  ✔ valid-00000.bin      18 B   sha256 ok

[TB-I302] Shard is corrupted: corpus/web/train-00001.bin
  cause: sha256 676af03409c4... differs from 74808de8d3e4... in meta.json
  fix:   download the shard again and repeat `tokbin verify corpus/web`

✖ Status: damaged · 1 of 3 shards
```

> **Note** A missing shard or a shard of the wrong size is noticed even without `verify`:
> `info`, `status` and `read_source` check the sizes of all files when they open a source.
> Only a change **inside** a file of the right size needs the full `verify`.

## Next

- [Archives](archives.md): pack a source to move it to another machine, with sha256 checks.
- [Command line](cli.md)
