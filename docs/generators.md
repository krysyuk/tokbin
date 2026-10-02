# Data generators

**In short:** `Dataset.write` reads documents from anything you can loop over. Use the
ready-made readers `txt_dir` and `jsonl` for files, or write a small Python generator for
anything else: CSV, a database, a Hugging Face dataset. This page shows each of them.

Why a generator and not a list? A list must fit in memory. A generator hands over one
document at a time, so you can tokenize a 500 GB corpus on a laptop.

## Your own generator in 5 lines

A generator is a function with `yield` instead of `return`. Every `yield` gives tokbin one
document:

```python
# write_hello.py
from tokbin import Dataset

def my_documents():
    for i in range(5):
        yield f"doc-{i}", f"Document number {i} says hello."

result = Dataset("corpus").write("hello", my_documents(), "gpt2/tokenizer.json")
print(result.stats)
```

```text
WriteStats(n_input=5, n_docs=5, n_skipped=0, n_items=35, n_shards=1, n_split_docs=0)
```

Note the parentheses: pass `my_documents()` (the generator), not `my_documents` (the
function).

What the generator may yield is described in
[Writing data](writing.md#what-a-document-can-be): `text`, `(id, text)`, bytes, or a
`SkipDocument`.

## Ready-made reader: a folder of text files (`txt_dir`)

Every `*.txt` file becomes one document. Say you have:

```text
data/books/
├── alice.txt
├── moby_dick.txt
└── pride.txt
```

Let's peek at what `txt_dir` yields:

```python
# peek_books.py
from tokbin import txt_dir

books = txt_dir("data/books")
for doc_id, text in books:
    print(doc_id, "->", text[:40])
```

```text
alice.txt -> b'Alice was beginning to get very tired of'
moby_dick.txt -> b'Call me Ishmael. Some years ago, never m'
pride.txt -> b'It is a truth universally acknowledged, '
```

- The id is the path of the file inside the folder.
- The text is raw bytes; tokbin decodes them as UTF-8.
- Files are always in the same order (sorted by path), on every operating system.
- Subfolders are searched too.

Write it:

```python
# write_books.py
from tokbin import Dataset, txt_dir

result = Dataset("corpus").write("books", txt_dir("data/books"), "gpt2/tokenizer.json")
print(result.stats)
```

```text
WriteStats(n_input=3, n_docs=3, n_skipped=0, n_items=133, n_shards=1, n_split_docs=0)
```

Use `pattern` for other extensions: `txt_dir("src", pattern="*.py")` turns every Python
file into a document.

## Ready-made reader: JSON Lines (`jsonl`)

A `.jsonl` file has one JSON object per line. This is how most web and chat datasets are
shipped:

```text
data/web/
├── part-00.jsonl
└── part-01.jsonl
```

```bash
$ head -2 data/web/part-00.jsonl
```

```text
{"url": "https://example.com/cats/0", "text": "Cats are older than most people think. Many books have been written about cats."}
{"url": "https://example.com/rivers/1", "text": "Rivers are older than most people think. Many books have been written about rivers. Children often ask questions about rivers."}
```

`jsonl` takes the `"text"` field of every line:

```python
# peek_web.py
from tokbin import jsonl

web = jsonl("data/web")
for i, doc in enumerate(web):
    print(doc)
    if i == 2:
        break
```

```text
('part-00.jsonl:0', 'Cats are older than most people think. Many books have been written about cats.')
('part-00.jsonl:1', 'Rivers are older than most people think. Many books have been written about rivers. Children often ask questions about rivers.')
('part-00.jsonl:2', 'Maps are older than most people think. Many books have been written about maps. Children often ask questions about maps. Scientists still study maps today.')
```

The id is `file:line`, with lines counted from 0. Now write it. The last line of
`part-01.jsonl` is broken on purpose: it has no `"text"` field.

```python
# write_web.py
from tokbin import Dataset, jsonl

result = Dataset("corpus").write("web", jsonl("data/web"), "gpt2/tokenizer.json")
print(result.status)
print(result.stats)
for issue in result.issues:
    print(issue.level, issue.code, issue.message)
```

```text
.../tokbin/write/stream_writer.py:516: DataQualityWarning: [TB-D206] Input record cannot be read: document 'part-01.jsonl:30' (no 'text' field); skipped, see train-skipped.jsonl
  self._flush()
complete_with_issues
WriteStats(n_input=61, n_docs=60, n_skipped=1, n_items=1746, n_shards=1, n_split_docs=0)
info TB-C215 Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name
warning TB-D206 Input record cannot be read: document 'part-01.jsonl:30' (no 'text' field); skipped, see train-skipped.jsonl
```

The broken line did not stop the write. It was skipped, Python printed a warning, and the
reason was saved:

```bash
$ cat corpus/web/train-skipped.jsonl
```

```text
{"input_index": 60, "id": "part-01.jsonl:30", "code": "TB-D206", "reason": "no 'text' field"}
```

Lines that are not valid JSON, not an object, or have a non-string field are skipped the
same way.

### Another field or extension

```python
# peek_chats.py
from tokbin import jsonl

chats = jsonl("data/chats", field="message", pattern="*.json")
print(list(chats))
```

```text
[('day1.json:0', 'Hi there'), ('day1.json:1', 'Bye')]
```

### Showing progress

`txt_dir` and `jsonl` know how many bytes they have read. Wrap them in your own generator to
print progress:

```python
# write_with_progress.py
import warnings
from tokbin import Dataset, DataQualityWarning, jsonl

warnings.simplefilter("ignore", DataQualityWarning)   # we know about the broken line

web = jsonl("data/web")

def with_progress(source):
    for n, doc in enumerate(source, start=1):
        yield doc
        if n % 20 == 0:
            print(f"{n} docs, {source.bytes_read} / {source.total_bytes} bytes, now reading {source.current}")

Dataset("corpus").write("web-progress", with_progress(web), "gpt2/tokenizer.json")
```

```text
20 docs, 3930 / 11851 bytes, now reading part-00.jsonl
40 docs, 7870 / 11851 bytes, now reading part-01.jsonl
60 docs, 11810 / 11851 bytes, now reading part-01.jsonl
```

> **Tip** The command `tokbin build` does all of this for you, with a progress bar:
> `tokbin build corpus/web --from-jsonl data/web --tokenizer gpt2/tokenizer.json`.
> See [Command line](cli.md#tokbin-build).

## Recipes for other sources

### CSV

```text
data/articles.csv
─────────────────
id,title,body
1,Tea,Tea is a drink made from leaves.
2,Empty,
3,Milk,Milk comes from cows and goats.
```

```python
# write_articles.py
import csv
from tokbin import Dataset, SkipDocument

def read_articles(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc_id = f"article-{row['id']}"
            if not row["body"].strip():
                yield SkipDocument(doc_id, "empty body")
                continue
            yield doc_id, f"{row['title']}\n\n{row['body']}"

result = Dataset("corpus").write("articles", read_articles("data/articles.csv"), "gpt2/tokenizer.json")
print(result.stats)
```

```text
.../tokbin/write/stream_writer.py:516: DataQualityWarning: [TB-D206] Input record cannot be read: document 'article-2' (empty body); skipped, see train-skipped.jsonl
  self._flush()
WriteStats(n_input=3, n_docs=2, n_skipped=1, n_items=25, n_shards=1, n_split_docs=0)
```

### A database

```python
# write_forum.py
import sqlite3
from tokbin import Dataset

db = sqlite3.connect("data/forum.db")
db.execute("CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, text TEXT)")
db.executemany("INSERT OR REPLACE INTO posts VALUES (?, ?)",
               [(1, "First post!"), (2, "Welcome to the forum."), (3, "Thanks, glad to be here.")])

def posts():
    for post_id, text in db.execute("SELECT id, text FROM posts ORDER BY id"):
        yield f"post-{post_id}", text

result = Dataset("corpus").write("forum", posts(), "gpt2/tokenizer.json")
print(result.stats)
```

```text
WriteStats(n_input=3, n_docs=3, n_skipped=0, n_items=18, n_shards=1, n_split_docs=0)
```

> **Warning** Always add `ORDER BY` (or any stable order). To
> [resume](resume.md) an interrupted write, the generator must give the same documents in the
> same order as before.

### A Hugging Face dataset

The [`datasets`](https://huggingface.co/docs/datasets) library (`pip install datasets`) can
read a dataset row by row with `streaming=True`, so nothing is loaded into memory. Here are
20,000 training and 1,000 validation stories from
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), whose parquet files
were downloaded into `TinyStories/data/`:

```python
# prepare_tinystories.py
from itertools import islice

from datasets import load_dataset
from tokbin import Dataset, WriterConfig

stories = load_dataset(
    "parquet",
    data_files={"train": "TinyStories/data/train-*.parquet",
                "validation": "TinyStories/data/validation-*.parquet"},
    streaming=True,                    # read row by row, never load everything
)

def tiny_stories(split, limit):
    for i, row in enumerate(islice(stories[split], limit)):
        yield f"{split}/{i}", row["text"]

ds = Dataset("corpus")
train = ds.write("tinystories", tiny_stories("train", 20_000), "gpt2/tokenizer.json")
valid = ds.write("tinystories", tiny_stories("validation", 1_000), "gpt2/tokenizer.json",
                 config=WriterConfig(split="valid"))
print("train:", train.stats.n_docs, "stories,", train.stats.n_items, "tokens")
print("valid:", valid.stats.n_docs, "stories,", valid.stats.n_items, "tokens")
```

```text
train: 20000 stories, 4466061 tokens
valid: 1000 stories, 194559 tokens
```

To read straight from the Hugging Face Hub without downloading first, replace the
`load_dataset(...)` call with `load_dataset("roneneldan/TinyStories", streaming=True)`. The
generator stays the same. This dataset is used on the
[PyTorch integration](pytorch.md) page.

## Skipping broken records

Sometimes your reader finds a record it cannot use. Instead of silently dropping it, yield a
`SkipDocument(id, reason)`, as the CSV recipe above does. It is:

- written to `<split>-skipped.jsonl` with your reason, so you can find it later;
- counted in `result.stats.n_skipped`;
- reported once as a `DataQualityWarning`.

tokbin also skips some documents by itself:

| Code | Why |
|---|---|
| `TB-D201` | bytes that are not valid UTF-8 |
| `TB-D203` | empty text, or text that gives no tokens (a whitespace-only text usually still gives tokens, so filter it yourself if you do not want it) |
| `TB-D204` | the tokenizer failed on this document |
| `TB-D206` | a `SkipDocument` from your generator, or a broken line in `jsonl` |

## Choosing how strict to be: `ErrorPolicy`

By default a broken document is skipped. If you would rather stop at the first one, pass
`ErrorPolicy(on_data_error="raise")`:

```python
# write_strict.py
from tokbin import Dataset, ErrorPolicy, jsonl

strict = ErrorPolicy(on_data_error="raise")
Dataset("corpus").write("web-strict", jsonl("data/web"), "gpt2/tokenizer.json", policy=strict)
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.DataError: [TB-D206] Input record cannot be read
  where: tokbin.write.stream_writer.StreamWriter.write
  cause: no 'text' field
  fix:   fix the input record, or ignore it if a few are expected
```

There is also a safety net: if **too many** documents are skipped, something is wrong with
the input, and tokbin stops instead of making a dataset of fragments. The default limit is
1% after the first 1000 documents. Here every third document is empty:

```python
# write_mostly_empty.py
import warnings
from tokbin import Dataset, DataQualityWarning, ErrorPolicy

docs = [(f"doc-{i}", "" if i % 3 == 0 else f"Text {i}.") for i in range(30)]   # every 3rd is empty
policy = ErrorPolicy(max_skip_ratio=0.2, min_docs_for_ratio=10)

warnings.simplefilter("ignore", DataQualityWarning)
Dataset("corpus").write("too-many-empty", docs, "gpt2/tokenizer.json", policy=policy)
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ContractError: [TB-K204] Too many documents were skipped: 4 of 10
  where: tokbin.write.stream_writer.StreamWriter.write
  cause: 40.0% of documents were skipped, above the limit of 20.0%; a systematically broken input must not produce a dataset made of fragments
  fix:   see train-skipped.jsonl in the partial directory for reasons; fix the input, or raise ErrorPolicy.max_skip_ratio
```

| `ErrorPolicy` setting | Default | Meaning |
|---|---|---|
| `on_data_error` | `"skip"` | `"skip"` records the document and goes on; `"raise"` stops with a `DataError`. |
| `max_skip_ratio` | `0.01` | Stop when more than this share of documents is skipped. |
| `min_docs_for_ratio` | `1000` | Start checking the share only after this many documents. |

> **Note** A failed write leaves its work in `corpus/<name>.partial/`. Remove it with
> `Dataset("corpus").clean("<name>")` or `tokbin clean corpus/<name>`, or continue it, see
> [Resuming interrupted writes](resume.md).

## Duplicate ids

tokbin notices when two documents share an id. Duplicates are still written (maybe you
meant it), but you get one warning at the end with the count and an example:

```python
# write_dupes.py
from tokbin import Dataset

docs = [("a", "One."), ("b", "Two."), ("a", "One again."), ("c", "Three."), ("a", "And again.")]
result = Dataset("corpus").write("dupes", docs, "gpt2/tokenizer.json")
for issue in result.issues:
    print(issue.level, issue.code, issue.message)
```

```text
.../tokbin/write/stream_writer.py:1014: DataQualityWarning: [TB-D205] Duplicate document id: 2 documents repeat an earlier id, e.g. 'a'
  return self._finish()
info TB-C215 Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name
warning TB-D205 Duplicate document id: 2 documents repeat an earlier id, e.g. 'a'
```

## Rules for a good generator

1. **Same order every time.** Sort files, add `ORDER BY`, avoid `set()`. Needed for
   [resume](resume.md).
2. **Give ids.** `(id, text)` instead of plain text: you can trace every token back to its
   document, and resume can check the input.
3. **One form for all documents.** Do not mix plain text and `(id, text)`.
4. **Yield `SkipDocument` for broken records** instead of dropping them silently.
5. **Do not load everything first.** Yield as you read.

## Next

- [Resuming interrupted writes](resume.md)
- [Reading data](reading.md)
