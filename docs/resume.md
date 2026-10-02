# Resuming interrupted writes

**In short:** tokenizing a big corpus takes hours, and things go wrong: you press Ctrl+C,
the laptop sleeps, the server reboots, the disk fills up. tokbin saves its progress after
every finished shard. Run the same write again with `resume=True` and it continues where it
stopped. The final dataset is **byte-for-byte identical** to one written without any
interruption.

## How it works, in pictures

While a write is running, everything goes to a hidden-from-readers folder
`<name>.partial/`. After each shard is finished, tokbin saves a small **checkpoint**: "these
shards are done, and they used the first N documents".

```text
 write starts          shard 0 done         shard 1 done      Ctrl+C!
     │                     │                    │                │
     ▼                     ▼                    ▼                ▼
 corpus/stories.partial/
     checkpoint.json  ─►  "1 shard, N docs"  ─►  "2 shards, M docs"     (kept)
     train-00000.bin                             train-00001.bin         train-00002.bin.open  (thrown away)
```

When you resume, tokbin:

1. keeps the finished shards;
2. throws away the half-written one;
3. skips the documents that are already saved (and checks their ids);
4. continues writing from there.

When the write finishes, `stories.partial/` is renamed to `stories/` in one step. Readers
never see a half-written source.

## Step by step

### 1. A write gets interrupted

This script pretends that you press Ctrl+C after 1500 documents:

```python
# write_stories.py
from tokbin import Dataset, WriterConfig

def stories():
    for i in range(3000):
        if i == 1500:
            raise KeyboardInterrupt        # pretend you pressed Ctrl+C here
        yield f"story-{i}", f"Story number {i}: once upon a time there was a robot."

config = WriterConfig(shard_bytes=8192)    # small shards, so we get several checkpoints
Dataset("corpus").write("stories", stories(), "gpt2/tokenizer.json", config=config)
```

```text
Traceback (most recent call last):
  ...
KeyboardInterrupt
```

### 2. Look at what is left

There is no `corpus/stories/` yet, only the unfinished write:

```bash
$ ls corpus/stories.partial
```

```text
checkpoint.json
tokenizer
train-00000.bin
train-00001.bin
train-00002.bin
train-ids.idx.i64
train-ids.jsonl
train-offsets.i64
train-skipped.jsonl
```

```bash
$ tokbin status corpus/stories
```

```text
corpus/stories/   partial

  ! unfinished write   corpus/stories.partial
                       split train · 3 shards closed · 12.3K tokens · updated 2026-10-02T11:30:21Z
                       resume: repeat the write with resume=True and the same input
                       discard: `tokbin clean corpus/stories`

! Status: partial
```

The same from Python:

```python
# check_status.py
from tokbin import Dataset

info = Dataset("corpus").status("stories")
print(info.state)
print(info.partial.n_closed_shards, "shards,", info.partial.n_items, "tokens,",
      info.partial.n_input_consumed, "documents saved")
print("resumable:", info.partial.resumable)
```

```text
partial
3 shards, 12288 tokens, 852 documents saved
resumable: True
```

Readers cannot open an unfinished source, so a training job never picks up half a dataset:

```python
from tokbin import read_source

read_source("corpus/stories")
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.FormatError: [TB-F104] Metadata file is missing: corpus/stories/meta.json
  where: tokbin.read.source.read_source
  cause: corpus/stories contains neither meta.json nor dataset.json
  fix:   check the path; if the write was interrupted, see `tokbin status`
```

### 3. Running the same write again is refused

If you simply start the write again, tokbin will not silently throw away your progress:

```python
# write_again.py  (the same write, without resume=True)
from tokbin import Dataset, WriterConfig

def stories():
    for i in range(3000):
        yield f"story-{i}", f"Story number {i}: once upon a time there was a robot."

config = WriterConfig(shard_bytes=8192)
Dataset("corpus").write("stories", stories(), "gpt2/tokenizer.json", config=config)
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ResumeError: [TB-R201] An unfinished write already exists: corpus/stories.partial
  where: tokbin.write.stream_writer.StreamWriter.open
  cause: a previous write of this source did not finish
  fix:   pass resume=True to continue it, or delete it with `tokbin clean corpus/stories.partial` to start over
```

### 4. Resume

Add `resume=True`. Everything else stays **exactly** the same: the same documents in the
same order, the same tokenizer, the same `WriterConfig`:

```python
# resume_stories.py
from tokbin import Dataset, WriterConfig

def stories():
    for i in range(3000):
        yield f"story-{i}", f"Story number {i}: once upon a time there was a robot."

config = WriterConfig(shard_bytes=8192)
result = Dataset("corpus").write("stories", stories(), "gpt2/tokenizer.json",
                                 config=config, resume=True)
print(result.stats)
for issue in result.issues:
    print(issue.level, issue.code, issue.message)
```

```text
WriteStats(n_input=3000, n_docs=3000, n_skipped=0, n_items=44328, n_shards=11, n_split_docs=10)
info TB-C215 Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name
info TB-R207 Write resumed from a checkpoint: 3 closed shards kept, 852 input documents skipped
info TB-S203 Documents are split across shards: their items continue in the next shard; readers handle this transparently
```

`TB-R207` tells you what happened: 3 shards were reused and the first 852 documents were
skipped instead of tokenized again. Now `corpus/stories/` exists and `stories.partial/` is
gone.

### 5. The result is identical

Let's write the same data without any interruption into another folder and compare the
sha256 of every shard:

```python
# compare.py
import json
from tokbin import Dataset, WriterConfig

def stories():
    for i in range(3000):
        yield f"story-{i}", f"Story number {i}: once upon a time there was a robot."

config = WriterConfig(shard_bytes=8192)
Dataset("reference").write("stories", stories(), "gpt2/tokenizer.json", config=config)

def shard_hashes(path):
    meta = json.load(open(f"{path}/meta.json"))
    return [s["sha256"] for s in meta["splits"]["train"]["shards"]]

print(shard_hashes("corpus/stories") == shard_hashes("reference/stories"))
```

```text
True
```

> **Tip** `resume=True` is safe to use always. If there is nothing to resume, the write
> simply starts from the beginning. A common pattern is to put `resume=True` in your
> preprocessing script once and just rerun the script after any failure.

## The rules

Resume works only if the second run is the **same write**:

| Must be the same | Why |
|---|---|
| documents and their order | tokbin skips the first N documents; they must be the ones already saved |
| tokenizer | the saved tokens must mean the same thing |
| `WriterConfig` (split, shard size, EOS/BOS, dtype) | the shards must be cut the same way |

What you **may** change: `batch_docs` (speed only) and the `ErrorPolicy`.

You don't need to remember this: tokbin checks it and tells you what differs.

### Different order of documents

```python
# resume_wrong.py
from tokbin import Dataset, WriterConfig

def stories():
    for i in range(3000):
        yield f"story-{i}", f"Story number {i}: once upon a time there was a robot."

def shuffled_stories():                       # oops: a different order
    docs = list(stories())
    yield from reversed(docs)

config = WriterConfig(shard_bytes=8192)
Dataset("corpus").write("stories", shuffled_stories(), "gpt2/tokenizer.json",
                        config=config, resume=True)
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ResumeError: [TB-R203] Documents differ from the interrupted write: corpus/stories.partial
  where: tokbin.write.stream_writer.StreamWriter.write
  cause: input #0 is 'story-2999', the interrupted write had another document at this position
  fix:   resume with a generator that yields the same documents in the same order, or delete the unfinished write (`tokbin clean corpus/stories.partial`) and start again
```

The checkpoint is not damaged by a failed resume: fix the generator and try again.

### Different settings

```python
# resume_config.py
from tokbin import Dataset, WriterConfig

def stories():
    for i in range(3000):
        yield f"story-{i}", f"Story number {i}: once upon a time there was a robot."

config = WriterConfig(shard_bytes=16384)      # oops: a different shard size
Dataset("corpus").write("stories", stories(), "gpt2/tokenizer.json", config=config, resume=True)
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ResumeError: [TB-R205] Settings differ from the interrupted write: corpus/stories.partial
  where: tokbin.write.stream_writer.StreamWriter.open
  cause: the interrupted write used different settings: WriterConfig (shard_bytes, append_eos, prepend_bos...)
  fix:   resume with exactly the settings and tokenizer of the interrupted write, or delete it (`tokbin clean corpus/stories.partial`) and start again
```

> **Warning** Ids are what make the order check reliable. If your documents are plain text
> without ids, tokbin can only check the position and warns `TB-R204`. Prefer
> `(id, text)`; see [Data generators](generators.md#rules-for-a-good-generator).

## Starting over instead

To throw the unfinished write away, use `clean`. The finished source (if there is one) is not
touched:

```python
# clean_it.py
from tokbin import Dataset

result = Dataset("corpus").clean("stories")
print(result)
```

```text
CleanResult(path=PosixPath('corpus/stories'), removed=(PosixPath('corpus/stories.partial'),), n_bytes=3613933)
```

Or from the command line: `tokbin clean corpus/stories`.

## Resume from the command line

`tokbin build` saves its settings (input, tokenizer, shard size, ...) inside the partial
folder. After Ctrl+C it prints the exact command to continue:

```bash
$ tokbin build corpus/big --from-jsonl data/big --tokenizer gpt2/tokenizer.json --shard-size 64K
^C
the last checkpoint is kept; continue with: tokbin build corpus/big --resume
✖ interrupted
```

```bash
$ tokbin status corpus/big
```

```text
corpus/big/   partial

  ! unfinished write   corpus/big.partial
                       split train · 47 shards closed · 1.54M tokens · updated 2026-10-02T11:31:11Z
                       resume: `tokbin build corpus/big --resume`
                       discard: `tokbin clean corpus/big`

! Status: partial
```

`--resume` alone is enough; there is no need to repeat the other options:

```bash
$ tokbin build corpus/big --resume
```

```text
✔ built corpus/big [train]
  400,000 docs · 7.32M tokens · 224 shards
  └ [TB-C215] Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name
  └ [TB-R207] Write resumed from a checkpoint: 47 closed shards kept, 84380 input documents skipped
  └ [TB-S203] Documents are split across shards: their items continue in the next shard; readers handle this transparently (x214)

✔ Status: complete
```

## Questions

**How much work do I lose?** Everything after the last finished shard. With the default
512 MiB shards that is at most a few minutes of tokenizing.

**What if the disk fills up?** Before starting each shard, tokbin checks for free space and
stops with `TB-C216` if there is not enough. Free some space and resume.

**Can two processes write the same source?** No. A `.lock` file inside the partial folder
stops the second one with `TB-R202`. A lock left by a process that died is taken over
automatically.

**Does `resume=True` work with `StreamWriter`?** Yes:
`StreamWriter("corpus/chat", tokenizer, resume=True)`, then `add` the same documents from the
start.

## Next

- [Reading data](reading.md)
- [Statistics and verification](statistics.md)
