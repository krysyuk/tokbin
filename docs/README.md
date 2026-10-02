# tokbin documentation

tokbin turns text into tokens **once** and stores them as flat binary files. Training code
then reads any piece of the data instantly, without loading the corpus into memory. On top of
that you get mixing of sources by weight, resumable writes, integrity checks and compressed
archives for moving data between machines.

```python
from tokbin import Dataset, read_source

Dataset("corpus").write("web", ["The cat sat on the mat.", "Dogs like to play."], "gpt2/tokenizer.json")
print(read_source("corpus/web").doc(0))
```

```text
[  464  3797  3332   319   262  2603    13 50256]
```

## Start here

0. [Is tokbin for you?](when-to-use.md): what tokbin is for, what it is not for, and what it
   costs. Two minutes; read this first.
1. [Core concepts](concepts.md): the words used everywhere: source, split, shard, window.
   Five minutes.
2. [Getting started](getting-started.md): install, write your first dataset, read it back.

## User guide

| Page | You will learn to |
|---|---|
| [Writing data](writing.md) | write a source with `Dataset.write`; splits, settings, special tokens, `StreamWriter` |
| [Data generators](generators.md) | feed text files, JSON Lines, CSV, databases and Hugging Face datasets; skip broken records |
| [Resuming interrupted writes](resume.md) | survive Ctrl+C and crashes; continue with `resume=True` |
| [Reading data](reading.md) | open a source; tokens, documents, windows, random batches |
| [Mixing sources](mixtures.md) | sample from several sources with weights from `mix.json` |
| [Statistics and verification](statistics.md) | see what is in a dataset; check every byte with sha256 |
| [Archives](archives.md) | pack a source, move it to another machine, unpack it safely |
| [PyTorch integration](pytorch.md) | `WindowDataset`, `MixtureDataset`, `DataLoader` and a full training loop |

## Reference

| Page | Contents |
|---|---|
| [Command line](cli.md) | every `tokbin` command with examples and options |
| [Errors and troubleshooting](troubleshooting.md) | how errors look, how to catch them, common problems |
| [Error codes](errors.md) | the full list of `TB-...` codes |

## How to read these docs

- Every example is a complete script. The first line names the file, e.g. `# write_web.py`;
  save it under that name in your project folder and run `python write_web.py`.
- The block right after an example is its **real output**, copied from a run with
  tokbin 0.1.0 and the GPT-2 tokenizer.
- Lines starting with `$` are typed in a terminal; don't type the `$` itself.
- All examples assume this project folder:

```text
my-project/
├── gpt2/
│   └── tokenizer.json      ← see "Get a tokenizer" in Getting started
├── data/                   ← your raw text
└── corpus/                 ← tokbin writes here
```
