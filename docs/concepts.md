# Core concepts

**In short:** tokbin turns text into numbers once, saves those numbers to disk in big flat
files, and later lets you grab any piece of them instantly. This page explains the handful
of words used everywhere else in the docs. Read it once; it takes five minutes.

## The big picture

A language model never sees text. It sees **tokens**: small integers produced by a
**tokenizer**. Turning gigabytes of text into tokens is slow, so you want to do it **once**
and keep the result. That is what tokbin does:

```text
 your text                 tokbin write                  tokbin read
 ───────────               ─────────────                 ──────────────
 "The cat sat."   ──►   [464, 3797, 3332, 13, 50256]  ──►  random windows
 "Dogs play."     ──►   [35, 18463, 711, 13, 50256]        for training
      ...                 saved as binary files            (no copying, no loading
                                                            the whole corpus)
```

## The words

| Word | What it means in plain words |
|---|---|
| **token** | One integer. `"The cat"` → `[464, 3797]` with the GPT-2 tokenizer. |
| **tokenizer** | The thing that turns text into tokens. tokbin uses Hugging Face `tokenizer.json` files. |
| **document** | One piece of text you feed in: an article, a book, a code file, a chat. |
| **id** | An optional name for a document, like `"wiki/Cat"` or `"part-00.jsonl:17"`. |
| **EOS** | "End of sequence" token. tokbin puts one after every document so the model sees where a document ends. For GPT-2 it is `<|endoftext|>` = `50256`. |
| **BOS** | "Beginning of sequence" token. Off by default; some models want it before every document. |
| **source** | One dataset on disk, for example all your web pages, or all your code. A directory. |
| **split** | A part of a source: `train`, `valid` or `test`. Each split is written separately. |
| **shard** | One binary file with tokens, e.g. `train-00000.bin`. A big split is cut into many shards (512 MiB each by default). |
| **corpus** | A directory that holds several sources side by side. |
| **window** | A run of consecutive tokens, e.g. "1024 tokens starting at position 5,000,000". This is what you feed to a model. |
| **mixture** | Reading from several sources at once with weights, e.g. 75% web and 25% code. |
| **partial write** | A write that has not finished yet. It lives in `<name>.partial/` and is invisible to readers. |
| **checkpoint** | A small file saved after every finished shard, so an interrupted write can continue instead of starting over. |
| **dtype** | How many bytes one token takes on disk. tokbin picks the smallest that fits the vocabulary: `uint16` (2 bytes) for GPT-2. |

## How documents become one long stream

tokbin glues all documents of a split into **one long stream** of tokens, with an EOS after
each one. Then it cuts that stream into shards of equal size, without caring where documents
end:

```text
 documents:  "The cat sat on the mat."   "Dogs like to play in the park."   "The sun is a star."

 stream:     464 3797 3332 319 262 2603 13 50256 | 35 18463 588 284 711 287 262 3952 13 50256 | 464 4252 ...
             └───────────── doc 0 ─────────────┘   └──────────────── doc 1 ───────────────┘   └─ doc 2

 shards:     [ train-00000.bin                  ][ train-00001.bin                  ][ train-00002.bin ]
```

A document may start in one shard and end in the next one. That is fine: the reader stitches
shards together for you, and it also remembers where every document starts (the
**offsets**), so you can still ask for "document number 1".

## What a corpus looks like on disk

```text
corpus/                          ← a corpus: just a folder with sources
├── mix.json                     ← optional: weights for mixing sources
├── web/                         ← a source
│   ├── meta.json                ← description: dtype, tokenizer, splits, shard list with sha256
│   ├── tokenizer/
│   │   ├── tokenizer.json       ← a copy of the tokenizer used to write this source
│   │   └── tokenizer_config.json
│   ├── train-00000.bin          ← tokens, raw little-endian integers
│   ├── train-00001.bin
│   ├── train-offsets.npy        ← where every document starts in the stream
│   ├── train-ids.jsonl          ← the id of every document, one per line
│   ├── train-ids.idx.npy        ← index into train-ids.jsonl, for instant lookup
│   ├── train-skipped.jsonl      ← documents that were skipped and why
│   ├── valid-00000.bin          ← the valid split: same set of files with "valid-"
│   └── ...
├── books/                       ← another source
└── code.partial/                ← an unfinished write of "code": readers ignore it
```

> **Note** You never need to touch these files by hand. They are described here so that
> nothing on disk looks mysterious.

## Why reading is instant

A `.bin` shard is just tokens laid out one after another. tokbin opens it with
[`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html): the
operating system maps the file into memory **without reading it**. When you ask for
1024 tokens at position 5,000,000, only those 2 KiB are read from disk. A 100 GB corpus
opens in milliseconds and uses almost no RAM.

## Two packages

| Package | What you get | Install |
|---|---|---|
| `tokbin` | everything: write, read, inspect, pack | `pip install tokbin` |
| `tokbin-core` | read, verify, unpack only (depends on numpy alone) | `pip install tokbin-core` |

Use `tokbin` on the machine that prepares data and `tokbin-core` on training machines, where
you only need to read.

## Next

- [Getting started](getting-started.md): write and read your first dataset.
