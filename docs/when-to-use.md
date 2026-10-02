# Is tokbin for you?

**In short:** tokbin prepares **pretraining data for language models**: raw text in, tokens on
disk, random fixed-length windows out. It is for people who train or continue training a
model on large amounts of plain text and want to do the slow tokenization **once, ahead of
time**, on a machine of their choice. If you fine-tune on chats or instructions, look
elsewhere.

tokbin was built for one workflow and does that workflow well. This page helps you decide in
two minutes whether it is yours.

## The workflow tokbin is made for

```text
 your machine (cheap, no hurry)                rented GPU server (expensive, every hour counts)
 ──────────────────────────────                ────────────────────────────────────────────────
 raw text ── tokbin build ── tokbin pack ──►  tokbin unpack ── training starts at once
            (hours, may crash:                (sha256 checked)  (data read straight from
             resumes where it stopped)                           disk, no preprocessing)
```

GPU time is the most expensive part of training. Tokenizing a large corpus takes hours of CPU
time. If you tokenize on the GPU server, you pay for GPUs that sit idle. tokbin moves that
work to a machine where time is cheap. It makes sure that what arrives on the server is
complete and intact, and that the server needs nothing but `numpy` to read it.

## Use tokbin if

- you **pretrain** a language model from scratch, or **continue pretraining** one, on plain
  text;
- the corpus is **too big for memory**, and tokenizing it takes long enough that a crash
  halfway would hurt;
- you **prepare data on one machine and train on another** (a rented GPU server, a cluster
  node, a colleague's machine);
- you train on **several sources with weights** ("70% web, 20% code, 10% books") and want to
  change the weights without rewriting the data;
- you want **reproducible** data loading: the same seed gives the same batches on any machine
  and with any number of workers.

## Do not use tokbin if

| Your case | Why tokbin does not fit |
|---|---|
| **Fine-tuning, SFT, chat or instruction data** | tokbin has no chat templates, no prompt/answer separation, no loss masks and no padding. It cuts one continuous stream into windows across document borders. |
| **You need attention masks at document borders** | Document offsets are stored, but the PyTorch adapters do not build masks. Documents in a window are separated only by the EOS token. |
| **Your tokenizer is not a Hugging Face `tokenizer.json`** | Sentencepiece `.model` files and custom vocabularies are not supported yet. |
| **Images, audio or other non-text data** | Only text is supported; the format reserves room for more, but it is not implemented. |
| **Distributed writing, or streaming from S3 or other object storage** | One process writes one source; reading needs the files on a local (or mounted) disk. |
| **You change the tokenizer often** | A dataset is tied to its tokenizer; a new tokenizer means tokenizing again. |
| **Small data that fits in memory** | Tokenize it in your training script; tokbin would only add a step. |
| **You need a stable, long-term format today** | tokbin is pre-alpha. The format is versioned and `tokbin migrate` exists for future changes, but the API may still change. |

## tokbin vs. a 30-line script

Many people prepare pretraining data with a short script in the style of nanoGPT's
`prepare.py`: tokenize everything into one `train.bin` and read it with `numpy.memmap`. For
many projects that is enough, and you should keep it if nothing in the right column matters
to you.

| | Hand-written `train.bin` + `np.memmap` | tokbin |
|---|---|---|
| Lines of code you maintain | ~30 | 0 |
| Crash or Ctrl+C after 5 hours of tokenizing | start over | [resume](resume.md) from the last shard; result byte-identical |
| Half-written file visible to training | possible | no: writes are published only when complete |
| Corrupted or truncated file after copying | silently wrong data | detected by sha256 ([verify](statistics.md#verification-are-the-files-intact), [unpack](archives.md)) |
| Moving data to a server | `scp` and hope | `pack` / `unpack` with compression and checksums |
| Several sources with weights | write your own sampler | `mix.json` + [`Mixture`](mixtures.md) |
| Which document is token 1,234,567 from? | unknown | `doc(i)`, `id_of_doc(i)` |
| Broken input records | crash, or silently dropped | skipped, logged with a reason, limited by a ratio |
| Dependencies on the training server | whatever your script imports | `tokbin-core`: only `numpy` |
| PyTorch | write a `Dataset` | [`WindowDataset`, `MixtureDataset`](pytorch.md) |

## What it costs

Measured in the project [benchmarks](../benchmarks/benchmarks.md) (Apple M4 Pro, TinyStories,
2.7 million documents). The numbers come from one machine; on your hardware they will
differ, but the ratios are what matter.

| | Cost |
|---|---|
| Writing, against a hand-written `.bin` writer | 5–15% slower end to end; tokenization itself dominates |
| Very short documents (~12 tokens) | tokbin's bookkeeping reaches 50% of tokenization time, and the index files outgrow the tokens |
| Random access to one window | 1.6–2.5 µs, against 0.2–0.8 µs for raw numpy on a memmap |
| PyTorch `WindowDataset` | matches a hand-written memmap dataset with 4+ workers |
| PyTorch `MixtureDataset` | 1.7–7× slower than `WindowDataset`, the price of weighted random sampling |

In short: you pay a few percent of write time and a few microseconds per read for safety,
resume and mixing. For documents of normal length, the GPU is the bottleneck, not tokbin.

## Quick decision

```text
Training on chats, instructions, preference pairs?          ── yes ──► not tokbin
        │ no
Plain text that fits in memory, tokenized in seconds?       ── yes ──► not needed
        │ no
Tokenizer is a Hugging Face tokenizer.json?                 ── no ───► not yet
        │ yes
Prepare on one machine, train on another, or afraid
of losing hours to a crash, or mixing several sources?      ── yes ──► tokbin fits
        │ no
A 30-line np.memmap script is probably enough.
```

## Next

- [Getting started](getting-started.md): your first dataset in five minutes.
- [Core concepts](concepts.md): the words used in these docs.
