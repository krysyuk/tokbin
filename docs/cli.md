# Command line

**In short:** the `tokbin` command does everything the Python API does, without writing
code: build a source from files, look at it, check it, pack it, clean up. Every command
follows the same pattern: `tokbin <command> <path> [options]`.

```bash
$ tokbin --help
```

```text
tokbin 0.1.0 · pretraining data for language models: tokenize once, read windows via memmap

Usage: tokbin [OPTIONS] <COMMAND>

Write:
  build    Tokenize text files or JSON Lines into a source
  migrate  Convert a source to the current format schema

Inspect:
  ls       List the sources of a corpus with states and sizes
  info     Describe a corpus or a source: tokens, shards, tokenizer
  status   Show the write state of a source or a corpus
  verify   Hash every shard and check all files of a source or a corpus

Transfer:
  pack     Compress a source into <name>.tbpack
  unpack   Unpack a .tbpack, check every sha256, publish the source

Maintenance:
  clean    Remove the unfinished write (<name>.partial) of a source
  rm       Delete a source and its unfinished write (requires --yes)
  doctor   Show version, mode, dependencies and features

Global options:
      --json      Print one JSON document to stdout
      --no-color  Disable colors (also: NO_COLOR=1)
      --strict    Exit with code 1 on warnings (for CI)
  -h, --help      Print help
  -V, --version   Print version

Examples:
  $ tokbin build corpus/web --from-jsonl data/ --tokenizer tokenizer.json
  $ tokbin ls corpus
  $ tokbin verify corpus/web

Use tokbin help <command> for more information on a command.
```

`tokbin help <command>` or `tokbin <command> --help` shows the options of one command.

The examples below use this project layout (the same data as in
[Data generators](generators.md)):

```text
my-project/
├── gpt2/tokenizer.json
└── data/
    ├── web/part-00.jsonl, part-01.jsonl     ← 61 JSON lines, one of them broken
    ├── web-valid.jsonl                      ← 2 JSON lines for validation
    ├── books/alice.txt, moby_dick.txt, pride.txt
    └── big/                                 ← 400,000 JSON lines, to show Ctrl+C
```

Commands at a glance:

| Command | What it does |
|---|---|
| [`build`](#tokbin-build) | tokenize files into a source |
| [`ls`](#tokbin-ls) | list the sources of a corpus |
| [`info`](#tokbin-info) | describe a corpus or a source |
| [`status`](#tokbin-status) | show whether writes are finished |
| [`verify`](#tokbin-verify) | check every byte with sha256 |
| [`clean`](#tokbin-clean) | remove an unfinished write |
| [`pack`](#tokbin-pack) | compress a source for transfer |
| [`unpack`](#tokbin-unpack) | unpack and check a pack |
| [`migrate`](#tokbin-migrate) | convert to the current format |
| [`rm`](#tokbin-rm) | delete a source |
| [`doctor`](#tokbin-doctor) | show version and installed extras |

---

## tokbin build

Tokenize a folder of `.txt` files or JSON Lines files into a source.

```bash
$ tokbin build corpus/web --from-jsonl data/web --tokenizer gpt2/tokenizer.json
```

```text
✔ built corpus/web [train]
  60 docs · 1.75K tokens · 1 shard · 1 skipped
  └ [TB-D206] Input record cannot be read: document 'part-01.jsonl:30' (no 'text' field); skipped, see train-skipped.jsonl
  └ [TB-C215] Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name

! Status: complete · 1 warning
```

The source is built. One broken line was skipped (the warning), and the end-of-text token was
found automatically (the note). While it runs, a progress line shows how much input has been
read.

From text files, each file is one document:

```bash
$ tokbin build corpus/books --from-txt data/books --tokenizer gpt2/tokenizer.json
```

```text
✔ built corpus/books [train]
  3 docs · 133 tokens · 1 shard
  └ [TB-C215] Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name

✔ Status: complete
```

Add a validation split to an existing source:

```bash
$ tokbin build corpus/web --from-jsonl data/web-valid.jsonl --tokenizer gpt2/tokenizer.json --split valid
```

```text
✔ built corpus/web [valid]
  2 docs · 26 tokens · 1 shard
  └ [TB-C215] Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name

✔ Status: complete
```

| Option | Meaning | Default |
|---|---|---|
| `<TARGET>` | the source folder to write, e.g. `corpus/web` | required |
| `--from-txt <DIR>` | every `*.txt` file is a document | |
| `--from-jsonl <PATH>` | every line of `*.jsonl` files is a document (a folder or one file) | |
| `--field <KEY>` | JSON key with the text (JSON Lines only) | `text` |
| `--pattern <GLOB>` | which files to read | `*.txt` / `*.jsonl` |
| `--tokenizer <PATH>` | path to `tokenizer.json` | required |
| `--no-eos` | do not add EOS after documents | EOS is added |
| `--bos` | add BOS before documents | off |
| `--eos-token <TEXT>` / `--bos-token <TEXT>` | the special token text | detected |
| `--batch-docs <N>` | documents per tokenizer batch (speed only) | `1024` |
| `--split <SPLIT>` | `train`, `valid` or `test` | `train` |
| `--shard-size <SIZE>` | shard size: `512M`, `1G`, `256MiB` or bytes | `512M` |
| `--overwrite` | replace an existing split | |
| `--resume` | continue an interrupted build | |

### Interrupted? Resume

Press Ctrl+C (or lose the connection) and tokbin tells you how to continue:

```bash
$ tokbin build corpus/big --from-jsonl data/big --tokenizer gpt2/tokenizer.json --shard-size 64K
^C
the last checkpoint is kept; continue with: tokbin build corpus/big --resume
✖ interrupted
```

`--resume` alone is enough, the settings were saved:

```bash
$ tokbin build corpus/big --resume
```

The full story is in [Resuming interrupted writes](resume.md#resume-from-the-command-line).

### A split that already exists

```bash
$ tokbin build corpus/books --from-txt data/books --tokenizer gpt2/tokenizer.json
```

```text
✖ [TB-C206] Split already exists in the source: corpus/books [train]
  where: tokbin.write.stream_writer.StreamWriter.open
  cause: the source already has a 'train' split
  fix:   pass overwrite=True to replace this split, or write another split
```

Add `--overwrite` to replace it.

---

## tokbin ls

List the sources of a corpus with their states and sizes.

```bash
$ tokbin ls corpus
```

```text
corpus/   2 sources · 1,905 tokens · 3.7 KiB

  ✔ books   complete     133 tokens    3 docs    1 shard     266 B
  ✔ web     complete   1.77K tokens   62 docs   2 shards   3.5 KiB
```

Unfinished writes are listed too (here `corpus/big`, a build interrupted with Ctrl+C):

```text
corpus/   3 sources · 1,905 tokens · 3.7 KiB

  ! big     partial    split train · 49 shards closed · 1.61M tokens · updated 2026-10-02T11:50:26Z
  ✔ books   complete     133 tokens    3 docs    1 shard     266 B
  ✔ web     complete   1.77K tokens   62 docs   2 shards   3.5 KiB
```

| Option | Meaning | Default |
|---|---|---|
| `[ROOT]` | the corpus folder | `.` (current folder) |

---

## tokbin info

Describe a corpus or one source: tokens, documents, shards, tokenizer, mixture.

```bash
$ tokbin info corpus
```

```text
corpus/   1,905 tokens · 2 sources

  ✔ books     133 tokens    3 docs    1 shard     266 B
  ✔ web     1.77K tokens   62 docs   2 shards   3.5 KiB
    └ [TB-D301] Documents were skipped when the source was written: the reasons are in train-skipped.jsonl

  tokenizer   gpt2 · 31dba887 · same in all sources
  mix         web 0.75 · books 0.25

✔ Status: ready for training
```

`mix` comes from `corpus/mix.json` (see [Mixing sources](mixtures.md)). Point it at a source to
see its splits:

```bash
$ tokbin info corpus/web
```

```text
corpus/web/   1,772 tokens · complete

  dtype       uint16 · vocab 50,257
  tokenizer   gpt2 · 31dba887 · eos 50256 · no bos
  format      schema 1 · written by tokbin 0.1.0

  split   tokens   docs   shards      size   skipped
  train    1,746     60        1   3.4 KiB         1
  valid       26      2        1      52 B         0

  └ [TB-D301] Documents were skipped when the source was written: the reasons are in train-skipped.jsonl

✔ Status: ready for training
```

| Option | Meaning |
|---|---|
| `<PATH>` | a corpus folder or a source folder |

`info` reads only metadata, so it is instant on any size. The same data from Python:
[Statistics and verification](statistics.md).

---

## tokbin status

Show whether the writes of a source or a corpus are finished, and what to do if not.

```bash
$ tokbin status corpus/big
```

```text
corpus/big/   partial

  ! unfinished write   corpus/big.partial
                       split train · 49 shards closed · 1.61M tokens · updated 2026-10-02T11:50:26Z
                       resume: `tokbin build corpus/big --resume`
                       discard: `tokbin clean corpus/big`

! Status: partial
```

A finished source is short and sweet:

```bash
$ tokbin status corpus/web
```

```text
corpus/web/   complete · train, valid

✔ Status: complete
```

For a whole corpus:

```bash
$ tokbin status corpus
```

```text
corpus/

  ✔ books   complete     133 tokens    3 docs    1 shard     266 B
  ✔ web     complete   1.77K tokens   62 docs   2 shards   3.5 KiB
    └ [TB-D301] Documents were skipped when the source was written: the reasons are in train-skipped.jsonl

✔ Status: 2 complete
```

| Option | Meaning |
|---|---|
| `<PATH>` | a source folder or a corpus folder |

---

## tokbin verify

Read every shard, compute its sha256 and compare it with `meta.json`; check all index files.

```bash
$ tokbin verify corpus/web
```

```text
corpus/web/

  ✔ train-00000.bin   3.4 KiB   sha256 ok
  ✔ valid-00000.bin      52 B   sha256 ok

✔ Status: intact · 2 shards · 3.5 KiB checked
```

A whole corpus at once:

```bash
$ tokbin verify corpus
```

```text
corpus/

  books/
    ✔ train-00000.bin   266 B   sha256 ok

  web/
    ✔ train-00000.bin   3.4 KiB   sha256 ok
    ✔ valid-00000.bin      52 B   sha256 ok

✔ Status: intact · 3 shards · 3.7 KiB checked
```

When a shard is damaged, `verify` shows which one and exits with code `3`; see
[Statistics and verification](statistics.md#when-something-is-broken) for an example.

| Option | Meaning |
|---|---|
| `<PATH>` | a source folder or a corpus folder |

---

## tokbin clean

Remove the unfinished write (`<name>.partial`) of a source. The finished source, if any, is
not touched.

```bash
$ tokbin clean corpus/big
```

```text
✔ removed corpus/big.partial

✔ Status: 10.4 MiB freed
```

Nothing to clean is not an error:

```bash
$ tokbin clean corpus/big
```

```text
– nothing to clean: corpus/big has no unfinished write
```

| Option | Meaning |
|---|---|
| `<PATH>` | a source folder, or its `.partial` folder |

A write that is running right now is never removed.

---

## tokbin pack

Compress a finished source into `<name>.tbpack` (a folder) or `<name>.tbpack.tar` (one file),
with a sha256 manifest.

```bash
$ tokbin pack corpus/web --tar
```

```text
✔ packed corpus/web → corpus/web.tbpack.tar

✔ Status: 9 files · 3.4 MiB → 657.7 KiB (19%) · zstd-3
```

Smaller, but slower:

```bash
$ tokbin pack corpus/web --method lzma --level 9 --out outbox
```

```text
✔ packed corpus/web → outbox/web.tbpack

✔ Status: 9 files · 3.4 MiB → 449.5 KiB (13%) · lzma-9
```

| Option | Meaning | Default |
|---|---|---|
| `<PATH>` | a complete source folder | required |
| `--out <DIR>` | where to put the pack | next to the source |
| `--tar` | one `.tbpack.tar` file instead of a folder | folder |
| `--method <METHOD>` | `zstd` or `lzma` | `zstd` if available, else `lzma` |
| `--level <N>` | compression level | `3` |
| `--overwrite` | replace an existing pack | |

More in [Archives](archives.md).

---

## tokbin unpack

Unpack a `.tbpack`, check every sha256, then publish the source.

```bash
$ tokbin unpack web.tbpack.tar --into corpus
```

```text
✔ unpacked web.tbpack.tar → corpus/web

✔ Status: 9 files · 3.4 MiB · sha256 ok
```

| Option | Meaning | Default |
|---|---|---|
| `<PACK>` | a `<name>.tbpack` folder or `<name>.tbpack.tar` file | required |
| `--into <DIR>` | the corpus folder to unpack into | next to the pack |
| `--name <NAME>` | the source name | the name stored in the pack |
| `--overwrite` | replace an existing source | |

If any check fails, nothing is created. Unpacking works with the light `tokbin-core`
package.

---

## tokbin migrate

Convert a source written by an older tokbin to the current format. By default it writes a
converted **copy** next to the original.

```bash
$ tokbin migrate corpus/web
```

```text
✔ corpus/web already uses schema 1; nothing to do
```

Format version 1 is the only one so far, so today there is nothing to convert. When a newer
tokbin changes the format, `tokbin info` will mark old sources as `outdated` and point you
here.

| Option | Meaning | Default |
|---|---|---|
| `<PATH>` | a source folder | required |
| `--out <DIR>` | where to write the converted copy | `<name>-v<N>` |
| `--in-place` | replace the source itself instead of copying | |

---

## tokbin rm

Delete a source and its unfinished write. Because this cannot be undone, it asks for
`--yes`:

```bash
$ tokbin rm corpus/books
```

```text
✖ not removed: corpus/books (3.4 MiB) would be deleted for good; repeat with --yes to confirm
```

```bash
$ tokbin rm corpus/books --yes
```

```text
✔ removed corpus/books

✔ Status: 3.4 MiB freed
```

| Option | Meaning |
|---|---|
| `<PATH>` | a source folder |
| `--yes` | confirm the deletion |

`rm` refuses folders that are not tokbin sources, so a typo cannot delete your home folder.

---

## tokbin doctor

Show the tokbin version, the mode, and which optional packages are installed.

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

- `mode full`: you can write and read. `mode core` (only `tokbin-core` installed): read,
  verify and unpack only.
- `✔` installed, `–` optional and not installed, `✖` required for writing and missing.

Attach this output to bug reports.

---

## Options for every command

### `--json`: output for scripts

Every command can print one JSON document instead of text. The document always has the same
envelope:

```bash
$ tokbin info corpus/missing --json
```

```json
{
  "format": 1,
  "tokbin": "0.1.0",
  "command": "info",
  "mode": "full",
  "exit_code": 1,
  "result": null,
  "warnings": [],
  "error": {
    "type": "ConfigError",
    "code": "TB-C305",
    "what": "Dataset path not found: corpus/missing",
    "why": "neither the source directory nor an unfinished write of it exists",
    "fix": "check the path; `tokbin ls <corpus>` lists the sources of a corpus",
    "where": "tokbin.ops.inspect.inspect_source"
  }
}
```

On success, `error` is `null` and `result` holds the data: the same as `to_dict()` of the
matching Python result. Combine it with [`jq`](https://jqlang.github.io/jq/):

```bash
$ tokbin info corpus --json | jq '.result.n_items'
```

```text
1905
```

```bash
$ tokbin info corpus/web --json | jq '.result.splits[] | {name, n_items, n_docs}'
```

```json
{
  "name": "train",
  "n_items": 1746,
  "n_docs": 60
}
{
  "name": "valid",
  "n_items": 26,
  "n_docs": 2
}
```

### `--strict`: warnings fail the command

Useful in CI, where nothing should pass with warnings:

```bash
$ tokbin build corpus/web --from-jsonl data/web --tokenizer gpt2/tokenizer.json --overwrite --strict
```

```text
✔ built corpus/web [train]
  60 docs · 1.75K tokens · 1 shard · 1 skipped
  └ [TB-D206] Input record cannot be read: document 'part-01.jsonl:30' (no 'text' field); skipped, see train-skipped.jsonl
  └ [TB-C215] Special tokens were detected automatically: EOS '<|endoftext|>' (id 50256) from well-known name

! Status: complete · 1 warning
✖ --strict: warnings are treated as errors
```

The source is still written; only the exit code becomes `1`.

### `--no-color`

Turns colors off. Setting the environment variable `NO_COLOR=1` does the same. Colors are also
off automatically when the output goes to a file or a pipe.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | success (warnings allowed, unless `--strict`) |
| `1` | an error, or a warning with `--strict` |
| `2` | wrong usage: unknown command, missing option, `rm` without `--yes` |
| `3` | an integrity check failed: a damaged or missing shard |
| `4` | a required package is missing |
| `130` | interrupted with Ctrl+C |

So a script can do:

```bash
tokbin verify corpus/web || echo "the dataset is damaged, download it again"
```

### Typos

```bash
$ tokbin bild corpus
```

```text
error: unrecognized command 'bild'

  tip: a similar command exists: 'build'

Usage: tokbin [OPTIONS] <COMMAND>

For more information, try 'tokbin --help'.
```
