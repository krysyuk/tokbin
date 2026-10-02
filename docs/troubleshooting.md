# Errors and troubleshooting

**In short:** every tokbin error tells you three things: **what** happened, **why**, and how
to **fix** it, plus a stable code like `TB-C206` that you can search for. This page shows how
to read and catch errors, what the warnings mean, and how to solve the most common problems.

## Anatomy of an error

```text
tokbin.errors.ConfigError: [TB-C206] Split already exists in the source: corpus/numbers [train]
  where: tokbin.write.stream_writer.StreamWriter.open
  cause: the source already has a 'train' split
  fix:   pass overwrite=True to replace this split, or write another split
```

| Part | Here | Meaning |
|---|---|---|
| class | `ConfigError` | the kind of problem (see the table below) |
| code | `TB-C206` | a stable id: it never changes meaning, so you can search for it and test for it |
| what | `Split already exists in the source: corpus/numbers [train]` | what happened |
| where | `tokbin.write.stream_writer.StreamWriter.open` | which tokbin function raised it |
| cause | `the source already has a 'train' split` | why |
| fix | `pass overwrite=True ...` | what to do |

The full list of codes is in [errors.md](errors.md).

## Catching errors in code

All tokbin errors inherit from `TokbinError`, and each part is an attribute:

```python
# catch_error.py
from tokbin import TokbinError, read_source

try:
    read_source("corpus/missing")
except TokbinError as err:
    print("type: ", type(err).__name__)
    print("code: ", err.code.id)
    print("what: ", err.what)
    print("why:  ", err.why)
    print("fix:  ", err.fix)
    print("where:", err.where)
```

```text
type:  FormatError
code:  TB-F104
what:  Metadata file is missing: corpus/missing/meta.json
why:   corpus/missing contains neither meta.json nor dataset.json
fix:   check the path; if the write was interrupted, see `tokbin status`
where: tokbin.read.source.read_source
```

Catch a specific class to react to one kind of problem:

```python
# catch_by_class.py
from tokbin import ConfigError, Dataset, IntegrityError, ResumeError

try:
    Dataset("corpus").write("bad name!", ["text"], "gpt2/tokenizer.json")
except ConfigError as err:
    print("a setting is wrong:", err.what)
except ResumeError as err:
    print("an unfinished write is in the way:", err.what)
except IntegrityError as err:
    print("files on disk are damaged:", err.what)
```

```text
a setting is wrong: Invalid source name: 'bad name!'
```

To check for one exact error, compare the code with a constant from `tokbin.codes`:

```python
from tokbin import codes

if err.code is codes.METADATA_MISSING:      # TB-F104
    ...
```

### The error classes

| Class | Code letter | Means | Example |
|---|---|---|---|
| `ConfigError` | `C` | a wrong argument, setting or path | split already exists, unknown split, bad source name |
| `OutOfRangeError` | `C` | an index past the end (also an `IndexError`) | `src.doc(10**9)` |
| `ContractError` | `K` | the input does not follow the rules | mixed document forms, too many skipped documents |
| `DataError` | `D` | one document is bad (raised only with `ErrorPolicy(on_data_error="raise")`) | invalid UTF-8, missing JSON field |
| `FormatError` | `F` | files on disk are not a valid tokbin source | no `meta.json` |
| `SchemaVersionError` | `F` | written by a different format version | needs `tokbin migrate` or a newer tokbin |
| `UnsupportedFeatureError` | `F` | something reserved for a future version | multimodal sources |
| `IntegrityError` | `I` | data is damaged | missing shard, wrong sha256 |
| `CompatibilityError` | `M` | things that do not fit together | sources with different tokenizers |
| `ResumeError` | `R` | about an unfinished write | partial exists, input differs on resume |
| `DependencyError` | `P` | an optional package is missing (also an `ImportError`) | writing with `tokbin-core` only |
| `InternalError` | `X` | a bug in tokbin; please report it | |

Errors that are not tokbin's own pass through unchanged: `OSError` (disk full, permission
denied), `KeyboardInterrupt`, and any exception raised inside **your** generator.

## Warnings

Some things are worth knowing but not worth stopping for. tokbin reports them as Python
warnings, once per kind, and also lists them in `result.issues`:

| Warning class | Codes | Example |
|---|---|---|
| `DataQualityWarning` | `TB-D2xx`, `TB-R204` | documents skipped, duplicate ids, resume without ids |
| `ShardingWarning` | `TB-S201`, `TB-S202` | a document larger than a shard, or more than 5% of one |
| `CompatibilityWarning` | `TB-P003` | zstd missing, lzma used instead when packing |

All of them inherit from `TokbinWarning`, a `UserWarning`. Control them with the standard
`warnings` module.

Hide the warnings you already know about:

```python
import warnings
from tokbin import DataQualityWarning

warnings.simplefilter("ignore", DataQualityWarning)
```

Or make them fatal, for example in a CI job that must not accept dirty data:

```python
# strict_data.py
import warnings
from tokbin import Dataset, DataQualityWarning

warnings.simplefilter("error", DataQualityWarning)      # turn data warnings into errors

docs = [("a", "Hello."), ("b", "")]                     # the second one is empty
Dataset("corpus").write("strict", docs, "gpt2/tokenizer.json")
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.DataQualityWarning: [TB-D203] Document is empty: document 'b' (the text is empty); skipped, see train-skipped.jsonl
```

On the command line, `--strict` does the same: any warning makes the command exit with
code `1`.

## Common problems

| You see | It means | Do this |
|---|---|---|
| `TB-R201` An unfinished write already exists | an earlier write was interrupted | pass `resume=True` to continue, or `tokbin clean corpus/<name>` to start over ([Resume](resume.md)) |
| `TB-R203` Documents differ from the interrupted write | on resume, the generator gave other documents or another order | make the generator deterministic (sort, `ORDER BY`) |
| `TB-R205` Settings differ from the interrupted write | on resume, another `WriterConfig` or tokenizer | use exactly the old settings, or clean and start over |
| `TB-R202` another process is writing | two writers for one source | wait, or stop the other process |
| `TB-C206` Split already exists | you are writing a split twice | `overwrite=True` to replace it, or `WriterConfig(split="valid")` |
| `TB-C208` / `TB-C209` no EOS / BOS token | tokbin could not find the special token | `WriterConfig(eos_token="...")` / `bos_token="..."` |
| `TB-C212` Tokenizer file cannot be loaded | wrong path, or not a `tokenizer.json` | check the path; tokbin needs the Hugging Face `tokenizer.json` |
| `TB-C301` Split is not in the source | `read_source(..., split="valid")` but there is no valid split | the error lists the existing splits |
| `TB-C304` Window is larger than the split | `block_size` is longer than the data | use a smaller `block_size` |
| `TB-F104` Metadata file is missing | wrong path, or the write has not finished yet | `tokbin status <path>` |
| `TB-I301` / `TB-I303` shard missing / wrong size | a file was lost or cut while copying | copy it again, then `tokbin verify` |
| `TB-I302` Shard is corrupted | the content of a shard changed | copy it again, then `tokbin verify` |
| `TB-K204` Too many documents were skipped | most of the input is broken | look at `<split>-skipped.jsonl` in the `.partial` folder; fix the input or raise `ErrorPolicy.max_skip_ratio` |
| `TB-M301` different tokenizers | mixing sources written with different tokenizers | give the odd sources weight 0, or rewrite them |
| `TB-P001` / `TB-P002` missing package | e.g. writing with only `tokbin-core`, or torch not installed | the message gives the exact `pip install` |
| `TB-C216` not enough disk space | the next shard would not fit | free space, then resume |

## Asking the tool itself

`tokbin doctor` shows your version, mode and optional packages, and `tokbin status` explains
the state of any source:

```bash
$ tokbin doctor
$ tokbin status corpus/web
```

Both are described in [Command line](cli.md).
