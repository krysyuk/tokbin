# Archives: pack and unpack

**In short:** to move a source to another machine, **pack** it: tokbin compresses every file
and writes a manifest with sha256 checksums. On the other side, **unpack** checks every
checksum and only then makes the source visible. A damaged or incomplete copy never turns
into a "dataset that looks fine".

```text
 laptop                                  GPU server
 ──────                                  ──────────
 corpus/web/  ── pack ──►  web.tbpack.tar  ── copy ──►  web.tbpack.tar ── unpack ──►  corpus/web/
              (compress +                 (scp, rsync,                (check sha256,
               sha256 manifest)            S3, USB...)                 then publish)
```

## The data used on this page

```python
# prepare_web.py
import warnings
from tokbin import Dataset, DataQualityWarning, WriterConfig, jsonl

warnings.simplefilter("ignore", DataQualityWarning)
Dataset("corpus").write("web", jsonl("data/web"), "gpt2/tokenizer.json",
                        config=WriterConfig(shard_bytes=2048))
```

## Pack a source

```python
# pack_web.py
from tokbin import pack_source

result = pack_source("corpus/web")

print(result.path)
print("files:      ", len(result.manifest.files))
print("original:   ", result.manifest.n_bytes, "bytes")
print("compressed: ", result.manifest.stored_bytes, "bytes")
print("method:     ", result.manifest.method, "level", result.manifest.level)
```

```text
corpus/web.tbpack
files:       9
original:    3565052 bytes
compressed:  673490 bytes
method:      zstd level 3
```

The pack is a folder next to the source. Every file is compressed separately, and
`manifest.json` lists them all:

```text
corpus/
├── web/                                  ← the source, untouched
└── web.tbpack/                           ← the pack
    ├── manifest.json                     ← list of files with sizes and sha256
    ├── meta.json.zst
    ├── tokenizer/
    │   ├── tokenizer.json.zst
    │   └── tokenizer_config.json.zst
    ├── train-00000.bin.zst
    ├── train-00001.bin.zst
    ├── train-ids.idx.npy.zst
    ├── train-ids.jsonl.zst
    ├── train-offsets.npy.zst
    └── train-skipped.jsonl.zst
```

The beginning of `manifest.json`:

```json
{
  "format": "tokbin-pack",
  "format_version": 1,
  "tokbin_version": "0.1.0.dev0",
  "created_at": "2026-10-02T11:35:16Z",
  "source": "web",
  "method": "zstd",
  "level": 3,
  "files": [
    {
      "path": "meta.json",
      "stored": "meta.json.zst",
      "n_bytes": 898,
      "sha256": "36748f8be2c11af3633b86d839a69fdf2d912998ac9c5bca082f7c86178ab213",
      "stored_bytes": 468,
      "stored_sha256": "7b321d2edfcf7a0516dfcf3a6cc24f2ab48254a65342d99ef7424845ad101c41"
    },
    ...
```

Each file has two checksums: of the original (`sha256`) and of the compressed copy
(`stored_sha256`). So unpack can tell "the archive was damaged in transit" from "the data
was wrong".

> **Note** In this tiny example almost all the bytes are the tokenizer copy (3.4 MB). In a
> real dataset the shards dominate. In the project
> [benchmarks](../benchmarks/benchmarks.md) zstd-3 shrinks `uint16` shards to about 51% of
> their size and `uint32` shards to about 28%.

### One file instead of a folder

One file is easier to upload. Use `tar=True` and, if you like, another output folder:

```python
# pack_tar.py
from tokbin import pack_source

result = pack_source("corpus/web", "outbox", tar=True)
print(result.path)
```

```text
outbox/web.tbpack.tar
```

The `.tar` is not compressed again; it just holds the already compressed files together.

### Compression method and level

| `method` | When |
|---|---|
| `"zstd"` (default) | fast to pack and unpack. Built into Python 3.14+; on older Python install `pip install 'tokbin-core[zstd]'`. |
| `"lzma"` | smaller files, much slower. Always available. Used automatically (with a `TB-P003` warning) when zstd is not installed. |

```python
# pack_lzma.py
from tokbin import pack_source

result = pack_source("corpus/web", "outbox", method="lzma", level=6)
print(result.path, result.manifest.method, result.manifest.level, result.manifest.stored_bytes)
```

```text
outbox/web.tbpack lzma 6 460324
```

The default `level` is `3` for both methods.

### What pack refuses to do

- Pack an unfinished source: `ConfigError` `TB-C309`. Finish or [resume](resume.md) the
  write first.
- Pack a damaged source: before compressing, every shard is checked against the sha256 in
  `meta.json`. A mismatch raises `IntegrityError` `TB-I302`, so you never ship broken data.
- Overwrite an existing pack: `ConfigError` `TB-C308`. Pass `overwrite=True` to replace it.

## Unpack

```python
# unpack_web.py
from tokbin import read_source, unpack_pack

result = unpack_pack("outbox/web.tbpack.tar", into="server/corpus")
print(result.path)

src = read_source(result.path)
print(src)
```

```text
server/corpus/web
Source('server/corpus/web', split='train', n_items=1746)
```

`unpack_pack` takes either a `.tbpack` folder or a `.tbpack.tar` file. It:

1. unpacks into a hidden `server/corpus/web.partial/` folder;
2. checks the size and sha256 of every compressed file and every unpacked file;
3. checks the result against `meta.json`;
4. only then renames it to `server/corpus/web`.

If anything fails, nothing is left behind.

### Under another name

The source keeps its original name unless you pass `name`:

```python
# unpack_as.py
from tokbin import unpack_pack

result = unpack_pack("outbox/web.tbpack.tar", into="server/corpus", name="web-copy")
print(result.path)
```

```text
server/corpus/web-copy
```

### An existing source is not overwritten

```python
unpack_pack("outbox/web.tbpack.tar", into="server/corpus")   # web is already there
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.ConfigError: [TB-C308] Output already exists: server/corpus/web
  where: tokbin.ops.pack.unpack_pack
  cause: refusing to overwrite it
  fix:   pass overwrite=True (--overwrite), or choose another output directory
```

### A damaged archive

Here 4 bytes in the middle of the archive were overwritten, as a bad network copy might do:

```python
unpack_pack("outbox/broken.tbpack.tar", into="server/corpus", name="broken")
```

```text
Traceback (most recent call last):
  ...
tokbin.errors.IntegrityError: [TB-I307] Packed file is corrupted: outbox/broken.tbpack.tar: tokenizer/tokenizer.json.zst
  where: tokbin.ops.pack.unpack_pack
  cause: it cannot be decompressed: zstd decompress error: Data corruption detected
  fix:   copy the pack again from its origin and repeat `tokbin unpack`
```

`server/corpus/broken` was not created.

> **Tip** Unpacking needs only `tokbin-core` (plus zstd on Python older than 3.14). Your
> training server does not need `tokenizers`.

## The whole trip from the command line

On the machine with the data:

```bash
$ tokbin pack corpus/web --tar
```

```text
✔ packed corpus/web → corpus/web.tbpack.tar

✔ Status: 9 files · 3.4 MiB → 657.7 KiB (19%) · zstd-3
```

Copy the file with any tool you like, for example:

```bash
$ scp corpus/web.tbpack.tar gpu-server:/data/
```

On the GPU server:

```bash
$ tokbin unpack web.tbpack.tar --into corpus
```

```text
✔ unpacked web.tbpack.tar → corpus/web

✔ Status: 9 files · 3.4 MiB · sha256 ok
```

`sha256 ok` means every file arrived intact. You can double-check at any time later:

```bash
$ tokbin verify corpus/web
```

```text
corpus/web/

  ✔ train-00000.bin   2.0 KiB   sha256 ok
  ✔ train-00001.bin   1.4 KiB   sha256 ok

✔ Status: intact · 2 shards · 3.4 KiB checked
```

Smaller archive at the cost of time:

```bash
$ tokbin pack corpus/web --method lzma --level 9 --out outbox
```

```text
✔ packed corpus/web → outbox/web.tbpack

✔ Status: 9 files · 3.4 MiB → 449.5 KiB (13%) · lzma-9
```

All options are in [Command line](cli.md#tokbin-pack).

## Safety

Archives can come from untrusted places, so unpack is strict: it refuses paths that escape
the target folder, links, unexpected files and "decompression bombs" that would expand far
beyond the sizes in the manifest.

## Next

- [PyTorch integration](pytorch.md)
- [Statistics and verification](statistics.md)
