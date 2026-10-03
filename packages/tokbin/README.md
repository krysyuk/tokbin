<!-- Local builds only: the release workflow replaces this file with the root
     README.md, so the PyPI page of `tokbin` shows the full description. -->

# tokbin

Pretraining data for language models: tokenize a text corpus once into resumable,
verifiable binary shards and read random training windows through `numpy.memmap`.
tokbin supports pretraining only; it is not a tool for fine-tuning, SFT or chat data.

Meta-package: installs `tokbin-core` and `tokenizers`.

```bash
pip install tokbin          # full: write and read
pip install tokbin-core     # lightweight: read, verify, unpack
```

The code and documentation live in the `tokbin-core` package, imported as `import tokbin`.
