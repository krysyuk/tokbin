# Benchmarks

Two kinds of measurements live here:

- **`regress.py`**: a fast synthetic suite (about a minute, no downloads) compared
  against `baseline.json`. A metric more than 10% slower than the baseline is measured
  again, and if it is still slower the run fails. Timings compare only on the machine
  that made the baseline.
- **`study_*.py`**: studies on real text that produce the numbers and charts of
  [docs/benchmarks.md](../docs/benchmarks.md). They take hours and need the data below.

```bash
uv sync --group bench                                      # matplotlib, psutil

uv run --group bench python benchmarks/regress.py          # compare with baseline.json
uv run --group bench python benchmarks/regress.py --save-baseline
```

## Studies

The studies use TinyStoriesV2 (`TinyStoriesV2-GPT4-train.txt`, 2.1 GB) and two
tokenizers: a large one (DeepSeek, 129,280 tokens, stored as `uint32`) and a small
BPE trained on TinyStories (12,000 tokens, `uint16`).

```bash
D=TinyStoriesV2-GPT4-train.txt
T="--tokenizer deepseek=deepseek/tokenizer.json --tokenizer bpe12k=bpe12k/tokenizer.json"

uv run --group bench python benchmarks/study_write.py --data $D $T          # all write experiments
uv run --group bench python benchmarks/study_read.py --source benchmarks/.work/full/deepseek/src
uv run --group bench python benchmarks/study_storage.py \
    --source deepseek=benchmarks/.work/full/deepseek/src \
    --source bpe12k=benchmarks/.work/full/bpe12k/src --text $D
python benchmarks/study_torch.py --source benchmarks/.work/full/deepseek/src \
    --raw benchmarks/.work/read/concat.bin                                  # needs torch
uv run --group bench python benchmarks/plot.py                              # docs/benchmarks/*.png
```

Each experiment writes `benchmarks/results/<study>-<experiment>.json` with the machine
description, every timing sample and the settings. Large intermediate data goes to
`benchmarks/.work/` (ignored by git, about 10 GB at the peak; delete it afterwards).

## Rules of measurement

- Variants that are compared run in rounds (A B C, A B C, ...), so a slow drift of
  the machine (heat, background work) affects them alike.
- Results report the best and the median of the samples; the JSON keeps all of them.
- Previous results are removed before a timed run, outside the timed region.
- Runs whose memory is reported start in a fresh subprocess.
- Read timings are warm (page cache) unless the experiment says cold. Cold reads use
  copies written with `F_NOCACHE` (macOS), since `purge` needs root.
- "Writer only" replays tokenization from a table: it is the cost of tokbin's own
  work, independent of the tokenizers package.
