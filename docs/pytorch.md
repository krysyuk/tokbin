# PyTorch integration

**In short:** tokbin gives you two ready PyTorch datasets. `WindowDataset` cuts one source
into windows of `block_size` tokens. `MixtureDataset` draws windows from several sources with
weights. Both plug straight into a `torch.utils.data.DataLoader`, work with many workers, and
read only the windows a batch needs, straight from disk.

## Install

```bash
$ pip install tokbin 'tokbin-core[torch]'
```

On a training server that only reads data, `pip install 'tokbin-core[torch]'` is enough.

## The data used on this page

20,000 short stories from the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset for training, 1,000 for validation. The script that writes them is in
[Data generators](generators.md#a-hugging-face-dataset):

```bash
$ tokbin info corpus/tinystories
```

```text
corpus/tinystories/   4,660,620 tokens · complete

  dtype       uint16 · vocab 50,257
  tokenizer   gpt2 · 31dba887 · eos 50256 · no bos
  format      schema 1 · written by tokbin 0.1.0.dev0

  split      tokens     docs   shards        size   skipped
  train   4,466,061   20,000        1     8.5 MiB         0
  valid     194,559    1,000        1   380.0 KiB         0

✔ Status: ready for training
```

## `WindowDataset`: one source, cut into windows

```python
# window_dataset.py
from tokbin import read_source
from tokbin.adapters.torch import WindowDataset

src = read_source("corpus/tinystories")
dataset = WindowDataset(src, block_size=8)

print(len(src), "tokens ->", len(dataset), "windows of 8")
print(dataset[0])
print(dataset[0].dtype, dataset[0].shape)
```

```text
4466061 tokens -> 558257 windows of 8
tensor([ 3198,  1110,    11,   257,  1310,  2576,  3706, 20037])
torch.int64 torch.Size([8])
```

- Item `i` is the window that starts at token `i * block_size`.
- Each item is a 1-D `torch.int64` tensor: the type `nn.Embedding` and `CrossEntropyLoss`
  expect. You never need `.long()`.
- The tail that is shorter than one window is dropped.

### With a `DataLoader`

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from tokbin import read_source
from tokbin.adapters.torch import WindowDataset

torch.manual_seed(0)                       # makes shuffle=True repeatable
src = read_source("corpus/tinystories")
loader = DataLoader(WindowDataset(src, block_size=8), batch_size=4, shuffle=True)

batch = next(iter(loader))
print(batch)
print(batch.shape, batch.dtype)
```

```text
tensor([[  290, 29382,   340,   329,   257,   890,   640,    13],
        [21264,  2474,   198,   198,    43,   813,  2936,   257],
        [  546,   340,   290,  4987,    13,   198,   198,  2990],
        [  523,  3772,   284,   423,  1043,   428,  6016, 21181]])
torch.Size([4, 8]) torch.int64
```

A batch is a `(batch_size, block_size)` tensor. `shuffle=True` visits the windows in a new
random order every epoch.

### Inputs and targets

A language model learns to predict the **next** token. tokbin gives you plain windows; you
shift them yourself. Ask for **one extra token** (`block_size + 1`), then split it:

```python
# shift.py
import torch
from torch.utils.data import DataLoader
from tokbin import read_source
from tokbin.adapters.torch import WindowDataset

torch.manual_seed(0)
src = read_source("corpus/tinystories")
loader = DataLoader(WindowDataset(src, block_size=9), batch_size=2, shuffle=True)

batch = next(iter(loader))       # 9 tokens per row
x = batch[:, :-1]                # the model sees tokens 0..7
y = batch[:, 1:]                 # and must predict tokens 1..8
print("x:", x[0].tolist())
print("y:", y[0].tolist())
```

```text
x: [760, 644, 284, 466, 13, 632, 13112, 510]
y: [644, 284, 466, 13, 632, 13112, 510, 284]
```

`y` is `x` moved one step to the left: for every position, the target is the token that comes
next.

### Overlapping windows: `stride`

By default windows do not overlap (`stride = block_size`). A smaller stride gives more
windows that share tokens:

```python
# stride.py
from tokbin import read_source
from tokbin.adapters.torch import WindowDataset

src = read_source("corpus/tinystories")

no_overlap = WindowDataset(src, block_size=8)              # stride = block_size
overlap = WindowDataset(src, block_size=8, stride=4)       # each window starts 4 tokens later

print(len(no_overlap), len(overlap))
print(no_overlap[1].tolist())
print(overlap[1].tolist())
print(overlap[2].tolist())
```

```text
558257 1116514
[1043, 257, 17598, 287, 607, 2119, 13, 1375]
[1310, 2576, 3706, 20037, 1043, 257, 17598, 287]
[1043, 257, 17598, 287, 607, 2119, 13, 1375]
```

```text
position:   0       4       8       12      16
stride 8:   [window 0      ][window 1      ]
stride 4:   [window 0      ]
                    [window 1      ]
                            [window 2      ]
```

## A complete training script

Everything together: data, a tiny model, the training loop, validation, and a few generated
words at the end. It runs on a laptop CPU in about a minute.

```python
# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from tokbin import read_source
from tokbin.adapters.torch import WindowDataset

BLOCK = 64            # tokens the model sees at once
BATCH = 32
STEPS = 300

# ---- 1. data ---------------------------------------------------------------
train_src = read_source("corpus/tinystories", split="train")
valid_src = read_source("corpus/tinystories", split="valid")
vocab_size = train_src.meta.vocab_size

# block_size = BLOCK + 1: one extra token to build the targets
train_loader = DataLoader(WindowDataset(train_src, block_size=BLOCK + 1),
                          batch_size=BATCH, shuffle=True, drop_last=True)
valid_loader = DataLoader(WindowDataset(valid_src, block_size=BLOCK + 1),
                          batch_size=BATCH)

# ---- 2. a tiny model -------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size, dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.mix = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h, _ = self.mix(self.embed(x))
        return self.head(h)

torch.manual_seed(0)
model = TinyLM(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

def loss_on(batch):
    x, y = batch[:, :-1], batch[:, 1:]          # inputs and next-token targets
    logits = model(x)
    return loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

# ---- 3. training loop ------------------------------------------------------
step = 0
while step < STEPS:
    for batch in train_loader:
        loss = loss_on(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step:4}  train loss {loss.item():.3f}")
        if step == STEPS:
            break

# ---- 4. validation ---------------------------------------------------------
model.eval()
with torch.no_grad():
    losses = [loss_on(batch).item() for batch in valid_loader]
print(f"valid loss {sum(losses) / len(losses):.3f} on {len(losses)} batches")

# ---- 5. generate a few tokens ---------------------------------------------
from tokenizers import Tokenizer

tok = Tokenizer.from_file("corpus/tinystories/tokenizer/tokenizer.json")
ids = tok.encode("Once upon a time").ids
with torch.no_grad():
    for _ in range(12):
        logits = model(torch.tensor([ids]))
        ids.append(int(logits[0, -1].argmax()))
print(tok.decode(ids))
```

```bash
$ python train.py
```

```text
step   50  train loss 5.606
step  100  train loss 4.952
step  150  train loss 4.752
step  200  train loss 4.534
step  250  train loss 4.134
step  300  train loss 4.038
valid loss 4.165 on 94 batches
Once upon a time, there was a little girl named Lily. She loved to
```

Where tokbin appears in this script:

| Section | tokbin's part |
|---|---|
| 1. data | `read_source` opens the splits; `WindowDataset` turns them into windows; `meta.vocab_size` sizes the model |
| 3. training loop | each batch is read from disk on demand; nothing else changes compared to any PyTorch loop |
| 4. validation | the `valid` split, read the same way, without shuffling |
| 5. generation | the tokenizer copy saved inside the source decodes the output |

Swap `TinyLM` for your real model; the data part stays the same.

## Many workers

Reading is cheap, but on a big model you want the next batches ready while the GPU works.
Set `num_workers`. A `WindowDataset` sends only the source's path to each worker, which reopens
the files there, so starting workers is fast and uses no extra memory.

```python
# load_with_workers.py
from torch.utils.data import DataLoader
from tokbin import read_source
from tokbin.adapters.torch import WindowDataset

def main():
    src = read_source("corpus/tinystories")
    loader = DataLoader(WindowDataset(src, block_size=1024), batch_size=16,
                        shuffle=True, num_workers=4)
    for i, batch in enumerate(loader):
        if i == 3:
            break
        print(i, batch.shape)

if __name__ == "__main__":      # required for num_workers > 0 on macOS and Windows
    main()
```

```text
0 torch.Size([16, 1024])
1 torch.Size([16, 1024])
2 torch.Size([16, 1024])
```

> **Warning** On macOS and Windows, PyTorch starts workers by re-running your script. Put
> the code in `main()` behind `if __name__ == "__main__":`, as above, or the script will
> start itself again and again.

## Mixing sources: `MixtureDataset`

To train on several sources with weights (see [Mixing sources](mixtures.md)), use
`MixtureDataset`. Here the corpus has `tinystories` and `books`, and `corpus/mix.json` says
`{"tinystories": 9, "books": 1}`. The first example uses explicit weights to keep it simple.

```python
# mixture_dataset.py
from tokbin import Mixture
from tokbin.adapters.torch import MixtureDataset

mix = Mixture("corpus", {"tinystories": 1})
dataset = MixtureDataset(mix, block_size=8, length=1000, seed=0)

print(len(dataset))
print(dataset[0])
print(dataset[0])       # the same index always gives the same window
```

```text
1000
tensor([ 366, 2061,  611,  314,  900,  510,  257,  481])
tensor([ 366, 2061,  611,  314,  900,  510,  257,  481])
```

A mixture has no natural "end", so you choose:

| Argument | Meaning |
|---|---|
| `mixture` | a `Mixture`, e.g. `Mixture.from_config("corpus")` |
| `block_size` | tokens per window |
| `length` | how many windows one pass ("epoch") has |
| `seed` | which windows; item `i` is fully defined by `(seed, i)` |

Because every item is defined by `(seed, i)`, the result is the same with any number of workers
and on any machine. No `worker_init_fn` is needed.

### New windows every epoch

Use a different `seed` per epoch:

```python
# mixture_epochs.py
from torch.utils.data import DataLoader
from tokbin import Mixture
from tokbin.adapters.torch import MixtureDataset

mix = Mixture.from_config("corpus")

for epoch in range(2):
    dataset = MixtureDataset(mix, block_size=8, length=4, seed=epoch)   # new seed, new windows
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    print(f"epoch {epoch}:", batch[:, :4].tolist())
```

```text
epoch 0: [[44484, 373, 3726, 284], [198, 198, 1, 29252], [836, 470, 588, 14343], [3211, 16337, 13, 383]]
epoch 1: [[750, 407, 765, 284], [290, 3521, 470, 1057], [2921, 607, 617, 9007], [198, 50, 3301, 2753]]
```

In the training script above, replace the train loader with:

```python
from tokbin import Mixture
from tokbin.adapters.torch import MixtureDataset

mix = Mixture.from_config("corpus")
train_loader = DataLoader(MixtureDataset(mix, block_size=BLOCK + 1, length=100_000, seed=0),
                          batch_size=BATCH, drop_last=True)
```

The windows are already random, so `shuffle=True` is not needed.

## Which one to use

| | `WindowDataset` | `MixtureDataset` |
|---|---|---|
| sources | one | several, with weights |
| windows | fixed grid: start at `i * stride` | random positions |
| an epoch | every window once | `length` windows |
| randomness | `DataLoader(shuffle=True)` | `seed` |
| typical use | validation, a single-source run | pre-training on a data mix |

## Things to know

- **Windows are `int64` tensors**, converted from the compact `uint16` on disk. A batch of
  32 × 1024 tokens is 256 KiB in memory.
- **No padding and no attention masks.** Windows are cut from one continuous stream, so every
  window is full. Documents inside a window are separated by the EOS token (`50256` for GPT-2).
- **`block_size` must fit in the split.** A window longer than the split raises
  `ConfigError` `TB-C304`.
- **Without the torch extra**, creating a dataset raises `DependencyError` `TB-P001` with the
  exact `pip install` line.
- **Plain NumPy instead of torch?** Use `src.sample_windows(...)` or `mix.batch(...)` from
  [Reading data](reading.md#random-windows-for-training).

## Next

- [Mixing sources](mixtures.md)
- [Archives](archives.md): move the prepared data to the GPU server.
