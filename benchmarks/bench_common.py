"""Shared helpers of the benchmarks: timing, environment, data, memory, cold copies.

Nothing here is part of the library. Timings use ``time.perf_counter``; a measurement
repeats a callable and keeps every sample, so reports can show the spread and not only
the best case.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"


# --- timing ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Timing:
    """Seconds per call over repeated runs."""

    samples: list[float]

    @property
    def best(self) -> float:
        return min(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def p90(self) -> float:
        return float(np.percentile(self.samples, 90))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "best": self.best,
            "median": self.median,
            "p90": self.p90,
            "max": max(self.samples),
            "n": len(self.samples),
        }


def measure(
    fn: Callable[[], object],
    *,
    repeat: int = 7,
    number: int | None = None,
    min_time: float = 0.2,
    warmup: int = 1,
) -> Timing:
    """Time ``fn`` per call: ``repeat`` samples, each the mean of ``number`` calls.

    ``number`` is chosen so that a sample lasts at least ``min_time`` seconds, which
    keeps timer resolution out of sub-microsecond results. The garbage collector is
    disabled while a sample runs, as in ``timeit``.
    """
    for _ in range(warmup):
        fn()
    if number is None:
        number = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(number):
                fn()
            if time.perf_counter() - t0 >= min_time or number >= 1 << 24:
                break
            number *= 4
    samples = []
    for _ in range(repeat):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            for _ in range(number):
                fn()
            samples.append((time.perf_counter() - t0) / number)
        finally:
            gc.enable()
    return Timing(samples)


def time_once(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# --- environment -------------------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def cpu_name() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def environment() -> dict[str, Any]:
    """What the numbers depend on: hardware, OS, interpreter, packages, commit."""
    import tokbin

    try:
        import psutil

        memory = psutil.virtual_memory().total
    except ImportError:
        memory = None
    try:
        import tokenizers

        tokenizers_version: str | None = tokenizers.__version__
    except ImportError:
        tokenizers_version = None
    return {
        "cpu": cpu_name(),
        "cores": os.cpu_count(),
        "memory_bytes": memory,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tokenizers": tokenizers_version,
        "tokbin": tokbin.__version__,
        "commit": _run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "dirty": bool(_run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"])),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)  # type: ignore[call-overload]
    raise TypeError(f"not JSON serializable: {type(value).__qualname__}")


# --- memory ------------------------------------------------------------------------------


def rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def peak_rss_bytes() -> int:
    """Peak resident memory of this process so far."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


@dataclass
class RssSampler:
    """Samples the resident memory of this process in a background thread."""

    interval: float = 0.25
    samples: list[tuple[float, int]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> RssSampler:
        t0 = time.perf_counter()

        def loop() -> None:
            while not self._stop.is_set():
                self.samples.append((time.perf_counter() - t0, rss_bytes()))
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak(self) -> int:
        return max((rss for _, rss in self.samples), default=0)


# --- page cache --------------------------------------------------------------------------

_F_NOCACHE = 48  # fcntl.h on macOS


def cold_copy(src: Path, dst: Path) -> Path:
    """Copy a directory so that the copy is not in the page cache (macOS only).

    Written with ``F_NOCACHE``, the new files go to disk without staying in memory; the
    first read of the copy really comes from the disk. ``purge`` would do the same for
    the whole system but needs root. On other systems coldness is not
    guaranteed and the caller must say so.
    """
    import fcntl

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(exist_ok=True)
            continue
        with path.open("rb") as fin, target.open("wb") as fout:
            if sys.platform == "darwin":
                fcntl.fcntl(fin.fileno(), _F_NOCACHE, 1)
                fcntl.fcntl(fout.fileno(), _F_NOCACHE, 1)
            while chunk := fin.read(16 << 20):
                fout.write(chunk)
            fout.flush()
            os.fsync(fout.fileno())
    return dst


# --- data --------------------------------------------------------------------------------

SEPARATOR = "<|endoftext|>"


def tinystories(path: Path, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """``(id, text)`` of every story of a TinyStories ``.txt`` file, in file order."""
    name = path.name
    buf: list[str] = []
    n = 0
    with path.open(encoding="utf-8", newline="") as f:
        for line in f:
            if line.strip() == SEPARATOR:
                text = "".join(buf).strip()
                buf = []
                if text:
                    yield f"{name}:{n}", text
                    n += 1
                    if limit is not None and n >= limit:
                        return
            else:
                buf.append(line)
    text = "".join(buf).strip()
    if text and (limit is None or n < limit):
        yield f"{name}:{n}", text


def word_tokenizer(vocab_size: int = 50_000) -> Any:
    """A deterministic WordLevel tokenizer: ``w<N>`` is id ``N + 2``; needs no files."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "<|endoftext|>": 1}
    vocab.update({f"w{i}": i + 2 for i in range(vocab_size - 2)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok


def synthetic_docs(
    n: int, *, mean_words: int = 200, vocab_size: int = 50_000, seed: int = 0
) -> list[tuple[str, str]]:
    """``n`` documents of Zipf-distributed words with lengths around ``mean_words``."""
    rng = np.random.default_rng(seed)
    lengths = np.maximum(1, rng.poisson(mean_words, size=n))
    words = np.minimum(rng.zipf(1.2, size=int(lengths.sum())), vocab_size - 3)
    out = []
    pos = 0
    for i, length in enumerate(lengths.tolist()):
        chunk = words[pos : pos + length]
        pos += length
        out.append((f"doc-{i:09d}", " ".join(f"w{w}" for w in chunk.tolist())))
    return out


def replay_tokenizer(real: Any, table: dict[str, npt.NDArray[np.int64]]) -> Any:
    """A tokbin tokenizer whose ``encode_batch`` looks results up in ``table``.

    It isolates the cost of the writer itself: tokenization becomes a dict lookup plus
    the same ``np.concatenate`` a real adapter does. Fingerprint, vocabulary and special
    tokens are those of ``real``, so the written bytes are identical.
    """
    from tokbin.tokenizer.hf import HFTokenizer

    class ReplayTokenizer(HFTokenizer):
        __slots__ = ()

        def encode_batch(self, texts: Any) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
            arrays = [table[t] for t in texts]
            lengths = np.fromiter(map(len, arrays), dtype=np.int64, count=len(arrays))
            return np.concatenate(arrays), lengths

    replay = ReplayTokenizer.__new__(ReplayTokenizer)
    for name in HFTokenizer.__slots__:
        object.__setattr__(replay, name, getattr(real, name))
    return replay


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
