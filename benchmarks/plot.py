"""Charts for docs/benchmarks.md from the JSON files in ``benchmarks/results``.

::

    uv run --group bench python benchmarks/plot.py

A chart whose results are missing is skipped.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench_common import RESULTS, ROOT

OUT = ROOT / "docs" / "benchmarks"
COLORS = {
    "tokbin": "#2563eb",
    "raw": "#9ca3af",
    "alt": "#f59e0b",
    "bad": "#dc2626",
    "ok": "#16a34a",
    "extra": "#7c3aed",
}

Chart = Callable[[Path], None]
CHARTS: dict[str, Chart] = {}


def chart(name: str) -> Callable[[Chart], Chart]:
    def register(fn: Chart) -> Chart:
        CHARTS[name] = fn
        return fn

    return register


def load(results: Path, name: str) -> dict[str, Any]:
    return json.loads((results / f"{name}.json").read_text())["data"]


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)


def style(ax: Any) -> None:
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# --- write -------------------------------------------------------------------------------


@chart("write-breakdown")
def write_breakdown(results: Path) -> None:
    data = load(results, "write-breakdown")
    names = list(data["tokenizers"])
    parts = [
        ("generate", "iterate the input", COLORS["raw"]),
        ("encode_batch", "tokenizers.encode_batch", COLORS["raw"]),
        ("tokenize", "+ ids to numpy", COLORS["raw"]),
        ("hand_written", "hand-written .bin writer", COLORS["alt"]),
        ("tokbin", "tokbin write", COLORS["tokbin"]),
        ("tokbin_replay", "tokbin, tokenization replayed", COLORS["extra"]),
    ]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        row = data["tokenizers"][name]
        values = [row[k]["best"] for k, _, _ in parts]
        y = np.arange(len(parts))[::-1]
        ax.barh(y, values, color=[c for _, _, c in parts])
        ax.set_yticks(y, [label for _, label, _ in parts])
        for yi, v in zip(y, values, strict=True):
            ax.text(v, yi, f" {v:.2f} s", va="center", fontsize=8)
        mtok = row["n_tokens"] / 1e6
        ax.set_title(f"{name} ({row['dtype']}): {data['n_docs']:,} docs, {mtok:.1f}M tokens")
        ax.set_xlabel("seconds (best of runs)")
        ax.set_xlim(0, max(values) * 1.25)
        style(ax)
    save(fig, "write-breakdown")


@chart("write-batch")
def write_batch(results: Path) -> None:
    data = load(results, "write-batch")
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, rows), marker in zip(data["tokenizers"].items(), "os", strict=False):
        x = [r["batch_docs"] for r in rows]
        w = [r["n_docs"] / r["write"]["best"] / 1e3 for r in rows]
        t = [r["n_docs"] / r["tokenize"]["best"] / 1e3 for r in rows]
        ax.plot(x, w, marker=marker, color=COLORS["tokbin"], label=f"{name}: tokbin write")
        ax.plot(x, t, marker=marker, ls="--", color=COLORS["raw"], label=f"{name}: tokenize only")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch_docs (documents per encode_batch call)")
    ax.set_ylabel("thousand documents / s")
    ax.axvline(1024, color=COLORS["ok"], lw=1, ls=":")
    ax.text(1024, ax.get_ylim()[1] * 0.95, " default", color=COLORS["ok"], fontsize=8)
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "write-batch")


@chart("write-threads")
def write_threads(results: Path) -> None:
    data = load(results, "write-threads")
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, rows), marker in zip(data["tokenizers"].items(), "os", strict=False):
        x = [r["threads"] for r in rows]
        base_w = rows[0]["write"]["best"]
        ax.plot(
            x,
            [base_w / r["write"]["best"] for r in rows],
            marker=marker,
            color=COLORS["tokbin"],
            label=f"{name}: tokbin write",
        )
        base_t = rows[0]["tokenize"]["best"]
        ax.plot(
            x,
            [base_t / r["tokenize"]["best"] for r in rows],
            marker=marker,
            ls="--",
            color=COLORS["raw"],
            label=f"{name}: tokenize only",
        )
    top = max(r["threads"] for rows in data["tokenizers"].values() for r in rows)
    ax.plot([1, top], [1, top], color="black", lw=0.8, ls=":", label="linear")
    ax.set_xlabel("RAYON_NUM_THREADS")
    ax.set_ylabel("speed-up over 1 thread")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "write-threads")


@chart("write-doclen")
def write_doclen(results: Path) -> None:
    data = load(results, "write-doclen")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for (name, rows), marker in zip(data["tokenizers"].items(), "os", strict=False):
        mean = [r["n_tokens"] / r["n_docs"] for r in rows]
        # The writer's own work (tokenization replayed) against tokenization: a stable
        # ratio. The difference of two end-to-end runs is within process noise.
        overhead = [r["writer_only"]["best"] / r["tokenize"]["best"] * 100 for r in rows]
        a1.plot(mean, overhead, marker=marker, color=COLORS["tokbin"], label=name)
        for m, o, r in zip(mean, overhead, rows, strict=True):
            if marker != "o":
                continue  # one set of labels is enough
            per_doc = r["writer_only"]["best"] / r["n_docs"] * 1e6
            a1.annotate(
                f"{r['variant']}\n{per_doc:.1f} us/doc",
                (m, o),
                fontsize=7,
                textcoords="offset points",
                xytext=(6, 4),
            )
        index = [
            sum(v for k, v in r["files"].items() if not k.endswith(".bin") and k != "meta.json")
            / r["shard_bytes"]
            * 100
            for r in rows
        ]
        a2.plot(mean, index, marker=marker, color=COLORS["bad"], label=name)
    a1.set_title("tokbin's own work (tokenization replayed)")
    a1.set_ylabel("% of the tokenization time")
    a2.set_title("index files (offsets, ids) relative to token data")
    a2.set_ylabel("% of shard bytes")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("mean tokens per document")
        ax.axhline(0, color="black", lw=0.6)
        ax.legend(fontsize=8)
        style(ax)
    save(fig, "write-doclen")


@chart("write-shards")
def write_shards(results: Path) -> None:
    data = load(results, "write-shards")
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = [r["shard_mib"] for r in rows]
    y = [r["n_tokens"] / r["write"]["best"] / 1e6 for r in rows]
    ax.plot(x, y, marker="o", color=COLORS["extra"])
    for xi, yi, r in zip(x, y, rows, strict=True):
        ax.annotate(
            f"{r['n_shards']} shards",
            (xi, yi),
            fontsize=7,
            textcoords="offset points",
            xytext=(3, -10),
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("shard_bytes, MiB")
    ax.set_ylabel("M tokens / s (tokenization replayed)")
    ax.set_ylim(0, max(y) * 1.15)
    style(ax)
    save(fig, "write-shards")


@chart("write-resume")
def write_resume(results: Path) -> None:
    data = load(results, "write-resume")
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = [f"{r['fraction']:.0%}" for r in rows]
    first = [r["first_part"]["best"] for r in rows]
    resume = [r["resume"]["best"] for r in rows]
    ax.bar(labels, first, color=COLORS["raw"], label="until the interruption")
    ax.bar(labels, resume, bottom=first, color=COLORS["tokbin"], label="resume")
    ax.axhline(data["full"]["best"], color="black", ls="--", lw=1, label="uninterrupted write")
    ax.set_xlabel("interrupted after this share of the input")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "write-resume")


@chart("write-full")
def write_full(results: Path) -> None:
    data = load(results, "write-full")
    names = list(data["tokenizers"])
    colors = {"tokbin": COLORS["tokbin"], "hand_written": COLORS["alt"], "tokenize": COLORS["raw"]}
    labels = {"tokbin": "tokbin", "hand_written": "hand-written .bin", "tokenize": "tokenize only"}
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        runs = data["tokenizers"][name]
        for variant, color in colors.items():
            for k, run in enumerate(runs[variant]):
                t, rss = zip(*run["rss"], strict=True)
                seconds = ", ".join(f"{r['seconds']:.0f}" for r in runs[variant])
                ax.plot(
                    t,
                    np.array(rss) / 2**20,
                    color=color,
                    alpha=1 if k == 0 else 0.45,
                    label=f"{labels[variant]} ({seconds} s)" if k == 0 else None,
                )
        stats = runs["tokbin"][0]["stats"]
        ax.set_title(f"{name}: {stats['n_docs']:,} docs, {stats['n_items'] / 1e6:.0f}M tokens")
        ax.set_xlabel("seconds")
        ax.set_ylabel("resident memory, MiB")
        ax.set_ylim(0, None)
        ax.legend(fontsize=8)
        style(ax)
    save(fig, "write-full")


@chart("write-idhashes")
def write_idhashes(results: Path) -> None:
    rows = load(results, "write-idhashes")["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    n = [r["n"] for r in rows]
    ax.plot(
        n,
        [r["held_bytes"] / 2**20 for r in rows],
        marker="o",
        color=COLORS["tokbin"],
        label="held during the write",
    )
    ax.plot(
        n,
        [(r["held_bytes"] + r["peak_growth_in_report"]) / 2**20 for r in rows],
        marker="s",
        color=COLORS["bad"],
        label="peak at the final duplicate check",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("documents with ids")
    ax.set_ylabel("MiB")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "write-idhashes")


# --- read --------------------------------------------------------------------------------


@chart("read-calls")
def read_calls(results: Path) -> None:
    d = load(results, "read-calls")["us"]
    pairs = [
        ("one item", "raw_item", "src_item"),
        ("256-item slice", "raw_slice_256", "src_window_256"),
        ("document", "raw_doc", "src_doc"),
        ("id of document", None, "src_id_of_doc"),
        ("open (read_source)", None, "open"),
        ("pickle round trip", None, "pickle_roundtrip"),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(pairs))[::-1]
    raw = [d[r]["median"] if r else 0 for _, r, _ in pairs]
    tb = [d[t]["median"] for _, _, t in pairs]
    ax.barh(y + 0.2, tb, height=0.4, color=COLORS["tokbin"], label="tokbin")
    ax.barh(y - 0.2, raw, height=0.4, color=COLORS["raw"], label="numpy on a memmap")
    for yi, v in zip(y, tb, strict=True):
        ax.text(v, yi + 0.2, f" {v:.1f} us", va="center", fontsize=7)
    for yi, v in zip(y, raw, strict=True):
        if v:
            ax.text(v, yi - 0.2, f" {v:.2f} us", va="center", fontsize=7)
    ax.set_yticks(y, [p[0] for p in pairs])
    ax.set_xscale("log")
    ax.set_xlabel("microseconds per call (median, warm cache)")
    ax.legend(fontsize=8, loc="lower right")
    style(ax)
    save(fig, "read-calls")


@chart("read-windows")
def read_windows(results: Path) -> None:
    rows = load(results, "read-windows")["rows"]
    x = [r["length"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [
        ("src_view", "tokbin window (view)", COLORS["tokbin"], "-"),
        ("raw_view", "memmap slice (view)", COLORS["raw"], "-"),
        ("src_int64", "tokbin window -> int64", COLORS["tokbin"], "--"),
        ("raw_int64", "memmap slice -> int64", COLORS["raw"], "--"),
        ("src_cross", "tokbin window across shards", COLORS["bad"], ":"),
    ]
    for key, label, color, ls in series:
        pts = [(xi, r[key]["median"]) for xi, r in zip(x, rows, strict=True) if r.get(key)]
        ax.plot(*zip(*pts, strict=True), marker="o", ms=3, color=color, ls=ls, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("window length, tokens")
    ax.set_ylabel("microseconds (median, warm cache)")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "read-windows")


@chart("read-sampling")
def read_sampling(results: Path) -> None:
    rows = load(results, "read-sampling")["rows"]
    labels = [f"{r['batch']}x{r['block']}" for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(rows))
    series = [
        ("tokbin", "Source.sample_windows", COLORS["tokbin"]),
        ("mixture", "Mixture.batch (1 source)", COLORS["extra"]),
        ("raw_loop", "memmap, Python loop (nanoGPT)", COLORS["raw"]),
        ("raw_fancy", "memmap, fancy indexing", COLORS["alt"]),
    ]
    width = 0.2
    for k, (key, label, color) in enumerate(series):
        rate = [r["batch"] * r["block"] / (r[key]["median"] / 1e6) / 1e6 for r in rows]
        ax.bar(x + (k - 1.5) * width, rate, width, color=color, label=label)
    ax.set_xticks(x, labels)
    ax.set_xlabel("batch x block")
    ax.set_ylabel("M tokens / s (warm cache)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "read-sampling")


@chart("read-shards")
def read_shards(results: Path) -> None:
    rows = load(results, "read-shards")["rows"]
    n = [r["n_shards"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    a1.plot(
        n,
        [r["open"]["median"] / 1e3 for r in rows],
        marker="o",
        color=COLORS["bad"],
        label="read_source",
    )
    a1.plot(
        n,
        [r["inspect"]["median"] / 1e3 for r in rows],
        marker="s",
        color=COLORS["alt"],
        label="inspect_source",
    )
    a1.set_ylabel("milliseconds")
    a1.set_title("opening grows with the number of shards")
    a2.plot(
        n,
        [r["window_2048"]["median"] for r in rows],
        marker="o",
        color=COLORS["tokbin"],
        label="window(start, 2048), shards mapped",
    )
    a2.plot(
        n,
        [r["window_first_pass_us"]["median"] for r in rows],
        marker="^",
        color=COLORS["bad"],
        label="window, first 4096 calls (shards being mapped)",
    )
    a2.plot(
        n,
        [r["sample_32x2048"]["median"] / 32 for r in rows],
        marker="s",
        color=COLORS["extra"],
        label="sample_windows(32, 2048) per row",
    )
    a2.set_ylabel("microseconds")
    a2.set_title("access does not, once the shards are mapped")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("shards (same 20M tokens)")
        ax.legend(fontsize=8)
        style(ax)
    a1.set_yscale("log")
    a2.set_yscale("log")
    save(fig, "read-shards")


@chart("read-docs")
def read_docs(results: Path) -> None:
    rows = load(results, "read-docs")["rows"]
    n = [r["n_docs"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(
        n, [r["doc"]["median"] for r in rows], marker="o", color=COLORS["tokbin"], label="doc(i)"
    )
    ax.plot(
        n,
        [r["id_of_doc"]["median"] for r in rows],
        marker="s",
        color=COLORS["extra"],
        label="id_of_doc(i)",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_xlabel("documents in the source")
    ax.set_ylabel("microseconds (median)")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "read-docs")


@chart("read-cold")
def read_cold(results: Path) -> None:
    d = load(results, "read-cold")
    blocks = [k for k in d if k.startswith("window_")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, color in zip(blocks, (COLORS["tokbin"], COLORS["extra"], COLORS["bad"]), strict=False):
        for kind, ls in (("cold_us", "-"), ("warm_us", ":")):
            v = np.sort(d[key][kind])
            ax.plot(
                v,
                np.linspace(0, 1, len(v)),
                color=color,
                ls=ls,
                label=f"{key.split('_')[1]} tokens, {kind.split('_')[0]}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("microseconds per window (read and summed)")
    ax.set_ylabel("share of windows")
    ax.legend(fontsize=7)
    style(ax)
    save(fig, "read-cold")


@chart("read-memory")
def read_memory(results: Path) -> None:
    d = load(results, "read-memory")
    sampled, rss = zip(*d["rss"], strict=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(
        np.array(sampled) / d["n_items"],
        np.array(rss) / 2**20,
        color=COLORS["tokbin"],
        label="resident memory of the process",
    )
    ax.axhline(d["data_bytes"] / 2**20, color=COLORS["bad"], ls="--", label="size of the shards")
    ax.set_xlabel("tokens sampled / tokens in the source")
    ax.set_ylabel("MiB")
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "read-memory")


@chart("read-mixture")
def read_mixture(results: Path) -> None:
    d = load(results, "read-mixture")
    rows = d["rows"]
    k = [r["sources"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(
        k,
        [r["batch_32x2048"]["median"] for r in rows],
        marker="o",
        color=COLORS["tokbin"],
        label="Mixture.batch(32, 2048)",
    )
    ax.plot(
        k,
        [r["batch_512x128"]["median"] for r in rows],
        marker="s",
        color=COLORS["extra"],
        label="Mixture.batch(512, 128)",
    )
    ax.axhline(
        d["single_32x2048"]["median"],
        color=COLORS["tokbin"],
        ls=":",
        label="Source.sample_windows(32, 2048)",
    )
    ax.axhline(
        d["single_512x128"]["median"],
        color=COLORS["extra"],
        ls=":",
        label="Source.sample_windows(512, 128)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sources in the mixture")
    ax.set_ylabel("microseconds per batch")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    style(ax)
    save(fig, "read-mixture")


# --- storage and torch -------------------------------------------------------------------


@chart("storage-pack")
def storage_pack(results: Path) -> None:
    d = load(results, "storage-pack")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    names = list(d["sources"])
    width = 0.8 / max(1, len(names))
    for k, name in enumerate(names):
        rows = [r for r in d["sources"][name] if not r.get("skipped")]
        labels = [f"{r['method']}-{r['level']}" for r in rows]
        x = np.arange(len(rows)) + (k - (len(names) - 1) / 2) * width
        a1.bar(x, [r["ratio"] * 100 for r in rows], width, label=name)
        a2.bar(x, [r["bytes"] / r["pack_s"] / 2**20 for r in rows], width, label=f"{name} pack")
        a2.scatter(
            x,
            [r["bytes"] / r["unpack_s"] / 2**20 for r in rows],
            color="black",
            marker="_",
            s=200,
            zorder=3,
        )
        a1.set_xticks(np.arange(len(rows)), labels)
        a2.set_xticks(np.arange(len(rows)), labels)
    a1.set_ylabel("packed size, % of the source")
    a2.set_ylabel("MiB of source / s  (bars: pack, dashes: unpack)")
    a2.set_yscale("log")
    for ax in (a1, a2):
        ax.legend(fontsize=8)
        style(ax)
    save(fig, "storage-pack")


@chart("torch-loader")
def torch_loader(results: Path) -> None:
    rows = json.loads((results / "torch-loader.json").read_text())["data"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    colors = {
        "raw memmap": COLORS["raw"],
        "WindowDataset": COLORS["tokbin"],
        "MixtureDataset": COLORS["extra"],
    }
    for name, color in colors.items():
        sub = [r for r in rows if r["dataset"] == name]
        w = [r["workers"] for r in sub]
        a1.plot(w, [r["batches_per_s"] for r in sub], marker="o", color=color, label=name)
        a2.plot(w, [r["first_batch_s"] for r in sub], marker="o", color=color, label=name)
    a1.set_ylabel("batches / s (32 x 2048)")
    a2.set_ylabel("seconds to the first batch")
    for ax in (a1, a2):
        ax.set_xlabel("DataLoader num_workers")
        ax.legend(fontsize=8)
        style(ax)
    save(fig, "torch-loader")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in CHARTS.items():
        try:
            fn(args.results)
        except FileNotFoundError:
            print(f"skip {name}: no results")
            continue
        print(f"ok   {name}")


if __name__ == "__main__":
    main()
