#!/usr/bin/env python3
"""Mechanism figures: rare-class hit probability vs n_c, and Jaccard-weight histogram."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N = 43410
B = 32
# Official-split training counts from Table labelfreq in the paper.
N_C = {
    "grief": 77,
    "pride": 111,
    "relief": 153,
    "nervousness": 164,
    "embarrassment": 303,
    "remorse": 545,
    "fear": 596,
    "joy": 1452,
    "admiration": 4130,
    "neutral": 14219,
}

OUT = ROOT / "outputs" / "figures"
PAPER = ROOT / "paper" / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )


def hit_prob(n_c: np.ndarray, k: int) -> np.ndarray:
    """P(at least one other class-c example in a pool of B-1 in-batch + K queue)."""
    p = n_c / N
    m = (B - 1) + k
    return 1.0 - np.power(1.0 - p, m)


def plot_hit_prob() -> None:
    style()
    n = np.logspace(np.log10(50), np.log10(N), 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ks = [(0, "K=0 (batch only)"), (512, "K=512"), (1024, "K=1024"), (4096, "K=4096")]
    colors = ["#7f8c8d", "#e67e22", "#2980b9", "#1a7a4c"]
    for (k, lab), c in zip(ks, colors):
        ax.plot(n, hit_prob(n, k), label=lab, color=c, lw=2.0)
    marks = [("grief", 77), ("nervousness", 164), ("fear", 596)]
    for name, nc in marks:
        ax.axvline(nc, color="#bdc3c7", lw=0.8, ls="--")
        ax.text(nc * 1.05, 0.08, name, rotation=90, va="bottom", fontsize=8, color="#555")
    ax.set_xscale("log")
    ax.set_xlim(50, N)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"class frequency $n_c$")
    ax.set_ylabel(r"$P$(at least one class-$c$ partner)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.62, 0.42), fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    for dest in (OUT, PAPER):
        fig.savefig(dest / "fig5_hit_prob.pdf", bbox_inches="tight")
        fig.savefig(dest / "fig5_hit_prob.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_hit_prob")


def collect_jaccard_weights(max_pairs: int = 80000) -> np.ndarray | None:
    data_dir = ROOT / "data" / "goemotions"
    try:
        from src.data.preprocess import load_split
    except Exception as exc:
        print("cannot import load_split:", exc)
        return None
    if not (data_dir / "train.tsv").exists() and not (data_dir / "train.csv").exists():
        print("GoEmotions train split not on disk; skip Jaccard histogram")
        return None
    samples = load_split(data_dir, "train")
    Y = np.stack([s["labels"] for s in samples]).astype(np.int8)
    n = len(Y)
    rng = random.Random(42)
    weights: list[float] = []
    n_batches = 2000
    batch_size = 32
    for _ in range(n_batches):
        idx = rng.sample(range(n), batch_size)
        batch = Y[idx]
        for i in range(batch_size):
            yi = batch[i]
            for j in range(batch_size):
                if i == j:
                    continue
                inter = float((yi * batch[j]).sum())
                if inter <= 0:
                    continue
                union = float(np.maximum(yi, batch[j]).sum())
                weights.append(inter / union)
                if len(weights) >= max_pairs:
                    return np.asarray(weights, dtype=np.float32)
    return np.asarray(weights, dtype=np.float32)


def plot_jaccard_hist(w: np.ndarray) -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, 1.0, 21)
    ax.hist(w, bins=bins, color="#1a7a4c", edgecolor="white", linewidth=0.6)
    exact = float((w >= 0.999).mean())
    ax.axvline(1.0, color="#c0392b", lw=1.4, ls="--")
    ax.text(0.98, 0.92, f"exact match {100 * exact:.1f}%", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_xlabel("Jaccard weight of directed positive pairs")
    ax.set_ylabel("count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for dest in (OUT, PAPER):
        fig.savefig(dest / "fig6_jaccard_hist.pdf", bbox_inches="tight")
        fig.savefig(dest / "fig6_jaccard_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "n_pairs": int(w.size),
        "exact_match_fraction": exact,
        "mean_weight": float(w.mean()),
    }
    (ROOT / "outputs" / "jaccard_hist_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("wrote fig6_jaccard_hist", summary)


def plot_jaccard_fallback() -> None:
    """Two-bin plot from the already-computed 5k-batch overlap JSON."""
    path = ROOT / "outputs" / "positive_pair_overlap.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    exact = data["among_jaccard_positive_pairs"]["exact_match_fraction"]
    partial = data["among_jaccard_positive_pairs"]["partial_only_fraction"]
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(
        ["exact match\n($w=1$)", "partial overlap\n($0<w<1$)"],
        [100 * exact, 100 * partial],
        color=["#1a7a4c", "#e67e22"],
        width=0.55,
    )
    ax.set_ylabel("% of Jaccard-positive directed pairs")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for dest in (OUT, PAPER):
        fig.savefig(dest / "fig6_jaccard_hist.pdf", bbox_inches="tight")
        fig.savefig(dest / "fig6_jaccard_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig6_jaccard_hist (two-bin fallback from overlap JSON)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    plot_hit_prob()
    weights = collect_jaccard_weights()
    if weights is not None and weights.size > 0:
        plot_jaccard_hist(weights)
    else:
        plot_jaccard_fallback()


if __name__ == "__main__":
    main()
