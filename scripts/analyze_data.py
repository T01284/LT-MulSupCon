#!/usr/bin/env python3
"""Analyze GoEmotions distribution and generate Fig.1 / Fig.4."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import (
    compute_cooccurrence_matrix,
    compute_label_statistics,
    download_goemotions,
    load_emotion_labels,
    load_split,
    save_statistics,
)


def plot_label_distribution(stats: dict, labels: list[str], output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    counts = [stats["label_counts"].get(l, 0) for l in labels]
    sorted_pairs = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_counts = zip(*sorted_pairs)

    colors = []
    for lbl in sorted_labels:
        if lbl in stats["groups"]["head"]:
            colors.append("#2ecc71")
        elif lbl in stats["groups"]["medium"]:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    ax.bar(range(len(sorted_labels)), sorted_counts, color=colors, width=0.86)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=90, ha="center", fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("training count (log scale)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)
    from matplotlib.patches import Patch

    legend = [
        Patch(color="#2ecc71", label="Head"),
        Patch(color="#f39c12", label="Medium"),
        Patch(color="#e74c3c", label="Tail"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_label_distribution.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig1_label_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cooccurrence(cooc: np.ndarray, labels: list[str], output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    sns.heatmap(
        cooc,
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Normalized Co-occurrence"},
    )
    plt.xticks(rotation=90, ha="center", fontsize=6)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_cooccurrence.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig4_cooccurrence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GoEmotions data")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "goemotions"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "figures"))
    parser.add_argument("--download", action="store_true", help="Download data if missing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download or not (data_dir / "train.tsv").exists():
        download_goemotions(data_dir)

    labels = load_emotion_labels(data_dir)
    train_samples = load_split(data_dir, "train")
    stats = compute_label_statistics(train_samples, labels)
    cooc = compute_cooccurrence_matrix(train_samples, len(labels))

    save_statistics(stats, output_dir / "dataset_statistics.json")
    with open(output_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_counts": stats["label_counts"],
                "groups": stats["groups"],
                "total_samples": stats["total_samples"],
                "multilabel_ratio": stats["multilabel_ratio"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    plot_label_distribution(stats, labels, output_dir)
    plot_cooccurrence(cooc, labels, output_dir)

    print(f"Statistics saved to {output_dir}")
    print(f"Head: {stats['groups']['head']}")
    print(f"Tail: {stats['groups']['tail']}")


if __name__ == "__main__":
    main()
