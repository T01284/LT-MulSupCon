#!/usr/bin/env python3
"""Generate architecture diagram and result figures for the paper."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


def plot_architecture(output_dir: Path) -> None:
    # Compact canvas + large relative fonts so labels stay readable in two-column layout.
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    boxes = [
        (0.25, 2.05, 1.85, 1.05, "Input Text"),
        (2.45, 2.05, 2.15, 1.05, "RoBERTa\nEncoder"),
        (5.05, 3.55, 2.05, 0.95, "Projection\nHead"),
        (5.05, 2.20, 2.05, 0.95, "Classifier"),
        (5.05, 0.85, 2.05, 0.95, "Label\nPrototypes"),
        (7.55, 3.55, 2.55, 0.95, "LT-MulSupCon"),
        (7.55, 2.20, 2.55, 0.95, "CB-Focal Loss"),
        (7.55, 0.85, 2.55, 0.95, "LPC Loss"),
        (5.05, 0.08, 5.05, 0.62, r"$L = L_{cls}+\lambda_1 L_{supcon}+\lambda_2 L_{proto}$"),
    ]

    for x, y, w, h, text in boxes:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04",
            linewidth=2.0,
            edgecolor="#2c3e50",
            facecolor="#ecf0f1",
        )
        ax.add_patch(box)
        fs = 13 if "L =" in text or text.startswith("$") else 15
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fs, fontweight="bold",
            linespacing=1.1,
        )

    arrows = [
        ((2.10, 2.57), (2.45, 2.57)),
        ((4.60, 2.57), (5.05, 4.02)),
        ((4.60, 2.57), (5.05, 2.67)),
        ((4.60, 2.57), (5.05, 1.32)),
        ((7.10, 4.02), (7.55, 4.02)),
        ((7.10, 2.67), (7.55, 2.67)),
        ((7.10, 1.32), (7.55, 1.32)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, color="#34495e", lw=1.8)
        )

    ax.set_title("CLFER-LT Model Architecture", fontsize=17, fontweight="bold", pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(output_dir / "fig2_architecture.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig2_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_results(results_files: dict[str, Path], output_dir: Path) -> None:
    models, head, medium, tail = [], [], [], []
    for name, path in results_files.items():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("test_metrics", data)
        models.append(name)
        head.append(metrics.get("head", 0))
        medium.append(metrics.get("medium", 0))
        tail.append(metrics.get("tail", 0))

    if not models:
        print("No result files found; skipping Fig.6")
        return

    x = range(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], head, width, label="Head", color="#2ecc71")
    ax.bar(x, medium, width, label="Medium", color="#f39c12")
    ax.bar([i + width for i in x], tail, width, label="Tail", color="#e74c3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Head / Medium / Tail Group Performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_group_f1.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig6_group_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "figures"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_architecture(output_dir)

    results_files = {
        "Baseline": ROOT / "outputs" / "checkpoints" / "baseline" / "results.json",
        "Focal": ROOT / "outputs" / "checkpoints" / "baseline_focal" / "results.json",
        "CLFER-LT": ROOT / "outputs" / "checkpoints" / "clfer_lt" / "results.json",
    }
    plot_group_results(results_files, output_dir)
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
