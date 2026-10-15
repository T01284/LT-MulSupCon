#!/usr/bin/env python3
"""Fig.4 (paper numbering): LT-MulSupCon multi-label contrastive illustration.

Node labels and relation notes are placed off all dashed/solid arrows.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]


def plot_fig3(output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))

    points = {
        "A": (-0.95, 0.70),
        "B": (0.90, 0.55),
        "C": (0.55, -0.95),
        "D": (-0.95, -0.70),
    }
    labels = {
        "A": "A  {joy}",
        "B": "B  {joy, love}",
        "C": "C  {grief}",
        "D": "D  {anger}",
    }
    # Labels sit well away from inter-node arrows
    label_pos = {
        "A": (-0.95, 1.25),
        "B": (0.90, 1.20),
        "C": (0.55, -1.40),
        "D": (-0.95, -1.30),
    }
    colors = {"A": "#3498db", "B": "#2ecc71", "C": "#e74c3c", "D": "#e67e22"}

    def draw_nodes(ax):
        for key, (x, y) in points.items():
            ax.scatter(x, y, s=320, c=colors[key], edgecolors="black", zorder=4, linewidths=1.3)
            lx, ly = label_pos[key]
            ax.text(
                lx, ly, labels[key], ha="center", va="center", fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.95),
                zorder=6,
            )

    # ----- Left -----
    ax = axes[0]
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.75, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Single-Label SupCon\n(Mis-treats partial overlap)", fontsize=16, pad=8)
    draw_nodes(ax)

    ax.annotate(
        "", xy=points["A"], xytext=points["C"],
        arrowprops=dict(arrowstyle="<->", color="red", lw=2.4, linestyle="--"),
        zorder=2,
    )
    # Completely left of the A--C chord
    ax.annotate(
        "false negative\n(no label conflict)",
        xy=(-0.15, -0.10),
        xytext=(-1.55, 0.05),
        fontsize=14, color="#c0392b", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec", edgecolor="#c0392b", alpha=0.95),
        zorder=7,
    )

    ax.annotate(
        "", xy=points["B"], xytext=points["A"],
        arrowprops=dict(arrowstyle="->", color="gray", lw=2.0, linestyle=":"),
        zorder=2,
    )
    ax.text(
        -0.05, 0.95, "weak positive\n(Jaccard=0.5)",
        fontsize=14, color="#555555", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#aaaaaa", alpha=0.95),
        zorder=7,
    )

    # ----- Right -----
    ax = axes[1]
    ax.set_xlim(-1.8, 2.05)
    ax.set_ylim(-1.75, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) LT-MulSupCon (Ours)\nJaccard-weighted + Memory Queue", fontsize=16, pad=8)
    draw_nodes(ax)

    ax.annotate(
        "", xy=points["B"], xytext=points["A"],
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=3.4),
        zorder=2,
    )
    ax.text(
        -0.05, 0.95, "w = 0.50",
        fontsize=15, color="#1e8449", fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#27ae60", alpha=0.95),
        zorder=7,
    )

    queue_box = mpatches.FancyBboxPatch(
        (1.15, 0.35), 0.75, 1.05, boxstyle="round,pad=0.02", linewidth=1.8,
        edgecolor="#8e44ad", facecolor="#f5eef8", linestyle="--"
    )
    ax.add_patch(queue_box)
    ax.text(1.52, 1.18, "Memory\nQueue", ha="center", fontsize=13, color="#8e44ad", fontweight="bold")
    ax.scatter([1.32, 1.58, 1.45], [0.60, 0.50, 0.78], s=80, c="#e74c3c", marker="s", zorder=4)

    ax.annotate(
        "", xy=points["C"], xytext=(1.32, 0.45),
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=2.2, linestyle="--"),
        zorder=2,
    )
    ax.text(
        1.65, -0.55, "tail positives\nfrom queue",
        fontsize=14, color="#6c3483", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#8e44ad", alpha=0.95),
        zorder=7,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_supcon.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig3_supcon.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "figures"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fig3(output_dir)
    print(f"Fig.3/4 SupCon diagram saved to {output_dir}")


if __name__ == "__main__":
    main()
