#!/usr/bin/env python3
"""Sync experiment figures into paper/figures for LaTeX compilation."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "figures"
DST = ROOT / "paper" / "figures"

FIGURES = [
    "fig1_label_distribution.pdf",
    "fig2_architecture.pdf",
    "fig3_supcon.pdf",
    "fig4_cooccurrence.pdf",
    "fig5_hit_prob.pdf",
    "fig6_jaccard_hist.pdf",
]

PNGS = [
    "fig1_label_distribution.png",
    "fig2_architecture.png",
    "fig3_supcon.png",
    "fig4_cooccurrence.png",
    "fig5_hit_prob.png",
    "fig6_jaccard_hist.png",
]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in FIGURES + PNGS:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, DST / name)
            print(f"copied {name}")
        elif (DST / name).exists():
            print(f"skip (already in paper/figures): {name}")
        else:
            print(f"missing: {name}")
    print(f"Done. Figures ready in {DST}")


if __name__ == "__main__":
    main()
