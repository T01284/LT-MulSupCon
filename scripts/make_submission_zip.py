#!/usr/bin/env python3
"""Build an Expert Systems LaTeX submission zip from paper/ sources."""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT_DIR = ROOT / "submission_pdfs"

# Figures referenced by main.tex or supplementary.tex
FIGURE_PDFS = (
    "fig1_label_distribution.pdf",
    "fig3_supcon.pdf",
    "fig4_cooccurrence.pdf",
    "fig5_hit_prob.pdf",
    "fig6_jaccard_hist.pdf",
)

ROOT_FILES = (
    "main.tex",
    "supplementary.tex",
    "references.bib",
    "sn-jnl.cls",
    "sn-apa.bst",
    "sn-apacite.bst",
)


def zip_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    return f"EXSY_CLFER-LT_latex_{stamp}.zip"


def validate() -> None:
    missing: list[str] = []
    for name in ROOT_FILES:
        if not (PAPER / name).is_file():
            missing.append(name)
    for name in FIGURE_PDFS:
        if not (PAPER / "figures" / name).is_file():
            missing.append(f"figures/{name}")
    if missing:
        raise FileNotFoundError("Missing submission files:\n  " + "\n  ".join(missing))


def build_zip(dest: Path) -> None:
    validate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ROOT_FILES:
            zf.write(PAPER / name, arcname=name)
        for name in FIGURE_PDFS:
            zf.write(PAPER / "figures" / name, arcname=f"figures/{name}")
    print(f"Created {dest}")
    print(f"  Files: {len(ROOT_FILES) + len(FIGURE_PDFS)}")
    with zipfile.ZipFile(dest) as zf:
        for info in zf.infolist():
            print(f"    {info.filename} ({info.file_size:,} bytes)")


def main() -> None:
    dest = OUT_DIR / zip_name()
    build_zip(dest)


if __name__ == "__main__":
    main()
