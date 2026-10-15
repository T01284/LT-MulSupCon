#!/usr/bin/env python3
"""Sync paper-aligned artifacts into outputs/ before a public GitHub push."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RERUN = ROOT / "outputs" / "rtx6000_rerun"
PACK_PRED = RERUN / "pack" / "predictions"
OUT_PRED = ROOT / "outputs" / "predictions"


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def main() -> None:
    print("Syncing RTX6000 paper artifacts...")
    copy_if_exists(RERUN / "key_results.json", ROOT / "outputs" / "key_results.json")
    copy_if_exists(RERUN / "experiment_summary.json", ROOT / "outputs" / "experiment_summary.json")

    if PACK_PRED.is_dir():
        OUT_PRED.mkdir(parents=True, exist_ok=True)
        n = 0
        for src in PACK_PRED.iterdir():
            if src.is_file():
                dst = OUT_PRED / src.name
                if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(src, dst)
                    n += 1
        print(f"  predictions: {n} files updated under outputs/predictions/")
    else:
        print("  skip predictions (no rtx6000 pack)")


if __name__ == "__main__":
    main()
