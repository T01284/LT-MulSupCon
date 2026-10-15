#!/usr/bin/env python3
"""Run all experiments sequentially."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "quick.yaml"

EXPERIMENTS = [
    "baseline",
    "baseline_focal",
    "clfer_lt",
    "clfer_lt_no_supcon",
    "clfer_lt_no_proto",
    "clfer_lt_no_queue",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = Path(args.config)

    summary = {}
    for model_type in EXPERIMENTS:
        print(f"\n{'='*60}\nRunning: {model_type}\n{'='*60}")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config", str(config),
            "--model_type", model_type,
        ]
        ret = subprocess.run(cmd, cwd=str(ROOT))
        if ret.returncode != 0:
            print(f"[warn] {model_type} failed with code {ret.returncode}")
            continue
        result_path = ROOT / "outputs" / "checkpoints" / model_type / "results.json"
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                summary[model_type] = json.load(f)

    out = ROOT / "outputs" / "experiment_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll experiments done. Summary: {out}")


if __name__ == "__main__":
    main()
