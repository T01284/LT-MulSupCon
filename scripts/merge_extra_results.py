#!/usr/bin/env python3
"""Merge round-2 results (hard SupCon, extended lambda, multi-seed) into experiment_summary.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "experiment_summary.json"
CKPT = ROOT / "outputs" / "checkpoints"
SEMV = ROOT / "outputs" / "checkpoints_semeval"

TUNE_KEYS = [
    "clfer_lt_l002",
    "clfer_lt_l005",
    "clfer_lt_l010",
    "clfer_lt_l015",
    "clfer_lt_l020",
]
BASELINE_KEYS = ["hard_supcon", "hard_supcon_focal"]
SEED_KEYS = [
    "baseline_focal_s42",
    "baseline_focal_s43",
    "baseline_focal_s44",
    "baseline_focal_s45",
    "baseline_focal_s46",
    "clfer_lt_l010_s42",
    "clfer_lt_l010_s43",
    "clfer_lt_l010_s44",
    "clfer_lt_l010_s45",
    "clfer_lt_l010_s46",
]
ABL_L010_KEYS = [
    "clfer_lt_l010_no_proto",
    "clfer_lt_l010_no_proto_s43",
    "clfer_lt_l010_no_proto_s44",
    "clfer_lt_l010_no_proto_s45",
    "clfer_lt_l010_no_proto_s46",
    "clfer_lt_l010_no_queue",
    "clfer_lt_l010_no_queue_s43",
    "clfer_lt_l010_no_queue_s44",
    "clfer_lt_l010_no_queue_s45",
    "clfer_lt_l010_no_queue_s46",
    "clfer_lt_l010_hard_match",
    "clfer_lt_l010_hard_match_s43",
    "clfer_lt_l010_hard_match_s44",
    "clfer_lt_l010_hard_match_s45",
    "clfer_lt_l010_hard_match_s46",
]
ABL_L002_NO_PROTO_KEYS = [
    "clfer_lt_l002_no_proto",
    "clfer_lt_l002_no_proto_s43",
    "clfer_lt_l002_no_proto_s44",
    "clfer_lt_l002_no_proto_s45",
    "clfer_lt_l002_no_proto_s46",
]
HARD_SEED_KEYS = [
    "hard_supcon",
    "hard_supcon_s43",
    "hard_supcon_s44",
    "hard_supcon_s45",
    "hard_supcon_s46",
]
SEMV_KEYS = ["semeval_baseline", "semeval_baseline_focal", "semeval_clfer_lt"]


def load_result(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    summary = {}
    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    for name in TUNE_KEYS + BASELINE_KEYS + SEED_KEYS + ABL_L010_KEYS + ABL_L002_NO_PROTO_KEYS + HARD_SEED_KEYS:
        data = load_result(CKPT / name / "results.json")
        if data:
            summary[name] = data
            m = data.get("test_metrics", {})
            print(f"{name}: macro={m.get('macro_f1', 0):.4f} tail={m.get('tail', 0):.4f}")

    for name in SEMV_KEYS:
        data = load_result(SEMV / name / "results.json")
        if data:
            summary[name] = data
            m = data.get("test_metrics", {})
            print(f"semeval {name}: macro={m.get('macro_f1', 0):.4f}")

    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {SUMMARY}")


if __name__ == "__main__":
    main()
