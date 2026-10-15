#!/usr/bin/env python3
"""Export paper-aligned evidence summary from five-seed key_results.json.

Legacy `paper_table_fragments.json` was generated from single-run
experiment_summary.json and contradicted the manuscript. This script now
writes a manifest whose numbers match `outputs/key_results.json` and the
dev lambda grid in `outputs/experiment_summary.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_RESULTS = ROOT / "outputs" / "key_results.json"
SUMMARY = ROOT / "outputs" / "experiment_summary.json"
OUT = ROOT / "outputs" / "paper_table_fragments.json"
IJCS_STATS = ROOT / "outputs" / "analysis" / "ijcs_revision_stats.json"

LAMBDA_DEV_KEYS = [
    ("clfer_lt_l002", "0.02"),
    ("clfer_lt_l005", "0.05"),
    ("clfer_lt_l010", "0.10"),
    ("clfer_lt_l015", "0.15"),
    ("clfer_lt_l020", "0.20"),
    ("clfer_lt", "0.50"),
]


def fmt(v, nd=3):
    if v is None:
        return None
    return round(float(v), nd)


def mean_std_block(entry: dict, field: str) -> dict:
    block = entry.get(field, {})
    return {
        "mean": fmt(block.get("mean")),
        "std": fmt(block.get("std")),
        "ci95": [fmt(x, 4) for x in block.get("ci95", [])],
        "n": block.get("n"),
    }


def best_dev_epoch(history: list[dict]) -> dict:
    return max(history, key=lambda r: r.get("macro_f1", 0.0))


def ijcs_revision_summary() -> dict | None:
    """Headline queue-slope and compensation numbers from scripts/analyze_ijcs_revision.py."""
    if not IJCS_STATS.exists():
        return None
    st = json.loads(IJCS_STATS.read_text(encoding="utf-8"))

    def slope(key: str) -> dict | None:
        s = st.get(key)
        if s is None:
            return None
        return {"mean": fmt(s["mean"], 4), "ci95": [fmt(s["lo"], 4), fmt(s["hi"], 4)], "p": fmt(s["p"], 5), "n": s["n"]}

    def comp(key: str) -> dict | None:
        c = st.get(key)
        if c is None:
            return None
        out = {"slope": fmt(c["slope"]), "r": fmt(c["r"]), "p": float(f"{c['p']:.2e}")}
        if "slope_se" in c:
            out["slope_se"] = fmt(c["slope_se"])
            out["n_cells"] = c["n"]
        return out

    return {
        "source": str(IJCS_STATS.relative_to(ROOT)).replace("\\", "/"),
        "queue_tail_slope_per_ln_lambda": {
            "RoBERTa_GoEmotions_seeds42-51": slope("slope_GoEmotions_seeds 42–51"),
            "RoBERTa_SemEval_seeds42-51": slope("slope_SemEval_seeds 42–51"),
            "BERT_GoEmotions_seeds42-46": slope("slope_bert_GoEmotions"),
            "BERT_SemEval_seeds42-46": slope("slope_bert_SemEval"),
        },
        "queue_size_slope": {
            "GoEmotions": slope("slope_qsize_goemo"),
            "SemEval": slope("slope_qsize_semeval"),
        },
        "compensation_recovery_on_damage": {
            "pooled_shared_seeds": comp("compensation_RoBERTa_and_BERT_shared_seeds"),
            "pooled_disjoint_seeds": comp("compensation_RoBERTa_and_BERT_split_seeds"),
            "RoBERTa_shared_seeds": comp("compensation_shared_seeds"),
            "BERT_shared_seeds": comp("compensation_BERT_shared_seeds"),
        },
    }


def main() -> None:
    if not KEY_RESULTS.exists():
        print(f"Missing {KEY_RESULTS}", file=sys.stderr)
        sys.exit(1)

    key = json.loads(KEY_RESULTS.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}

    go = key.get("goemotions", {})
    five_seed = {}
    for model, metrics in go.items():
        five_seed[model] = {
            "macro_f1": mean_std_block(metrics, "macro_f1"),
            "tail": mean_std_block(metrics, "tail"),
            "micro_f1": mean_std_block(metrics, "micro_f1"),
        }

    dev_grid = {}
    for model_key, lam in LAMBDA_DEV_KEYS:
        if model_key not in summary:
            continue
        dev = best_dev_epoch(summary[model_key]["history"])
        dev_grid[lam] = {
            "model_key": model_key,
            "dev_macro_f1": fmt(dev.get("macro_f1")),
            "dev_tail_f1": fmt(dev.get("tail")),
            "dev_micro_f1": fmt(dev.get("micro_f1")),
        }

    paired = {}
    for name, fields in key.get("paired_deltas", {}).items():
        paired[name] = {
            "macro_delta": mean_std_block(fields, "macro_f1"),
            "tail_delta": mean_std_block(fields, "tail"),
        }

    focal = go.get("baseline_focal", {}).get("macro_f1", {})
    bce = go.get("baseline", {}).get("macro_f1", {})
    manifest = {
        "source": "outputs/key_results.json (five-seed test means) + experiment_summary.json (dev grid)",
        "hardware": key.get("hardware"),
        "generated_from": str(KEY_RESULTS.relative_to(ROOT)).replace("\\", "/"),
        "protocol_notes": {
            "test_metrics": "mean ± SD over seeds 42–46 on GoEmotions official test split",
            "dev_lambda_grid": "development Macro-F1 from experiment_summary history; peaks at λ₁=0.02 and 0.05 (0.514 each on RTX PRO 6000 re-run)",
            "isolation_lambda": (
                "0.10, locked from the initial three-point dev scan {0.10, 0.15, 0.20} before drop-one "
                "testing; 0.02 and 0.05 were added to the dev grid afterwards, so contrasts at other "
                "weights are post-hoc sensitivity checks"
            ),
            "cb_focal_minus_bce_macro_delta_mean": fmt(
                (focal.get("mean") or 0) - (bce.get("mean") or 0), 4
            ),
        },
        "five_seed_test_means": five_seed,
        "dev_lambda_grid": dev_grid,
        "paired_deltas": paired,
        "paper_table_values": {
            "roberta_bce_macro_f1": five_seed.get("baseline", {}).get("macro_f1"),
            "roberta_cb_focal_macro_f1": five_seed.get("baseline_focal", {}).get("macro_f1"),
            "cb_focal_minus_bce_macro_mean": fmt(
                (focal.get("mean") or 0) - (bce.get("mean") or 0), 4
            ),
            "hard_supcon_macro_f1": five_seed.get("hard_supcon", {}).get("macro_f1"),
            "clfer_lt_l010_macro_f1": five_seed.get("clfer_lt_l010", {}).get("macro_f1"),
        },
        "ijcs_revision": ijcs_revision_summary(),
    }

    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        "CB-Focal minus BCE Macro-F1 (five-seed means):",
        manifest["paper_table_values"]["cb_focal_minus_bce_macro_mean"],
    )


if __name__ == "__main__":
    main()
