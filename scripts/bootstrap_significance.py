#!/usr/bin/env python3
"""Paired bootstrap significance test on exported test predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.metrics import compute_multilabel_metrics


def bootstrap_macro_diff(y_true, pred_a, pred_b, n: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap for Macro-F1 difference: model_b minus model_a (B - A)."""
    rng = np.random.default_rng(seed)
    n_samples = y_true.shape[0]
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        m_a = compute_multilabel_metrics(y_true[idx], pred_a[idx])["macro_f1"]
        m_b = compute_multilabel_metrics(y_true[idx], pred_b[idx])["macro_f1"]
        diffs.append(m_b - m_a)
    diffs = np.array(diffs)
    point_a = compute_multilabel_metrics(y_true, pred_a)["macro_f1"]
    point_b = compute_multilabel_metrics(y_true, pred_b)["macro_f1"]
    point_diff = point_b - point_a
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value_two_sided": float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
        "point_macro_a": float(point_a),
        "point_macro_b": float(point_b),
        "point_diff": float(point_diff),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="First .npz (predictions)")
    parser.add_argument("--b", required=True, help="Second .npz")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "bootstrap_results.json"))
    args = parser.parse_args()

    da = np.load(args.a)
    db = np.load(args.b)
    y_true = da["y_true"]
    assert np.array_equal(y_true, db["y_true"]), "y_true mismatch between files"

    stats = bootstrap_macro_diff(y_true, da["y_pred"], db["y_pred"], n=args.n)
    stats["model_a"] = args.a
    stats["model_b"] = args.b
    stats["method_a"] = Path(args.a).stem.replace("_test", "")
    stats["method_b"] = Path(args.b).stem.replace("_test", "")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
