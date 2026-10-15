#!/usr/bin/env python3
"""Estimate Jaccard vs exact-match positive-pair overlap on GoEmotions training batches."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import compute_label_statistics, load_emotion_labels, load_split  # noqa: E402


def jaccard_pos(y_i: np.ndarray, y_j: np.ndarray) -> bool:
    return float((y_i * y_j).sum()) > 0


def exact_match(y_i: np.ndarray, y_j: np.ndarray) -> bool:
    return bool(np.array_equal(y_i, y_j))


def batch_stats(labels: np.ndarray) -> dict[str, int]:
    b = labels.shape[0]
    jaccard_pairs = 0
    exact_pairs = 0
    partial_pairs = 0
    for i in range(b):
        for j in range(b):
            if i == j:
                continue
            if jaccard_pos(labels[i], labels[j]):
                jaccard_pairs += 1
                if exact_match(labels[i], labels[j]):
                    exact_pairs += 1
                else:
                    partial_pairs += 1
    return {
        "jaccard_pairs": jaccard_pairs,
        "exact_pairs": exact_pairs,
        "partial_pairs": partial_pairs,
    }


def main() -> None:
    data_dir = ROOT / "data" / "goemotions"
    samples = load_split(data_dir, "train")
    label_names = load_emotion_labels(data_dir)
    stats = compute_label_statistics(samples, label_names)
    Y = np.stack([s["labels"] for s in samples])
    n = len(samples)
    batch_size = 32
    rng = random.Random(42)
    totals = {"jaccard_pairs": 0, "exact_pairs": 0, "partial_pairs": 0}
    n_batches = 5000
    for _ in range(n_batches):
        idx = rng.sample(range(n), batch_size)
        batch = Y[idx]
        bs = batch_stats(batch)
        for k in totals:
            totals[k] += bs[k]

    j = totals["jaccard_pairs"]
    e = totals["exact_pairs"]
    p = totals["partial_pairs"]
    overlap = e / j if j else 0.0
    partial_frac = p / j if j else 0.0

    # Per-sample: at least one other sample in batch with shared label (dropout view always has self)
    multilabel = stats["multilabel_ratio"]

    out = {
        "train_samples": n,
        "multilabel_ratio": multilabel,
        "simulated_batches": n_batches,
        "batch_size": batch_size,
        "directed_positive_pairs_per_batch_mean": {
            "jaccard": j / n_batches,
            "exact_match": e / n_batches,
            "partial_only": p / n_batches,
        },
        "among_jaccard_positive_pairs": {
            "exact_match_fraction": overlap,
            "partial_only_fraction": partial_frac,
        },
    }
    out_path = ROOT / "outputs" / "positive_pair_overlap.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
