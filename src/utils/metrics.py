"""Evaluation metrics for multi-label emotion classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard multi-label classification metrics."""
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    if y_prob is not None:
        try:
            metrics["mAP"] = float(average_precision_score(y_true, y_prob, average="macro"))
        except ValueError:
            metrics["mAP"] = 0.0
    return metrics


def compute_group_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_indices: dict[str, list[int]],
) -> dict[str, float]:
    """Compute macro-F1 restricted to label groups (head/medium/tail)."""
    results: dict[str, float] = {}
    for group_name, indices in group_indices.items():
        if not indices:
            results[group_name] = 0.0
            continue
        yt = y_true[:, indices]
        yp = y_pred[:, indices]
        results[group_name] = float(f1_score(yt, yp, average="macro", zero_division=0))
    return results


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return f1_score(y_true, y_pred, average=None, zero_division=0)
