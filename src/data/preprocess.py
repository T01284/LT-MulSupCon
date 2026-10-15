"""GoEmotions data loading and preprocessing."""

from __future__ import annotations

import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
)

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

DOWNLOAD_URLS = {
    "emotions.txt": f"{GITHUB_RAW_BASE}/emotions.txt",
    "train.tsv": f"{GITHUB_RAW_BASE}/train.tsv",
    "dev.tsv": f"{GITHUB_RAW_BASE}/dev.tsv",
    "test.tsv": f"{GITHUB_RAW_BASE}/test.tsv",
}


def _download_file(url: str, target: Path) -> None:
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, target)


def _download_via_huggingface(data_dir: Path) -> bool:
    """Fallback: export splits from HuggingFace datasets hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        return False

    print("[download] Using HuggingFace datasets: google-research-datasets/go_emotions")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    label_names = ds["train"].features["labels"].feature.names
    emotions_file = data_dir / "emotions.txt"
    with open(emotions_file, "w", encoding="utf-8") as f:
        f.write("\n".join(label_names) + "\n")

    id_to_name = {i: name for i, name in enumerate(label_names)}
    for split_name, hf_split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        out_path = data_dir / f"{split_name}.tsv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in ds[hf_split]:
                active_ids = [str(i) for i, v in enumerate(row["labels"]) if v == 1]
                writer.writerow([row["text"], ",".join(active_ids), str(row.get("id", ""))])
        print(f"[saved] {out_path} ({len(ds[hf_split])} samples)")
    return True


def download_goemotions(data_dir: str | Path) -> Path:
    """Download official GoEmotions train/dev/test splits."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_exist = all((data_dir / f).exists() for f in DOWNLOAD_URLS)
    if all_exist:
        print(f"[skip] All files exist in {data_dir}")
        return data_dir

    try:
        for filename, url in DOWNLOAD_URLS.items():
            target = data_dir / filename
            if target.exists():
                print(f"[skip] {target} already exists")
                continue
            _download_file(url, target)
    except Exception as exc:
        print(f"[warn] Direct download failed: {exc}")
        if not _download_via_huggingface(data_dir):
            _write_builtin_labels(data_dir)
            raise RuntimeError(
                "Failed to download GoEmotions. Install `datasets` and retry, "
                "or check network access."
            ) from exc

    if not (data_dir / "emotions.txt").exists():
        _write_builtin_labels(data_dir)

    return data_dir


def _write_builtin_labels(data_dir: Path) -> None:
    emotions_file = data_dir / "emotions.txt"
    with open(emotions_file, "w", encoding="utf-8") as f:
        f.write("\n".join(EMOTION_LABELS) + "\n")


def load_emotion_labels(data_dir: str | Path) -> list[str]:
    emotions_file = Path(data_dir) / "emotions.txt"
    if not emotions_file.exists():
        return EMOTION_LABELS.copy()
    with open(emotions_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels or EMOTION_LABELS.copy()


def parse_tsv_row(row: list[str], num_labels: int) -> tuple[str, np.ndarray, str]:
    text, label_ids_str, example_id = row[0], row[1], row[2]
    label_vec = np.zeros(num_labels, dtype=np.float32)
    if label_ids_str.strip():
        for idx in label_ids_str.split(","):
            label_vec[int(idx.strip())] = 1.0
    return text, label_vec, example_id


def load_split(data_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Load one split (train/dev/test) into list of dicts."""
    data_dir = Path(data_dir)
    labels = load_emotion_labels(data_dir)
    num_labels = len(labels)
    split_file = data_dir / f"{split}.tsv"

    samples: list[dict[str, Any]] = []
    with open(split_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            text, label_vec, example_id = parse_tsv_row(row, num_labels)
            active = [labels[i] for i, v in enumerate(label_vec) if v > 0]
            samples.append(
                {
                    "text": text,
                    "labels": label_vec,
                    "id": example_id,
                    "active_labels": active,
                    "num_active": int(label_vec.sum()),
                }
            )
    return samples


def compute_label_statistics(
    samples: list[dict[str, Any]], labels: list[str]
) -> dict[str, Any]:
    """Compute per-label counts and long-tail groups."""
    counts = Counter()
    for sample in samples:
        for i, label in enumerate(labels):
            if sample["labels"][i] > 0:
                counts[label] += 1

    sorted_labels = sorted(labels, key=lambda x: counts[x], reverse=True)
    n = len(sorted_labels)
    third = n // 3
    groups = {
        "head": sorted_labels[:third],
        "medium": sorted_labels[third : 2 * third],
        "tail": sorted_labels[2 * third :],
    }
    group_indices = {
        g: [labels.index(lbl) for lbl in group_labels] for g, group_labels in groups.items()
    }

    return {
        "label_counts": dict(counts),
        "sorted_labels": sorted_labels,
        "groups": groups,
        "group_indices": group_indices,
        "total_samples": len(samples),
        "multilabel_ratio": float(np.mean([s["num_active"] > 1 for s in samples])),
    }


def compute_cooccurrence_matrix(
    samples: list[dict[str, Any]], num_labels: int
) -> np.ndarray:
    cooc = np.zeros((num_labels, num_labels), dtype=np.float64)
    for sample in samples:
        active = np.where(sample["labels"] > 0)[0]
        for i in active:
            for j in active:
                cooc[i, j] += 1
    row_sums = cooc.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cooc / row_sums


def effective_number_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """Class-balanced weights from Cui et al., CVPR 2019."""
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(weights)
    return weights.astype(np.float32)


def save_statistics(stats: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: v for k, v in stats.items() if k != "group_indices"
    }
    serializable["groups"] = stats["groups"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
