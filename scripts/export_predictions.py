#!/usr/bin/env python3
"""Export test-set predictions and per-class F1 from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GoEmotionsDataset
from src.data.preprocess import compute_label_statistics, load_emotion_labels, load_split
from src.training.trainer import _build_model, build_class_weights, set_seed
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.hf_env import configure_hf_mirror, pretrained_kwargs
from src.utils.metrics import compute_multilabel_metrics, compute_group_macro_f1, per_class_f1

configure_hf_mirror()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model_type", required=True, help="baseline | baseline_focal | clfer_lt | ...")
    parser.add_argument("--experiment_name", default="", help="Override checkpoint subdir name")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--out_dir", default="", help="Override prediction output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["training"].get("device", "auto"))
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    eval_cfg = config["evaluation"]
    hf_kw = pretrained_kwargs(model_cfg)

    run_name = args.experiment_name or args.model_type
    save_dir = Path(config["training"]["save_dir"]) / run_name
    ckpt = save_dir / "best_model.pt"
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")

    data_dir = data_cfg["data_dir"]
    label_names = load_emotion_labels(data_dir)
    stats = compute_label_statistics(load_split(data_dir, "train"), label_names)
    class_weights = build_class_weights(data_dir, loss_cfg["cb_beta"]).to(device)
    model = _build_model(args.model_type, model_cfg, data_cfg, loss_cfg, class_weights, hf_kw)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"], **hf_kw)
    ds = GoEmotionsDataset(data_dir, args.split, tokenizer, data_cfg["max_length"])
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            out = model(ids, mask)
            probs = torch.sigmoid(out.logits).cpu().numpy()
            preds = (probs >= eval_cfg["threshold"]).astype(np.int32)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    metrics = compute_multilabel_metrics(y_true, y_pred, y_prob)
    group = compute_group_macro_f1(y_true, y_pred, stats["group_indices"])
    metrics.update(group)
    class_f1 = per_class_f1(y_true, y_pred)
    per_class = {label_names[i]: float(class_f1[i]) for i in range(len(label_names))}

    pred_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_npz = pred_dir / f"{run_name}_{args.split}.npz"
    np.savez(out_npz, y_true=y_true, y_pred=y_pred, y_prob=y_prob, labels=np.array(label_names))

    summary = {
        f"{args.split}_metrics": metrics,
        "per_class_f1": per_class,
        "checkpoint": str(ckpt),
        "split": args.split,
    }
    out_json = pred_dir / f"{run_name}_{args.split}_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    results_path = save_dir / "results.json"
    if results_path.exists() and args.split == "test":
        results = json.loads(results_path.read_text(encoding="utf-8"))
        results["per_class_f1"] = per_class
        results["test_metrics"] = metrics
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"npz": str(out_npz), "metrics": metrics, "per_class_count": len(per_class)}, indent=2))


if __name__ == "__main__":
    main()
