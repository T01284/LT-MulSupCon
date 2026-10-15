#!/usr/bin/env python3
"""Re-evaluate a saved checkpoint and print group metrics."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.hf_env import configure_hf_mirror
configure_hf_mirror()

from src.training.trainer import _build_model, build_class_weights, evaluate_model, set_seed
from src.utils.config import load_config
from src.utils.hf_env import pretrained_kwargs
from src.utils.device import resolve_device
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.dataset import GoEmotionsDataset
from src.data.preprocess import compute_label_statistics, load_emotion_labels, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model_type", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["training"].get("device", "auto"))
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    eval_cfg = config["evaluation"]
    hf_kw = pretrained_kwargs(model_cfg)

    data_dir = data_cfg["data_dir"]
    stats = compute_label_statistics(load_split(data_dir, "train"), load_emotion_labels(data_dir))
    class_weights = build_class_weights(data_dir, loss_cfg["cb_beta"]).to(device)
    model = _build_model(args.model_type, model_cfg, data_cfg, loss_cfg, class_weights, hf_kw)
    ckpt = Path(config["training"]["save_dir"]) / args.model_type / "best_model.pt"
    model.load_state_dict(__import__("torch").load(ckpt, map_location=device, weights_only=True))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"], **hf_kw)
    test_ds = GoEmotionsDataset(data_dir, "test", tokenizer, data_cfg["max_length"])
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)
    metrics = evaluate_model(model, test_loader, device, eval_cfg["threshold"], stats["group_indices"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
