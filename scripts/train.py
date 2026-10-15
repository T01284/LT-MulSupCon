#!/usr/bin/env python3
"""Train CLFER-LT or baseline models."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.hf_env import configure_hf_mirror
configure_hf_mirror()

from src.training.trainer import train
from src.utils.config import load_config
from src.utils.json_util import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="clfer_lt",
        choices=[
            "clfer_lt",
            "clfer_lt_no_supcon",
            "clfer_lt_no_proto",
            "clfer_lt_no_queue",
            "baseline",
            "baseline_focal",
            "hard_supcon",
            "hard_supcon_focal",
            "clfer_lt_hard_match",
        ],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None, help="Checkpoint subfolder name")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.experiment_name:
        config["training"]["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config["seed"] = args.seed

    results = train(config, model_type=args.model_type)

    run_name = config["training"].get("experiment_name", args.model_type)
    output_dir = Path(config["training"]["save_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(
        {
            "test_metrics": results["test_metrics"],
            "per_class_f1": results.get("per_class_f1", {}),
            "history": results["history"],
            "config_snapshot": results.get("config_snapshot", {}),
        }
    )
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
