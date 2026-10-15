"""Training and evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import GoEmotionsDataset
from src.data.preprocess import (
    compute_label_statistics,
    effective_number_weights,
    load_emotion_labels,
    load_split,
)
from src.models.clfer_lt import BaselineModel, CLFERLT, HardSupConModel, ModelOutput
from src.utils.device import resolve_device
from src.utils.hf_env import configure_hf_mirror, pretrained_kwargs
from src.utils.json_util import to_jsonable
from src.utils.metrics import (
    compute_group_macro_f1,
    compute_multilabel_metrics,
    per_class_f1,
)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_class_weights(data_dir: str, beta: float) -> torch.Tensor:
    labels = load_emotion_labels(data_dir)
    train_samples = load_split(data_dir, "train")
    counts = np.zeros(len(labels), dtype=np.float64)
    for sample in train_samples:
        counts += sample["labels"]
    weights = effective_number_weights(counts, beta=beta)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    group_indices: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output: ModelOutput = model(input_ids, attention_mask)
        probs = torch.sigmoid(output.logits).cpu().numpy()
        preds = (probs >= threshold).astype(np.int32)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    metrics = compute_multilabel_metrics(y_true, y_pred, y_prob)
    if group_indices:
        metrics.update(compute_group_macro_f1(y_true, y_pred, group_indices))
    return metrics


def _build_model(
    model_type: str,
    model_cfg: dict,
    data_cfg: dict,
    loss_cfg: dict,
    class_weights: torch.Tensor,
    hf_kwargs: dict | None = None,
) -> torch.nn.Module:
    hf_kwargs = hf_kwargs or {}
    common = {
        "backbone": model_cfg["backbone"],
        "num_labels": data_cfg["num_labels"],
        "class_weights": class_weights,
        "hf_kwargs": hf_kwargs,
    }
    if model_type == "baseline":
        return BaselineModel(**common, loss_type="bce")
    if model_type == "baseline_focal":
        return BaselineModel(
            **common,
            loss_type="focal",
            focal_gamma=loss_cfg["focal_gamma"],
        )
    if model_type == "hard_supcon":
        return HardSupConModel(
            **common,
            loss_type="bce",
            lambda_supcon=loss_cfg.get("lambda_supcon", 0.5),
            temperature=loss_cfg["temperature"],
        )
    if model_type == "hard_supcon_focal":
        return HardSupConModel(
            **common,
            loss_type="focal",
            lambda_supcon=loss_cfg.get("lambda_supcon", 0.5),
            temperature=loss_cfg["temperature"],
            focal_gamma=loss_cfg["focal_gamma"],
        )
    if model_type == "clfer_lt_hard_match":
        return CLFERLT(
            **common,
            projection_dim=model_cfg["projection_dim"],
            queue_size=model_cfg["queue_size"],
            lambda_supcon=loss_cfg["lambda_supcon"],
            lambda_proto=loss_cfg["lambda_proto"],
            temperature=loss_cfg["temperature"],
            proto_temperature=loss_cfg["proto_temperature"],
            focal_gamma=loss_cfg["focal_gamma"],
            supcon_mode="hard",
            enable_supcon=True,
            enable_proto=True,
            use_memory_queue=True,
        )
    if model_type.startswith("clfer_lt"):
        ablation = {
            "clfer_lt": dict(enable_supcon=True, enable_proto=True, use_memory_queue=True),
            "clfer_lt_no_supcon": dict(enable_supcon=False, enable_proto=True, use_memory_queue=True),
            "clfer_lt_no_proto": dict(enable_supcon=True, enable_proto=False, use_memory_queue=True),
            "clfer_lt_no_queue": dict(enable_supcon=True, enable_proto=True, use_memory_queue=False),
        }
        flags = ablation.get(model_type, ablation["clfer_lt"])
        return CLFERLT(
            **common,
            projection_dim=model_cfg["projection_dim"],
            queue_size=model_cfg["queue_size"],
            lambda_supcon=loss_cfg["lambda_supcon"],
            lambda_proto=loss_cfg["lambda_proto"],
            temperature=loss_cfg["temperature"],
            proto_temperature=loss_cfg["proto_temperature"],
            focal_gamma=loss_cfg["focal_gamma"],
            **flags,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def train(
    config: dict[str, Any],
    model_type: str = "clfer_lt",
) -> dict[str, Any]:
    set_seed(config["seed"])
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    eval_cfg = config["evaluation"]
    device = resolve_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    data_dir = data_cfg["data_dir"]
    labels = load_emotion_labels(data_dir)
    stats = compute_label_statistics(load_split(data_dir, "train"), labels)
    group_indices = stats["group_indices"]
    class_weights = build_class_weights(data_dir, loss_cfg["cb_beta"]).to(device)

    configure_hf_mirror()
    hf_kw = pretrained_kwargs(model_cfg)

    max_train = data_cfg.get("max_train_samples")
    max_eval = data_cfg.get("max_eval_samples")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"], **hf_kw)
    train_ds = GoEmotionsDataset(
        data_dir, "train", tokenizer, data_cfg["max_length"], max_samples=max_train
    )
    dev_ds = GoEmotionsDataset(
        data_dir, "dev", tokenizer, data_cfg["max_length"], max_samples=max_eval
    )
    test_ds = GoEmotionsDataset(
        data_dir, "test", tokenizer, data_cfg["max_length"], max_samples=max_eval
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    use_cuda = device.type == "cuda"
    loader_extra = {}
    if num_workers > 0:
        loader_extra["persistent_workers"] = True
        loader_extra["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **loader_extra,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **loader_extra,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **loader_extra,
    )

    model = _build_model(model_type, model_cfg, data_cfg, loss_cfg, class_weights, hf_kw)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    run_name = train_cfg.get("experiment_name", model_type)
    save_dir = Path(train_cfg["save_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    best_dev_f1 = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=use_cuda)
            attention_mask = batch["attention_mask"].to(device, non_blocking=use_cuda)
            label_tensors = batch["labels"].to(device, non_blocking=use_cuda)

            output: ModelOutput = model(input_ids, attention_mask, label_tensors)
            loss = output.loss
            assert loss is not None
            (loss / grad_accum).backward()

            if step % grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        dev_metrics = evaluate_model(
            model, dev_loader, device, eval_cfg["threshold"], group_indices
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss / len(train_loader)),
            **{k: float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v for k, v in dev_metrics.items()},
        }
        history.append(to_jsonable(record))
        print(f"Dev Macro-F1: {dev_metrics['macro_f1']:.4f}, Micro-F1: {dev_metrics['micro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    try:
        state = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    test_metrics = evaluate_model(
        model, test_loader, device, eval_cfg["threshold"], group_indices
    )
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}, Micro-F1: {test_metrics['micro_f1']:.4f}")

    # Per-class F1 for tail analysis
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.sigmoid(out.logits).cpu().numpy()
            all_preds.append((probs >= eval_cfg["threshold"]).astype(np.int32))
            all_labels.append(batch["labels"].numpy())
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    label_names = labels
    class_f1 = per_class_f1(y_true, y_pred)
    per_class = {label_names[i]: float(class_f1[i]) for i in range(len(label_names))}

    return to_jsonable(
        {
            "history": history,
            "test_metrics": test_metrics,
            "per_class_f1": per_class,
            "group_indices": group_indices,
            "labels": labels,
            "stats": stats,
            "config_snapshot": {
                "backbone": model_cfg["backbone"],
                "max_train_samples": max_train,
                "epochs": train_cfg["epochs"],
            },
        }
    )
