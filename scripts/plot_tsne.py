#!/usr/bin/env python3
"""Fig.5: t-SNE visualization of embeddings (baseline vs CLFER-LT)."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import GoEmotionsDataset
from src.data.preprocess import load_emotion_labels
from src.models.clfer_lt import BaselineModel, CLFERLT
from src.training.trainer import build_class_weights


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=800):
    model.eval()
    embs, label_ids = [], []
    count = 0
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        if hasattr(model, "encoder"):
            pooled = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        else:
            pooled = model._get_pooled(ids, mask)
        embs.append(pooled.cpu().numpy())
        primary = labels.argmax(dim=1).numpy()
        label_ids.append(primary)
        count += ids.size(0)
        if count >= max_samples:
            break
    return np.vstack(embs)[:max_samples], np.concatenate(label_ids)[:max_samples]


def plot_tsne(emb2d, label_ids, label_names, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        emb2d[:, 0], emb2d[:, 1],
        c=label_ids, cmap="tab20", s=12, alpha=0.7
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.colorbar(scatter, ax=ax, label="Primary emotion id")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "goemotions"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "figures"))
    parser.add_argument("--backbone", type=str, default="roberta-base")
    parser.add_argument("--baseline_ckpt", type=str, default="")
    parser.add_argument("--clfer_ckpt", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=800)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_emotion_labels(args.data_dir)
    weights = build_class_weights(args.data_dir, 0.9999).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    test_ds = GoEmotionsDataset(args.data_dir, "test", tokenizer, max_samples=args.max_samples)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    models_to_plot = []
    ckpt_base = Path(args.baseline_ckpt) if args.baseline_ckpt else ROOT / "outputs" / "checkpoints" / "baseline" / "best_model.pt"
    ckpt_clfer = Path(args.clfer_ckpt) if args.clfer_ckpt else ROOT / "outputs" / "checkpoints" / "clfer_lt_l010" / "best_model.pt"

    if ckpt_base.exists():
        m = BaselineModel(args.backbone, len(labels), weights, "bce").to(device)
        m.load_state_dict(torch.load(ckpt_base, map_location=device, weights_only=True))
        models_to_plot.append(("Baseline", m, "fig5_tsne_baseline.png"))
    if ckpt_clfer.exists():
        m = CLFERLT(args.backbone, len(labels), weights).to(device)
        m.load_state_dict(torch.load(ckpt_clfer, map_location=device, weights_only=True))
        models_to_plot.append(("CLFER-LT", m, "fig5_tsne_clfer_lt.png"))

    if not models_to_plot:
        print("No checkpoints found. Train models first.")
        return

    for name, model, fname in models_to_plot:
        embs, label_ids = extract_embeddings(model, loader, device, args.max_samples)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb2d = tsne.fit_transform(embs)
        plot_tsne(emb2d, label_ids, labels, f"t-SNE: {name}", output_dir / fname)
        print(f"Saved {output_dir / fname}")


if __name__ == "__main__":
    main()
