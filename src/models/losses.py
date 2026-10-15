"""Loss functions for CLFER-LT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def jaccard_similarity(labels: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Jaccard similarity for multi-hot label vectors."""
    intersection = labels @ labels.t()
    label_sum = labels.sum(dim=1, keepdim=True)
    union = label_sum + label_sum.t() - intersection
    return intersection / union.clamp(min=1e-8)


class ClassBalancedFocalLoss(nn.Module):
    """Multi-label class-balanced focal loss."""

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.gamma) * bce
        weighted = focal * self.class_weights.unsqueeze(0)
        return weighted.mean()


class HardSupConLoss(nn.Module):
    """Single-label-style SupCon: positives only for exactly matching multi-hot labels."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        memory_features: torch.Tensor | None = None,
        memory_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        sim = features @ features.t() / self.temperature
        exact = (labels.unsqueeze(1) == labels.unsqueeze(0)).all(dim=2).float()

        if memory_features is not None and memory_labels is not None:
            mem_norm = F.normalize(memory_features, dim=1)
            cross_sim = features @ mem_norm.t() / self.temperature
            cross_exact = (labels.unsqueeze(1) == memory_labels.unsqueeze(0)).all(dim=2).float()
            sim = torch.cat([sim, cross_sim], dim=1)
            exact = torch.cat([exact, cross_exact], dim=1)

        batch_size = labels.size(0)
        self_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
        pos_mask = exact > 0
        pos_mask[:, :batch_size] = pos_mask[:, :batch_size] & ~self_mask

        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = pos_mask.float().sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return features.new_tensor(0.0)

        loss = -(pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp(min=1e-8)
        return loss[valid].mean()


class LTMultiLabelSupConLoss(nn.Module):
    """Long-tail aware multi-label supervised contrastive loss with Jaccard weights."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        memory_features: torch.Tensor | None = None,
        memory_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        sim = features @ features.t() / self.temperature
        label_sim = jaccard_similarity(labels)

        if memory_features is not None and memory_labels is not None:
            mem_norm = F.normalize(memory_features, dim=1)
            cross_sim = features @ mem_norm.t() / self.temperature
            cross_label_sim = jaccard_similarity(
                torch.cat([labels, memory_labels], dim=0)
            )[: labels.size(0), labels.size(0) :]
            sim = torch.cat([sim, cross_sim], dim=1)
            label_sim = torch.cat([label_sim, cross_label_sim], dim=1)

        batch_size = labels.size(0)
        self_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
        pos_mask = label_sim > 0
        pos_mask[:, :batch_size] = pos_mask[:, :batch_size] & ~self_mask

        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_weights = label_sim * pos_mask.float()
        pos_count = pos_weights.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return features.new_tensor(0.0)

        loss = -(pos_weights * log_prob).sum(dim=1) / pos_count.clamp(min=1e-8)
        return loss[valid].mean()


class LabelPrototypeContrastLoss(nn.Module):
    """Contrastive loss between sample embeddings and learnable label prototypes."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        logits = features @ prototypes.t() / self.temperature

        losses = []
        for i in range(labels.size(0)):
            pos_indices = torch.where(labels[i] > 0)[0]
            if len(pos_indices) == 0:
                continue
            sample_logits = logits[i]
            log_probs = F.log_softmax(sample_logits, dim=0)
            losses.append(-log_probs[pos_indices].mean())

        if not losses:
            return features.new_tensor(0.0)
        return torch.stack(losses).mean()
