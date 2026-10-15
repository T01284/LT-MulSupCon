"""CLFER-LT model architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from src.models.losses import (
    ClassBalancedFocalLoss,
    HardSupConLoss,
    LTMultiLabelSupConLoss,
    LabelPrototypeContrastLoss,
)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_dict: dict[str, float] | None = None


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MemoryQueue:
    """FIFO queue for storing historical embeddings (MoCo-style)."""

    def __init__(self, dim: int, queue_size: int, num_labels: int, device: torch.device):
        self.queue_size = queue_size
        self.ptr = 0
        self.full = False
        self.features = torch.zeros(queue_size, dim, device=device)
        self.labels = torch.zeros(queue_size, num_labels, device=device)

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        batch = features.size(0)
        if batch >= self.queue_size:
            features = features[-self.queue_size :]
            labels = labels[-self.queue_size :]
            batch = self.queue_size

        end = self.ptr + batch
        if end <= self.queue_size:
            self.features[self.ptr : end] = features
            self.labels[self.ptr : end] = labels
        else:
            first = self.queue_size - self.ptr
            self.features[self.ptr :] = features[:first]
            self.labels[self.ptr :] = labels[:first]
            remain = batch - first
            self.features[:remain] = features[first:]
            self.labels[:remain] = labels[first:]
            self.full = True

        self.ptr = (self.ptr + batch) % self.queue_size

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.queue_size if self.full else self.ptr
        return self.features[:size].clone(), self.labels[:size].clone()


class CLFERLT(nn.Module):
    """Contrastive Learning Enhanced Fine-grained Emotion Recognition under Long-Tail."""

    def __init__(
        self,
        backbone: str,
        num_labels: int,
        class_weights: torch.Tensor,
        projection_dim: int = 256,
        queue_size: int = 4096,
        use_memory_queue: bool = True,
        enable_supcon: bool = True,
        enable_proto: bool = True,
        supcon_mode: str = "jaccard",
        lambda_supcon: float = 0.5,
        lambda_proto: float = 0.3,
        temperature: float = 0.07,
        proto_temperature: float = 0.1,
        focal_gamma: float = 2.0,
        hf_kwargs: dict | None = None,
    ):
        super().__init__()
        hf_kwargs = hf_kwargs or {}
        config = AutoConfig.from_pretrained(backbone, **hf_kwargs)
        self.encoder = AutoModel.from_pretrained(backbone, **hf_kwargs)
        hidden_size = config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.projection = ProjectionHead(hidden_size, projection_dim, projection_dim)
        self.prototypes = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)

        self.cls_loss_fn = ClassBalancedFocalLoss(class_weights, gamma=focal_gamma)
        if supcon_mode == "hard":
            self.supcon_loss_fn = HardSupConLoss(temperature=temperature)
        else:
            self.supcon_loss_fn = LTMultiLabelSupConLoss(temperature=temperature)
        self.proto_loss_fn = LabelPrototypeContrastLoss(temperature=proto_temperature)

        self.num_labels = num_labels
        self.lambda_supcon = lambda_supcon
        self.lambda_proto = lambda_proto
        self.use_memory_queue = use_memory_queue
        self.enable_supcon = enable_supcon
        self.enable_proto = enable_proto
        self.queue_size = queue_size
        self._queue: MemoryQueue | None = None

    def _get_pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def _init_queue(self, device: torch.device, dim: int) -> None:
        if self._queue is None:
            self._queue = MemoryQueue(
                dim=dim,
                queue_size=self.queue_size,
                num_labels=self.num_labels,
                device=device,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> ModelOutput:
        pooled = self._get_pooled(input_ids, attention_mask)
        logits = self.classifier(pooled)

        if labels is None:
            return ModelOutput(logits=logits)

        cls_loss = self.cls_loss_fn(logits, labels)

        supcon_loss = pooled.new_tensor(0.0)
        proto_loss = pooled.new_tensor(0.0)

        if self.enable_supcon and self.lambda_supcon > 0:
            z1 = self.projection(pooled)
            z2 = self.projection(pooled)
            proj = F.normalize((z1 + z2) / 2.0, dim=1)

            mem_feats, mem_labels = None, None
            if self.use_memory_queue and self.training:
                self._init_queue(pooled.device, proj.size(1))
                assert self._queue is not None
                mem_feats, mem_labels = self._queue.get()
                self._queue.enqueue(proj.detach(), labels.detach())

            supcon_loss = self.supcon_loss_fn(proj, labels, mem_feats, mem_labels)

        if self.enable_proto and self.lambda_proto > 0:
            proto_loss = self.proto_loss_fn(pooled, labels, self.prototypes)

        total = cls_loss + self.lambda_supcon * supcon_loss + self.lambda_proto * proto_loss

        loss_dict = {
            "total": float(total.item()),
            "cls": float(cls_loss.item()),
            "supcon": float(supcon_loss.item()),
            "proto": float(proto_loss.item()),
        }
        return ModelOutput(logits=logits, loss=total, loss_dict=loss_dict)


class BaselineModel(nn.Module):
    """RoBERTa + BCE/Focal baseline without contrastive learning."""

    def __init__(
        self,
        backbone: str,
        num_labels: int,
        class_weights: torch.Tensor,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        hf_kwargs: dict | None = None,
    ):
        super().__init__()
        hf_kwargs = hf_kwargs or {}
        self.encoder = AutoModel.from_pretrained(backbone, **hf_kwargs)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_type = loss_type
        self.focal_loss = ClassBalancedFocalLoss(class_weights, gamma=focal_gamma)
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        if labels is None:
            return ModelOutput(logits=logits)

        if self.loss_type == "focal":
            loss = self.focal_loss(logits, labels)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        return ModelOutput(logits=logits, loss=loss, loss_dict={"total": float(loss.item())})


class HardSupConModel(nn.Module):
    """RoBERTa + hard SupCon (exact label match) + BCE or CB-Focal; no LPC / queue."""

    def __init__(
        self,
        backbone: str,
        num_labels: int,
        class_weights: torch.Tensor,
        loss_type: str = "bce",
        lambda_supcon: float = 0.5,
        temperature: float = 0.07,
        focal_gamma: float = 2.0,
        hf_kwargs: dict | None = None,
    ):
        super().__init__()
        hf_kwargs = hf_kwargs or {}
        self.encoder = AutoModel.from_pretrained(backbone, **hf_kwargs)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.projection = ProjectionHead(hidden_size, 256, 256)
        self.loss_type = loss_type
        self.lambda_supcon = lambda_supcon
        self.focal_loss = ClassBalancedFocalLoss(class_weights, gamma=focal_gamma)
        self.supcon_loss_fn = HardSupConLoss(temperature=temperature)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        if labels is None:
            return ModelOutput(logits=logits)

        if self.loss_type == "focal":
            cls_loss = self.focal_loss(logits, labels)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        supcon_loss = pooled.new_tensor(0.0)
        if self.lambda_supcon > 0:
            z = F.normalize(self.projection(pooled), dim=1)
            supcon_loss = self.supcon_loss_fn(z, labels)

        total = cls_loss + self.lambda_supcon * supcon_loss
        return ModelOutput(
            logits=logits,
            loss=total,
            loss_dict={
                "total": float(total.item()),
                "cls": float(cls_loss.item()),
                "supcon": float(supcon_loss.item()),
            },
        )
