"""PyTorch Dataset for GoEmotions."""

from __future__ import annotations

import numpy as np
import torch
from typing import Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.data.preprocess import load_split


class GoEmotionsDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 64,
        max_samples: int | None = None,
    ):
        self.samples = load_split(data_dir, split)
        if max_samples is not None and max_samples < len(self.samples):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
        self.max_length = max_length
        encoded = tokenizer(
            [sample["text"] for sample in self.samples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self._enc = {key: value for key, value in encoded.items()}
        self._labels = torch.tensor(
            [sample["labels"] for sample in self.samples], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: value[idx] for key, value in self._enc.items()}
        item["labels"] = self._labels[idx]
        item["id"] = self.samples[idx]["id"]
        return item
