"""HuggingFace mirror and cache configuration for restricted networks."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import project_root


def hf_cache_dir() -> Path:
    env = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if env:
        return Path(env)
    return project_root() / "models" / "hf_cache"


def configure_hf_mirror() -> Path:
    """Use hf-mirror.com when HuggingFace is unreachable (common on CN cloud GPUs)."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cache = hf_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    return cache


def pretrained_kwargs(model_cfg: dict) -> dict:
    cache = model_cfg.get("cache_dir") or str(configure_hf_mirror())
    kwargs: dict = {"cache_dir": cache}
    if model_cfg.get("local_files_only"):
        kwargs["local_files_only"] = True
    return kwargs
