#!/usr/bin/env python3
"""Pre-download HuggingFace models via mirror (for cloud servers without huggingface.co access)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoModel, AutoTokenizer

from src.utils.hf_env import configure_hf_mirror, pretrained_kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--also", nargs="*", default=[], help="Extra models, e.g. distilroberta-base")
    args = parser.parse_args()

    cache = configure_hf_mirror()
    print(f"HF_ENDPOINT={__import__('os').environ.get('HF_ENDPOINT')}")
    print(f"cache_dir={cache}")

    for name in [args.model, *args.also]:
        print(f"\nDownloading {name} ...")
        model_cfg = {"cache_dir": str(cache)}
        kw = pretrained_kwargs(model_cfg)
        AutoTokenizer.from_pretrained(name, **kw)
        AutoModel.from_pretrained(name, **kw)
        print(f"OK: {name}")

    print("\nAll models cached. Set model.local_files_only: true in yaml for offline training.")


if __name__ == "__main__":
    main()
