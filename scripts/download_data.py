#!/usr/bin/env python3
"""Download GoEmotions dataset."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import download_goemotions


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GoEmotions dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(ROOT / "data" / "goemotions"),
        help="Directory to save dataset",
    )
    args = parser.parse_args()
    download_goemotions(args.data_dir)
    print(f"Dataset saved to {args.data_dir}")


if __name__ == "__main__":
    main()
