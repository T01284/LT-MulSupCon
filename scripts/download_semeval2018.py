#!/usr/bin/env python3
"""Download SemEval-2018 Task 1 (E-c) English and export to project TSV format."""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SEMEVAL_LABELS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
]

USER_AGENT = "Mozilla/5.0 (compatible; CLFER-LT/1.0; +https://github.com/)"

ALL_DATA_ZIP = "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"
SPLIT_ZIPS = {
    "train": "http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-train.zip",
    "dev": "http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-dev.zip",
}
GITHUB_RAW = {
    "train": "https://raw.githubusercontent.com/mithunpaul08/twitter_psentiment_multi_label/master/2018-E-c-En-train.txt",
    "dev": "https://raw.githubusercontent.com/mithunpaul08/twitter_psentiment_multi_label/master/2018-E-c-En-dev.txt",
}
ZIP_SPLIT_FILES = {
    "train": "2018-E-c-En-train.txt",
    "dev": "2018-E-c-En-dev.txt",
    "test": "2018-E-c-En-test-gold.txt",
}


def _fetch_url(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _label_is_active(value: str) -> bool:
    token = value.strip().upper()
    if not token or token in {"NONE", "NULL", "NA", "N/A"}:
        return False
    return token in {"1", "TRUE", "T", "YES", "Y"}


def parse_semeval_table(content: str) -> list[tuple[str, list[int], str]]:
    """Parse official SemEval E-c TSV: ID, Tweet, 11 binary emotion columns."""
    rows: list[tuple[str, list[int], str]] = []
    reader = csv.reader(io.StringIO(content), delimiter="\t")
    header = next(reader, None)
    if header is None:
        return rows

    header_lower = [h.strip().lower() for h in header]
    if "tweet" in header_lower:
        text_idx = header_lower.index("tweet")
        id_idx = header_lower.index("id") if "id" in header_lower else None
        label_indices = [header_lower.index(name) for name in SEMEVAL_LABELS]
    elif len(header) >= 13:
        id_idx, text_idx = 0, 1
        label_indices = list(range(2, 13))
    else:
        return rows

    for line_no, fields in enumerate(reader, start=1):
        if len(fields) <= max(label_indices):
            continue
        text = fields[text_idx].strip()
        if not text:
            continue
        ex_id = fields[id_idx].strip() if id_idx is not None and id_idx < len(fields) else str(line_no)
        active = [i for i, col in enumerate(label_indices) if _label_is_active(fields[col])]
        rows.append((text, active, ex_id))
    return rows


def export_rows(rows: list[tuple[str, list[int], str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for text, active, ex_id in rows:
            writer.writerow([text, ",".join(str(i) for i in active), ex_id])


def save_split(rows: list[tuple[str, list[int], str]], data_dir: Path, split: str) -> None:
    fname = "dev.tsv" if split == "dev" else f"{split}.tsv"
    out = data_dir / fname
    export_rows(rows, out)
    print(f"[saved] {out} ({len(rows)} samples)")


def download_from_all_data_zip(data_dir: Path) -> None:
    zip_path = data_dir / "_semeval2018_all.zip"
    print(f"[download] {ALL_DATA_ZIP}")
    zip_path.write_bytes(_fetch_url(ALL_DATA_ZIP))
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for split, fname in ZIP_SPLIT_FILES.items():
            match = next((n for n in names if n.endswith(fname)), None)
            if not match:
                raise FileNotFoundError(f"{fname} not found in all-data zip")
            raw = zf.read(match).decode("utf-8", errors="replace")
            rows = parse_semeval_table(raw)
            if not rows:
                raise ValueError(f"No rows parsed from {fname}")
            save_split(rows, data_dir, split)
    zip_path.unlink(missing_ok=True)


def download_split_zip(url: str) -> str:
    print(f"[download] {url}")
    payload = _fetch_url(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        txt_name = next(n for n in zf.namelist() if n.endswith(".txt"))
        return zf.read(txt_name).decode("utf-8", errors="replace")


def download_from_split_zips(data_dir: Path) -> None:
    for split, url in SPLIT_ZIPS.items():
        raw = download_split_zip(url)
        rows = parse_semeval_table(raw)
        if not rows:
            raise ValueError(f"No rows parsed from {split} zip")
        save_split(rows, data_dir, split)

    print("[download] fetching test-gold from all-data zip ...")
    payload = _fetch_url(ALL_DATA_ZIP)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        match = next((n for n in zf.namelist() if n.endswith(ZIP_SPLIT_FILES["test"])), None)
        if not match:
            raise FileNotFoundError("test-gold file missing in all-data zip")
        raw = zf.read(match).decode("utf-8", errors="replace")
    rows = parse_semeval_table(raw)
    if not rows:
        raise ValueError("No rows parsed from test-gold")
    save_split(rows, data_dir, "test")


def download_from_github(data_dir: Path) -> None:
    for split, url in GITHUB_RAW.items():
        print(f"[download] {url}")
        raw = _fetch_url(url).decode("utf-8", errors="replace")
        rows = parse_semeval_table(raw)
        if not rows:
            raise ValueError(f"No rows parsed from GitHub {split}")
        save_split(rows, data_dir, split)

    print("[download] fetching test-gold from all-data zip ...")
    payload = _fetch_url(ALL_DATA_ZIP)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        match = next((n for n in zf.namelist() if n.endswith(ZIP_SPLIT_FILES["test"])), None)
        if not match:
            raise FileNotFoundError("test-gold file missing in all-data zip")
        raw = zf.read(match).decode("utf-8", errors="replace")
    rows = parse_semeval_table(raw)
    if not rows:
        raise ValueError("No rows parsed from test-gold")
    save_split(rows, data_dir, "test")


def verify_outputs(data_dir: Path) -> None:
    expected = {
        "train.tsv": 6000,
        "dev.tsv": 800,
        "test.tsv": 3000,
    }
    for fname, min_rows in expected.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with open(path, encoding="utf-8") as f:
            count = sum(1 for _ in f)
        if count < min_rows:
            raise ValueError(f"{fname} has only {count} rows (expected >= {min_rows})")
        print(f"[ok] {fname}: {count} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "semeval2018"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "emotions.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(SEMEVAL_LABELS) + "\n")

    errors: list[str] = []
    for name, fn in (
        ("all-data zip", download_from_all_data_zip),
        ("split zips", download_from_split_zips),
        ("GitHub + test-gold", download_from_github),
    ):
        try:
            print(f"\n=== trying {name} ===")
            fn(data_dir)
            verify_outputs(data_dir)
            print(f"\nSemEval-2018 ready at {data_dir}")
            return
        except (HTTPError, URLError, OSError, zipfile.BadZipFile, ValueError, FileNotFoundError) as exc:
            msg = f"{name} failed: {exc}"
            print(f"[warn] {msg}")
            errors.append(msg)

    print("\n[error] all download sources failed:")
    for msg in errors:
        print(" -", msg)
    sys.exit(1)


if __name__ == "__main__":
    main()
