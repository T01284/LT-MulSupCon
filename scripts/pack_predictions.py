#!/usr/bin/env python3
"""Bundle all test-prediction dumps into one archive for a GitHub Release / Zenodo upload.

Writes release/clfer_lt_predictions.zip (git-ignored) and outputs/predictions_manifest.json
(tracked), which lists every file with its size and SHA-256 so a downloaded archive can be
checked against the repository.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("outputs/predictions", "outputs/predictions_semeval")
ARCHIVE = ROOT / "release" / "clfer_lt_predictions.zip"
MANIFEST = ROOT / "outputs" / "predictions_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(p for src in SOURCES for p in (ROOT / src).glob("*.npz"))
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            zf.write(p, rel)
            entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest = {
        "archive": ARCHIVE.name,
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "n_files": len(entries),
        "content": "np.savez dumps with y_true, y_pred, y_prob, labels for each test run",
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(entries)} files -> {ARCHIVE} ({ARCHIVE.stat().st_size / 2**20:.1f} MB)")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
