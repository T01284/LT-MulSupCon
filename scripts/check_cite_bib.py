#!/usr/bin/env python3
"""Ensure every bib key is cited and every cite has a bib entry."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
bib = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")

cited = set()
for block in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
    for k in block.split(","):
        cited.add(k.strip())
for block in re.findall(r"\\refcite\{([^}]+)\}", tex):
    for k in block.split(","):
        cited.add(k.strip())

bib_keys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
uncited = [k for k in bib_keys if k not in cited]
missing = sorted(cited - set(bib_keys))

print(f"bib={len(bib_keys)} cited={len(cited)}")
print("uncited:", uncited or "(none)")
print("missing_from_bib:", missing or "(none)")
raise SystemExit(1 if uncited or missing else 0)
