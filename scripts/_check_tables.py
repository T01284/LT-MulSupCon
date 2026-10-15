#!/usr/bin/env python3
import re
from pathlib import Path

tex = Path("paper/main.tex").read_text(encoding="utf-8")
cites = [(m.start(), m.group(1)) for m in re.finditer(r"\\ref\{(tab:[^}]+)\}", tex)]
seen = set()
first = []
for pos, lab in cites:
    if lab not in seen:
        seen.add(lab)
        first.append((lab, tex[:pos].count("\n") + 1))
print("=== First citation order ===")
for i, (lab, ln) in enumerate(first, 1):
    print(f"{i:2d}. L{ln}: {lab}")
defs = [
    (m.group(1), tex[: m.start()].count("\n") + 1)
    for m in re.finditer(r"\\label\{(tab:[^}]+)\}", tex)
]
print("=== Definition (numbering) order ===")
for i, (lab, ln) in enumerate(defs, 1):
    print(f"{i:2d}. L{ln}: {lab}")
def_rank = {lab: i for i, (lab, _) in enumerate(defs, 1)}
print("=== Mismatch ===")
for i, (lab, ln) in enumerate(first, 1):
    m = def_rank.get(lab, "?")
    mark = " OK" if m == i else " BAD"
    print(f"{mark} first-cite#{i} {lab} -> float Table {m}")
