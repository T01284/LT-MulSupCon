#!/usr/bin/env python3
import re
from pathlib import Path

tex = Path("paper/main.tex").read_text(encoding="utf-8")
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
if m is None:
    start = tex.find(r"\abstract{")
    end = tex.find(r"\keywords{")
    if start < 0 or end < 0:
        raise SystemExit("abstract block not found")
    abs_text = tex[start + len(r"\abstract{") : end].rstrip().rstrip("}")
else:
    abs_text = m.group(1)
abs_plain = re.sub(r"\\method\{\}", "CLFER-LT", abs_text)
abs_plain = re.sub(r"\\textit\{([^}]*)\}", r"\1", abs_plain)
abs_plain = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", abs_plain)
abs_plain = re.sub(r"\\[a-zA-Z]+", "", abs_plain)
abs_plain = re.sub(r"[{}$\\]", " ", abs_plain)
abs_plain = re.sub(r"\s+", " ", abs_plain).strip()
words = abs_plain.split()
print("ABSTRACT_WORDS", len(words))
print("BLANK_LINES_IN_ABS", abs_text.count("\n\n"))
print(abs_plain)
