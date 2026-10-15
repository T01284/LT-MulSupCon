#!/usr/bin/env python3
"""Count manuscript words excluding appendix (and Acknowledgements optional?)."""
import re
from pathlib import Path

tex = Path("paper/main.tex").read_text(encoding="utf-8")

# Drop preamble
body = tex.split(r"\begin{document}", 1)[1]
# Drop appendix and after (acks are before appendix? check)
# Structure: main ... conclusion, appendix, acks, bibliography
# User: 全文（包含图表，文献列表，不包含附件）
# So include: title to refs, exclude appendix content
# Acknowledgements is usually included in main; AI disclosure is a declaration

parts = re.split(r"\\appendix\b", body, maxsplit=1)
main = parts[0]
# If acks are after appendix in this file, pull them back
# Current order: conclusion, appendix..., Acknowledgements, bibliography
# So acks are AFTER appendix - need to include acks+bib from after appendix
rest = parts[1] if len(parts) > 1 else ""
ack_bib = ""
m = re.search(r"(\\section\*\{Acknowledgements\}.*)", rest, re.S)
if m:
    ack_bib = m.group(1)
else:
    # bibliography only if no acks match
    m2 = re.search(r"(\\bibliographystyle.*)", rest, re.S)
    if m2:
        ack_bib = m2.group(1)

text = main + "\n" + ack_bib

# Remove comments
text = re.sub(r"(?<!\\)%.*", "", text)
# Remove common latex
text = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", text)
text = re.sub(r"\\(?:cite|ref|label|path|url|email|texttt|textbf|textit|emph|textsc|mathrm|mathbb|mathcal|mathbf|mathrm|operatorname)\{[^}]*\}", " ", text)
text = re.sub(r"\\method\{\}", "CLFER-LT", text)
text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r" \1 ", text)
text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
text = re.sub(r"\$[^$]*\$", " ", text)
text = re.sub(r"[{}&~^_]", " ", text)
text = re.sub(r"\s+", " ", text).strip()
words = text.split()
print("WORD_COUNT", len(words))
print("SAMPLE_TAIL", " ".join(words[-30:]))
