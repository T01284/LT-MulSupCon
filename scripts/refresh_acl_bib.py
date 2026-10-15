#!/usr/bin/env python3
"""Refresh ACL Anthology BibTeX with a robust field parser."""
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"

KEY_TO_ANTHOLOGY = {
    "demszky2020goemotions": "2020.acl-main.372",
    "mohammad2018semeval": "S18-1001",
    "abdulmageed2017emonet": "P17-1067",
    "devlin2019bert": "N19-1423",
    "gao2021simcse": "2021.emnlp-main.552",
    "yan2021consert": "2021.acl-long.393",
    "su2022contrastive": "2022.acl-short.75",
    "gong2025taco": "2025.acl-long.247",
    "lin2023effective": "2023.findings-acl.556",
    "hu2023sacl": "2023.acl-long.606",
    "chen2023hyperbolic": "2023.acl-long.613",
    "kumar2025semantic": "2025.acl-srw.55",
    "sharma2024linear": "2024.emnlp-main.1161",
    "zhang2023depl": "2023.findings-eacl.81",
    "li2024openworld": "2024.emnlp-main.841",
    "zhang2024message": "2024.emnlp-main.162",
    "yu2024emotion": "2024.findings-naacl.282",
    "reimers2019sentencebert": "D19-1410",
    "saravia2018carer": "D18-1404",
}

KEEP = [
    "title",
    "author",
    "editor",
    "booktitle",
    "month",
    "year",
    "address",
    "publisher",
    "pages",
    "doi",
    "url",
]


def fetch_anthology_bib(anth_id: str) -> str:
    url = f"https://aclanthology.org/{anth_id}.bib"
    req = urllib.request.Request(url, headers={"User-Agent": "CLFER-ref-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_fields(entry: str) -> dict[str, str]:
    """Parse BibTeX fields supporting {braces} and \"quotes\"."""
    fields: dict[str, str] = {}
    # strip header @type{key,
    body = re.sub(r"^@\w+\{[^,]+,\s*", "", entry.strip(), count=1)
    body = re.sub(r"\}\s*$", "", body)
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        m = re.match(r"(\w+)\s*=\s*", body[i:])
        if not m:
            break
        key = m.group(1).lower()
        i += m.end()
        if i >= n:
            break
        if body[i] == "{":
            depth = 0
            j = i
            while j < n:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        fields[key] = body[i + 1 : j]
                        i = j + 1
                        break
                j += 1
            else:
                raise ValueError(f"unbalanced brace for {key}")
        elif body[i] == '"':
            j = i + 1
            while j < n:
                if body[j] == '"' and body[j - 1] != "\\":
                    fields[key] = body[i + 1 : j]
                    i = j + 1
                    break
                j += 1
            else:
                raise ValueError(f"unbalanced quote for {key}")
        else:
            # bare value until comma
            j = i
            while j < n and body[j] not in ",\n":
                j += 1
            fields[key] = body[i:j].strip()
            i = j
    return fields


def format_entry(key: str, fields: dict[str, str]) -> str:
    lines = [f"@inproceedings{{{key},"]
    used = []
    for f in KEEP:
        if f in fields and fields[f].strip():
            used.append(f)
    for idx, f in enumerate(used):
        comma = "," if idx < len(used) - 1 else ""
        lines.append(f"  {f:<9}= {{{fields[f]}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def replace_entry(bib: str, key: str, new_entry: str) -> str:
    pat = re.compile(rf"@\w+\{{{re.escape(key)},.*?^\}}", re.M | re.S)
    m = pat.search(bib)
    if not m:
        raise KeyError(key)
    return bib[: m.start()] + new_entry + bib[m.end() :]


def main() -> None:
    bib = BIB.read_text(encoding="utf-8")
    bib = re.sub(
        r"^% CLFER-LT references \(verified against.*\)\n+",
        "% CLFER-LT references\n",
        bib,
        count=1,
    )

    for key, anth_id in KEY_TO_ANTHOLOGY.items():
        raw = fetch_anthology_bib(anth_id)
        fields = parse_fields(raw)
        if len(fields) < 4:
            raise RuntimeError(f"parse failed for {key}: {fields.keys()}")
        fields["url"] = f"https://aclanthology.org/{anth_id}/"
        new_entry = format_entry(key, fields)
        bib = replace_entry(bib, key, new_entry)
        print(
            "OK",
            key,
            "fields",
            len(fields),
            "booktitle=",
            fields.get("booktitle", "")[:70],
        )
        time.sleep(0.3)

    aud = """@inproceedings{audibert2024exploring,
  title     = {Exploring Contrastive Learning for Long-Tailed Multi-label Text Classification},
  author    = {Audibert, Alexandre and Gauffre, Aur{\\'e}lien and Amini, Massih-Reza},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
  series    = {Lecture Notes in Computer Science},
  volume    = {14947},
  pages     = {245--261},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-70368-3_15},
  url       = {https://doi.org/10.1007/978-3-031-70368-3_15}
}"""
    bib = replace_entry(bib, "audibert2024exploring", aud)

    # Keep previously corrected MoCo / Kang URLs if present; ensure Kang is correct
    bib = bib.replace(
        "https://openreview.net/forum?id=HkgjGBtwrw",
        "https://openreview.net/forum?id=r1gRTCVFvB",
    )

    BIB.write_text(bib, encoding="utf-8")
    print("Wrote", BIB)

    # Verify requested fixes
    checks = {
        "gong2025taco": "Volume 1: Long Papers",
        "chen2023hyperbolic": "Chih-Yao",
        "hu2023sacl": "Volume 1: Long Papers",
        "abdulmageed2017emonet": "Volume 1: Long Papers",
        "su2022contrastive": "Volume 2: Short Papers",
        "kumar2025semantic": "Semantic alignment",
        "audibert2024exploring": "Multi-label",
        "kang2020decoupling": "r1gRTCVFvB",
    }
    for key, needle in checks.items():
        m = re.search(rf"@\w+\{{{key},.*?^\}}", bib, re.M | re.S)
        ok = bool(m and needle in m.group(0))
        print(("PASS" if ok else "FAIL"), key, "=>", needle)


if __name__ == "__main__":
    main()
