#!/usr/bin/env python3
"""Verify DOIs in paper/references.bib via Crossref API."""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "paper" / "references.bib"


def extract_dois(text: str) -> list[tuple[str, str]]:
    """Return (bib_key, doi) pairs in file order."""
    entries: list[tuple[str, str]] = []
    key = None
    for line in text.splitlines():
        m = re.match(r"@\w+\{([^,]+),", line)
        if m:
            key = m.group(1)
        m = re.match(r"\s*doi\s*=\s*\{([^}]+)\}", line, re.I)
        if m and key:
            entries.append((key, m.group(1).strip()))
    return entries


def crossref(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "CLFER-LT/1.0 mailto:verify@example.com"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())["message"]


def main() -> None:
    text = BIB.read_text(encoding="utf-8")
    dois = extract_dois(text)
    print(f"Found {len(dois)} DOI entries in {BIB.name}\n")
    ok, fail = 0, 0
    for key, doi in dois:
        time.sleep(0.6)
        try:
            m = crossref(doi)
            title = m.get("title", [""])[0]
            authors = "; ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in m.get("author", [])[:8]
            )
            line = (
                f"[OK] {key}\n"
                f"     DOI:   {doi}\n"
                f"     TITLE: {title}\n"
                f"     AUTH:  {authors}\n"
            )
            print(line.encode("utf-8", errors="replace").decode("utf-8"))
            ok += 1
        except Exception as e:
            print(f"[FAIL] {key}  DOI {doi}  ERR {e}\n")
            fail += 1
    print(f"Summary: {ok} OK, {fail} FAIL, {len(dois)} total with DOI")


if __name__ == "__main__":
    main()
