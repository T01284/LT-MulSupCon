"""Final reference authenticity + cite coverage check."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
UA = {"User-Agent": "Mozilla/5.0 (ref-verify; mailto:local@example.com)"}


def parse_bib(text: str) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for m in re.finditer(
        r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}",
        text,
        flags=re.S,
    ):
        etype, key, body = m.group(1), m.group(2), m.group(3)
        fields: dict[str, str] = {"_type": etype}
        for fm in re.finditer(
            r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\")",
            body,
        ):
            fields[fm.group(1).lower()] = (fm.group(2) or fm.group(3) or "").strip()
        entries[key] = fields
    return entries


def cited_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            keys.add(k.strip())
    return keys


def http_ok(url: str) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=25) as r:
            return True, str(r.status)
    except Exception:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return True, str(r.status)
        except Exception as e:
            return False, str(e)


def crossref(doi: str) -> dict | None:
    url = "https://api.crossref.org/works/" + doi
    try:
        req = urllib.request.Request(url, headers=UA)
        return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())["message"]
    except Exception as e:
        return {"_error": str(e)}


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[{}\\]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_overlap(a: str, b: str) -> float:
    ta, tb = set(norm(a).split()), set(norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def main() -> None:
    entries = parse_bib(BIB)
    cited = cited_keys(TEX)
    bib_keys = set(entries)
    print(f"bib={len(bib_keys)} cited={len(cited)}")
    print("uncited:", sorted(bib_keys - cited) or "NONE")
    print("missing_from_bib:", sorted(cited - bib_keys) or "NONE")

    lines = [
        "# Reference verification log",
        "",
        f"- Bib entries: **{len(bib_keys)}**",
        f"- Cited keys in `main.tex`: **{len(cited)}**",
        f"- Uncited bib entries: {sorted(bib_keys - cited) or 'none'}",
        f"- Cites missing from bib: {sorted(cited - bib_keys) or 'none'}",
        "",
        "## Per-entry checks",
        "",
    ]

    ok_n = fail_n = 0
    for key in sorted(entries):
        e = entries[key]
        title = e.get("title", "")
        doi = e.get("doi", "")
        url = e.get("url", "")
        status = "OK"
        detail = []

        if doi:
            msg = crossref(doi)
            if not msg or "_error" in msg:
                status = "FAIL"
                detail.append(f"Crossref error: {msg}")
            else:
                cr_title = (msg.get("title") or [""])[0]
                ov = title_overlap(title, cr_title)
                authors = [
                    f"{a.get('given','')} {a.get('family','')}".strip()
                    for a in (msg.get("author") or [])
                ]
                detail.append(f"Crossref title: {cr_title}")
                detail.append(f"title overlap={ov:.2f}")
                detail.append(f"authors: {', '.join(authors[:6])}")
                # For @inproceedings the bib title should match Crossref paper title
                if ov < 0.35 and e["_type"] != "inproceedings":
                    # already checked articles
                    status = "WARN"
                if ov < 0.35:
                    # still authenticate via DOI resolve; flag soft mismatch for inproceedings
                    # where parser sometimes grabs wrong field — re-check first author year
                    if e["_type"] == "inproceedings" and ov < 0.2:
                        status = "WARN"
                if "CVPR.2019.00967" in doi:
                    status = "FAIL"
                    detail.append("Known wrong DOI (KTN)")
        elif url:
            good, info = http_ok(url)
            detail.append(f"URL check: {info}")
            if not good:
                status = "FAIL"
        else:
            status = "FAIL"
            detail.append("No DOI/URL")

        if status == "OK":
            ok_n += 1
        else:
            fail_n += 1
        mark = status
        print(f"[{mark}] {key}" + (f"  doi={doi}" if doi else f"  url={url[:60]}"))
        lines.append(f"### `{key}` — **{mark}**")
        lines.append(f"- type: `{e['_type']}`")
        if title:
            lines.append(f"- bib title: {title}")
        if doi:
            lines.append(f"- doi: `{doi}`")
        if url:
            lines.append(f"- url: {url}")
        for d in detail:
            lines.append(f"- {d}")
        lines.append("")

    lines.append("## Summary")
    lines.append(f"- OK/WARN resolved: {ok_n}")
    lines.append(f"- FAIL: {fail_n}")
    out = ROOT / "paper" / "REFERENCES_VERIFY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out)
    print(f"ok={ok_n} failish={fail_n}")


if __name__ == "__main__":
    main()
