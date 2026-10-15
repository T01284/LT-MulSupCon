"""Apply the verified APA 6 cleanup rules used for the submission bibliography."""
from __future__ import annotations

import re
from pathlib import Path


BIB = Path(__file__).resolve().parents[1] / "paper" / "references.bib"


def clean_entry(entry: str) -> str:
    if entry.lstrip().startswith("@inproceedings"):
        entry = re.sub(r"(?m)^\s*month\s*=\s*\{[^}]*\},?\s*\n", "", entry)

    doi_match = re.search(r"(?m)^\s*doi\s*=\s*\{([^}]+)\},?\s*$", entry)
    url_match = re.search(r"(?m)^\s*url\s*=\s*\{https://doi\.org/([^}]+)\},?\s*$", entry)
    if doi_match and url_match and doi_match.group(1).lower() == url_match.group(1).lower():
        entry = entry[: url_match.start()] + entry[url_match.end() :]
        entry = re.sub(
            r"(?m)^(\s*doi\s*=\s*\{[^}]+\}),\s*$",
            r"\1",
            entry,
        )

    # Keep a comma between the final field and any following field, but not
    # immediately before the entry-closing brace.
    entry = re.sub(r",\s*\n(\s*\})", r"\n\1", entry)
    return entry


def main() -> None:
    text = BIB.read_text(encoding="utf-8")
    parts = re.split(r"(?=^@)", text, flags=re.MULTILINE)
    cleaned = "".join(clean_entry(part) if part.startswith("@") else part for part in parts)
    BIB.write_text(cleaned, encoding="utf-8")
    print(f"Cleaned {BIB}")


if __name__ == "__main__":
    main()
