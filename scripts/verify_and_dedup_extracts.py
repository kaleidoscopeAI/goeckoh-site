#!/usr/bin/env python3
"""Verify extraction with looser heuristics and deduplicate snippets.

Produces:
- extracted_code/additional_candidates.json — code-like regions found by looser heuristics but not in the original index
- extracted_code/deduped_index.json — grouping of identical snippets with occurrences
- extracted_code/dedupe_report.md — human-readable summary of deduplication
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
EXDIR = ROOT / "extracted_code"
INDEX = EXDIR / "index.json"

# Looser code-like regexes to find additional candidates
LOOSE_PATTERNS = [
    re.compile(r"\bif\s*\(|\bfor\s*\(|\bwhile\s*\(|\breturn\b"),
    re.compile(r";\s*$"),  # languages that use semicolons
    re.compile(r"\bSELECT\b|\bINSERT\b|\bUPDATE\b", re.I),
    re.compile(r"<\w+\b.*?>"),  # HTML-like
    re.compile(r"\bfunc\b|\bfn\b|\bpackage\b|\bimport\b"),
]


def read_index():
    return json.loads(INDEX.read_text())


def find_existing_ranges_by_source(index):
    d = defaultdict(list)
    for e in index:
        d[e["source"]].append((e["start_line"], e["end_line"]))
    return d


def overlaps(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or b_end < a_start)


def find_additional_candidates():
    index = read_index()
    existing = find_existing_ranges_by_source(index)
    candidates = []
    txt_files = list(ROOT.rglob("*.txt"))
    for p in txt_files:
        rel = str(p.relative_to(ROOT))
        lines = p.read_text(errors="ignore").splitlines()
        n = len(lines)
        i = 0
        while i < n:
            matched = any(pat.search(lines[i]) for pat in LOOSE_PATTERNS)
            if matched:
                start = i
                block = [lines[i]]
                i += 1
                # extend up to 30 lines or until blank line threshold
                while i < n and len(block) < 60 and (lines[i].strip() != "" or len(block) < 3):
                    block.append(lines[i])
                    i += 1
                end = start + len(block) - 1
                # check overlap with existing
                overlaps_existing = any(overlaps(start+1, end+1, a, b) for (a,b) in existing.get(rel, []))
                if not overlaps_existing:
                    candidates.append({
                        "source": rel,
                        "start_line": start+1,
                        "end_line": end+1,
                        "lines": block,
                    })
            else:
                i += 1
    (EXDIR / "additional_candidates.json").write_text(json.dumps(candidates, indent=2))
    print(f"Found {len(candidates)} additional candidates (written to extracted_code/additional_candidates.json)")
    return candidates


def normalize_content(text: str) -> str:
    # strip trailing whitespace, collapse multiple blank lines, normalize EOL
    lines = [l.rstrip() for l in text.splitlines()]
    out = []
    prev_blank = False
    for l in lines:
        blank = (l.strip() == "")
        if blank and prev_blank:
            continue
        out.append(l)
        prev_blank = blank
    return "\n".join(out).strip() + "\n"


def compute_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deduplicate_index():
    index = read_index()
    groups = defaultdict(list)
    for e in index:
        p = ROOT / e["snippet_file"]
        if not p.exists():
            continue
        content = p.read_text(errors="ignore")
        norm = normalize_content(content)
        h = compute_hash(norm)
        groups[h].append({
            "snippet_file": e["snippet_file"],
            "source": e["source"],
            "lang": e["lang"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
            "content_preview": norm[:300],
        })

    deduped = []
    for h, occ in groups.items():
        deduped.append({
            "hash": h,
            "count": len(occ),
            "occurrences": occ,
            "lang": occ[0]["lang"],
            "example": occ[0]["snippet_file"],
        })

    deduped_sorted = sorted(deduped, key=lambda x: x["count"], reverse=True)
    (EXDIR / "deduped_index.json").write_text(json.dumps(deduped_sorted, indent=2))

    # report
    total_snips = sum(d["count"] for d in deduped_sorted)
    unique = len(deduped_sorted)
    top = deduped_sorted[:20]
    report_lines = [
        "# Deduplication report",
        "",
        f"Total snippets (from index): {total_snips}",
        f"Unique snippet groups: {unique}",
        "",
        "Top duplicated snippet groups (top 20):",
    ]
    for g in top:
        report_lines.append(f"- count={g['count']} lang={g['lang']} example={g['example']}")

    report_lines.append("")
    report_lines.append("To inspect a group: open extracted_code/deduped_index.json and follow 'occurrences' -> 'snippet_file'.")
    (EXDIR / "dedupe_report.md").write_text("\n".join(report_lines))
    print(f"Deduplicated {total_snips} snippets into {unique} unique groups (written to extracted_code/deduped_index.json and dedupe_report.md)")


def main():
    find_additional_candidates()
    deduplicate_index()


if __name__ == "__main__":
    main()
