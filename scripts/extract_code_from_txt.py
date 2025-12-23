#!/usr/bin/env python3
"""Extract code snippets from .txt files and organize them under extracted_code/ by language.

Usage: python scripts/extract_code_from_txt.py

This script finds fenced code blocks (```), shebang lines, and code-like contiguous line groups
and writes them to files under extracted_code/<language>/. It also produces extracted_code/index.json
with metadata and a README describing heuristics.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "extracted_code"

FENCE_RE = re.compile(r"^```\s*(?P<lang>\w+)?\s*$")
CODE_LIKE = re.compile(r"\b(def |class |import |from |function |fn |let |const |package |#include|using |println!|console\.log|async def|public |private |fmt::|fmt\.Print|SELECT |INSERT |UPDATE )")
SUSPICIOUS_SECRET = re.compile(r"(TOKEN|API_KEY|SECRET|PRIVATE_KEY|ACCESS_TOKEN|PASSWORD)", re.I)

EXT_BY_LANG = {
    "python": "py",
    "py": "py",
    "js": "js",
    "javascript": "js",
    "ts": "ts",
    "typescript": "ts",
    "rust": "rs",
    "rs": "rs",
    "go": "go",
    "golang": "go",
    "sh": "sh",
    "bash": "sh",
    "c": "c",
    "cpp": "cpp",
    "java": "java",
    "sql": "sql",
}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_language_from_snippet(lines: List[str]) -> str:
    text = "\n".join(lines[:20])
    if re.search(r"\b(def |import |from )", text):
        return "python"
    if re.search(r"\b(function |const |let |console\.log)", text):
        return "javascript"
    if re.search(r"\bpackage \w+;|func \w+\(|fmt\.Print|fmt::", text):
        return "go"
    if re.search(r"fn \w+\(|println!|use \w+::", text):
        return "rust"
    if re.search(r"#include|int main\(|std::", text):
        return "c"
    if re.search(r"class \w+\{|public |private |System\.", text):
        return "java"
    if re.search(r"SELECT \w+|INSERT INTO|UPDATE \w+ SET", text, re.I):
        return "sql"
    if re.search(r"^#!", lines[0]):
        if "python" in lines[0]:
            return "python"
        if "bash" in lines[0] or "sh" in lines[0]:
            return "sh"
    return "text"


def extension_for_lang(lang: str) -> str:
    return EXT_BY_LANG.get(lang.lower(), "txt")


def is_suspicious(lines: List[str]) -> bool:
    return any(SUSPICIOUS_SECRET.search(l) for l in lines)


def extract_from_file(path: Path) -> List[dict]:
    out = []
    lines = path.read_text(errors="ignore").splitlines()
    n = len(lines)
    i = 0
    # First: fenced code blocks
    while i < n:
        m = FENCE_RE.match(lines[i])
        if m:
            lang = m.group("lang") or ""
            start = i
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            end = i
            # skip closing fence
            i += 1
            entry = {
                "type": "fenced",
                "lang_hint": lang or None,
                "start_line": start + 1,
                "end_line": end + 1,
                "lines": block,
            }
            out.append(entry)
            continue
        i += 1

    # Second: detect contiguous code-like regions (sequence of lines where at least one line matches CODE_LIKE in a block)
    i = 0
    while i < n:
        if CODE_LIKE.search(lines[i]):
            start = i
            block = [lines[i]]
            i += 1
            while i < n and (lines[i].strip() == "" or lines[i].startswith(" ") or lines[i].startswith("\t") or CODE_LIKE.search(lines[i])):
                block.append(lines[i])
                i += 1
            end = i - 1
            # avoid duplicates if same region overlaps with fenced already
            if any(e["start_line"] - 1 <= start <= e["end_line"] - 1 for e in out):
                continue
            entry = {
                "type": "inline",
                "lang_hint": None,
                "start_line": start + 1,
                "end_line": end + 1,
                "lines": block,
            }
            out.append(entry)
        else:
            i += 1

    # Indented code blocks (Markdown): lines starting with 4 spaces or a tab
    i = 0
    while i < n:
        if lines[i].startswith("    ") or lines[i].startswith("\t"):
            start = i
            block = [lines[i][4:] if lines[i].startswith("    ") else lines[i][1:]]
            i += 1
            while i < n and (lines[i].startswith("    ") or lines[i].startswith("\t") or lines[i].strip() == ""):
                if lines[i].strip() == "":
                    block.append("")
                else:
                    block.append(lines[i][4:] if lines[i].startswith("    ") else lines[i][1:])
                i += 1
            end = i - 1
            entry = {
                "type": "indented",
                "lang_hint": None,
                "start_line": start + 1,
                "end_line": end + 1,
                "lines": block,
            }
            out.append(entry)
        else:
            i += 1

    # HTML blocks: <pre><code class="language-..."> ... </code></pre> and <script>..</script>
    html_open_re = re.compile(r"<(pre|script)(?:\s+[^>]*)?>", re.I)
    html_close_re = re.compile(r"</(pre|script)>", re.I)
    i = 0
    while i < n:
        if html_open_re.search(lines[i]):
            start = i
            tag = html_open_re.search(lines[i]).group(1).lower()
            block = []
            i += 1
            while i < n and not html_close_re.search(lines[i]):
                line = re.sub(r"</?code[^>]*>", "", lines[i])
                block.append(line)
                i += 1
            end = i
            i += 1
            entry = {
                "type": "html",
                "lang_hint": None,
                "start_line": start + 1,
                "end_line": end + 1,
                "lines": block,
            }
            out.append(entry)
        else:
            i += 1

    return out


def write_snippets(file_path: Path, snippets: List[dict], index: list):
    rel = file_path.relative_to(ROOT)
    base = file_path.stem
    count = defaultdict(int)
    for sn in snippets:
        lines = sn["lines"]
        if not lines:
            continue
        redacted = False
        if is_suspicious(lines):
            # redact entire snippet
            lines = ["# <REDACTED: potential secret detected>\n"]
            redacted = True

        lang = sn.get("lang_hint")
        if not lang:
            lang = detect_language_from_snippet(lines)
        ext = extension_for_lang(lang)
        count[lang] += 1
        idx = count[lang]
        safe_name = f"{base}__snippet-{idx:03d}.{ext}"
        outdir = OUTDIR / lang
        safe_mkdir(outdir)
        outpath = outdir / safe_name
        outpath.write_text("\n".join(lines) + "\n")

        index.append({
            "source": str(rel),
            "snippet_file": str(outpath.relative_to(ROOT)),
            "lang": lang,
            "start_line": sn["start_line"],
            "end_line": sn["end_line"],
            "type": sn.get("type"),
            "redacted": redacted,
        })


def main():
    # Reset output dir so re-running replaces previous extraction cleanly
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)

    # Process both .txt and .md files (and .markdown)
    txt_files = list(ROOT.rglob("*.txt")) + list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.markdown"))
    txt_files = [p for p in txt_files if "extracted_code" not in p.parts and "node_modules" not in p.parts]
    summary = {"processed_files": 0, "snippets": 0}
    index = []
    seen_hashes = set()
    for p in txt_files:
        try:
            snippets = extract_from_file(p)
        except Exception as e:
            print(f"Error parsing {p}: {e}")
            continue
        if not snippets:
            continue
        # filter duplicates by content hash so re-running doesn't produce duplicates
        filtered = []
        for sn in snippets:
            content = "\n".join(sn.get("lines", [])).strip()
            if not content:
                continue
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            filtered.append(sn)
        if not filtered:
            continue
        write_snippets(p, filtered, index)
        summary["processed_files"] += 1
        summary["snippets"] += len(filtered)

    safe_mkdir(OUTDIR)
    (OUTDIR / "index.json").write_text(json.dumps(index, indent=2))
    readme = OUTDIR / "README.md"
    readme.write_text("""# Extracted Code

This directory contains code snippets extracted from `.txt`, `.md`, and `.markdown` files in the repository.

Heuristics used:
- Extract fenced code blocks with triple backticks (```), respecting optional language hints.
- Find contiguous blocks of lines that match common code patterns (imports, defs, class, function, let, const, etc.).
- Detect language by fence hint or simple heuristics over snippet contents.
- If a snippet appears to contain secrets (API keys, tokens, private keys), the snippet is redacted and not stored.

Structure:
- `extracted_code/<language>/<original_basename>__snippet-XXX.<ext>` - individual snippets
- `extracted_code/index.json` - metadata mapping snippets to source files and line ranges

To regenerate, run: `python scripts/extract_code_from_txt.py`
""")

    print(f"Processed {summary['processed_files']} files, extracted {summary['snippets']} snippets.")


if __name__ == "__main__":
    main()
