# Extracted Code

This directory contains code snippets extracted from `.txt` files in the repository.

Heuristics used:
- Extract fenced code blocks with triple backticks (```), respecting optional language hints.
- Find contiguous blocks of lines that match common code patterns (imports, defs, class, function, let, const, etc.).
- Detect language by fence hint or simple heuristics over snippet contents.
- If a snippet appears to contain secrets (API keys, tokens, private keys), the snippet is redacted and not stored.

Structure:
- `extracted_code/<language>/<original_basename>__snippet-XXX.<ext>` - individual snippets
- `extracted_code/index.json` - metadata mapping snippets to source files and line ranges

To regenerate, run: `python scripts/extract_code_from_txt.py`
