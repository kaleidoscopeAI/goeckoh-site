#!/usr/bin/env python3
"""
Extract code from .txt files in the repository.

This script scans all .txt files, identifies code blocks, and extracts them
into organized files in the extracted_code/ directory.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json


# File extensions and their common indicators
CODE_INDICATORS = {
    'python': [
        r'#!/usr/bin/env python',
        r'import\s+\w+',
        r'from\s+\w+\s+import',
        r'def\s+\w+\s*\(',
        r'class\s+\w+',
        r'if\s+__name__\s*==\s*["\']__main__["\']',
    ],
    'javascript': [
        r'function\s+\w+\s*\(',
        r'const\s+\w+\s*=',
        r'let\s+\w+\s*=',
        r'var\s+\w+\s*=',
        r'import\s+.*\s+from',
        r'export\s+(default\s+)?',
    ],
    'cpp': [
        r'#include\s*<\w+\.h>',
        r'#include\s*<\w+>',
        r'std::',
        r'int\s+main\s*\(',
        r'class\s+\w+\s*{',
        r'namespace\s+\w+',
    ],
    'java': [
        r'public\s+class\s+\w+',
        r'public\s+static\s+void\s+main',
        r'import\s+java\.',
        r'package\s+\w+',
    ],
    'rust': [
        r'fn\s+\w+\s*\(',
        r'use\s+\w+::',
        r'mod\s+\w+',
        r'pub\s+fn\s+\w+',
        r'impl\s+\w+',
    ],
    'shell': [
        r'#!/bin/bash',
        r'#!/bin/sh',
        r'#!/usr/bin/env bash',
    ],
    'sql': [
        r'SELECT\s+',
        r'INSERT\s+INTO',
        r'CREATE\s+TABLE',
        r'UPDATE\s+\w+\s+SET',
    ],
}


def detect_language(content: str) -> str:
    """Detect the programming language of code content."""
    content_lower = content.lower()
    
    # Count matches for each language
    scores = {}
    for lang, patterns in CODE_INDICATORS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            score += len(matches)
        if score > 0:
            scores[lang] = score
    
    if not scores:
        return 'unknown'
    
    # Return language with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def extract_code_blocks(content: str, filename: str) -> List[Tuple[str, str, int]]:
    """
    Extract code blocks from text content.
    Returns list of (code, language, start_position) tuples.
    """
    blocks = []
    
    # Strategy 1: Look for markdown code blocks (```language ... ```)
    markdown_pattern = r'```(\w+)?\n(.*?)```'
    for match in re.finditer(markdown_pattern, content, re.DOTALL):
        lang = match.group(1) or 'unknown'
        code = match.group(2).strip()
        if code:
            blocks.append((code, lang, match.start()))
    
    # Strategy 2: Look for script separators (like in all_scripts_combined.txt)
    script_pattern = r'SCRIPT:\s*(.*?)\n[▀═\-]{3,}\n(.*?)(?=\n[▄═\-]{3,}\nSCRIPT:|$)'
    for match in re.finditer(script_pattern, content, re.DOTALL):
        script_name = match.group(1).strip()
        code = match.group(2).strip()
        if code:
            # Detect language from script name or content
            lang = detect_language(code)
            if script_name.endswith('.py'):
                lang = 'python'
            elif script_name.endswith('.js'):
                lang = 'javascript'
            elif script_name.endswith('.cpp') or script_name.endswith('.cc'):
                lang = 'cpp'
            blocks.append((code, lang, match.start()))
    
    # Strategy 3: Look for continuous code-like blocks (if no blocks found yet)
    if not blocks:
        # Split content into potential code blocks
        lines = content.split('\n')
        current_block = []
        block_start = 0
        
        for i, line in enumerate(lines):
            # Check if line looks like code
            if (line.strip() and 
                (re.match(r'^\s*(import|from|def|class|function|const|let|var|#include)', line) or
                 re.match(r'^\s*[a-zA-Z_]\w*\s*[=\(]', line) or
                 re.match(r'^\s*[{}()\[\];]', line))):
                if not current_block:
                    block_start = i
                current_block.append(line)
            elif current_block and len(current_block) > 5:
                # End of code block
                code = '\n'.join(current_block)
                lang = detect_language(code)
                if lang != 'unknown':
                    blocks.append((code, lang, block_start))
                current_block = []
        
        # Don't forget the last block
        if current_block and len(current_block) > 5:
            code = '\n'.join(current_block)
            lang = detect_language(code)
            if lang != 'unknown':
                blocks.append((code, lang, block_start))
    
    return blocks


def extract_from_file(txt_file: Path, output_dir: Path) -> Dict:
    """Extract code from a single .txt file."""
    try:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
        return {'error': str(e), 'blocks': 0}
    
    # Extract code blocks
    blocks = extract_code_blocks(content, txt_file.name)
    
    if not blocks:
        return {'blocks': 0, 'message': 'No code blocks found'}
    
    # Create output directory for this file
    relative_path = txt_file.relative_to(Path.cwd())
    file_output_dir = output_dir / relative_path.parent / txt_file.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each block
    saved_blocks = []
    for idx, (code, lang, pos) in enumerate(blocks, 1):
        ext = {
            'python': 'py',
            'javascript': 'js',
            'cpp': 'cpp',
            'java': 'java',
            'rust': 'rs',
            'shell': 'sh',
            'sql': 'sql',
        }.get(lang, 'txt')
        
        output_file = file_output_dir / f"block_{idx:03d}.{ext}"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        saved_blocks.append({
            'file': str(output_file.relative_to(output_dir)),
            'language': lang,
            'lines': len(code.split('\n')),
            'position': pos
        })
    
    return {
        'source': str(relative_path),
        'blocks': len(blocks),
        'saved': saved_blocks
    }


def main():
    """Main extraction function."""
    repo_root = Path.cwd()
    output_dir = repo_root / 'extracted_code'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Code Extraction from .txt Files")
    print("=" * 70)
    print()
    
    # Find all .txt files
    txt_files = []
    for txt_file in repo_root.rglob('*.txt'):
        # Skip .git directory
        if '.git' in txt_file.parts:
            continue
        txt_files.append(txt_file)
    
    print(f"Found {len(txt_files)} .txt files")
    print()
    
    # Process each file
    results = []
    total_blocks = 0
    
    for txt_file in sorted(txt_files):
        print(f"Processing: {txt_file.relative_to(repo_root)}")
        result = extract_from_file(txt_file, output_dir)
        
        if result.get('blocks', 0) > 0:
            print(f"  ✓ Extracted {result['blocks']} code blocks")
            total_blocks += result['blocks']
        else:
            print(f"  - No code blocks found")
        
        results.append(result)
    
    print()
    print("=" * 70)
    print(f"Extraction Complete!")
    print(f"Total files processed: {len(txt_files)}")
    print(f"Total code blocks extracted: {total_blocks}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Save summary
    summary_file = output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'total_files': len(txt_files),
            'total_blocks': total_blocks,
            'results': results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Create README for extracted code
    readme_file = output_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write("# Extracted Code from .txt Files\n\n")
        f.write("This directory contains code extracted from various .txt files in the repository.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total .txt files processed**: {len(txt_files)}\n")
        f.write(f"- **Total code blocks extracted**: {total_blocks}\n\n")
        f.write("## Structure\n\n")
        f.write("Each .txt file's extracted code is organized in a subdirectory matching its original path.\n")
        f.write("Code blocks are numbered sequentially and saved with appropriate file extensions.\n\n")
        f.write("## Details\n\n")
        
        for result in results:
            if result.get('blocks', 0) > 0:
                f.write(f"### {result['source']}\n\n")
                f.write(f"Extracted {result['blocks']} code blocks:\n\n")
                for block in result.get('saved', []):
                    f.write(f"- `{block['file']}` ({block['language']}, {block['lines']} lines)\n")
                f.write("\n")
    
    print(f"README created: {readme_file}")


if __name__ == '__main__':
    main()
