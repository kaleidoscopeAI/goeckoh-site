def can_transform(self, code_file: CodeFile) -> bool:
    return code_file.language == LanguageType.PYTHON

def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
    content = code_file.content
    transformations = []

    content, type_transforms = self._add_type_hints(content)
    if type_transforms:
        transformations.append("Added type hints")

    content, fstring_count = self._convert_to_fstrings(content)
    if fstring_count > 0:
        transformations.append(f"Converted {fstring_count} string formats to f-strings")

    content, modern_transforms = self._modernize_python_features(content)
    transformations.extend(modern_transforms)

    return content, transformations

def _add_type_hints(self, content: str) -> Tuple[str, List[str]]:
    if 'from typing import ' not in content and 'import typing' not in content:
        content = "from typing import List, Dict, Tuple, Optional, Any, Union\n" + content
        return content, ["Added typing imports"]
    return content, []

def _convert_to_fstrings(self, content: str) -> Tuple[str, int]:
    pattern = r'([\'"].*?[\'"])\s*\.\s*format\s*\((.*?)\)'
    count = 0

    def replace_format(match):
        nonlocal count
        string = match.group(1)[1:-1]
        args = match.group(2).strip()
        if re.match(r'^[\w\s,]+$', args):
            vars = [v.strip() for v in args.split(',')]
            new_str = f"f'{string.format(*['{' + v + '}' for v in vars])}'"
            count += 1
            return new_str
        return match.group(0)

    content = re.sub(pattern, replace_format, content)
    return content, count

def _modernize_python_features(self, content: str) -> Tuple[str, List[str]]:
    transformations = []
    if re.search(r'^print\s+[^(\n]', content, re.MULTILINE):
        content = re.sub(r'^print\s+(.+)$', r'print(\1)', content, flags=re.MULTILINE)
        transformations.append("Converted print statements to Python 3 style")
    return content, transformations
