"""
Given an iterable of lines from a Metadata file, return
the value of the Version field, if present, or None otherwise.
"""

def is_version_line(line):
    return line.lower().startswith('version:')

version_lines = filter(is_version_line, lines)
line = next(iter(version_lines), '')
_, _, value = line.partition(':')
return safe_version(value.strip()) or None


