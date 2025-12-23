"""Position in source code"""
line: int
column: int

def __str__(self) -> str:
    return f"{self.line}:{self.column}"

