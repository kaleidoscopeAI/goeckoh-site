"""
Exception thrown by :class:`ParserElement.validate` if the
grammar could be left-recursive; parser may need to enable
left recursion using :class:`ParserElement.enable_left_recursion<ParserElement.enable_left_recursion>`
"""

def __init__(self, parseElementList):
    self.parseElementTrace = parseElementList

def __str__(self) -> str:
    return f"RecursiveGrammarException: {self.parseElementTrace}"


