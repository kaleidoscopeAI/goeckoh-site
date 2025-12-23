"""Abstract :class:`ParserElement` subclass, for defining atomic
matching patterns.
"""

def __init__(self):
    super().__init__(savelist=False)

def _generateDefaultName(self) -> str:
    return type(self).__name__


