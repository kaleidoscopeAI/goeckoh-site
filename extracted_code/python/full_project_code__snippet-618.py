"""
Default filter. Subclass this class or use the `simplefilter`
decorator to create own filters.
"""

def __init__(self, **options):
    self.options = options

def filter(self, lexer, stream):
    raise NotImplementedError()


