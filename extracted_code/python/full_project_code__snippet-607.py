"""
A helper object that holds lexer position data.
"""

def __init__(self, text, pos, stack=None, end=None):
    self.text = text
    self.pos = pos
    self.end = end or len(text)  # end=0 not supported ;-)
    self.stack = stack or ['root']

def __repr__(self):
    return 'LexerContext(%r, %r, %r)' % (
        self.text, self.pos, self.stack)


