"""Generic class to defer some work.

Handled specially in RegexLexerMeta, to support regex string construction at
first use.
"""
def get(self):
    raise NotImplementedError


