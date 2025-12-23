"""
Indicates a list of literal words that is transformed into an optimized
regex that matches any of the words.

.. versionadded:: 2.0
"""
def __init__(self, words, prefix='', suffix=''):
    self.words = words
    self.prefix = prefix
    self.suffix = suffix

def get(self):
    return regex_opt(self.words, prefix=self.prefix, suffix=self.suffix)


