"""
A pseudo match object constructed from a string.
"""

def __init__(self, start, text):
    self._text = text
    self._start = start

def start(self, arg=None):
    return self._start

def end(self, arg=None):
    return self._start + len(self._text)

def group(self, arg=None):
    if arg:
        raise IndexError('No such group')
    return self._text

def groups(self):
    return (self._text,)

def groupdict(self):
    return {}


