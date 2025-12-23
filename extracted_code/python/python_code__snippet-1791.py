"""
>>> bool(NoDists())
False

>>> list(NoDists()('anything'))
[]
"""

def __bool__(self):
    return False

def __call__(self, fullpath):
    return iter(())


