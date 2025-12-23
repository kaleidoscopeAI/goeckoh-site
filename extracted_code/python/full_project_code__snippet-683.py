"""
Wrapper for parse actions, to ensure they are only called once.
"""

def __init__(self, method_call):
    from .core import _trim_arity

    self.callable = _trim_arity(method_call)
    self.called = False

def __call__(self, s, l, t):
    if not self.called:
        results = self.callable(s, l, t)
        self.called = True
        return results
    raise ParseException(s, l, "OnlyOnce obj called multiple times w/out reset")

def reset(self):
    """
    Allow the associated parse action to be called once more.
    """

    self.called = False


