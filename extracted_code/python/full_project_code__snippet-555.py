"""Base exception for errors caused within a pool."""

def __init__(self, pool, message):
    self.pool = pool
    HTTPError.__init__(self, "%s: %s" % (pool, message))

def __reduce__(self):
    # For pickling purposes.
    return self.__class__, (None, None)


