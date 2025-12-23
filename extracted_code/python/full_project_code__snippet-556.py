"""Base exception for PoolErrors that have associated URLs."""

def __init__(self, pool, url, message):
    self.url = url
    PoolError.__init__(self, pool, message)

def __reduce__(self):
    # For pickling purposes.
    return self.__class__, (None, self.url, None)


