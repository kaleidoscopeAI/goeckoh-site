"""
A simple cache mapping names and .dist-info paths to distributions
"""

def __init__(self):
    """
    Initialise an instance. There is normally one for each DistributionPath.
    """
    self.name = {}
    self.path = {}
    self.generated = False

def clear(self):
    """
    Clear the cache, setting it to its initial state.
    """
    self.name.clear()
    self.path.clear()
    self.generated = False

def add(self, dist):
    """
    Add a distribution to the cache.
    :param dist: The distribution to add.
    """
    if dist.path not in self.path:
        self.path[dist.path] = dist
        self.name.setdefault(dist.key, []).append(dist)


