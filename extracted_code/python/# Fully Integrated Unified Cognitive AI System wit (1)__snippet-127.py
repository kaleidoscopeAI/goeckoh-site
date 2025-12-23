def __init__(self, seed=None):
    self.seed = seed
    self._rng = np.random.default_rng(seed)

def randint(self, low, high):
    return self._rng.integers(low, high)

def uniform(self, low, high):
    return self._rng.uniform(low, high)

