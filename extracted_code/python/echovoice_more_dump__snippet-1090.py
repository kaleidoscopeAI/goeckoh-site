def __init__(self, d0=64, d1=64, d2=16, d3=8):
    self.tensor = np.zeros((d0, d1, d2, d3), dtype=np.float32)

def update(self, idxs, value):
    d0, d1, d2, d3 = idxs
    self.tensor[d0, d1, d2, d3] = value

def marginalize(self, axis):
    return self.tensor.sum(axis=axis)

