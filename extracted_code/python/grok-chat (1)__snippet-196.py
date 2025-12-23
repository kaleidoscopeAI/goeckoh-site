def __init__(self, node1, node2, weight=1.0):
    self.node1 = node1
    self.node2 = node2
    self.weight = weight

def similarity(self):
    hamming = sum(b1.x != b2.x for b1, b2 in zip(self.node1.genome, self.node2.genome)) / len(self.node1.genome)
    return 1 - hamming

def energy(self):
    bit_term = self.weight * (1 - self.similarity())
    spatial_term = self.weight * sum((p1 - p2)**2 for p1, p2 in zip(self.node1.position, self.node2.position))
    return bit_term + spatial_term

