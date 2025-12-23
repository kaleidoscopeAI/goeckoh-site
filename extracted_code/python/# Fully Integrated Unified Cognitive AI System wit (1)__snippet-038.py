import numpy as np

class Vector:
    def __init__(self, components):
        self.components = np.array(components, dtype=np.float64)

    def __sub__(self, other):
        return Vector(self.components - other.components)

    def __mul__(self, scalar):
        return Vector(self.components * scalar)

    def dot(self, other):
        return np.dot(self.components, other.components)

    def norm(self):
        return np.linalg.norm(self.components)
