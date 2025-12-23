import itertools
import numpy as np

class Vector:
    def __init__(self, components):
        self.components = np.array(components, dtype=float)

    def __sub__(self, other):
        return Vector(self.components - other.components)

    def __mul__(self, scalar):
        return Vector(self.components * scalar)

    def dot(self, other):
        return float(np.dot(self.components, other.components))

    def norm(self):
        return np.linalg.norm(self.components)

class E8Lattice:
    def __init__(self):
        # Generate all 240 roots of E8 lattice
        self.roots = self.generate_roots()

    def generate_roots(self):
        roots = []
        # Type 1 roots: permutations of (±1, ±1, 0,0,0,0,0,0)
        for i, j in itertools.combinations(range(8), 2):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    root = [^25_0]*8
                    root[i] = s1
                    root[j] = s2
                    roots.append(Vector(root))

        # Type 2 roots: (±½, ..., ±½) with even number of minus signs
        for signs in itertools.product([-0.5, 0.5], repeat=8):
            if sum(1 for s in signs if s < 0) % 2 == 0:
                roots.append(Vector(signs))

        return roots

    def reflect(self, v: Vector, alpha: Vector) -> Vector:
        dot_va = v.dot(alpha)
        dot_aa = alpha.dot(alpha)
        scale = 2 * dot_va / dot_aa
        reflected = v - (alpha * scale)
        return reflected

    def mirror_state(self, vector_3d: Vector) -> Vector:
        # Project 3D vector to 8D by padding with zeros
        vector_8d = Vector(list(vector_3d.components) + [0.0] * 5)

        # Select random root vector
        root = np.random.choice(self.roots)

        # Reflect 8D vector across chosen root
        reflected = self.reflect(vector_8d, root)

        # Project back to 3D by taking first three coordinates
        mirrored_3d = Vector(reflected.components[:3])
        return mirrored_3d
