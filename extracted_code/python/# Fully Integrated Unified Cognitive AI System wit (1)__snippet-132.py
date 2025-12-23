def __init__(self):
    self.roots = self.generate_roots()

def generate_roots(self):
    roots = []
    for i, j in itertools.combinations(range(8), 2):
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                root = [0]*8
                root[i] = s1
                root[j] = s2
                roots.append(Vector(root))

    for signs in itertools.product([-0.5, 0.5], repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            roots.append(Vector(signs))

    return roots

def reflect(self, v: Vector, alpha: Vector) -> Vector:
    dot_va = v.dot(alpha)
    dot_aa = alpha.dot(alpha)
    scale = 2 * dot_va / dot_aa
    return v - (alpha * scale)

def mirror_state(self, vector_3d: Vector) -> Vector:
    vector_8d = Vector(list(vector_3d.components) + [0.0]*5)
    root = np.random.choice(self.roots)
    reflected = self.reflect(vector_8d, root)
    return Vector(reflected.components[:3])
