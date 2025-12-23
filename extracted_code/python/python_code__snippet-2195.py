"""Single node in the emotional regulation lattice"""
def __init__(self, node_id: int):
    self.id = node_id
    self.emotion = np.zeros(5, dtype=np.float32)  # [a, v, d, c, r]
    self.energy = 0.0
    self.neighbors = []
    self.weights = []

def compute_local_stress(self) -> float:
    """Tension from neighbors"""
    if not self.neighbors:
        return 0.0
    tension = sum(
        w * np.linalg.norm(self.emotion - n.emotion)
        for n, w in zip(self.neighbors, self.weights)
    )
    return tension / len(self.neighbors)


