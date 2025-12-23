def __init__(self, node_id: str, pos: np.ndarray = None, fixed=False):
    self.id = node_id
    self.pos = np.array(pos) if pos is not None else np.random.randn(3).astype(float)
    self.fixed = fixed
    self.energy = 1.0
    self.valence = 0.0
    self.arousal = 0.0
    self.bonds: List[Bond] = []

def add_bond(self, b: Bond):
    self.bonds.append(b)

