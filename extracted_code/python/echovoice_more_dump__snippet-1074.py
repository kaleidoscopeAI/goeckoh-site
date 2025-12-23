def __init__(self, a: str, b: str, rest_length: float = 1.0, stiffness: float = 1.0, validated: bool = False):
    self.a = a
    self.b = b
    self.rest_length = rest_length
    self.stiffness = stiffness
    self.validated = validated
    self.tension = 0.0

def compute_tension(self, pos_a: np.ndarray, pos_b: np.ndarray):
    dist = float(np.linalg.norm(pos_a - pos_b))
    self.tension = self.stiffness * max(0.0, abs(dist - self.rest_length))
    return self.tension

