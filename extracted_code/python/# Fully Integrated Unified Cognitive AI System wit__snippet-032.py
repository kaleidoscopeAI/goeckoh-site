from perspective_engine import PerspectiveEngine
from emotional_transformer import EmotionalTransformer

class KaleidoscopeEngine:
    def __init__(self, nodes, grid, rng, r_dim=3, dt=0.01, seed=None):
        self.rng = rng
        self.nodes = nodes
        self.grid = grid
        self.r_dim = r_dim
        self.dt = dt

        self.e8_lattice = E8Lattice(seed=self.rng.randint(0, 100000))
        self.perspective_engine = PerspectiveEngine(self.e8_lattice, self.rng)
        self.emotional_transformer = EmotionalTransformer()

        # Other initializations...
