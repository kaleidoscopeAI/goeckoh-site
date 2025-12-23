from vector import Vector
from e8_lattice import E8Lattice

class PerspectiveEngine:
    def __init__(self, e8_lattice: E8Lattice, rng, k_mirror=0.5):
        self.e8_lattice = e8_lattice
        self.rng = rng
        self.k_mirror = k_mirror

    def generate_hypothesis(self, node_r: Vector) -> Vector:
        hypothesis_r = self.e8_lattice.mirror_state(node_r)
        return hypothesis_r

    def evaluate_hypothesis(self, current_node_r: Vector, hypothesis_r: Vector,
                            node_E: float, node_K: float, emotional_state) -> float:
        distance_to_hypothesis = (current_node_r - hypothesis_r).norm()
        confidence = (distance_to_hypothesis * 0.1) + (node_K * 0.5) - (node_E * 0.2)
        confidence += emotional_state.valence * 0.1
        confidence += emotional_state.coherence * 0.15
        return max(0.0, min(1.0, confidence))
