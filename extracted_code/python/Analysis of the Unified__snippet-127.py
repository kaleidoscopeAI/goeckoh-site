def __init__(self, lattice):
    self.lattice = lattice

def generate_hypothesis(self, node_state):
    try:
        perspectives = [self.lattice.mirror_state(node_state[:3]) for _ in range(3)]
        score = sum(math.cos(sum(p)) for p in perspectives) / 3  # Valence-based
        return f"Hypothesis: {score > 0.5}"
    except:
        return "Default hypothesis"

