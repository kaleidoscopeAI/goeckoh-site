def __init__(self, node_id: str, dna: NodeDNA):
    self.id = node_id
    self.dna = dna
    self.energy = 1.0
    self.memory = []

def metabolize(self, stimulus: float):
    # Grow or decay based on stimulus (entropy)
    self.energy += stimulus * self.dna.traits[0] # Trait 0 = Metabolism efficiency
    self.energy = max(0.0, min(10.0, self.energy))

def replicate(self) -> Optional['OrganicNode']:
    if self.energy > 8.0:
        self.energy *= 0.5
        return OrganicNode(f"{self.id}_child", self.dna.mutate())
    return None

