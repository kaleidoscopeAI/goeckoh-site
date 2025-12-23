def __init__(self, seed: int = 0):
    self.rng = random.Random(int(seed))
    self.generation = 0

async def evolve_population(self, nodes: Dict[str, Any]) -> None:
    if len(nodes) < 2:
        return
    self.generation += 1
    items = list(nodes.items())
    def _tournament(k: int = 3):
        chosen = self.rng.sample(items, min(k, len(items)))
        chosen = sorted(chosen, key=lambda kv: len(getattr(kv[1], '_local_memory', [])), reverse=True)
        return chosen[0][1]
    parent1 = _tournament()
    parent2 = _tournament()
    for node_id, node in nodes.items():
        if self.rng.random() < 0.1:
            for trait in ['valence_bias', 'arousal_sensitivity', 'coherence_preference']:
                v1 = float(parent1.emotional_traits.get(trait, 0.0))
                v2 = float(parent2.emotional_traits.get(trait, 0.0))
                node.emotional_traits[trait] = max(-1.0, min(1.0, 0.5 * (v1 + v2) + (self.rng.random() - 0.5) * 0.01))

async def calculate_potential(self) -> float:
    return float(self.rng.random())

async def get_generation(self) -> int:
    return int(self.generation)
