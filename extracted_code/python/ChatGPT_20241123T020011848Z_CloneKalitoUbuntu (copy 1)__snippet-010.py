class OrganicCore:
    def __init__(self, node_id, dna):
        self.node_id = node_id
        self.dna = dna
        self.memory = {}  # Memory now stores experiences with weights

    def learn(self, experience, impact=1):
        # Add experience to memory with a weight (impact level)
        if experience in self.memory:
            self.memory[experience] += impact
        else:
            self.memory[experience] = impact
        self.adapt()

    def adapt(self):
        # Adjust DNA based on the weighted experiences in memory
        for experience, weight in self.memory.items():
            self.dna.evolve_from_experience(experience, weight)

class NodeDNA:
    def __init__(self, traits):
        self.traits = traits

    def evolve_from_experience(self, experience, weight):
        # Adjust traits based on experience and weight
        self.traits = [trait + (experience * 0.1 * weight) for trait in self.traits]

