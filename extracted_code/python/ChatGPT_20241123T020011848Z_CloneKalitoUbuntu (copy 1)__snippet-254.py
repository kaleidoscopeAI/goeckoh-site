def __init__(self, node_id, dna):
    self.node_id = node_id
    self.dna = dna
    self.memory = {}  # Memory to store experiences with weights

def learn(self, experience, impact=1):
    # Add experience to memory with a weight (impact level)
    if experience in self.memory:
        self.memory[experience] += impact
    else:
        self.memory[experience] = impact
    self.adapt()

def adapt(self):
    # Adjust DNA based on weighted experiences in memory
    for experience, weight in self.memory.items():
        self.dna.evolve_from_experience(experience, weight)

def replicate(self):
    # Replicate with slight DNA adaptation
    new_dna = self.dna.mutate()
    return OrganicCore(node_id=self.node_id + "_child", dna=new_dna)


