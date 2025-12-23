class OrganicCore:
    def __init__(self, node_id, dna):
        self.node_id = node_id
        self.dna = dna
        self.memory = []  # Memory to store experiences

    def replicate(self):
        # Replicate with slight adaptation in DNA
        new_dna = self.dna.mutate()
        return OrganicCore(node_id=self.node_id + "_child", dna=new_dna)

    def learn(self, experience):
        self.memory.append(experience)
        self.adapt()

    def adapt(self):
        # Adjust DNA based on accumulated experiences
        self.dna.evolve_from_experiences(self.memory)


class NodeDNA:
    def __init__(self, traits):
        self.traits = traits

    def mutate(self):
        # Produce a variant of DNA for child nodes
        return NodeDNA(traits=[trait + self.variation() for trait in self.traits])

    def evolve_from_experiences(self, experiences):
        # Modify DNA traits based on experiences
        self.traits = [self.adjust_trait(trait, experiences) for trait in self.traits]

    def variation(self):
        # Placeholder function for DNA trait variation
        return 0.1

    def adjust_trait(self, trait, experiences):
        # Adjust trait based on experiences
        return trait + sum(experiences) * 0.05  # Simple learning adjustment

