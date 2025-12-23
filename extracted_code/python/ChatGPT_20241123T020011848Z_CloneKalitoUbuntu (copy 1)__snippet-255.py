def __init__(self, traits):
    self.traits = traits

def mutate(self):
    # Generate a variant of DNA for child nodes
    return NodeDNA(traits=[trait + self.variation() for trait in self.traits])

def evolve_from_experience(self, experience, weight):
    # Adjust DNA traits based on experiences and weights
    self.traits = [trait + (experience * 0.1 * weight) for trait in self.traits]

def variation(self):
    # Simple placeholder for DNA variation during mutation
    return 0.1

