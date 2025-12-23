def __init__(self, traits):
    self.traits = traits

def evolve_from_experience(self, experience, weight):
    # Adjust traits based on experience and weight
    self.traits = [trait + (experience * 0.1 * weight) for trait in self.traits]

