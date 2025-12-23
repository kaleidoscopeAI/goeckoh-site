def __init__(self, id):
    self.id = id
    self.position = [random.uniform(-1, 1) for _ in range(3)]
    self.energy = random.uniform(0, 1)
    self.awareness = random.uniform(0, 1)
    self.knowledge = random.uniform(0, 1)
    self.emotional_state = {'valence': random.uniform(-1, 1), 'arousal': random.uniform(0, 1)}
    self.quantum_state = [random.uniform(0, 1) for _ in range(2)]  # Simplified qubit

def state_vector(self):
    vec = self.position + [self.energy, self.awareness, self.knowledge, self.emotional_state['valence'], self.emotional_state['arousal']] + self.quantum_state
    # Bit-level binarization (from September 28)
    binarized = [1 if v > 0.5 else 0 for v in vec]  # Thresholding
    return binarized

