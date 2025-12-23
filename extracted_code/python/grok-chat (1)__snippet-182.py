def __init__(self):
    self.bits = [random.choice([0, 1]) for _ in range(128)]  # Genome
    self.position = [random.uniform(-1, 1) for _ in range(3)]  # 3D
    self.spin = random.choice([-1, 1])  # Ising
    self.emotion = [0.0] * 5  # Arousal, valence, etc.

