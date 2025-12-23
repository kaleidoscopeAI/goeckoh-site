def __init__(self):
    self.bits = [random.choice([0, 1]) for _ in range(128)]
    self.position = [random.uniform(-1, 1) for _ in range(3)]
    self.spin = random.choice([-1, 1])
    self.emotion = [0.0] * 5

