def __init__(self):
    self.x = random.choice([0, 1])  # State 0/1
    self.s = random.uniform(0, 1)  # Confidence
    self.p = [random.uniform(-1, 1) for _ in range(3)]  # Embedding R^3

def binarize(self):
    return 1 if self.s > 0.5 else 0  # Threshold rule (sign sim if float)

