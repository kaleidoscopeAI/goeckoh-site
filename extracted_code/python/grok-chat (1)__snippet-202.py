def __init__(self, x=random.choice([0, 1]), s=random.uniform(0, 1), p=None):
    self.x = x
    self.s = s
    self.p = p or [random.uniform(-1, 1) for _ in range(3)]

def binarize(self):
    return 1 if self.s > 0.5 else 0

