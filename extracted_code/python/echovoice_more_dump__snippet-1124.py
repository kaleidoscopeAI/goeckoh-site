def __init__(self, alpha: float = 0.2):
    self.alpha = float(alpha)
    self.estimate: Optional[float] = None
    self.squared: float = 0.0
    self.count: int = 0

def update(self, value: float) -> float:
    value = float(value)
    if self.estimate is None:
        self.estimate = value
        self.squared = value * value
        self.count = 1
    else:
        self.estimate = self.alpha * value + (1 - self.alpha) * self.estimate
        self.squared = self.alpha * (value * value) + (1 - self.alpha) * self.squared
        self.count += 1
    return float(self.estimate)

def variance(self) -> float:
    if self.count <= 1 or self.estimate is None:
        return 0.0
    return max(0.0, self.squared - self.estimate * self.estimate)

def predict(self) -> float:
    return float(self.estimate) if self.estimate is not None else 0.0

