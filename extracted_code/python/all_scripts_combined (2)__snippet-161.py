def __init__(self):
    self.DA = 0.5
    self.Ser = 0.5
    self.NE = 0.5

def step(self, reward: float, mood_signal: float, arousal: float, dt: float = 0.1):
    self.DA += (0.9 * reward - 0.12 * self.DA) * dt
    self.Ser += (0.4 * mood_signal - 0.06 * self.Ser) * dt
    self.NE += (0.65 * arousal - 0.08 * self.NE) * dt
    self.DA = float(np.clip(self.DA, 0.0, 1.0))
    self.Ser = float(np.clip(self.Ser, 0.0, 1.0))
    self.NE = float(np.clip(self.NE, 0.0, 1.0))

def vector(self) -> List[float]:
    return [self.DA, self.Ser, self.NE]

