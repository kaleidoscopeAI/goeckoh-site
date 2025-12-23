def __init__(self, components: List[float]):
    self.components = components

def __sub__(self, other: "Vector") -> "Vector":
    return Vector([a - b for a, b in zip(self.components, other.components)])

def __mul__(self, scalar: float) -> "Vector":
    return Vector([a * scalar for a in self.components])

def dot(self, other: "Vector") -> float:
    return sum(a * b for a, b in zip(self.components, other.components))

def norm(self) -> float:
    return sum(a * a for a in self.components) ** 0.5

