def __init__(self, capacity: int = 1024, rng: Optional[random.Random] = None):
    self.capacity = int(capacity)
    self.buffer: deque = deque(maxlen=self.capacity)
    self.rng = rng or random.Random(0)

def append(self, item: Dict[str, Any]) -> None:
    self.buffer.append(item)

def sample(self, k: int = 32) -> List[Dict[str, Any]]:
    if not self.buffer:
        return []
    k = min(int(k), len(self.buffer))
    return self.rng.sample(list(self.buffer), k)

def __len__(self) -> int:
    return len(self.buffer)

