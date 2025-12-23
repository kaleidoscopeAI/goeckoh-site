weight: float
capacity: int
data: List[Any] = field(default_factory=list)
position: float = 0.0
connected_gears: List['MemoryGear'] = field(default_factory=list)

def add_data(self, item: Any) -> bool:
    if len(self.data) < self.capacity:
        self.data.append(item)
        self.position += 1.0 / self.capacity # Simulate rotation
        return True
    return False

