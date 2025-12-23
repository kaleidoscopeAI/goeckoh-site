memory: List[Any] = field(default_factory=list)
memory_threshold: int = 10

def gather_data(self, data: Dict):
    self.memory.append(data)

