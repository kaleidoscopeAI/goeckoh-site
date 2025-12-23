def __init__(self, genome_size=128):
    self.genome = [AtomicBit() for _ in range(genome_size)]
    self.activation = sum(b.x for b in self.genome) / genome_size  # Mean bits
    self.embeddings = [sum(b.p[i] for b in self.genome) / genome_size for i in range(3)]  # Avg embed
    self.position = self.embeddings  # 3D from basis (sim identity)
    self.soft_prob = math.exp(-self.activation) / (1 + math.exp(-self.activation))  # Sigmoid flip prob

