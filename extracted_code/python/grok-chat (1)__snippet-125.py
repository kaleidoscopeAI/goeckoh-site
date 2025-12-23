class Node:
    def __init__(self, genome_size=128):
        self.genome = [AtomicBit() for _ in range(genome_size)]
        self.activation = sum(b.x for b in self.genome) / genome_size
        self.embeddings = [sum(b.p[i] for b in self.genome) / genome_size for i in range(3)]
        self.position = self.embeddings
        self.soft_prob = 1 / (1 + math.exp(-self.activation))

