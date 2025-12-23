import time
import math
import random
import hashlib  # Built-in SHA

# Definitions from Mindmap: Layer 1 Atomic
class AtomicBit:
    def __init__(self):
        self.x = random.choice([0, 1])  # State 0/1
        self.s = random.uniform(0, 1)  # Confidence
        self.p = [random.uniform(-1, 1) for _ in range(3)]  # Embedding R^3

    def binarize(self):
        return 1 if self.s > 0.5 else 0  # Threshold rule (sign sim if float)

# Layer 2 Node: Genome mean activation, embeddings avg, pos from basis
class Node:
    def __init__(self, genome_size=128):
        self.genome = [AtomicBit() for _ in range(genome_size)]
        self.activation = sum(b.x for b in self.genome) / genome_size  # Mean bits
        self.embeddings = [sum(b.p[i] for b in self.genome) / genome_size for i in range(3)]  # Avg embed
        self.position = self.embeddings  # 3D from basis (sim identity)
        self.soft_prob = math.exp(-self.activation) / (1 + math.exp(-self.activation))  # Sigmoid flip prob

