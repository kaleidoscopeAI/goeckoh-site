def __init__(self, node_id, dna, position):
    self.node_id = node_id  # Unique ID for the node
    self.dna = dna  # DNA of the node
    self.position = position  # 2D position of the node
    self.fitness = None  # Fitness score
    self.logs = []  # Decision logs

def evaluate_fitness(self, target):
    self.fitness = -np.linalg.norm(self.dna - target)
    self.logs.append(f"Fitness evaluated: {self.fitness:.2f}")

def mutate(self, mutation_rate):
    for i in range(len(self.dna)):
        if np.random.rand() < mutation_rate:
            self.dna[i] += np.random.normal(0, 0.1)
    self.dna = np.clip(self.dna, 0, 1)
    self.logs.append(f"Mutated DNA: {self.dna}")

def replicate(self, mutation_rate, node_id_counter):
    child_dna = self.dna.copy()
    for i in range(len(child_dna)):
        if np.random.rand() < mutation_rate:
            child_dna[i] += np.random.normal(0, 0.1)
    child_position = self.position + np.random.uniform(-0.1, 0.1, size=2)
    child_position = np.clip(child_position, 0, 1)
    self.logs.append(f"Replicated to position: {child_position}")
    return Node(node_id_counter, dna=np.clip(child_dna, 0, 1), position=child_position)

