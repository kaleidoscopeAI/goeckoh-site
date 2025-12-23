"""
Represents the genetic code of a node, influencing its behavior and capabilities.
"""
learning_rate: float = 0.1
mutation_rate: float = 0.01
energy_efficiency: float = 1.0
memory_capacity: int = 100
initial_energy: float = 100.0
replication_threshold: float = 80.0
min_memory_for_replication: int = 50
energy_consumption_rate: float = 0.1

def mutate(self) -> 'GeneticCode':
    """
    Creates a new instance of GeneticCode with slight mutations.
    """
    mutation_factor = 0.1  # Adjust for more or less drastic mutations

    new_code = GeneticCode(
        learning_rate=max(0.01, self.learning_rate + random.uniform(-mutation_factor, mutation_factor)),
        mutation_rate=max(0.001, self.mutation_rate + random.uniform(-0.005, 0.005)),
        energy_efficiency=max(0.1, self.energy_efficiency + random.uniform(-0.1, 0.1)),
        memory_capacity=int(max(50, self.memory_capacity + random.uniform(-50, 50))),
        initial_energy=max(10.0, self.initial_energy + random.uniform(-10, 10)),
        replication_threshold=max(50.0, self.replication_threshold + random.uniform(-5, 5)),
        min_memory_for_replication=int(max(10, self.min_memory_for_replication + random.uniform(-5, 5))),
        energy_consumption_rate=max(0.01, self.energy_consumption_rate + random.uniform(-0.01, 0.01))
    )

    return new_code


